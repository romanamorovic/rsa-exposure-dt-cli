
# ------------------------------------------------------------------------------
# UTILITY FUNCTIONS USED THROUGHOUT THE NOTEBOOKS



import logging
import os
import pickle
import re 
import sys
import textwrap
import warnings

# import abnumber
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm

import bin.params as p



# ------------------------------------------------------------------------------
# CONSTANTS



AHO_SCHEME_REGIONS = dict(
    CDR1 = dict(start=27, end=40), 
    CDR2 = dict(start=58, end=68),
    CDR3 = dict(start=107, end=138)
)

CDR_REGION_NAMES = ['CDR1', 'CDR2', 'CDR3']
CDR_TEXT_OFFSETS = [9, 8, 19]
CDR_TEXT_OFFSETS_X = [3, 2, 8]

Y_PLOT_LABEL_MAX_CHARS_PER_ROW = 30

PATHS_BASED_ON_DATASET = {
    'classic': {
        'FASTA_ALIGNED_CLEANED_DIR_PATH': f'{p.DATA_DIR}/csv/fasta_aligned_cleaned',
        'FASTA_ALIGNED_CLEANED_FIN_DIR_PATH': f'{p.DATA_DIR}/csv/fasta_aligned_cleaned',
         #'FASTA_ALIGNED_CLEANED_DL_DIR_PATH': f'{p.DATA_DIR}/csv/fasta_aligned_cleaned_dl', TODO
        'FASTA_ALIGNED_CLEANED_DL_DIR_PATH': f'{p.DATA_DIR}/csv/fasta_aligned_cleaned',
        'SASA_ALIGNED_DIR_PATH': f'{p.DATA_DIR}/csv/sasa_aligned',
        'SASA_ALIGNED_DL_DIR_PATH': f'{p.DATA_DIR}/csv/sasa_aligned',
        'SASA_ALIGNED_FIN_DIR_PATH': f'{p.DATA_DIR}/csv/sasa_aligned',
         #'SASA_ALIGNED_DL_DIR_PATH': f'{p.DATA_DIR}/csv/sasa_aligned_dl',
         #'SASA_ALIGNED_FIN_DIR_PATH': f'{p.DATA_DIR}/csv/sasa_aligned_fin',
        'METADATA_DIR_PATH': f'{p.DATA_DIR}/csv/metadata'
    },
    'test_new_234': {
        'FASTA_ALIGNED_CLEANED_DIR_PATH': f'{p.DATA_DIR}/csv/fasta_aligned_cleanedJuly2024',
        'FASTA_ALIGNED_CLEANED_DL_DIR_PATH': f'{p.DATA_DIR}/csv/fasta_aligned_cleaned_dlJuly2024',
        'FASTA_ALIGNED_CLEANED_FIN_DIR_PATH': f'{p.DATA_DIR}/csv/fasta_aligned_cleaned_finJuly2024',
        'SASA_ALIGNED_DIR_PATH': f'{p.DATA_DIR}/csv/sasa_alignedJuly2024',
        'SASA_ALIGNED_DL_DIR_PATH': f'{p.DATA_DIR}/csv/sasa_aligned_dlJuly2024',
        'SASA_ALIGNED_FIN_DIR_PATH': f'{p.DATA_DIR}/csv/sasa_aligned_finJuly2024',
        'METADATA_DIR_PATH': f'{p.DATA_DIR}/csv/metadataJuly2024'
    },
    'ib_aho': {
        'FASTA_ALIGNED_CLEANED_DIR_PATH': f'{p.DATA_DIR}/csv/fasta_alignedIB2July2024',
        
    }
}

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------



def select_only_chain_sequences(df: pd.DataFrame, chain_id: str) -> pd.DataFrame:
    """
    Select only chains of one specific type from the 'df' 
        - as defined by 'chain_id' parameter value
    """
    if chain_id not in ('L', 'H'):
        raise ValueError('`chain_id` must be either L or H')
        
    chain_keys = [index_key for index_key in df.index 
                  if index_key.endswith(':' + chain_id.upper())]
    chain_df = df.loc[chain_keys, :]
    return chain_df
    
    
def nondash_counts(df: pd.DataFrame) -> pd.Series:
    """
    Get number of residues present within each of the samples
    """
    return df.count(axis=1) - df.apply(lambda x: x.str.contains('-').sum(), axis=1)

def nondash_counts_columns(fasta_df):
    """Get numbers of occupied residues per position column"""
    all_counts = fasta_df.count(axis=0)
    gap_counts = fasta_df.apply(lambda x: x.str.contains('-').sum(), axis=0)
    return all_counts - gap_counts

def get_empty_fasta_positions(df) -> list:
    ncc = nondash_counts_columns(df)
    zero_columns_series = ncc[ncc == 0]
    zero_columns = zero_columns_series.keys()
    return list(zero_columns)


def wrap(text: str, n: int = Y_PLOT_LABEL_MAX_CHARS_PER_ROW):
    return '\n'.join(textwrap.wrap(text, n))


# ------------------------------------------------------------------------------
# DATA METRICS


def avg_deviations(actual, predictions, axis: int = 1):
    """
    Scoring function that compute mean loss for each of the data-points.
    The loss is defined as mean difference between pairs of individual 
       positions in `actual` and `prediction` SASA dataframes.
    `axis` parameter defines what parts of the DFs shall be compared:
       - if set to 0 - ANARCI positions (DF columns) are compared
       - if set to 1 - chains/samples (DF rows) are compared
    """    
    # We want `actual` and `predictions` to be of the same datatype
    # Convert `actual` to `np.ndarray` if the types do not match
    if isinstance(actual, pd.DataFrame) and isinstance(predictions, np.ndarray):
        actual = actual.to_numpy()
    
    # Replace NaN by default values - to easen up the score computation
    if type(predictions) is np.ndarray:
        if len(predictions.shape) == 2 and predictions.shape[1] == 1:
            # some models do provide different shape of predicts
            predictions = predictions.flatten()       
        predictions = np.nan_to_num(predictions, nan=p.SASA_NAN_DEFAULT)
    else:
        predictions = predictions.fillna(p.SASA_NAN_DEFAULT)
        
    # Compute deviations
    if type(actual) is np.ndarray:
        if actual.ndim == 2 and actual.shape[1] == 1:
            # it is nicer to have such array flattened
            actual = actual.flatten()
        # numpy array - some internet-made models do not use pandas
        actual = np.nan_to_num(actual, nan=p.SASA_NAN_DEFAULT)
        #raise ValueError(f'actual shape {actual.shape} preds shape {predictions.shape}')
        diffs = np.abs(actual - predictions)
        deviations_per_record = np.mean(diffs, axis=axis) if diffs.ndim > 1 else diffs
    else:
        # pandas dataframe
        actual = actual.fillna(p.SASA_NAN_DEFAULT)
        #print('inside scoring', actual.shape, predictions.shape)
        #raise ValueError(f'actual {type(actual)}, predictions {type(predictions)}')
        differences = actual.subtract(predictions).abs()
        deviations_per_record = differences.mean(axis=axis)
    
    return deviations_per_record   


def avg_deviation(actual, predictions) -> float:
    """
    Return single-value total loss for all the `predictions` vs. `actual`
    """
    devs = avg_deviations(actual, predictions)
    if type(devs) is np.ndarray:
        return np.mean(devs)
    else: 
        # pandas df
        return devs.mean()

 
# ------------------------------------------------------------------------------
# ONE-HOT ENCODING OF THE PROTEIN MATRIX


AAs = 'ACDEFGHIKLMNOPQRSTUVWY-'
aa_to_int = dict((c, i) for i, c in enumerate(AAs))


def _one_hot_encode_protein(seq):
    """
    (function by Kveta Brazdilova)
    One-hot encode one protein sequence
    """
    int_seq = [aa_to_int[aa] for aa in seq]
    onehot_encoded = []
    for val in int_seq:
        vect = [0 for i in range(len(AAs) - 1)]
        if val != 22:
            vect[val] = 1
        onehot_encoded += vect
    return np.asarray(onehot_encoded)


def _encode_proteins(data):
    """
    (function by Kveta Brazdilova)
    One-hot encode the whole protein matrix
    """
    res = data.apply(_one_hot_encode_protein, axis=1, result_type="expand")
    return res


def _get_col_names(n):
    """
    (function by Kveta Brazdilova)
    Generate column names for one-hot encoding of protein sequence with length 'n'
    """
    names = []
    for i in range(n):
        letters = [f"{ch}_{i + 1}" for ch in "ACDEFGHIKLMNOPQRSTUVWY"]
        names += letters
    return names


def get_oh_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    One-hot encode `df` matrix. 
    Do not one-hot-encode columns that are not the protein residues itself
    This is higher-level wrapper function for the one-hot-encoding, and thus this
       is the function that should be used in other notebooks
    """
    if 'Id' in df.columns:
        df_oh = _encode_proteins(df.iloc[:, 1:])
        df_oh.insert(0, "Id", df["Id"])
        df_oh.columns = ["Id"] + _get_col_names(len(df.columns) - 1)
        return df_oh
    else:
        df_oh = _encode_proteins(df)
        return df_oh


# ------------------------------------------------------------------------------
# DATA CLEANUP


def anarci_column_sorter(col_name):
    """
    Internal function that generates sort keys for built-in SORTED call
        for dataframe ANARCI position column sorting
    ANARCI position number has much higher weight (1000000 fold) than 
        subposition
    """
    position = re.findall(r'\d+', col_name)[0]
    subposition = re.findall(r'[A-Z]', col_name)
    if subposition:
        subposition = ord(subposition[0]) - ord('A')
    else:
        subposition = 0
    score = int(position) * 1_000_000 + subposition
    return score


def sort_numbering_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    In some cases, one might end up with DataFrame columns sorted in the following
        way: 1 2 3 ... 149 26A 85A 85B (i.e. all position columns containing subposition (letter)
        are put at the back
    However, the correct order of the columns is 1 2 3 ... 26A 85A 85B ... 149
    This function will reorder the dataframe columns so the order is correct.
    New dataframe with fixed columns order is returned [NOT INPLACE]
    """
    reordered_columns = sorted(df.columns, key=anarci_column_sorter)
    reordered_df = df.reindex(reordered_columns, axis=1)
    
    return reordered_df
    
    
def remove_useless_columns(train_data: pd.DataFrame, test_data: pd.DataFrame = None):
    """
    Remove columns that contain only zeros. 
    Doing so is expected to speed up the model execution,
        while preserving all the information in data
    """
    train_zero_cols = train_data.columns[train_data.sum() == 0]    
    train_cleaned = train_data.drop(columns=train_zero_cols, errors='ignore')
    if test_data:
        test_cleaned = test_data.drop(columns=train_zero_cols, errors='ignore')
        return train_cleaned, test_cleaned
    else:
        return train_cleaned
    

def handle_sasa_matrix(Y: pd.DataFrame) -> pd.DataFrame:
    """
    Replace NaNs and 0 within SASA matrix by other numerical values,
    so the models will not complain about Y data they get. 
    """
    Y = Y.fillna(-1)
    Y = Y.replace(0, 0.00001)
    return Y


def drop_nondata_columns(dataframes: list) -> list:
    """
    Remove the columns that are not used by the machine learning model
    """
    if isinstance(dataframes, pd.DataFrame):
        dataframes = [dataframes]
        
    new_dataframes = []
    for df in dataframes:
        df_new = df.drop(columns='Id', errors='ignore')
        new_dataframes.append(df_new)
        
    if len(new_dataframes) == 1:
        return new_dataframes[0]
        
    return new_dataframes


# ------------------------------------------------------------------------------
# SETUP FUNCTIONS - LOAD DATASETS, INITIALIZE THE ENVIRONMENT

def load_dataset(dataset, chains='L', options=dict()) -> tuple:
    """
    Load the dataset as defined in 'metadata.csv' (there is 'dataset' column). 
    We select only the data samples the 'dataset' column value of which matches the 'dataset'
       parameter in their respective record in 'metadata.csv' 
    `dataset` may be of type `string` if one wishes to load just one dataset, or
       tuple if more than one dataset is desired (e.g. ['train', 'val'] ))
    """
    def _get_dir_path(datatype, pathkey, options):
        # options: avoid_dl_removal -> bool
        # options: get_raw_data -> bool
        if datatype == 'fasta':
            reserve_dir_key = 'FASTA_ALIGNED_CLEANED_DIR_PATH'
            first_dir_key = 'FASTA_ALIGNED_CLEANED_DL_DIR_PATH'
            if options.get('get_raw_data', False):
                first_dir_key = 'FASTA_ALIGNED_CLEANED_DIR_PATH'
            elif pathkey == 'test_new_234' and options.get('avoid_dl_removal', False):
                first_dir_key = 'FASTA_ALIGNED_CLEANED_FIN_DIR_PATH'
            return PATHS_BASED_ON_DATASET[path_key].get(first_dir_key, PATHS_BASED_ON_DATASET[path_key][reserve_dir_key])
        elif datatype == 'sasa':
            reserve_dir_key = 'SASA_ALIGNED_DIR_PATH'
            first_dir_key = 'SASA_ALIGNED_DL_DIR_PATH'
            if options.get('get_raw_data', False):
                first_dir_key = 'SASA_ALIGNED_DIR_PATH'
            elif pathkey == 'test_new_234' and options.get('avoid_dl_removal', False):
                first_dir_key = 'SASA_ALIGNED_FIN_DIR_PATH'
            return PATHS_BASED_ON_DATASET[path_key].get(first_dir_key, PATHS_BASED_ON_DATASET[path_key][reserve_dir_key])
        else:
            raise ValueError('datatype must be fasta/sasa')

    # convert ['test'] to 'test' (unpack the array) if that was the case
    if type(dataset) == list and len(dataset) == 1:
        dataset = dataset[0]

    if chains == 'all':
        raise ValueError('chains="all" not yet supported')
    if chains not in ('L', 'H', ):
        # TODO add 'all' here later on
        raise ValueError('`chains` must either be "L", "H", "all" (the last one not yet supported)')
        
    if dataset == 'test_new_234':
        path_key = 'test_new_234'
    else: 
        path_key = 'classic'

    METADATA_DIR_PATH = PATHS_BASED_ON_DATASET[path_key]['METADATA_DIR_PATH']
    FASTA_ALIGNED_CLEANED_DIR_PATH = _get_dir_path('fasta', path_key, options)
    SASA_ALIGNED_DIR_PATH = _get_dir_path('sasa', path_key, options)

    metadata_file_path = f'{METADATA_DIR_PATH}/metadata_{chains}.csv'
    metadata_df = pd.read_csv(metadata_file_path, index_col=0)

    X_file_path = f'{FASTA_ALIGNED_CLEANED_DIR_PATH}/fasta_{p.FINAL_NUMBERING_SCHEME}_{chains}.csv'
    X = pd.read_csv(X_file_path)

    Y_file_path = f'{SASA_ALIGNED_DIR_PATH}/sasa_{chains}.csv'
    Y = pd.read_csv(Y_file_path).rename(columns={'Unnamed: 0':'Id'})

    tm_df = None
    if type(dataset) == str:
        tm_df = metadata_df if dataset == '*' else metadata_df[metadata_df['dataset'] == dataset]
    elif type(dataset) in (tuple, list):
        tm_df = metadata_df[metadata_df.dataset.isin(dataset)]
        
    tm_df.columns = ['tm_' + c for c in tm_df.columns]
    X = X.merge(tm_df, left_on='Id', right_index=True).drop(columns=tm_df.columns).reset_index(drop=True)
    Y = Y.merge(tm_df, left_on='Id', right_index=True).drop(columns=tm_df.columns).reset_index(drop=True)
    assert X.shape == Y.shape and X.shape[0] <= tm_df.shape[0], f'X.shape: {X.shape}, Y.shape: {Y.shape}, M.shape: {tm_df.shape}'

    print(f'load_dataset: {dataset}, metadata file path: {metadata_file_path}, chains: {chains}, shape: {tm_df.shape}')
    print(f'load_dataset: {dataset}, X file path: {X_file_path}, chains: {chains}, shape: {X.shape}')
    print(f'load_dataset: {dataset}, Y file path: {Y_file_path}, chains: {chains}, shape: {Y.shape}')
    return X, Y


def include_clusters(clusters_df: pd.DataFrame, 
                     X: pd.DataFrame,
                     Y: pd.DataFrame) -> tuple:
    """
    Copy and modify `clusters_df`, `X` and `Y` dataframes so they contain only shared 
       index keys and it is possible to pass them directly into `cross_validate`
       sklearn library function.
    Returns copies of modified three dataframes (tuple of three items = 3 dataframes)
    """
    c = clusters_df.copy(); 
    c.columns = ['c_' + col for col in c.columns]
    
    X = X.merge(c, left_on='Id', right_on='c_sequence_id').drop(columns=c.columns)
    Y = Y.merge(c, left_on='Id', right_on='c_sequence_id').drop(columns=c.columns)
    c = c.merge(X, left_on='c_sequence_id', right_on='Id').loc[:, ['c_sequence_id', 'c_cluster']]
    return c, X, Y


def setup_logging(model_name: str, input_cluster_file: str, output_file: str):
    """
    Logging-related boilerplate code used at every model-training/predicting notebooks
    
    Disables sklearn warning
    Sets up logger for papermill
    Logs basic experiment info
    """
    if not sys.warnoptions:
        warnings.simplefilter("ignore")
        os.environ["PYTHONWARNINGS"] = "ignore" # affects subprocesses as well

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger('papermill')

    logger.info(f'model {model_name} input: {input_cluster_file} output: {output_file}')
    
    return logger


def positionize_sasa_df(df: pd.DataFrame, ids: pd.Series) -> pd.DataFrame: 
    unique_ids = ids.unique()
    ui_index = np.arange(len(unique_ids))
    n_positions = len(df.columns)
    
    res_df = df.melt()
    res_df['sequence_id'] = pd.concat([pd.Series(unique_ids, index=ui_index)] * n_positions).reset_index(drop=True)
    res_df.columns = ['position', 'prediction', 'sequence_id']
    res_df = res_df[['sequence_id', 'position', 'prediction']]
    
    return res_df


# ------------------------------------------------------------------------------
# GENERATE ABNUMBER DATA
    
    
def _create_abnumber_chains_dict(X: pd.DataFrame) -> dict:      
    X_clean = X.copy(); X_clean.index = X_clean['Id']
    X_records = X_clean.drop(columns='Id').to_dict(orient='records')
    chains_dict = dict()
    chain_data_pairs = list(zip(X['Id'], X_records))
    chain_id = X.loc[0, 'Id'][-1]

    cdr_definition = 'chothia' if p.FINAL_NUMBERING_SCHEME == 'aho' else p.FINAL_NUMBERING_SCHEME
    abnumber = __import__('abnumber')
    for chain_full_id, chain_dict in tqdm(chain_data_pairs, desc=f'Generating abnumber chains {chain_id}...'):
        sequence =  ''.join(chain_dict.values()).replace('-', '')
        chain = abnumber.Chain(sequence, 
                               scheme=p.FINAL_NUMBERING_SCHEME, 
                               cdr_definition=cdr_definition, 
                               assign_germline=True)
        chains_dict[chain_full_id] = chain
        
    # chains_dict = LFAQ: HAKBLKAGKLHLFHLAF (sequence) ...  
    return chains_dict 


def generate_abnumber_chains(chains_filename: str, X: pd.DataFrame):    
    if os.path.exists(chains_filename):
        with open(chains_filename, 'rb') as chains_file:
            chains_dict = pickle.load(chains_file)
    else:
        # construct chains dictionary
        if type(X) is list:
            assert len(X) == 2
            chains_dict = _create_abnumber_chains_dict(X[0])
            chains_dict.update(_create_abnumber_chains_dict(X[1]))
        else:
            chains_dict = _create_abnumber_chains_dict(X)
            
        # pickle it
        with open(chains_filename, 'wb') as chains_file:
            pickle.dump(chains_dict, chains_file)
    
    
    return chains_dict


# ------------------------------------------------------------------------------
# PLOT-RELATED FUNCTIONS


def show_only_nth_ticklabel(plot, n=3, rotation=0, ax='x'):
    """
    Some plots have a lot of labels at some of the axes.
    The purpose of this function is to make such axis less overcrowded,
        by leaving only every n-th label visible, hiding all the rest.
    If `rotation` parameter is specified, label are rotated `rotation` degress in clockwise direction,
        thus saving even more horizontal space on the axis
    `ax` parameter should be `x` (default) if you want this function to apply to the x-axis,
        or the `y` if you want it to apply to the y-axis
    """
    if ax not in ('x', 'y'):
        raise ValueError('ax must be either "x" or "y"')
    
    labels = plot.get_xticklabels() if ax == 'x' else plot.get_yticklabels()
    for i, t in enumerate(labels):
        if (i % n) != 0:
            t.set_visible(False)
        t.set_rotation(rotation)
        
        
def annotate_plot(plot, x_rotation = 0, title='', xlabel='', ylabel='', 
                  show_bar_values: bool = False, bar_decimals = 2, 
                  clean_ax_settings = dict()):
    """
    Set up basic plot properties
    `x_rotation`
    `title`
    `xlabel`
    `ylabel`
    `show_bar_values`
    `bar_decimal`
    `clean_x_settings`: dict(`ax`, `nth_label_only`)
    """
    for item in plot.get_xticklabels(): 
        item.set_rotation(x_rotation)
    
    if title:
        plot.set_title(title)
    if xlabel:
        plot.set_xlabel(xlabel)
    if ylabel:
        plot.set_ylabel(ylabel) 
    if show_bar_values:
        show_barplot_values(plot, bar_decimals)
    if clean_ax_settings:
        ax = clean_ax_settings.get('ax', 'x')
        nth_only = clean_ax_settings.get('nth_label_only', 5)
        show_only_nth_ticklabel(plot, ax=ax, n=nth_only, rotation=x_rotation)
        
    return plot


def show_barplot_values(plot, bar_decimals=2):
    """
    Show actual counts at the top of individual bars in barplot
    """
    for p in plot.patches:
        barplot_value = np.round(p.get_height(), decimals=bar_decimals)
        plot.annotate(barplot_value, 
                   (p.get_x()+p.get_width()/2., p.get_height()),                              
                   ha='center', va='center',                              
                   xytext=(0, 8), textcoords='offset points')
     
        
def show_ab_regions(columns: list, ax, chain: str, scheme: str, regions: list, 
                    region_label_coord, color='black', alpha=0.1, 
                    position_ax: str = 'x', text_offsets : list = []):
    """
    Annotate plot with `regions` for given `chain` type and numbering `scheme`
    So the region boundaries can be computed on the axes, you need to provide the list of 
        `columns` used in the relevant numbering sceheme
    Desired `regions` are highlighted with `color` background color, 
        opacity of the colored layer is `alpha`
    Text labels with region names are shown at the far end of the color-highlighted region ->
        the exact coordinate (that remain constant)
        on the respective axis is controlled by `region_label_coord`. 
        The placement within the region itself may be controlled by `text_offsets` setting,
        the default value for which are all zeros -> that makes the text to place at the edge
        of the area
    `position_ax` should be either 'x' or 'y', based on which ax you wish to have the position
        labels displayed at
    ------------------------    
    Technical note: The plot object is not passed to the function - plot properties are set 
                    via matplotlib.pyplot module variable
    """
    if position_ax not in ('x', 'y'):
        raise ValueError('`position_ax` must be either `x` or `y`')
    
    region_key = f'{scheme}_{chain}'
    
    if scheme == 'aho':
        R = AHO_SCHEME_REGIONS
    else:
        R = common.SCHEME_REGIONS[region_key]
    
    if not text_offsets:
        text_offsets = [0] * len(regions)
        
    for region, text_offset in zip(regions, text_offsets):
        start_col, end_col = str(R[region]['start']), str(R[region]['end'])
        if start_col not in columns or end_col not in columns:
            continue
        start, end = columns.index(start_col),  columns.index(end_col)
        
        if position_ax == 'x':
            # fill highlighted area
            ax.axvspan(start, end, facecolor=color, alpha=alpha)
            # add text label with the name of the region
            ax.text(x=start+text_offset, y=region_label_coord, s=region)
        else:
            # 'y'
            ax.axhspan(start, end, facecolor=color, alpha=alpha)
            ax.text(
                x=region_label_coord, 
                # add some offset so the text actually starts
                # at the region boundary, not ends at it
                y=start+text_offset, 
                s=region, 
                # so the text goes vertically from the top the bottom,
                # that looks better
                rotation=270
            ) 
                                  
            
def show_cdr_regions(columns: list, ax, chain: str, scheme: str,
                     cdr_label_coord, color='black', alpha=0.1,
                     position_ax: str = 'x', text_offsets = []):
    """
    Annotate plot with `regions` for given `chain` type and numbering `scheme`
    So the region boundaries can be computed on the axes, you need to provide the list of 
        `columns` used in the relevant numbering sceheme
    Desired `regions` are highlighted with `color` background color, 
        opacity of the colored layer is `alpha`
    Text labels with region names are shown at the far end of the color-highlighted region ->
        the exact coordinate (that remain constant)
        on the respective axis is controlled by `region_label_coord`. 
        The placement within the region itself may be controlled by `text_offsets` setting,
        the default value for which are all zeros -> that makes the text to place at the edge
        of the area
    `position_ax` should be either 'x' or 'y', based on which ax you wish to have the position
        labels displayed at
    ------------------------    
    Technical note: The plot object is not passed to the function - plot properties are set 
                    via matplotlib.pyplot module variable
    """
    if not text_offsets:
        text_offsets = CDR_TEXT_OFFSETS
    show_ab_regions(columns, ax, chain, scheme, CDR_REGION_NAMES, cdr_label_coord, 
                    color, alpha, position_ax, text_offsets)

        
def get_position_columns(df: pd.DataFrame):
    columns = [col for col in df.columns if col[0].isnumeric()]
    return columns


def show_unique_columns(fasta_H_new_df = None, fasta_H_classic_df = None, fasta_L_new_df = None, fasta_L_classic_df = None):
    if fasta_H_new_df is not None and fasta_H_classic_df is not None:
        H_new_c, H_classic_c = set(get_position_columns(fasta_H_new_df)), set(get_position_columns(fasta_H_classic_df))
    if fasta_L_new_df is not None and fasta_L_classic_df is not None:
        L_new_c, L_classic_c = set(get_position_columns(fasta_L_new_df)), set(get_position_columns(fasta_L_classic_df))
        
    col_diffs = dict()
    if fasta_H_new_df is not None and fasta_H_classic_df is not None:
        col_diffs['H_yes_old_no_new'] = list(H_classic_c.difference(H_new_c))
        col_diffs['H_no_old_yes_new'] = list(H_new_c.difference(H_classic_c))
    if fasta_L_new_df is not None and fasta_L_classic_df is not None:
        col_diffs['L_yes_old_no_new'] = list(L_classic_c.difference(L_new_c))        
        col_diffs['L_no_old_yes_new'] = list(L_new_c.difference(L_classic_c))
    
    return col_diffs



        
        
        
        
        
        
        
        
        
        
        