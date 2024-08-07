
from collections import defaultdict

import numpy as np
import pandas as pd

import bin.params as p
import bin.utils as utils

# -------------------------------------------

def _get_shape_safe(df):
    if df is None:
        return '--'
    else:
        return df.shape


def _gen_whole_sequence(X: pd.DataFrame, Y: pd.DataFrame, 
                        c: pd.DataFrame,
                        model_name: str,
                        params: dict = dict()) -> tuple:
    """
    Generate 'wholeSequence' features
    This is almost leaving the original SAbDab processed data as they are - just doing the cleanup,
        removing non-data columns and values that cannot by handled by models 
    - and then one-hot encoding the data
    """
    print(f'X.shape: {X.shape} | Y.shape: {_get_shape_safe(Y)}')
    
    ids = X['Id']
    
    if Y is None:
        X = utils.drop_nondata_columns(X)
    else:
        X, Y = utils.drop_nondata_columns([X, Y])
        Y = utils.handle_sasa_matrix(Y)
    print(f'after non-data column drop | X.shape: {X.shape} | Y.shape: {_get_shape_safe(Y)}')

    if not model_name.startswith('BL') and not params.get('raw', False):
        # baseline models should not be one-hot encoded
        X = utils.get_oh_matrix(X)
        print(f'after one-hot encode | X.shape: {X.shape} | Y.shape: {_get_shape_safe(Y)}')
        
        if params.get('compress', False):
            X = utils.remove_useless_columns(X) 
            print(f'after column compression | X.shape: {X.shape} | Y.shape: {_get_shape_safe(Y)}')
    else:
        print('[NOTE] Skipping one-hot encoding, since this is baseline model')
        
    if params.get('preserve_seq_ids', False):
        X['sequence_id'] = ids
       
    print(f'[FINAL] | X.shape: {X.shape} | Y.shape: {_get_shape_safe(Y)}')
    if c is not None: print(f'[FINAL] c.shape {c.shape}')
    
    return X, Y, c

def _gen_cont_windows(X: pd.DataFrame, Y: pd.DataFrame, 
                      c: pd.DataFrame,
                      model_name: str,
                      params: dict = dict()) -> tuple:
    if 'radius' not in params:
        raise ValueError('gen_cont_windows: `radius` not specified')
        
    window_radius = params['radius']
    print(f'X.shape: {X.shape} | Y.shape: {_get_shape_safe(Y)}')
    
    X_orig = X.copy()
    Y_orig = None if Y is None else Y.copy()
    X = utils.drop_nondata_columns(X)
    if Y is not None:
        Y = utils.drop_nondata_columns(Y)
    print(f'after drop_nondata_columns: X.shape {X.shape} Y.shape {_get_shape_safe(Y)}')
    
    # add empty sequence ends
    _add_sequence_end(X, 'start', window_radius, '-')
    _add_sequence_end(X, 'end', window_radius, '-')
    if Y is not None:
        _add_sequence_end(Y, 'start', window_radius, np.nan)
        _add_sequence_end(Y, 'end', window_radius, np.nan)
        assert list(X.columns) == list(Y.columns), 'X and Y columns must match'
    print(f'after _add_sequence_end: X.shape {X.shape} Y.shape {_get_shape_safe(Y)}')
    
    # transforms (the integral part of all this)
    X_window = _x_window_transform(X, X_orig, window_radius)
    Y_window = None
    if Y is not None:
        Y_window = _y_window_transform(Y_orig, Y, X_window, window_radius)
    print(f'after window transforms: X_window.shape {X_window.shape} Y_window.shape {_get_shape_safe(Y_window)}')
    
    c_window = None
    if c is not None:
        c_window = _c_window_transform(c, X_window, window_radius)
        print(f'c_window.shape {c_window.shape}')
    
    # final cleanup
    # ------
    WINDOW_LENGTH = window_radius + 1 + window_radius
    
    # 1. drop sequence_id from X_window
    ids = X_window[WINDOW_LENGTH+1] 
    X_window = X_window.drop(columns=WINDOW_LENGTH+1, errors='ignore')
    
    # 2. sanity check
    N_POSITIONS = X_window[WINDOW_LENGTH].nunique()
    assert N_POSITIONS == X_orig.shape[1]-1, f'Some ANARCI positions are missing (before: {X_orig.shape[1]} now: {N_POSITIONS}), run all the cells again'
    
    # 3. sanity-check invalid Y values (np.NaN, 0) in Y_window
    if Y is not None:
        Y_window = utils.handle_sasa_matrix(Y_window)
    
    # 4. scale the last column in X_window (position ID) to (0, 1) interval
    X_window[WINDOW_LENGTH] = (X_window[WINDOW_LENGTH] - window_radius) / int(N_POSITIONS)
    
    # 5. one-hot X_window
    X_window_oh = utils.get_oh_matrix(X_window.drop(columns=WINDOW_LENGTH))
    X_window_oh[len(X_window_oh.columns)] = X_window[WINDOW_LENGTH]
    
    if params.get('preserve_seq_ids', False):
        X_window_oh[len(X_window_oh.columns)+1] = ids
    
    print(f'[FINAL] X.shape {X_window_oh.shape} Y.shape {_get_shape_safe(Y_window)}')
    if c is not None: print(f'[FINAL] c.shape {c_window.shape}')
    
    return X_window_oh, Y_window, c_window


def _gen_regional_simple(X: pd.DataFrame, Y: pd.DataFrame, 
                      c: pd.DataFrame,
                      model_name: str,
                      params: dict = dict()) -> tuple:
    assert 'features' in params
    print(f'X.shape {X.shape} Y.shape {Y.shape}')
    
    features_str = params['features']
    features_list = features_str.split(',')
    
    # TODO - what about custom-made sequences???
    chains_dict = utils.generate_abnumber_chains(f'{p.DATA_DIR}/pickles/abnumber_chains.p', X)
    
    # remove invalid Y values and enable sequence_id index
    Y_clean = utils.handle_sasa_matrix(Y.copy())
    Y_clean.index = Y_clean['Id']
    
    # generate dictionaries for the new dataframe
    X_data, Y_data = defaultdict(list), defaultdict(list)
    # for chain_full_id, chain in tqdm(chains_dict.items()): 
    for chain_full_id, chain in chains_dict.items(): 
        # process only those data that are in X/Y dataframe
        if chain_full_id not in Y_clean.index:
            continue
            
        for region, positions in chain.regions.items():
            for position, residue in positions.items():
                chain_type, pos_code = position.chain_type, position.format(chain_type=False)
                if 'region' in features_list: X_data['region'].append(region)
                if 'position' in features_list: X_data['position'].append(pos_code)
                if 'chain' in features_list: X_data['chain'].append(chain_type)
                if 'species' in features_list: X_data['species'].append(chain.species)
                X_data['chain_full_id'].append(chain_full_id)
                X_data['residue'].append(residue)
                sasa_value = Y_clean.loc[chain_full_id, pos_code]
                assert np.isfinite(sasa_value), f'y values must be finite, {chain_full_id} {pos_code}'
                Y_data['sasa'].append(sasa_value)
    
    # convert dictionaries to dataframes
    X_regional = pd.DataFrame(X_data) # .drop(columns='chain_full_id')
    Y_regional = pd.DataFrame(Y_data, index=range(len(Y_data['sasa'])))   
    if c is None:
        c_regional = None
        print(f'after transformation X.shape {X.shape} Y.shape {Y.shape}')
    else:
        X_regional_c_df = X_regional.merge(c, left_on='chain_full_id', right_on='c_sequence_id')
        c_regional = X_regional_c_df[['c_sequence_id', 'c_cluster']].reset_index(drop=True)
        print(f'after transformation X.shape {X.shape} Y.shape {Y.shape} c.shape {c.shape}')

    # one-hot encode using pandas method
    X_regional.drop(columns='chain_full_id', inplace=True)
    X_regional_oh = pd.get_dummies(X_regional, prefix=features_list + ['residue'])
    X_regional_oh.head(n=2)
    print(f'[FINAL] after one-hot encode X.shape {X_regional_oh.shape} Y.shape {Y_regional.shape}')
    if c is not None: print(f'[FINAL] c.shape {c_regional.shape}')
    
    return X_regional_oh, Y_regional, c_regional
    
    
    
# ------------- GENERATE function ------------------------------------
# (the only one that should be used from this module)

           
def generate(X: pd.DataFrame, Y: pd.DataFrame, 
             c: pd.DataFrame,
             model_name: str, features: str,
             params: dict = dict()) -> tuple:
    """
    Generate features as requested by 'features' parameter
    Return tuple (c, X, Y) - where:
        c - filtered clustering records
        X - filtered and transformed X dataframe
        Y - filtered and transformed Y dataframe
    """
    if c is not None:
        print(f'before merge with clusters X.shape {X.shape} Y.shape {Y.shape} c.shape {c.shape}')
        c, X, Y = utils.include_clusters(c, X, Y)
        print(f'after merge with clusters X.shape {X.shape} Y.shape {Y.shape} c.shape {c.shape}')
        
    print(features)
    if 'whole_sequence' in features or 'wholeSequence' in features:
        X, Y, c = _gen_whole_sequence(X, Y, c, model_name, params)
    elif 'lco_cont_window_' in features:
        # features: lco_cont_window_r3_all_H
        # get radius from experiment name 
        params['radius'] = int(features.split('_')[3][1:])
        X, Y, c = _gen_cont_windows(X, Y, c, model_name, params)
    elif 'regional_simple' in features:
        # lco_regional_simple_<FEATURES_LIST>
        feature_string = features.split('_')[3]
        params['features'] = feature_string
        X, Y, c = _gen_regional_simple(X, Y, c, model_name, params)
    
    # flatten if Y is 1d dataframe
    if type(Y) is pd.DataFrame and Y.shape[1] == 1:
        Y = Y.to_numpy().flatten()
        
    return X, Y, c


# ---------------------------------------------------------------------
# ----- utility functions used here 

def _add_sequence_end(df: pd.DataFrame, where: str, length: int, value):
    if where == 'start':
        beg_column_names = ['0_' + str(col_name) for col_name in range(1, length+1)]
        for col_name in beg_column_names[::-1]:
            df.insert(0, col_name, value)
    elif where == 'end': 
        end_column_names = ['999_' + str(col_name) for col_name in range(1, length+1)]
        for col_name in end_column_names:
            df.insert(len(df.columns), col_name, value)
    else:
        raise ValueError('`where` parameter must be one of ("start", "end")')
       
    
def _x_window_transform(X: pd.DataFrame, X_orig: pd.DataFrame,
                        window_radius: int) -> pd.DataFrame:
    windows = []
    WINDOWS_COUNT = X.shape[1] - window_radius*2
    WINDOW_LENGTH = window_radius + 1 + window_radius

    for window_start in range(WINDOWS_COUNT):
        window_columns = X.columns[window_start : window_start+WINDOW_LENGTH]
        window = X[window_columns].copy()
        window['window_center'] = window_start+window_radius
        window['Id'] = X_orig['Id']
        window.columns = np.arange(0, window.shape[1])
        windows.append(window)

    X_window = pd.concat(windows, sort=False)
    return X_window


def _y_window_transform(Y_orig: pd.DataFrame, Y: pd.DataFrame, 
                       X_window: pd.DataFrame, window_radius: int) -> pd.DataFrame:
    # Y transform - this may take a while
    WINDOW_LENGTH = window_radius + 1 + window_radius
    
    Y_tmp = Y_orig.copy()
    Y_tmp.index = Y_tmp['Id']

    def gen_y_window_record(row):
        Y_column_index = row[WINDOW_LENGTH]
        column = Y.columns[Y_column_index]
        seq_id = row[WINDOW_LENGTH+1]
        return Y_tmp.loc[seq_id, column]

    Y_window = X_window.apply(gen_y_window_record, axis=1).reset_index().drop(columns='index')
    return Y_window


def _c_window_transform(c: pd.DataFrame, X_window: pd.DataFrame, 
                       window_radius: int) -> pd.Series:
    # cluster transform - this may take a while
    WINDOW_LENGTH = window_radius + 1 + window_radius
    c.index = c['c_sequence_id']

    def gen_cluster_window_record(row):
        seq = row[WINDOW_LENGTH+1]
        return c.loc[seq]

    c_window = X_window.apply(gen_cluster_window_record, axis=1).reset_index(drop=True)
    return c_window
                      