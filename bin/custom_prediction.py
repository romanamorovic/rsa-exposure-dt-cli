import argparse
import math
import os
import pickle
import random
import sys; sys.path.append('..')

# import abnumber
import numpy as np
import pandas as pd
import subprocess

import bin.params as p
p.DATA_DIR = p.DATA_DIR.replace('../..', '..')

import bin.utils as u
import bin.feature_generators as fg

from Bio import SeqIO
from Bio.PDB.PDBExceptions import PDBConstructionWarning
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

# from tqdm.notebook import tqdm
import tqdm

TRAINED_MODELS_DIR_PATH = f'{p.DATA_DIR}/pickles/trained-test-models'

PARAMS = {'compress': False, 'preserve_seq_ids': True}

# python3 custom_prediction.py -i seq_seq1.fasta -o seq_seq1.csv -m BLmeansamerespos -f lco_whole_sequence_all_H 

def get_model(model_name, features):
    model_file_path = f'{TRAINED_MODELS_DIR_PATH}/{features}_{model_name}.p'
    print('model_name:', model_name, '| features:', features)
    print('model loaded from:', model_file_path)
    with open(model_file_path, 'rb') as trained_model_file:
        model = pickle.load(trained_model_file)
        return model


def save_fasta(seq_id, seq_data, seq_desc=''):
    # Define the sequence data and identifier
    
    # Create a SeqRecord object
    seq_record = SeqRecord(Seq(seq_data), id=seq_id, description=seq_desc)
    
    # Write the SeqRecord to a FASTA file
    input_file_path = f"seq_{seq_id}.fasta"
    SeqIO.write(seq_record, input_file_path, "fasta")
    print(f"FASTA file saved as {input_file_path}")
    return input_file_path

def predict(fasta_path, anarci_output_path, model_name='randomforestN30', features='lco_cont_window_r3_all_H'):
    model = get_model(model_name, features)
    scheme = p.FINAL_NUMBERING_SCHEME
    anarci_command = f'anarci -i {fasta_path} -o {anarci_output_path} --csv --scheme={scheme}'
    print('ANARCI command:', anarci_command)
    subprocess.run(anarci_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, shell=True)
    
    newest_file_command = "ls -Art | tail -n 1"
    newest_file_command_result = subprocess.run(
        newest_file_command, 
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, 
        text=True, shell=True)
    newest_file_command_output = newest_file_command_result.stdout.rstrip() # remove the newline from the end of the line
    print('ROMAN newest file:', newest_file_command_output)
    newest_file_df = pd.read_csv(newest_file_command_output, index_col=0)
    print('newest_file_df.shape:', newest_file_df.shape)
    
    position_columns = u.get_position_columns(newest_file_df)
    cols_to_remove = [c for c in newest_file_df if c not in position_columns]
    newest_file_df = newest_file_df.drop(columns=cols_to_remove)
    newest_file_df_ids = newest_file_df.index
    print(newest_file_df.head(n=1))
    
    Xr, Yr = u.load_dataset('test_new_234', chains='H')
    Xr.index = Xr['Id']; Xr = Xr.drop(columns=['Id'])
    print(Xr.head(n=1))
    
    cols = set(Xr.columns).difference(set(newest_file_df.columns))
    print('columns to add:', list(cols))
    for col in cols:
        newest_file_df[col] = '-'
    newest_file_df = u.sort_numbering_columns(newest_file_df).reset_index() # (BLmeansaremerespos)
    print('newest_file_df.shape:', newest_file_df.shape)
    X_custom = newest_file_df
    print(X_custom.head(n=1))
    
    # transform 
    X_custom_final, _, _ = fg.generate(X_custom, Y=None, c=None,
                                       model_name=model_name, features=features, params=PARAMS)

    last_column = X_custom_final.columns[-1]
    print('Removing last column from X_custom_final. it probably contains IDs')
    print(X_custom_final[last_column])
    X_custom_final = X_custom_final.drop(columns=[last_column])

    print('X_custom_final right before the prediction:', X_custom_final.head(n=1))
    
    predictions = model.predict(X_custom_final)
    predictions[predictions == -1] = np.nan
    print('len(predictions):', len(predictions))

    # todo
    if type(predictions) is np.ndarray:
        N_SEQUENCES = X_custom.shape[0]
        ids = list(newest_file_df_ids)
        print('predictions type: np.ndarray | N_SEQUENCES:', N_SEQUENCES, '| len(ids):', len(ids))
        # convert to dataframe
        Y_pred = Yr.copy().head(newest_file_df.shape[0]).drop(columns=['Id'])
        Y_pred.index = ids
        #Y_pred.index = Y_orig['Id']
        #Y_pred.drop(columns='Id', inplace=True)
        for i, _ in tqdm(enumerate(predictions), total=len(predictions), 
                         desc='Processing individual predictions...'):
            seq_id = ids[i % N_SEQUENCES]
            x_index = math.floor(i / N_SEQUENCES)
            pos_id = newest_file_df.columns[x_index+1] # starting from 1 as 0 is 'id'
            Y_pred.loc[seq_id, pos_id] = predictions[i]
        Y_pred = Y_pred.replace(-1, np.nan)
        predictions = Y_pred
    else:
        # dataframe
        assert isinstance(predictions, pd.DataFrame)
        print('predictions type: pd.DataFrame')
        assert predictions.index[0] == 0
        predictions.index = newest_file_df_ids
        
    return predictions.round(2)

def predict_single_sequence(seq_id, seq_data, seq_desc='', model_name='randomforestN30', features='lco_cont_window_r3_all_H'):
    fasta_path = save_fasta(seq_id, seq_data, seq_desc)
    anarci_output_path = f"seq_{seq_id}_numbered"
    return predict(fasta_path, anarci_output_path, model_name, features)
    
def main():
    # Initialize the parser
    parser = argparse.ArgumentParser(description="")

    # Add arguments
    parser.add_argument('-i', '--inputfile', type=str, help='FASTA input file containing input sequences')
    parser.add_argument('-o', '--outputfile', type=str, help='CSV output file to store the RSA prediction')
    parser.add_argument('-m', '--model', type=str, help='Which model to use for predictions')
    parser.add_argument('-f', '--features', type=str, help='Which feature representation to use for prediction')    

    # Parse the arguments
    args = parser.parse_args()

    input_file_path, output_file_path = args.inputfile, args.outputfile
    model_name, features = args.model, args.features

    anarci_filename = f"seqs_prediction_{random.randint(1000009, 2000009)}"
    predictions = predict(input_file_path, anarci_filename, model_name, features)
    print(predictions)
    print(anarci_filename)
    predictions.to_csv(output_file_path)
    os.system(f'rm -rf {anarci_filename}*')

if __name__ == "__main__":
    main()