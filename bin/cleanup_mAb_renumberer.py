import argparse
import os
import re
import sys
from collections import OrderedDict
from dataclasses import dataclass

import anarci
import numpy as np
import pandas as pd
from biopandas.pdb import PandasPdb


# ---------------------------------------- Anarci ------------------------------------------------


def do_anarci(
        chain_id: str,
        chain_seq: str,
        scheme: str):
    """
    Compute correct numbering using Anarci tool.
    Return both numbering and summary statistics
    @param chain_id: str - descriptor of a chain, eg H
    @param chain_seq: str - aminoacid sequence of a chain
    @param scheme: str - numbering scheme to use (raw + all schemes that anarci supports)
    """
    numbering, alignment, _hittables = anarci.anarci(
        [(chain_id, chain_seq)],
        ('chothia' if scheme == 'incremental' else scheme),
        assign_germline=True)

    if not alignment[0]:
        print('Anarci alignment failed', 'numbering ', numbering[0])
        sys.exit(6)

    # alignment stats
    alignment = alignment[0][0]
    stats = {
        'chain': alignment['query_name'],
        'start': alignment['query_start'],
        # see -1 below. It is there so the compatibility between
        # legacy script and this is maintained until we rewrite the rest
        # of the script
        'stop': alignment['query_end']-1,
        'species': alignment['species'],
        'v_gene': alignment['germlines']['v_gene'][0][1],
        'v_id': alignment['germlines']['v_gene'][1],
        'j_gene': alignment['germlines']['j_gene'][0][1],
        'j_id': alignment['germlines']['j_gene'][1],
    }

    # chain-related
    anarci_dict = {'chain': [], 'new_residue_number': [], 'new_insertion': []}

    raw_pos = 0
    numbering = numbering[0][0][0]

    if not numbering:
        print(f'Numbering of chain {chain_id} failed')
        sys.exit(8)

    for residue in numbering:
        pos, subpos, aminoacid = residue[0][0], residue[0][1], residue[1]
        if aminoacid == '-':
            continue

        raw_pos += 1
        anarci_dict['chain'].append(alignment['query_name'])
        if scheme == 'incremental':
            anarci_dict['new_residue_number'].append(str(raw_pos))
            anarci_dict['new_insertion'].append('')
        else:
            anarci_dict['new_residue_number'].append(pos)
            anarci_dict['new_insertion'].append(subpos.replace(' ', ''))

    numbering = OrderedDict(anarci_dict)

    if len(numbering['new_residue_number']) < len(chain_seq):
        shift = len(chain_seq) - len(numbering['new_residue_number'])
        start = int(numbering['new_residue_number'][-1])
        stop = int(numbering['new_residue_number'][-1]) + shift

        add = np.arange(start + 1, stop + 1)
        numbering['new_residue_number'].extend(add)

        for _ in range(len(add)):
            numbering['new_insertion'].append('')
            numbering['chain'].append(numbering['chain'][0])

    return stats, numbering


# ---------------------------------- Functions ----------------------------------------------------

def seq_order(chain_df):
    chain_df['residue_insertion'] = chain_df['residue_number'].astype(str) + \
                                    chain_df['insertion'].astype(str)
    ordered_seq = list(OrderedDict.fromkeys(chain_df['residue_insertion']))
    seq_dict = {ordered_seq[i]: i + 1 for i in range(0, len(ordered_seq))}
    chain_df['residue_insertion'] = chain_df['residue_insertion'].map(seq_dict)
    chain_df['residue_number'] = chain_df['residue_insertion']
    chain_df.drop(['residue_insertion'], axis=1, inplace=True)
    chain_df['insertion'] = ''
    return chain_df


def atom_renum(chain_df):
    """
    @param chain_df
    Renumber atoms by their position
    """
    atom_list = list(OrderedDict.fromkeys(chain_df['atom_number']))
    atom_dict = {atom_list[i]: i + 1 for i in range(0, len(atom_list))}
    chain_df['atom_number'] = chain_df['atom_number'].map(atom_dict)
    return chain_df


# ----------------------------- mAb Cleanup ------------------------------------------------------


@dataclass
class Header:
    """HEADER line record in PDB file"""
    experiment_name: str
    date: str
    structure_code: str


def standardize_pdb(
        input_file_path: str,
        output_file_path: str,
        scheme: str,
        header_obj: Header):
    """
    Cleanup and renumber chains contained in PDB file
    @param input_file_path: str - input PDB file path
    @param output_file_path: str - output PDB file path
    @param scheme: str - chothia, raw, aho...
    @param header_obj: Header - containing experiment-name/date/structure-code
    """

    ppdb = PandasPdb()
    ppdb.read_pdb(input_file_path)
    ppdb.df['ATOM'] = ppdb.df['ATOM'].sort_values(by=['chain_id', 'residue_number'])
    chains = ppdb.df['ATOM']['chain_id'].unique()
    chain_num = len(chains)
    print("Number of chains:", chain_num)
    print(f'\nRaw PDB file contents:\n\n{ppdb.pdb_text[:1000]}\n...')

    header_df = ppdb.df['OTHERS'].values
    # order by chain, position-within-chain
    ab_complex = []

    for _record_type, row, index in header_df:
        if re.search(r'\bPAIRED_HL\b', row):
            fields = row.split()
            hchain = fields[2].split('=')[1]
            lchain = fields[3].split('=')[1]
            agchain = fields[4].split('=')[1]
            agtype = re.split('=|;', fields[5])[1]
            ab_complex.append([index + 1, hchain, lchain, agchain, agtype])

    if not ab_complex:
        # no light-heavy chain pairs in the data
        print('No light-heavy chain in the data')
        sys.exit(7)

    # reformat ATOM records
    #['ATOM' 1927 '' 'N' '' 'GLU' '' 'H' 1 '' '' - 3.985 - 130.133 17.337 1.0, 31.6 '' '' 'N' nan
    # 1930]
    #['ATOM' 12589 '' 'N' '' 'VAL' '' 'A' 518 '' '' 1201.879 251.262 637.385, 1.0 467.12 '' '' 'N'
    # nan 12600]
    del ppdb.df['HETATM']
    del ppdb.df['ANISOU']

    bestcmplx = None
    for cmplx in ab_complex:
        compchains = ppdb.df['ATOM'][((ppdb.df['ATOM']['chain_id'] == cmplx[0]) |
                                      (ppdb.df['ATOM']['chain_id'] == cmplx[1]) |
                                      (ppdb.df['ATOM']['chain_id'] == cmplx[2])) &
                                     ((ppdb.df['ATOM']['atom_name'] == 'C') |
                                      (ppdb.df['ATOM']['atom_name'] == 'O') |
                                      (ppdb.df['ATOM']['atom_name'] == 'N') |
                                      (ppdb.df['ATOM']['atom_name'] == 'CA'))]
        bfact_avg = compchains['b_factor'].mean()
        cmplx.append(bfact_avg)
        bestcmplx = ab_complex[0]

    for cmplx in ab_complex:
        if cmplx[5] < bestcmplx[5]:
            bestcmplx = cmplx

    if bestcmplx[1] == bestcmplx[2]:
        with open('single_chain.txt', 'a+', encoding='utf-8') as single_chain_file:
            single_chain_file.write(''.join([input_file_path, '\n']))
        print('single chain, exiting')
        sys.exit(5)

    print("Best Complex is:", bestcmplx)

    # leave ONLY the best pair of chains in the ATOM records,
    # remove all the others
    ppdb.df['ATOM'] = ppdb.df['ATOM'][(ppdb.df['ATOM']['chain_id'] == bestcmplx[1]) |
                                      (ppdb.df['ATOM']['chain_id'] == bestcmplx[2])]

    # label first chain as H, second chain as L
    chain1_pos = ppdb.df['ATOM'].chain_id == bestcmplx[1]
    chain2_pos = ppdb.df['ATOM'].chain_id == bestcmplx[2]
    ppdb.df['ATOM'].loc[chain1_pos, ['chain_id']] = 'H'
    ppdb.df['ATOM'].loc[chain2_pos, ['chain_id']] = 'L'

    chain_df = ppdb.df['ATOM'].copy()
    hdf = chain_df[(chain_df['chain_id'] == 'H')].reset_index(drop=True)
    ldf = chain_df[(chain_df['chain_id'] == 'L')].reset_index(drop=True)
    ppdb.df['ATOM'] = pd.DataFrame(columns=ppdb.df['ATOM'].columns)

    hdf = seq_order(hdf)
    ldf = seq_order(ldf)

    # ppdb.df['ATOM'] = ppdb.df['ATOM'].append(ldf)
    ppdb.df['ATOM'] = pd.concat([ppdb.df['ATOM'], ldf])
    # ppdb.df['ATOM'] = ppdb.df['ATOM'].append(hdf).reset_index(drop=True)
    ppdb.df['ATOM'] = pd.concat([ppdb.df['ATOM'], hdf]).reset_index(drop=True)
    
    def df_to_seq(chain):
        amino_seq = ppdb.amino3to1()
        amino_seq = str(''.join(amino_seq.loc[amino_seq['chain_id'] == chain, 'residue_name']))
        return amino_seq

    # ----------------------------------- mAb Renumber --------------------------------------------

    chain_info = []
    chain_df = ppdb.df['ATOM'].copy()
    ppdb.df['ATOM'].drop(ppdb.df['ATOM'].index, inplace=True)
    for chain in chain_df['chain_id'].unique():
        # cdf = chain + 'df' TODO ???
        cdf = chain_df[(chain_df['chain_id'] == chain)].reset_index(drop=True)
        cdf = seq_order(cdf)
        #ppdb.df['ATOM'] = ppdb.df['ATOM'].append(cdf).reset_index(drop=True)
        ppdb.df['ATOM'] = pd.concat([ppdb.df['ATOM'], cdf]).reset_index(drop=True)
        seq = df_to_seq(cdf['chain_id'].unique()[0])
        chain_info.append([chain, chain + 'df', seq])

    chain_df = ppdb.df['ATOM'].copy()

    for chain in chain_info:
        chain_id, chain_seq = chain[0], chain[2]
        stats, num = do_anarci(chain_id, chain_seq, scheme)

        if stats['chain'] == 'H':
            print(f'Chain {chain_id} is a Heavy chain')
            chain_df.loc[chain_df.chain_id == chain_id, 'chain_id'] = 'H'
            hdf_renum = pd.DataFrame.from_dict(num)
            hdf = chain_df[(chain_df['chain_id'] == 'H')].reset_index(drop=True)
            hdf_renum.drop(['chain'], inplace=True, axis=1)
            hdf_renum['residue_number'] = np.arange(len(hdf_renum)) + 1
            hdf = hdf.merge(hdf_renum, how='left') \
                .drop(columns=['residue_number', 'insertion']) \
                .rename(columns={
                    'new_residue_number': 'residue_number',
                    'new_insertion': 'insertion'
                })

        elif stats['chain'] in ['K', 'L']:
            lchain_type = 'Lambda' if stats['chain'] == 'L' else 'Kappa'
            print(f'Chain {chain_id} is a {lchain_type} Light chain')
            chain_df.loc[chain_df.chain_id == chain_id, 'chain_id'] = 'L'
            ldf_renum = pd.DataFrame.from_dict(num)
            ldf = chain_df[(chain_df['chain_id'] == 'L')].reset_index(drop=True)
            ldf_renum.drop(['chain'], inplace=True, axis=1)
            ldf_renum['residue_number'] = np.arange(len(ldf_renum)) + 1
            ldf = ldf.merge(ldf_renum, how='left') \
                .drop(columns=['residue_number', 'insertion']) \
                .rename(columns={
                    'new_residue_number': 'residue_number',
                    'new_insertion': 'insertion'
                })

        else:
            print(f'Chain {chain_id} is not an antibody chain, sorry')
            sys.exit(4)

    # delete all the records from the dataframe
    ppdb.df['ATOM'].drop(ppdb.df['ATOM'].index, inplace=True)
    ppdb.df['OTHERS'].drop(ppdb.df['OTHERS'].index, inplace=True)

    # construct new well prepared set of ATOM records,
    # where only L and H-chain atoms are present
    # hdf = hdf.append(ldf, sort=False)
    hdf = pd.concat([hdf, ldf], sort=False)
    hdf = atom_renum(hdf)

    # compute correct line IDs
    # the old ones are not OK anymore because we did some line-mingling
    hdf.line_idx = np.arange(1, len(hdf) + 1)
    #ppdb.df['ATOM'] = ppdb.df['ATOM'].append(hdf, sort=False)
    ppdb.df['ATOM'] = pd.concat([ppdb.df['ATOM'], hdf], sort=False)

    # generate PDB file
    ppdb.to_pdb(output_file_path, records=['ATOM', 'OTHERS'])

    # reformat PDB
    pdb_fs = "{:6s}{:5d} {:^4s}{:1s}{:3s} {:1s}{:4d}{:1s}   {:8.3f}{:8.3f}{:8.3f}{:6.2f}{:6.2f}          {:>2s}{:2f}"
    with open(output_file_path, 'w') as pdb:
        for idx, r in ppdb.df['ATOM'].iterrows():
            line = pdb_fs.format(r.record_name, r.atom_number, r.atom_name, r.alt_loc,
                                 r.residue_name, r.chain_id, int(r.residue_number),
                                 r.insertion, float(r.x_coord), float(r.y_coord), float(r.z_coord),
                                 float(r.occupancy), float(r.b_factor),
                                 r.element_symbol, r.charge)
            pdb.write(line + os.linesep)

    # add header
    _write_pdb_header(output_file_path, header_obj)

def _write_pdb_header(output_file_path: str,
                      header: Header):
    exp_name = header.experiment_name.ljust(40, ' ')
    str_code = header.structure_code.upper()
    header_line = f'HEADER    {exp_name}{header.date}   {str_code}{os.linesep}'
    with open(output_file_path, 'r', encoding='utf-8') as original:
        pdb_text = original.read()
    with open(output_file_path, 'w', encoding='utf-8') as modified:
        modified.write(header_line + pdb_text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Renumber antibody given in PDB file into different scheme')
    parser.add_argument('input_pdb_file', help='PDB input records that are to be renumbered')
    parser.add_argument('numbering_scheme', help='Antibody numbering scheme used in the input file')
    parser.add_argument('output_pdb_file', help='PDB output file to be generated')
    parser.add_argument('--experiment_name', default='UNKNOWN EXPERIMENT',
                        help='Experiment name - used in HEADER record. Max 40 characters')
    parser.add_argument('--date', default='UNKNOWN  ',
                        help='Date of experiment - used in HEADER record. ' +
                             'Format: 22-JAN-98. 9 characters')
    parser.add_argument('--structure_code', default='', help='Structure code. 4 characters')
    args = parser.parse_args()

    if not args.structure_code:
        # take 4 characters right before .pdb extension
        # those usually denote structure code
        args.structure_code = args.input_pdb_file[-8:-4]

    header = Header(args.experiment_name, args.date, args.structure_code)
    standardize_pdb(
        input_file_path=args.input_pdb_file,
        scheme=args.numbering_scheme,
        output_file_path=args.output_pdb_file,
        header_obj=header)
