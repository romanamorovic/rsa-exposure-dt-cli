
###########################################
# PARAMETERS USED THROUGHOUT THE PIPELINE #
###########################################
###########################################

import numpy as np


# ---------------------------------------------------------------------------------------------
# DIRECTORY PATHS
# relative to the notebooks/<SUBCATEGORY> directory
DATA_DIR = '../../data'
BIN_DIR = '../../bin'
PROCESSING_DIR = '../../notebooks/processing'


# ---------------------------------------------------------------------------------------------
# PROTEIN STRUCTURE RESOLUTION
# maximum allowed resolution for structures we work with
# this will affect which structures will be downloaded from the database
RESOLUTION_CUTOFF = 3.0



# ---------------------------------------------------------------------------------------------
# NUMBERING SCHEMES
# downloaded data consists of three subdirs - RAW, CHOTHIA, IMGT
# this setting determines which of these 3 will be used in further processing steps
INITIAL_NUMBERING_SCHEME = 'chothia'
# chains are later renumbered again from 'incremental' numbering to 
# some more common one
FINAL_NUMBERING_SCHEME = 'aho'



# ---------------------------------------------------------------------------------------------
# CLUSTERING SETTINGS
# when clustering the data, use this as N_NEIGHBORS parameter to the clustering algorithm
# ('umap' is currently used)
CLUSTERING_N_NEIGHBORS = 10
# bottom identity threshold (e.g. 0.8 means 80% identity required)
IDENTITY_START = 0.8
# top identity threshold (e.g. 1.0 means 100% identity required)
IDENTITY_END = 1
# how many clusterings to perform in total, interpolating uniformly within interval defined above
# (IDENTITY_BOTTOM, IDENTITY_END)
IDENTITY_STEPS_NUMBER = 11
# threshold list used by Snakefile to command thresholds
CLUSTERING_THRESHOLDS = [int(x) for x in np.linspace(IDENTITY_START*100, IDENTITY_END*100, IDENTITY_STEPS_NUMBER)]



# ---------------------------------------------------------------------------------------------
# HOW WE CLUSTER SEQUENCES
# possible values:
# 'all' - light-sim-mat and heavy-sim-mat are computed and then the average of the two values
#         is used to compute the similarity between the pair of sequences.
#         Both chains are always in the same cluster.
#.        The output will contain both L and H chains, the value for chains from the same structure
#.        will always be the same (XXXX:L == XXXX:H)
# 'L'   - light-sim-mat is used only - clustering is generated based on L chain similarities only
#         Only L chains will be outputted
# 'H'  - same as above but for the heavy chains
CLUSTERING_CHAINS = 'all' 
assert CLUSTERING_CHAINS in ('all', 'L', 'H')



# ---------------------------------------------------------------------------------------------
# WHICH SEQUENCE CHAINS ARE USED BY DEFAULT
# This setting will be usually overriden when running the notebook from Snakemake using Papermill,
# depending on the parameter value set in Snakemake
# possible values:
# 'L' - light chains
# 'H' - heavy chains (default value)
CHAINS = 'H'


# ----------------------------------------------------------------------------------------------
# WHICH VALUE IS USED AS DEFAULT SASA IF NaN IS PRESENT AT THE POSITION
# some of the interesting values may be 0 or 25
SASA_NAN_DEFAULT = 0


# ----------------------------------------------------------------------------------------------
# HOW MANY BEST-PERFORMING NON-BASELINE SCENARIOS SHOULD BE SELECTED FROM THE GATHERING OF 
# CLUSTERING TRAINING RESULTS
# at least 5 is recommended
HOW_MANY_BEST_NONBASELINES = 5

# ----------------------------------------------------------------------------------------------
# what is the maximum allowed similarity between TEST and TRAIN datasets (to prevent data leakage)
DL_MAXIMUM_ALLOWED_SIMILARITY = 0.95