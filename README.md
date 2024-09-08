1. Go to data/pickles, unzip:

- lco_cont_window_r2_all_H_randomForestN30.zip
- lco_cont_window_r3_all_H_randomForestN30.zip
- lco_cont_window_r4_all_H_randomForestN30.zip

2. Run from the directory root:

```
cd bin
python3 custom_prediction.py -i <INPUT> -o <OUTPUT_WITHOUT_EXTENSION> -m <MODEL> -f <FEATURES>
```

----


Example usage of the script:

```
python3 custom_prediction.py -i seq_seq1.fasta -o seq1_preds -m randomForestN30 -f lco_cont_window_r4_all_H
```

Allowed combinations of <MODEL> and <FEATURES>

```
lco_cont_window_r0_all_H x randomForestN30
lco_cont_window_r1_all_H x randomForestN30
lco_cont_window_r2_all_H x randomForestN30 
lco_cont_window_r3_all_H x randomForestN30
lco_cont_window_r4_all_H x randomForestN30
lco_whole_sequence_all_H x BLavgpos
lco_whole_sequence_all_H x BLknnwholeseqn10
lco_whole_sequence_all_H_BLknnwholeseqn3.p
lco_whole_sequence_all_H_BLmeansamerespos.p
```

lco_whole_sequence_all_H_BLmediansamerespos.p


