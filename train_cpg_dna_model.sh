##generate training and validation data
mkdir $WD/$2
cd $WD/$2
python3 $WD/src/generate_data_array.py train_small $WD/$1/data_final $WD/$1/data_final_coord 1 50000 45
python3 $WD/src/generate_data_array.py val_small $WD/$1/data_final $WD/$1/data_final_coord 50001 100000 45
python3 $WD/src/generate_data_array.py train_large $WD/$1/data_final $WD/$1/data_final_coord 1 2000000 45
python3 $WD/src/generate_data_array.py val_large $WD/$1/data_final $WD/$1/data_final_coord 2000001 2700000 45


##train model
##cpg model
export CUDA_VISIBLE_DEVICES="1"
python3 -u $WD/src/train_model_cpgModule.py cpg_pred2_label1 cpg_pred2_label2 cpg_model train_small.npz val_small.npz $WD/$3/genome_encoding.npz
python3 -u $WD/src/continue_training_cpgModule.py cpg_pred3_label1 cpg_pred3_label2 cpg_model train_large.npz val_large.npz $WD/$3/genome_encoding.npz >cpg_pred3_out
#find the epoch that has the smallest loss and use this epoch for prediction
grep final cpg_pred3_out >cpg_pred3_out_grep


##dna model
export CUDA_VISIBLE_DEVICES="0"
python3 -u $WD/src/train_model_dnaModule.py dna_pred2_label1 dna_pred2_label2 dna_model train_small.npz val_small.npz $WD/$3/genome_encoding.npz

python3 -u $WD/src/continue_training_dnaModule.py dna_pred3_label1 dna_pred3_label2 dna_model train_large.npz val_large.npz $WD/$3/genome_encoding.npz >dna_pred3_out
#find the epoch that has the smallest loss and use this epoch for prediction
grep final dna_pred3_out >dna_pred3_out_grep
#modified 4-20-20
python3 $WD/src/find_best_epoch.py cpg_pred3_out_grep cpg_model
python3 $WD/src/find_best_epoch.py dna_pred3_out_grep dna_model
