##generate training and validation data
mkdir $PWD/$2
cd $PWD/$2
python3 $PWD/src/generate_data_array.py train_small $PWD/$1/data_final $PWD/$1/data_final_coord 1 50000 45
python3 $PWD/src/generate_data_array.py val_small $PWD/$1/data_final $PWD/$1/data_final_coord 50001 100000 45
python3 $PWD/src/generate_data_array.py train_large $PWD/$1/data_final $PWD/$1/data_final_coord 1 2000000 45
python3 $PWD/src/generate_data_array.py val_large $PWD/$1/data_final $PWD/$1/data_final_coord 2000001 2700000 45


##train model
##cpg model
export CUDA_VISIBLE_DEVICES="1"
python3 -u $PWD/src/train_model_cpgModule.py cpg_pred2_label1 cpg_pred2_label2 cpg_model train_small.npz val_small.npz $PWD/$3/mm9_genome_encoding.npz
python3 -u $PWD/src/continue_training_cpgModule.py cpg_pred3_label1 cpg_pred3_label2 cpg_model train_large.npz val_large.npz $PWD/$3/mm9_genome_encoding.npz >cpg_pred3_out
#find the epoch that has the smallest loss and use this epoch for prediction
grep final cpg_pred3_out >cpg_pred3_out_grep
python3 $PWD/src/find_best_epoch.py cpg_pred3_out_grep


##dna model
export CUDA_VISIBLE_DEVICES="0"
python3 -u $PWD/src/train_model_dnaModule.py dna_pred2_label1 dna_pred2_label2 dna_model train_small.npz val_small.npz $PWD/$3/mm9_genome_encoding.npz

python3 -u $PWD/src/continue_training_dnaModule.py dna_pred3_label1 dna_pred3_label2 dna_model train_large.npz val_large.npz $PWD/$3/mm9_genome_encoding.npz >dna_pred3_out
#find the epoch that has the smallest loss and use this epoch for prediction
grep final dna_pred3_out >dna_pred3_out_grep
python3 $PWD/src/find_best_epoch.py dna_pred3_out_grep
