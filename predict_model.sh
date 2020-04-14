##predict all cpg
mkdir $WD/$5
cd $WD/$5
Rscript $WD/src/normalize_from_template.R $WD/$6/data_final $WD/$7/data_norm_info data_final_norm
#generate all data
python3 $WD/src/generate_data_array.py all_data data_final_norm $WD/$8/cpg_no_chrM 1 21000000 45

#pred
#use the best epoch from rnn and cnn and joint module for below prediction
cp $WD/$1/$2* .
cp $WD/$1/$3* .
cp $WD/$1/$4* .

python3 -u $WD/src/continue_training_pred.py all_label1 all_label2 $4 $WD/$1/train_small.npz all_data.npz $3 $4 $WD/$9/mm9_genome_encoding.npz

cut -f2 all_label1 >all_label1_cut
cut -f2 all_label2 >all_label2_cut
paste all_label1_cut all_label2_cut >all_pred
paste $WD/$6/data_final_coord all_pred |head -21000000 >all_coord_pred_hmc_mc
#all_coord_pred_hmc_mc: coord, hmc, mc
