##pick high coverage data
mkdir $PWD/$2
cd $PWD/$2
#generate data with high cov
Rscript $PWD/src/get_high_cov_cpg.R $PWD/$1/data_mlml_mc_hmc_cov high_cov_cpg 

#intersect
bedtools intersect -a $PWD/$1/data_final_coord_data -b high_cov_cpg -wa -wb -sorted >data_highcov
cut -f1-3 data_highcov >data_highcov_coord
cut -f4-48,52-53 data_highcov >data_highcov_final


##normalize data
#normalize
Rscript $PWD/src/normalize_nondist_feature.R data_highcov_final data_final_norm data_norm_info data_highcov_coord 
paste data_final_norm_coord data_final_norm >data_final_norm_final
cut -f46-47 data_final_norm >data_final_norm_hmc_mc


##balance data
cut -f1 data_final_norm_hmc_mc >high_cov_hmc

#extract oversampled balanced coord
paste data_final_norm_coord high_cov_hmc >high_cov_coord_hmc
#below command is modified to be
Rscript $PWD/src/extract_balance_hmc_class_oversample.R high_cov_coord_hmc $PWD/$3/hmc_interval cpg_balance

bedtools intersect -a cpg_balance -b data_final_norm_final -wa -wb >cpg_balance_data
#shuffle data for training 
shuf cpg_balance_data >cpg_balance_data_shuffle
cut -f7- cpg_balance_data_shuffle >data_final
cut -f1-3 cpg_balance_data_shuffle >data_final_coord
