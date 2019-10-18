#created by Yu He on Oct 17 2019
#pipeline#
#make sure $PWD is pipeline folder
#
#
#


#1.mlml for tab-seq and wgbs
mkdir $PWD/mlml
cd $PWD/mlml
bedtools intersect -a $PWD/data/wgbs_YoungCB_final -b data/tab_YoungCB_final -wa -wb -sorted >wgbs_inter_tab
cut -f1-5 wgbs_inter_tab >wgbs
cut -f6-10 wgbs_inter_tab >tab
python3 $PWD/src/format_as_mlml.py wgbs wgbs_format
python3 $PWD/src/format_as_mlml.py tab tab_format
$PWD/MethPipe/bin/mlml -v -u wgbs_format -h tab_format  -o data_mlml
cut -f5,10 wgbs_inter_tab >wgbs_tab_cov
paste data_mlml wgbs_tab_cov >data_mlml_cov
cut -f1-5,8-9 data_mlml_cov >data_mlml_mc_hmc_cov


#2.process medip,hmcSeal and mre features
mkdir $PWD/process_methylation_feature
cd $PWD/process_methylation_feature
##6w cerebellum
#hmcSeal
sort -k1,1 -k2,2n $PWD/data/TWDL-mCB-hmC-6w-cerebellum-1_sort.extended.bedGraph >hmcSeal_6w_cerebellum1.bedGraph
python3 $PWD/src/extract_feature_from_bp3.py hmcSeal_6w_cerebellum1.bedGraph $PWD/mm9/cpg_no_chrM $PWD/mm9/mm9_chrom_sizes hmcSeal_window_6w_cerebellum1
cut -f4- hmcSeal_window_6w_cerebellum1 >hmcSeal_window_6w_cerebellum1_cut

#medip
sort -k1,1 -k2,2n $PWD/data/mCB-MeDIP-6w-cerebellum-1_sort.extended.bedGraph >medip_6w_cerebellum1.bedGraph
python3 $PWD/src/extract_feature_from_bp3.py medip_6w_cerebellum1.bedGraph $PWD/mm9/cpg_no_chrM $PWD/mm9/mm9_chrom_sizes medip_window_6w_cerebellum1
cut -f4- medip_window_6w_cerebellum1 >medip_window_6w_cerebellum1_cut

#mre
ml bedtools
cut -f1-3 $PWD/data/WangT_6w-cerebellum-1_sort.filter.bed >mre_WangT_6w-cerebellum-1_cut.filter.bed
sort -k1,1 -k2,2n mre_WangT_6w-cerebellum-1_cut.filter.bed >mre_WangT_6w-cerebellum-1_sort.filter.bed
bedtools genomecov -i mre_WangT_6w-cerebellum-1_sort.filter.bed -g $PWD/mm9/mm9_chrom_sizes -bg >mre_WangT_6w-cerebellum-1.bedGraph
python3 $PWD/src/extract_feature_from_bp3.py mre_WangT_6w-cerebellum-1.bedGraph $PWD/mm9/cpg_no_chrM $PWD/mm9/mm9_chrom_sizes mre_window_6w_cerebellum1
cut -f4- mre_window_6w_cerebellum1 >mre_window_6w_cerebellum1_cut


#3.process genomic features like gc percent, cpg num, dist to cpg island
mkdir $PWD/process_genomic_feature
cd $PWD/process_genomic_feature
#gc percent
python3 $PWD/src/extract_feature_from_bp3.py $PWD/mm9/gc5Base.sort.bedGraph $PWD/mm9/cpg_no_chrM $PWD/mm9/mm9_chrom_sizes cortex_gcContent_window2

#cpg num at each window
python3 $PWD/src/generate_cpg_window.py $PWD/mm9/cpg_no_chrM cpg_window
bedtools coverage -a cpg_window -b $PWD/mm9/cpg_no_chrM  -counts >cpg_num_window
cut -f4 cpg_num_window >cpg_num_window_cut
Rscript $PWD/src/format_cpgNum_data.R cpg_num_window_cut cpg_num_window_final

#dist to cpg island
bedtools intersect -a $PWD/mm9/cpg_no_chrM -b $PWD/mm9/cpgIsland.bed -wao -sorted >cpg_inter_cgi
cut -f1-3,7 cpg_inter_cgi >cpg_inter_cgi_cut
python3 $PWD/src/cal_dist_to_cgi.py cpg_inter_cgi_cut cpg_dist_cgi


#4.combine genomic and methylation features
mkdir $PWD/combine_features
cd $PWD/combine_features
#coord,dist_cgi,cpg_num, gc_content,medip,mre site, mre,hmcSeal
paste $PWD/mm9/cpg_no_chrM $PWD/process_genomic_feature/cpg_dist_cgi $PWD/process_genomic_feature/cpg_num_window_final $PWD/process_methylation_feature/medip_window_6w_cerebellum1_cut $PWD/mm9/data_mreSite_final $PWD/process_methylation_feature/mre_window_6w_cerebellum1_cut $PWD/process_methylation_feature/hmcSeal_window_6w_cerebellum1_cut >data_feature1

bedtools intersect -a $PWD/process_genomic_feature/cortex_gcContent_window2 -b data_feature1 -wa -wb -sorted >data_feature2
cut -f5-12,16- data_feature2 >data_feature2_cut
Rscript $PWD/src/convert_cpgNum_to_cpgDensity.R data_feature2_cut data_final 
cut -f1-3 data_feature2 >data_final_coord
paste data_final_coord data_final >data_final_coord_data


#4.pick high coverage data
mkdir $PWD/select_high_cov
cd $PWD/select_high_cov
#generate data with high cov
Rscript $PWD/src/get_high_cov_cpg.R $PWD/mlml/data_mlml_mc_hmc_cov high_cov_cpg 

#intersect
bedtools intersect -a $PWD/combine_features/data_final_coord_data -b high_cov_cpg -wa -wb -sorted >data_highcov
cut -f1-3 data_highcov >data_final_coord
cut -f4-48,52-53 data_highcov >data_final


#5.normalize data
mkdir $PWD/normalize_data
cd $PWD/normalize_data
#normalize
Rscript $PWD/src/normalize_nondist_feature.R $PWD/select_high_cov/data_final data_final_norm data_norm_info $PWD/select_high_cov/data_final_coord 
paste data_final_norm_coord data_final_norm >data_final_norm_final
cut -f46-47 data_final_norm >data_final_norm_hmc_mc


#6.balance data
##use the balanced cpg coord from (http://wang.wustl.edu/mediawiki/index.php/YuHe_June_2019#train_young_cerebellum_model_2)
mkdir $PWD/balance_data
cd $PWD/balance_data
cut -f1 $PWD/normalize_data/data_final_norm_hmc_mc >high_cov_hmc

#extract oversampled balanced coord
paste $PWD/normalize_data/data_final_norm_coord high_cov_hmc >high_cov_coord_hmc
#below command is modified to be
Rscript $PWD/src/extract_balance_hmc_class_oversample.R high_cov_coord_hmc $PWD/mm9/hmc_interval cpg_balance

bedtools intersect -a cpg_balance -b $PWD/normalize_data/data_final_norm_final -wa -wb >cpg_balance_data
#shuffle data for training 
shuf cpg_balance_data >cpg_balance_data_shuffle
cut -f7- cpg_balance_data_shuffle >data_final
cut -f1-3 cpg_balance_data_shuffle >data_final_coord


#7.generate training and validation data
mkdir $PWD/train
cd $PWD/train
~/Tool/deep_learning/deepHM_6_20_19/young_cb_model/correction/balance/dna_seq
python3 $PWD/src/generate_data_array.py train_small $PWD/balance_data/data_final $PWD/balance_data/data_final_coord 1 50000 45
python3 $PWD/src/generate_data_array.py val_small $PWD/balance_data/data_final $PWD/balance_data/data_final_coord 50001 100000 45
python3 $PWD/src/generate_data_array.py train_large $PWD/balance_data/data_final $PWD/balance_data/data_final_coord 1 2000000 45
python3 $PWD/src/generate_data_array.py val_large $PWD/balance_data/data_final $PWD/balance_data/data_final_coord 2000001 2700000 45


#8.train model
#8.1.cpg model
mkdir $PWD/train/cpg_module
cd $PWD/train/cpg_module
~/Tool/deep_learning/deepHM_6_20_19/young_cb_model/correction/balance/dna_seq/rnn
export CUDA_VISIBLE_DEVICES="1"
python3 -u $PWD/src/train_model_cpgModule.py pred2_label1 pred2_label2 cpg_model ../train_small.npz ../val_small.npz $PWD/mm9/mm9_genome_encoding.npz
python3 -u $PWD/src/continue_training_cpgModule.py pred3_label1 pred3_label2 cpg_model ../train_large.npz ../val_large.npz $PWD/mm9/mm9_genome_encoding.npz >pred3_out
#find the epoch that has the smallest loss and use this epoch for prediction
grep final pred3_out >pred3_out_grep
python3 $PWD/src/find_best_epoch.py pred3_out_grep

#8.2.dna model
mkdir $PWD/train/dna_module
cd $PWD/train/dna_module
~/Tool/deep_learning/deepHM_6_20_19/young_cb_model/correction/balance/dna_seq/cnn
export CUDA_VISIBLE_DEVICES="0"
python3 -u $PWD/src/train_model_dnaModule.py pred2_label1 pred2_label2 dna_model ../train_small.npz ../val_small.npz $PWD/mm9/mm9_genome_encoding.npz

python3 -u $PWD/src/continue_training_dnaModule.py pred3_label1 pred3_label2 dna_model ../train_large.npz ../val_large.npz $PWD/mm9/mm9_genome_encoding.npz >pred3_out
#find the epoch that has the smallest loss and use this epoch for prediction
grep final pred3_out >pred3_out_grep
python3 $PWD/src/find_best_epoch.py pred3_out_grep

#8.3.joint model
mkdir $PWD/train/joint_module
cd $PWD/train/joint_module
~/Tool/deep_learning/deepHM_6_20_19/young_cb_model/correction/balance/dna_seq/combine
#use the best epoch from rnn and cnn for below training
cp ../dna_module/dna_model-epoch21* .
cp ../cpg_module/cpg_model-epoch83* .
python3 -u $PWD/src/train_model_jointModule.py pred2_label1 pred2_label2 joint_model ../train_small.npz ../val_small.npz cpg_model-epoch83 dna_model-epoch21 $PWD/mm9/mm9_genome_encoding.npz

python3 -u $PWD/src/continue_training_jointModule.py pred3_label1 pred3_label2 joint_model ../train_large.npz ../val_large.npz cpg_model-epoch83 dna_model-epoch21 $PWD/mm9/mm9_genome_encoding.npz >pred3_out
#find the epoch that has the smallest loss and use this epoch for prediction
grep final pred3_out >pred3_out_grep
python3 $PWD/src/find_best_epoch.py pred3_out_grep


#9.predict all cpg
mkdir $PWD/predict
cd $PWD/predict
~/Tool/deep_learning/deepHM_6_20_19/young_cb_model/correction/balance/pred_all_6w_cb1
Rscript $PWD/src/normalize_from_template.R $PWD/combine_features/data_final $PWD/normalize_data/data_norm_info data_final_norm

~/Tool/deep_learning/deepHM_6_20_19/young_cb_model/correction/balance/dna_seq/combine/pred_all
#generate all data
python3 $PWD/src/generate_data_array.py all_data data_final_norm $PWD/mm9/cpg_no_chrM 1 21000000 45

#pred
#use the best epoch from rnn and cnn and joint module for below prediction
cp $PWD/train/joint_module/dna_model-epoch21* .
cp $PWD/train/joint_module/cpg_model-epoch83* .
cp $PWD/train/joint_module/joint_model-epoch39* .

python3 -u $PWD/src/continue_training_pred.py all_label1 all_label2 joint_model-epoch39 $PWD/train/train_small.npz all_data.npz cpg_model-epoch83 dna_model-epoch21 $PWD/mm9/mm9_genome_encoding.npz


