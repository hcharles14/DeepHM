##mlml for tab-seq and wgbs
mkdir $PWD/$3
cd $PWD/$3
bedtools intersect -a $PWD/$2/WGBS_data -b $PWD/$2/TAB_data -wa -wb -sorted >wgbs_inter_tab
cut -f1-5 wgbs_inter_tab >wgbs
cut -f6-10 wgbs_inter_tab >tab
python3 $PWD/src/format_as_mlml.py wgbs wgbs_format
python3 $PWD/src/format_as_mlml.py tab tab_format
$PWD/MethPipe/bin/mlml -v -u wgbs_format -h tab_format  -o data_mlml
cut -f5,10 wgbs_inter_tab >wgbs_tab_cov
paste data_mlml wgbs_tab_cov >data_mlml_cov
cut -f1-5,8-9 data_mlml_cov >data_mlml_mc_hmc_cov


##process medip,hmcSeal and mre features
#6w cerebellum
#hmcSeal
sort -k1,1 -k2,2n $PWD/$2/hmC-Seal.bedGraph >hmcSeal_6w_cerebellum1.bedGraph
python3 $PWD/src/extract_feature_from_bp3.py hmcSeal_6w_cerebellum1.bedGraph $PWD/$1/cpg_no_chrM $PWD/$1/mm9_chrom_sizes hmcSeal_window_6w_cerebellum1
cut -f4- hmcSeal_window_6w_cerebellum1 >hmcSeal_window_6w_cerebellum1_cut

#medip
sort -k1,1 -k2,2n $PWD/$2/MeDIP.bedGraph >medip_6w_cerebellum1.bedGraph
python3 $PWD/src/extract_feature_from_bp3.py medip_6w_cerebellum1.bedGraph $PWD/$1/cpg_no_chrM $PWD/$1/mm9_chrom_sizes medip_window_6w_cerebellum1
cut -f4- medip_window_6w_cerebellum1 >medip_window_6w_cerebellum1_cut

#mre
ml bedtools
cut -f1-3 $PWD/$2/MRE.bed >mre_WangT_6w-cerebellum-1_cut.filter.bed
sort -k1,1 -k2,2n mre_WangT_6w-cerebellum-1_cut.filter.bed >mre_WangT_6w-cerebellum-1_sort.filter.bed
bedtools genomecov -i mre_WangT_6w-cerebellum-1_sort.filter.bed -g $PWD/$1/mm9_chrom_sizes -bg >mre_WangT_6w-cerebellum-1.bedGraph
python3 $PWD/src/extract_feature_from_bp3.py mre_WangT_6w-cerebellum-1.bedGraph $PWD/$1/cpg_no_chrM $PWD/$1/mm9_chrom_sizes mre_window_6w_cerebellum1
cut -f4- mre_window_6w_cerebellum1 >mre_window_6w_cerebellum1_cut


##combine genomic and methylation features
#coord,dist_cgi,cpg_num, gc_content,medip,mre site, mre,hmcSeal
paste $PWD/$1/cpg_no_chrM $PWD/$4/cpg_dist_cgi $PWD/$4/cpg_num_window_final medip_window_6w_cerebellum1_cut $PWD/$1/data_mreSite_final mre_window_6w_cerebellum1_cut hmcSeal_window_6w_cerebellum1_cut >data_feature1

bedtools intersect -a $PWD/$4/cortex_gcContent_window2 -b data_feature1 -wa -wb -sorted >data_feature2
cut -f5-12,16- data_feature2 >data_feature2_cut
Rscript $PWD/src/convert_cpgNum_to_cpgDensity.R data_feature2_cut data_final 
cut -f1-3 data_feature2 >data_final_coord
paste data_final_coord data_final >data_final_coord_data
