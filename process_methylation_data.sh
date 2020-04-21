##mlml for tab-seq and wgbs
mkdir $WD/$3
cd $WD/$3
bedtools intersect -a $WD/$2/WGBS_data -b $WD/$2/TAB_data -wa -wb -sorted >wgbs_inter_tab
cut -f1-5 wgbs_inter_tab >wgbs
cut -f6-10 wgbs_inter_tab >tab
python3 $WD/src/format_as_mlml.py wgbs wgbs_format
python3 $WD/src/format_as_mlml.py tab tab_format
$WD/MethPipe/bin/mlml -v -u wgbs_format -h tab_format  -o data_mlml
cut -f5,10 wgbs_inter_tab >wgbs_tab_cov
paste data_mlml wgbs_tab_cov >data_mlml_cov
cut -f1-5,8-9 data_mlml_cov >data_mlml_mc_hmc_cov


##process medip,hmcSeal and mre features
#hmcSeal
sort -k1,1 -k2,2n $WD/$2/hmC-Seal.bedGraph >hmcSeal.bedGraph
python3 $WD/src/extract_feature_from_bp3.py hmcSeal.bedGraph $WD/$1/cpg_no_chrM $WD/$1/chrom_sizes hmcSeal_window
cut -f4- hmcSeal_window >hmcSeal_window_cut

#medip
sort -k1,1 -k2,2n $WD/$2/MeDIP.bedGraph >medip.bedGraph
python3 $WD/src/extract_feature_from_bp3.py medip.bedGraph $WD/$1/cpg_no_chrM $WD/$1/chrom_sizes medip_window
cut -f4- medip_window >medip_window_cut

#mre
cut -f1-3 $WD/$2/MRE.bed >mre.bed
sort -k1,1 -k2,2n mre.bed >mre_sort.filter.bed
bedtools genomecov -i mre_sort.filter.bed -g $WD/$1/chrom_sizes -bg >mre_sort.bedGraph
python3 $WD/src/extract_feature_from_bp3.py mre_sort.bedGraph $WD/$1/cpg_no_chrM $WD/$1/chrom_sizes mre_window
cut -f4- mre_window >mre_window_cut


##combine genomic and methylation features
#coord (3 column),gc_content (8 column),dist_cgi (1 column), cpg_num(8 column), medip (9 column), mre site (1 column), mre (9 column), hmcSeal (9 column)
paste $WD/$1/cpg_no_chrM $WD/$4/gcContent_window_cut $WD/$4/cpg_dist_cgi $WD/$4/cpg_num_window_final medip_window_cut $WD/$4/data_mreSite_final mre_window_cut hmcSeal_window_cut >data_feature
cut -f4- data_feature >data_feature_cut
Rscript $WD/src/convert_cpgNum_to_cpgDensity.R data_feature_cut data_final 
cut -f1-3 data_feature >data_final_coord
paste data_final_coord data_final >data_final_coord_data
