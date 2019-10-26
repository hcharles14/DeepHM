##process genomic features like gc percent, cpg num, dist to cpg island
mkdir $PWD/$2
cd $PWD/$2
#gc percent
python3 $PWD/src/extract_feature_from_bp3.py $PWD/$1/gc5Base.sort.bedGraph $PWD/$1/cpg_no_chrM $PWD/$1/mm9_chrom_sizes cortex_gcContent_window2

#cpg num at each window
python3 $PWD/src/generate_cpg_window.py $PWD/$1/cpg_no_chrM cpg_window
bedtools coverage -a cpg_window -b $PWD/$1/cpg_no_chrM  -counts >cpg_num_window
cut -f4 cpg_num_window >cpg_num_window_cut
Rscript $PWD/src/format_cpgNum_data.R cpg_num_window_cut cpg_num_window_final

#dist to cpg island
bedtools intersect -a $PWD/$1/cpg_no_chrM -b $PWD/$1/cpgIsland.bed -wao -sorted >cpg_inter_cgi
cut -f1-3,7 cpg_inter_cgi >cpg_inter_cgi_cut
python3 $PWD/src/cal_dist_to_cgi.py cpg_inter_cgi_cut cpg_dist_cgi

