## DeepH&M

README file created by Yu He on Oct 18, 2019

Please contact yu.he@wustl.edu with questions or for clarifications.


This folder "DeepHM" contains the code for running DeepH&M in the server. Download the code, which will create DeepHM folder .


a. After downloading the code in the folder, download mm9 datasets (for processing genomic features) and sample datasets (MeDIP-seq, MRE-seq, hmC-Seal, TAB-seq and WGBS) and MethPipe tool at the link http://wangftp.wustl.edu/~yhe/DeepHM_datasets/dataset.tar.gz. Uncompress the compressed .tar.gz file in DeepHM folder using command "tar -xzvf dataset.tar.gz". This will generate three folders: mm9, data, MethPipe. Folder mm9 lists files for processing genomic features for mm9. Folder data lists sample data for MeDIP-seq, MRE-seq, hmC-Seal, TAB-seq and WGBS data. Folder MethPipe lists the tool for running mlml.


b. Dependencies for DeepHM include BedTools, bash, python and R. R packages ggplot2, data.table and parallel packages are required. Please make sure these are installed and working before trying to run DeepHM.


c. After finishing the setup, simply run "bash  run_DeepHM.sh" to start the analysis. 



