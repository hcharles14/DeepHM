## DeepH&M

README file created by Yu He on Oct 18, 2019

Please contact yu.he@wustl.edu with questions or for clarifications.


This folder "DeepHM" contains the code for running DeepH&M in the server. Download the code, which will create DeepHM folder .


a. After downloading the code in the folder, download mm9 datasets (for processing genomic features) and sample datasets (MeDIP-seq, MRE-seq, hmC-Seal, TAB-seq and WGBS) and MethPipe tool at the link http://wangftp.wustl.edu/~yhe/DeepHM_datasets/dataset.tar.gz. Uncompress the compressed .tar.gz file in DeepHM folder using command "tar -xzvf dataset.tar.gz". This will generate three folders: mm9, data, MethPipe. Folder mm9 lists files for processing genomic features for mm9. Folder data lists sample data for MeDIP-seq, MRE-seq, hmC-Seal, TAB-seq and WGBS data. Folder MethPipe lists the tool for running mlml.




b. Dependencies for DeepHM include BedTools, bash, python and R. R packages ggplot2, data.table and parallel packages are required. Please make sure these are installed and working before trying to run DeepHM.




c. After finishing the setup, follow the steps below to run DeepH&M. 



1: process genomic features for mm9 or others

Script: bash process_genome.sh mm9_folder genome_process_folder

Inputs:

mm9_folder: genome folder which stores mm9 data

genome_process_folder: output folder where processed mm9 data will be stored




2: process tab-seq, wgbs, medip,hmcSeal and mre data.

Script: bash process_methylation_data.sh mm9_folder data_folder data_process_folder genome_process_folder

Inputs:

mm9_folder: genome folder which stores mm9 data

data_folder: data folder which stores tab-seq, wgbs, medip,hmcSeal and mre data

data_process_folder: output folder where processed data will be stored  

genome_process_folder: processed genome folder which stores output data from step 1




3: select high cov data, normalize and balance data .

Script: bash tune_data.sh data_process_folder data_tune_folder mm9_folder

Inputs:

data_process_folder: processed data folder which stores output data from step 2

data_tune_folder: output folder where tuned data will be stored  

mm9_folder: genome folder which stores mm9 data




4: train cpg and dna model .

Script: bash train_cpg_dna_model.sh data_tune_folder train_folder genome_process_folder

Inputs:

data_tune_folder: tuned data folder which stores output data from step 3

train_folder: output folder where training data will be stored  

genome_process_folder: processed genome folder which stores output data from step 1



5: train joint model .

Script: bash train_joint_model.sh train_folder cpg_model dna_model genome_process_folder

Inputs:

train_folder: training data folder which stores output data from step 4

cpg_model: trained cpg model from step 4

dna_model: trained dna model from step 4  

genome_process_folder: processed genome folder which stores output data from step 1



6: predict from cpg, dna and joint model . For predicting on current dataset, use following script.

Script: bash predict_model.sh train_folder cpg_model dna_model joint_model pred_folder data_process_folder data_tune_folder mm9_folder genome_process_folder

Inputs:

train_folder: training data folder which stores output data from step 4 

cpg_model: trained cpg model from step 4 

dna_model: trained cpg model from step 4 

joint_model: trained cpg model from step 5 

pred_folder: output folder where prediction data will be stored   

data_process_folder: processed data folder which stores output data from step 2 

data_tune_folder: tuned data folder which stores output data from step 3 

mm9_folder: genome folder which stores mm9 data

genome_process_folder: processed genome folder which stores output data from step 1


7: predict from cpg, dna and joint model . For predicting on new dataset, use following scripts.

Scripts:

a.process data for new dataset. Input variables are same as above.

bash process_methylation_data.sh mm9_folder data_folder data_process_folder genome_process_folder

b.predict data for new dataset using trained model. Input variables are same as above.

bash predict_model.sh train_folder cpg_model dna_model joint_model pred_folder data_process_folder data_tune_folder mm9_folder genome_process_folder




