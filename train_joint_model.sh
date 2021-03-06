##joint model
cd $WD/$1
#use the best epoch from rnn and cnn for below training
export CUDA_VISIBLE_DEVICES="1"
python3 -u $WD/src/train_model_jointModule.py joint_pred2_label1 joint_pred2_label2 joint_model train_small.npz val_small.npz $2 $3 $WD/$4/genome_encoding.npz

python3 -u $WD/src/continue_training_jointModule.py joint_pred3_label1 joint_pred3_label2 joint_model train_large.npz val_large.npz $2 $3 $WD/$4/genome_encoding.npz >joint_pred3_out
#find the epoch that has the smallest loss and use this epoch for prediction
grep final joint_pred3_out >joint_pred3_out_grep
#modified 4-20-20
python3 $WD/src/find_best_epoch.py joint_pred3_out_grep joint_model
