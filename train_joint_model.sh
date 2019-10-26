##joint model
#use the best epoch from rnn and cnn for below training
python3 -u $PWD/src/train_model_jointModule.py joint_pred2_label1 joint_pred2_label2 joint_model train_small.npz val_small.npz $2 $3 $PWD/$4/mm9_genome_encoding.npz

python3 -u $PWD/src/continue_training_jointModule.py joint_pred3_label1 joint_pred3_label2 joint_model train_large.npz val_large.npz $2 $3 $PWD/$4/mm9_genome_encoding.npz >joint_pred3_out
#find the epoch that has the smallest loss and use this epoch for prediction
grep final joint_pred3_out >joint_pred3_out_grep
python3 $PWD/src/find_best_epoch.py joint_pred3_out_grep
