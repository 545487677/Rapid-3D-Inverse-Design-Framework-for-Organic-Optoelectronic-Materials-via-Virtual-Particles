data_path=../Detector_Finetune_Data
n_gpu=1
layers=9
results_path="./infer"  # replace to your results path
weight_path='../weight/finetune.pt'  # replace to your ckpt path
batch_size=16
task_name='opt_dft' # data folder name  finetune_dft/general_dft/opt_dft
task_num=2
loss_func='finetune_smooth_mae_infer'
dict_name='pretrain_dict.txt'
conf_size=1
only_polar=0

MASTER_PORT=10086
python -m torch.distributed.launch --nproc_per_node=$n_gpu --master_port=$MASTER_PORT ../unimol/infer.py --user-dir ../unimol $data_path --task-name $task_name --valid-subset valid \
       --results-path $results_path \
       --num-workers 8 --ddp-backend=c10d --batch-size $batch_size \
       --task mol_finetune_infer --loss finetune_smooth_mae_infer --arch unimol_base \
       --classification-head-name finetune_dft --num-classes $task_num \
       --dict-name $dict_name --conf-size $conf_size \
       --only-polar $only_polar \
       --finetune-from-model $weight_path  \
       --fp16 --fp16-init-scale 4 --fp16-scale-window 256 \
       --seed 1 \
       --log-interval 50 --log-format simple  --encoder-layers ${layers}


##
# R2 for Column 0: 0.8875827504081546
# R2 for Column 1: 0.9603829024778971
# MAE for column 0: 0.06575834071146651
# MAE for column 1: 0.060517746736450204
##