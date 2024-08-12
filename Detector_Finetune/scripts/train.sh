data_path="../Detector_Finetune_Data"  # replace to your data path
save_dir="./save_finetune"  # replace to your save path
n_gpu=1
MASTER_PORT=10086
dict_name="pretrain_dict.txt"
weight_path="../weight/pretrain_backbone.pt"  # replace to your ckpt path (Pretraining Backbone)
task_name="finetune_dft"  # molecular property prediction task name 
task_num=2 
loss_func="finetune_smooth_mae"
lr=$1  # 1e-4
local_batch_size=$2
epoch=$3    #100
dropout=$4 #0
warmup=$5 # 0.06
encoder_layers=$6 # 15
consistent_loss=$7
only_polar=0
conf_size=1 
seed=0
batch_size=32 #32
consistent_loss_formatted=$(echo "${consistent_loss}" | sed 's/\./_/g')
noise_level=0
log_path=./
exp=test_rmnorm_consistent_loss_${consistent_loss_formatted}_opv_rdkit_gen_cond_homo_lumo_gap_epoch_${epoch}_lr_${lr}_bsz_${local_batch_size}_dropout_${dropout}_warmup_${warmup}_encoder_layers_${encoder_layers}
mkdir -p ${log_path}/$exp



if [ "$task_name" == "qm7dft" ] || [ "$task_name" == "qm8dft" ] || [ "$task_name" == "qm9dft" ] || [ "$task_name" == "finetune_dft" ]; then
	metric="valid_agg_mae"
elif [ "$task_name" == "esol" ] || [ "$task_name" == "freesolv" ] || [ "$task_name" == "lipo" ]; then
    metric="valid_agg_rmse"
else 
    metric="valid_agg_auc"
fi

export NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS=1
python -m torch.distributed.launch --nproc_per_node=$n_gpu --master_port=$MASTER_PORT $(which unicore-train) $data_path --task-name $task_name --user-dir ../unimol --train-subset train --valid-subset valid \
       --conf-size $conf_size \
       --num-workers 8 --ddp-backend=c10d \
       --dict-name $dict_name \
       --task mol_finetune --loss $loss_func --arch unimol_base  \
       --classification-head-name $task_name \
       --num-classes $task_num \
       --optimizer adam --adam-betas "(0.9, 0.99)" --adam-eps 1e-6 --clip-norm 1.0 \
       --lr-scheduler polynomial_decay --lr $lr --warmup-ratio $warmup --max-epoch $epoch --batch-size $local_batch_size --pooler-dropout $dropout\
       --update-freq 1 --seed $seed \
       --fp16 --fp16-init-scale 4 --fp16-scale-window 256 \
       --tensorboard-logdir "${log_path}/${exp}/tsb" \
       --log-interval 10 --log-format simple \
       --validate-interval 1 --keep-last-epochs 1 \
       --finetune-from-model $weight_path \
       --best-checkpoint-metric $metric --patience 50 \
       --save-dir "${log_path}/${exp}" --only-polar $only_polar \
       --max-atoms 510 \
       --encoder-layers ${encoder_layers} \
       --consistent-loss ${consistent_loss} \
       2>&1 | tee "${log_path}/${exp}/train.log"



## bash train.sh 1e-4 32 1000 0 0.06 9 0