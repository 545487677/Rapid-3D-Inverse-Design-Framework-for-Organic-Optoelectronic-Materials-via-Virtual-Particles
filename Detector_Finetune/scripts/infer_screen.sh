data_path=$1
task_name=$2 # data folder name 
todft=$3     # todft or skip (1/0 or any other string)
tolmdb=$4    # tolmdb or skip (1/0 or any other string)
gap_type=$5  # min/max or skip (1/0 or any other string)
strict_filter=$6 # strict_filter or skip (1/0 or any other string)


echo "Activated options:"

optional_args=""

if [ "$todft" == "todft" ]; then
    optional_args+=" --todft"
    echo " - TODFT computation enabled"
fi
if [ "$tolmdb" == "tolmdb" ]; then
    optional_args+=" --tolmdb"
    echo " - TOLMDB computation enabled"
fi
if [ "$gap_type" == "min" ]; then
    optional_args+=" --gap_type min"
    echo " - GAP type set to minimum"
elif [ "$gap_type" == "max" ]; then
    optional_args+=" --gap_type max"
    echo " - GAP type set to maximum"
fi

if [ "$strict_filter" == "strict_filter" ]; then
    optional_args+=" --strict_filter"
    echo " - Strict filtering enabled"
fi

if [ -z "$optional_args" ]; then
    echo " - No additional options activated"
fi

n_gpu=1
layers=9 #9
results_path="${data_path}/infer"  # replace to your results path
weight_path='../weight/finetune.pt'  # replace to your ckpt path
batch_size=16
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
       --log-interval 50 --log-format simple  --encoder-layers ${layers} $optional_args













