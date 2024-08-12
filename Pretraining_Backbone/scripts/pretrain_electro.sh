export NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS=1
data_path='./pretrain_data' # replace to your data path
MASTER_PORT=10086
lr=1e-4
wd=1e-4
batch_size=$1
update_freq=$2
masked_token_loss=1
masked_coord_loss=5
masked_dist_loss=10
x_norm_loss=0.01
delta_pair_repr_norm_loss=0.01
mask_prob=0.15
noise_type="uniform"
noise=1.0
kernel="gaussian"
arch="unimol"
seed=1
warmup_steps=20000
max_steps=2000000 #Stopped at 990542 steps/ 48 epoch

global_batch_size=`expr $batch_size \* $MLP_WORKER_GPU \* $update_freq \* $MLP_WORKER_NUM`
save_dir="./unimol_pretrain_electro_v1_${arch}_lr_${lr}_kernel_${kernel}_mp_${mask_prob}_noise_${noise_type}_${noise}_bs_${global_batch_size}_loss_${masked_token_loss}_${masked_coord_loss}_${masked_dist_loss}_${x_norm_loss}_${delta_pair_repr_norm_loss}_${warmup_steps}_${max_steps}" 
mkdir -p $save_dir


torchrun --nproc_per_node=$MLP_WORKER_GPU --nnodes=$MLP_WORKER_NUM --node_rank=$MLP_ROLE_INDEX  --master_addr=$MLP_WORKER_0_HOST --master_port=$MLP_WORKER_0_PORT  $(which unicore-train) $data_path  --user-dir ../unimol --train-subset train --valid-subset valid \
       --num-workers 4 --ddp-backend=c10d \
       --task unimol --loss unimol --arch $arch  \
       --kernel $kernel \
       --optimizer adam --adam-betas "(0.9, 0.99)" --adam-eps 1e-6 --clip-norm 1.0 --weight-decay $wd \
       --lr-scheduler polynomial_decay --lr $lr --warmup-updates $warmup_steps --total-num-update $max_steps \
       --update-freq $update_freq --seed $seed \
       --keep-last-epochs 2  \
       --fp16 --fp16-init-scale 4 --fp16-scale-window 256 --tensorboard-logdir "${save_dir}/tsb" \
       --max-update $max_steps --log-interval 10 --log-format simple \
       --masked-token-loss $masked_token_loss --masked-coord-loss $masked_coord_loss --masked-dist-loss $masked_dist_loss \
       --x-norm-loss $x_norm_loss --delta-pair-repr-norm-loss $delta_pair_repr_norm_loss \
       --mask-prob $mask_prob --noise-type $noise_type --noise $noise --batch-size $batch_size \
       --save-dir "${save_dir}"  2>& 1 | tee ${save_dir}/train.log


# env: unicore:20230607-pytorch2.0.1-cuda11.7-rdma
# Docker image
# We also provide the docker image. you can pull it by docker pull docker pull dptechnology/unicore:latest-pytorch2.0.1-cuda11.7-rdma. To use GPUs within docker, you need to install nvidia-docker-2 first.
# cd to the path of scripts folder
# bash pretrain_electro.sh 16 1
