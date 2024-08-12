set -e

generate_results_name="res_optim"
compress=$1 # 1.0 /0.0
sample_length=$2
grid_size=$3
grid_offset_size=$4
seed=$5
project_path=./test_compress_${compress}_${sample_length}_with_pretrain_${grid_size}_grid_size_${grid_offset_size}_grid_offset_size_seed_${seed}_1
mkdir -p $project_path
select_results_name="moleuclar_optimization_res_select_test"
task_id=1234

export OMP_NUM_THREADS=1
gpu=1 # 4
i=0
j=7500


python -m torch.distributed.launch --nproc_per_node=$gpu --master_port=$(($i+7500))  '../Generator/infer.py' '../OPV_DATA_4_BENCH/'  --valid-subset train \
    --num-workers 8 --ddp-backend=c10d \
    --log-format simple \
    --task unimol_diff_sample \
    --loss unimol_diff_opt_sample_e2e2 \
    --arch unimol_diff_sample_e2e3 \
    --pretrain_path '../weight/checkpoint_best.pt' \
    --batch-size 8 --sample-atom-type 1.0 \
    --results-path $project_path \
    --results-name $generate_results_name \
    --seed ${seed} --use-max-atom 0 --use-real-atom 1 \
    --distributed-world-size 8  --user-dir '../Generator' \
    --contrastive-loss -0.1  --mask-prob 1.0 --noise-type uniform  --neg-num 8 \
    --temperature 100 --masked-dist-loss 1.0 --cos-mask 1.0 --dist-regular-loss 1 \
    --remove-polar-hydrogen --max-atoms 510 --not-use-checkpoint True --use-initial 0.0 --max-dist 1 --coord-clamp 2 \
    --cos-distance 1.0 --refine-type 1.0 --encoder-layers 12 --refine-center 1.0 --pre-set-label 1.0 --use-focal-loss 1.0 \
    --pred-merge-loss 10 --merge-pos-weight 1.0 --use-pred-atom-type 1.0 --not-sample-label 1.0 --refine-edge-type 1.0 --freeze-param 0 \
    --low-neg-bound 4.3 --high-neg-bound 5  --null-pred-loss 0.0 --weighted-distance 0.05 \
    --weighted-distance-temperature 1.0 --x-norm-loss 0.01 --null-dis-range 2 \
    --null-dist-clip 1.0 --no-dist-head 1.0 \
    --delta-pair-repr-norm-loss 0.01  --pred-atom-num 0.0 \
    --dist-bin 8 --dist-bin-val 0.5 --masked-loss 0.1 --reduce-refine-loss 0.1 --use-pred-atom 0 --method-num 1 \
    --low-bound 0.5 --high-bound 0.8 \
    --calculate-metric 0 --ncpu 8 --use-add-num 0 \
    --compress_xy $compress \
    --remove-hydrogen \
    --tune_virtual 1.0 \
    --sample_length $sample_length \
    --cubic 0.0 \
    --grid_size $grid_size \
    --grid_offset_size $grid_offset_size \
    2>&1 | tee ${project_path}/log.txt


# bash infer_opv_with_pretrain.sh  1.0  10000 2 0.1 1

