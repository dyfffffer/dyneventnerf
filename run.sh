export CUDA_VISIBLE_DEVICES=0,1
unset LD_LIBRARY_PATH

# 单GPU
# export common="--train_split train_0 --N_iters 150001 --N_anneal_lambda 30000 --use_lr_scheduler False --event_threshold 0.5 --tstart 0 --tend 1000 --neg_ratio 0.9 --tonemap_eps 1e-2 --use_viewdirs False --damping_strength 1.0"
# export base="--config configs/mlp2_lambda1e-3.txt --lrate 1e-4 --max_freq_log2_pos 14 --max_freq_log2_time 7"
# export fullmodel=${base}" --lambda_reg 1e-2"

# export scene=lego_dyn2
# export sceneargs=""

# export CUDA_VISIBLE_DEVICES=1,2
# python ./ddp_train_nerf.py --expname fullablations_full_${scene} --scene dynsyn/${scene} $common $sceneargs $fullmodel

# 多GPU 使用 torchrun 启动 DDP，每张卡一个进程
# export common="--train_split train_0 --N_iters 150001 --N_anneal_lambda 30000 --use_lr_scheduler False --event_threshold 0.5 --tstart 0 --tend 1000 --neg_ratio 0.9 --tonemap_eps 1e-2 --use_viewdirs False --damping_strength 1.0"
# export base="--config configs/mlp2_lambda1e-3.txt --lrate 1e-4 --max_freq_log2_pos 14 --max_freq_log2_time 7"
# export fullmodel="${base} --lambda_reg 1e-2"
# export scene=lego_dyn2
# export sceneargs=""
# tensorboard --logdir="/data/dyf/DATA/DynEventnerf/logs/testcrf_${scene}" --port=6006 &

# torchrun --nproc_per_node=2 --master_port=12345 ./ddp_train_nerf1.py \
#     --expname testcrf_${scene} \
#     --scene dynsyn/${scene} \
#     $common $sceneargs $fullmodel

# render test views
export x="testcrf_lego_dyn2"
torchrun --nproc_per_node=2 ddp_test_nerf_video1.py --render_split validation --write_video True --render_bullet_time False --testskip 1 --config /data/dyf/DATA/DynEventnerf/logs/$x/args.txt 

# compute quality 需要validation渲染的图像
# WORK_DIR=/data/dyf/DATA/DynEventnerf/logs/exp_lego_dyn2
# GT_DIR=/data/dyf/DATA/DynEventnerf/processed_synth/dynsyn/lego_dyn2

# python tools/compute_quality.py \
#     --work_dir ${WORK_DIR} \
#     --gt_dir ${GT_DIR} \
#     --splits test