# =========================安装其他需要的库==================================
# pip install numpy==1.24.4 scipy==1.7.3 pandas==1.3.5 pywavelets==1.2.0 # 数据处理与科学计算
# pip install tinycudann torch-optimizer==0.3.0 # PyTorch 相关
# pip install ffmpeg-python==0.2.0 colour-science==0.4.3 pycollada==0.7.2 pyglm==2.5.7 # 图像处理与视觉
# pip install flask==3.1.0 werkzeug==3.1.3 jinja2==3.1.5 itsdangerous==2.2.0 click==8.1.8 # Web 与服务
# pip install dill==0.3.7 isort==5.12.0 mccabe==0.7.0 astroid==3.0.1 pylint==3.0.2 platformdirs==3.11.0 tomlkit==0.12.2 # 工具类与开发辅助
# pip install urwid==2.1.2 urwid-readline==0.13 urwid-utils==0.1.3.dev0 wcmatch==10.0 # 命令行与终端交互
# pip install msgpack==1.0.3 flatbuffers==2.0 zstd==1.5.2.5 # 数据格式与压缩
# pip install aedat==2.0.3 blinker==1.9.0 bracex==2.5.post1 dv==1.0.10 lz4==4.0.0 markupsafe==3.0.2 orderedattrdict==1.6.0 panwid==0.3.5 pymcubes==0.1.2 pynvim==0.4.3 raccoon==3.0.0 roma==1.4.2 tabulate==0.8.9 # 其他杂项
# ===========================================================

# ==========================数据集划分=================================
# python tools/copy_src_to_rgb.py --src /root/data/processed_synth/dynsyn/lego_dyn2/src --dst /root/data/processed_synth/dynsyn/lego_dyn2/train_0/rgb


# ======================单GPU运行=====================================
# export common="--train_split train_0 --N_iters 150001 --N_anneal_lambda 30000 --use_lr_scheduler False --event_threshold 0.5 --tstart 0 --tend 1000 --neg_ratio 0.9 --tonemap_eps 1e-2 --use_viewdirs False --damping_strength 1.0"
# export base="--config configs/mlp2_lambda1e-3.txt --lrate 1e-4 --max_freq_log2_pos 14 --max_freq_log2_time 7"
# export fullmodel=${base}" --lambda_reg 1e-2"
# export scene=lego_dyn2
# export sceneargs=""
# python ./ddp_train_nerf.py --expname fullablations_full_${scene} --scene dynsyn/${scene} $common $sceneargs $fullmodel
# ===========================================================


# ======================多GPU运行=====================================
#  使用 torchrun 启动 DDP，每张卡一个进程
export CUDA_VISIBLE_DEVICES=2,3 # A40
export scene=lego_dyn2
export common="--train_split train_0 --N_iters 150001 --N_anneal_lambda 30000 --use_lr_scheduler False --event_threshold 0.5 --tstart 0 --tend 1000 --neg_ratio 0.9 --tonemap_eps 1e-2 --use_viewdirs False --damping_strength 1.0"
export base="--config configs/mlp2_lambda1e-3.txt --lrate 1e-4 --max_freq_log2_pos 14 --max_freq_log2_time 7"
export fullmodel="${base} --lambda_reg 1e-2"
export sceneargs=""

torchrun --nproc_per_node=2 --master_port=12345 ./ddp_train_nerf1.py \
    --expname add_comp_${scene} \
    --scene dynsyn/${scene} \
    --use_cta_fusion True \
    --cta_event_bins 8 \
    --cta_feat_ch 32 \
    --cta_loss_high_w 0.05 \
    --cta_loss_low_w 0.1 \
    --cta_warmup_iters 5000 \
    $common $sceneargs $fullmodel

## tensorboard --logdir="/data/dyf/DATA/DynEventnerf/logs/exp_${scene}" --port=6006 &
# # 访问地址：localhost:6009
## tensorboard --logdir="/root/log/exp_add_gray_color${scene}" --port=6011 &
## tensorboard --logdir="/root/log/exp_${scene}" --port=6012 &
# ===========================================================


# ======================渲染测试集=====================================
# export x="exp_add_tem_loss_lego_dyn2"
# torchrun --nproc_per_node=2 ddp_test_nerf_video1.py --render_split validation --write_video False --render_bullet_time False --testskip 1 --config /root/log/$x/args.txt --render_timestamp_frames 5 --render_timestamp_periods 1.25
# ===========================================================


# =======================计算指标====================================
# python metric/main.py /root/log/eval_lego_dyn2/add_tem_loss /root/log/eval_lego_dyn2/gt
# python metric/compssim.py /root/log/eval_lego_dyn2/add_tem_loss_corr /root/log/eval_lego_dyn2/gt
# python metric/complpips.py /root/log/eval_lego_dyn2/add_tem_loss_corr /root/log/eval_lego_dyn2/gt
