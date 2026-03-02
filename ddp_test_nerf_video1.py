#!/usr/bin/env python3
import os
import time

import torch
import numpy as np
import ffmpeg
import imageio

from data_loader_split import load_event_data_split
from utils import mse2psnr, colorize_np, to8b
from ddp_config import setup_logger, logger, config_parser
from render_single_image import render_single_image
from create_nerf import create_nerf
from nerf_sample_ray_split import CameraManager
from tonemapping import Gamma22


# ============================================================
# 写图 / 写视频工具（只在 rank0 使用）
# ============================================================
class SequenceWriter:
    def __init__(self, out_dir, family, write_video):
        self.out_dir = out_dir
        self.family = family
        self.write_video = write_video
        self.video_stream = None
        self.width = None
        self.height = None

    def write(self, image, image_number):
        if not (image_number.lower().endswith('.png') or image_number.lower().endswith('.jpg')):
            image_number = image_number + '.png'

        def prepend_family(fn):
            return f"{self.family}_{fn}" if self.family else fn

        def append_family(fn):
            return f"{fn}_{self.family}" if self.family else fn

        image = to8b(image)
        outfn = os.path.join(self.out_dir, prepend_family(image_number))
        imageio.imwrite(outfn, image)

        if self.write_video:
            if self.video_stream is None:
                self.height, self.width = image.shape[:2]
                out_filename = append_family(self.out_dir.rstrip('/')) + '.mp4'
                self.video_stream = (
                    ffmpeg
                    .input(
                        'pipe:',
                        format='rawvideo',
                        pix_fmt='rgb24',
                        s=f'{self.width}x{self.height}',
                        r=30
                    )
                    .output(out_filename, pix_fmt='yuv444p', crf=10, blocksize=2048)
                    .overwrite_output()
                    .run_async(pipe_stdin=True)
                )

            self.video_stream.stdin.write(
                image.astype(np.uint8).tobytes()
            )

    def close(self):
        if self.video_stream is not None:
            self.video_stream.stdin.close()
            self.video_stream.wait()
            self.video_stream = None


# ============================================================
# DDP test / render 主逻辑
# ============================================================
def ddp_test_nerf(rank, args):
    """
    每个 rank 都会进入这个函数：
    - 所有 rank：负责 render 自己那一部分 rays
    - 只有 rank0：负责 merge + 写图 / 视频
    """
    is_main = (rank == 0)

    # ---------- logger ----------
    setup_logger()

    # ---------- 根据 GPU 显存动态调 batch ----------
    total_mem_gb = torch.cuda.get_device_properties(rank).total_memory / 1e9
    if total_mem_gb > 14:
        args.N_rand = 1024
        args.chunk_size = 8192
        logger.info("Using large-GPU config")
    else:
        args.N_rand = 512
        args.chunk_size = 4096
        logger.info("Using small-GPU config")

    # ---------- 创建模型（每个 rank 都要） ----------
    camera_mgr = CameraManager(learnable=False)
    start, models = create_nerf(
        rank,
        args,
        camera_mgr,
        load_camera_mgr=False,
        load_optimizer=False
    )

    render_splits = [x.strip() for x in args.render_splits.split(',')]

    for split in render_splits:
        suffix = '_bt' if args.render_bullet_time else ''
        out_dir = os.path.join(
            args.basedir,
            args.expname,
            f'render_{split}{suffix}_{start:06d}'
        )

        # 只有 rank0 创建目录
        if is_main:
            os.makedirs(out_dir, exist_ok=True)

        # ---------- 加载数据 ----------
        ray_samplers = load_event_data_split(
            args.datadir,
            args.scene,
            split,
            view_filter=args.render_view,
            camera_mgr=models['camera_mgr'],
            use_ray_jitter=args.use_ray_jitter,
            polarity_offset=args.polarity_offset,
            skip=args.testskip
        )

        writers_by_family = {}

        for idx in range(len(ray_samplers)):
            viewname = ''

            def write_image(image, family, fname):
                writer = writers_by_family.setdefault(
                    family,
                    SequenceWriter(out_dir, f'{viewname}_{family}', args.write_video)
                )
                writer.write(image, fname)

            # ---------- 时间戳设置 ----------
            if args.render_bullet_time:
                frame_indices = np.array([idx], dtype=np.int64)
                timestamps = frame_indices / (len(ray_samplers) - 1) * args.render_timestamp_periods
            else:
                frame_indices = np.arange(args.render_timestamp_frames)
                timestamps = frame_indices / args.render_timestamp_frames * args.render_timestamp_periods

            for frame_idx, timestamp in zip(frame_indices, timestamps):
                fname = '{:06d}.png'.format(idx)
                if ray_samplers[idx].view_name is not None:
                    fname = os.path.basename(ray_samplers[idx].view_name)

                a, b = os.path.splitext(fname)
                viewname = a

                ts_abs = int(np.round(timestamp * (args.tend - args.tstart) + args.tstart))
                if ts_abs < args.render_tstart or ts_abs > args.render_tend:
                    continue

                timestamp = (ts_abs - args.tstart) / (args.tend - args.tstart)
                fname = f'{ts_abs:04d}{b}'

                if is_main and os.path.isfile(os.path.join(out_dir, fname)):
                    logger.info(f"Skipping existing {fname}")
                    continue

                t0 = time.time()

                # ========== 核心：多卡 render + gather ==========
                ret = render_single_image(
                    rank,
                    args.world_size,
                    models,
                    ray_samplers[idx],
                    args.chunk_size,
                    start,
                    args,
                    timestamp=timestamp
                )

                if is_main:
                    logger.info(f"Rendered {fname} in {time.time() - t0:.2f}s")

                    # ---------- RGB ----------
                    im = Gamma22.from_linear(ret[-1]['rgb_linear']).cpu().numpy()
                    write_image(im, '', fname)

                    # ---------- FG RGB ----------
                    im = Gamma22.from_linear(ret[-1]['fg_rgb_linear']).cpu().numpy()
                    write_image(im, 'fg', fname)

                    # ---------- FG Depth ----------
                    im = ret[-1]['fg_depth'].cpu().numpy()
                    im = colorize_np(im, cmap_name='jet', append_cbar=True, vmin=0.0, vmax=2.0)
                    write_image(im, 'fg_depth', fname)

                    # ---------- optional ----------
                    if 'fg_ldist' in ret[-1]:
                        im = ret[-1]['fg_ldist'].cpu().numpy()
                        im = colorize_np(im, cmap_name='jet', append_cbar=True)
                        write_image(im, 'fg_ldist', fname)

            torch.cuda.empty_cache()

        # ---------- 关闭 writer ----------
        if is_main:
            for w in writers_by_family.values():
                w.close()
            writers_by_family.clear()


# ============================================================
# 程序入口（DDP bootstrap）
# ============================================================
def test():
    parser = config_parser()
    args = parser.parse_args()

    # ---------- 读取 torchrun 环境变量 ----------
    if "RANK" in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.local_rank = int(os.environ["LOCAL_RANK"])
    else:
        args.rank = 0
        args.world_size = 1
        args.local_rank = 0

    torch.cuda.set_device(args.local_rank)

    # ---------- 初始化进程组 ----------
    if args.world_size > 1:
        torch.distributed.init_process_group(
            backend="nccl",
            init_method="env://"
        )

    logger.info(
        f"[DDP test] rank={args.rank}, local_rank={args.local_rank}, world_size={args.world_size}"
    )

    if args.render_tstart < 0:
        args.render_tstart = args.tstart
    if args.render_tend < 0:
        args.render_tend = args.tend

    ddp_test_nerf(args.rank, args)

    # ---------- 清理 ----------
    if args.world_size > 1:
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


if __name__ == '__main__':
    test()
