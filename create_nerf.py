import os
import os.path
from collections import OrderedDict
import torch
import torch.optim
from ddp_model import NerfNet
from ddp_config import logger
from network.crf import CRF


def create_nerf(rank, args, camera_mgr, load_camera_mgr=True, load_optimizer=True, use_lr_scheduler=True):
    ###### create network and wrap in ddp; each process should do this
    # fix random seed just to make sure the network is initialized with same weights at different processes
    torch.manual_seed(777+args.seed_offset)
    # very important!!! otherwise it might introduce extra memory in rank=0 gpu
    torch.cuda.set_device(rank)

    models = OrderedDict()
    models['cascade_level'] = args.cascade_level
    models['cascade_samples'] = [int(x.strip()) for x in args.cascade_samples.split(',')]

    net = NerfNet(args).to(rank)
    # 创建 CRF 网络
    crf_net = CRF().to(rank)
    # net = DDP(net, device_ids=[rank], output_device=rank, find_unused_parameters=False)
    # net = DDP(net, device_ids=[rank], output_device=rank)
    # optim = torch.optim.Adam(net.parameters(), lr=args.lrate)

    #增加crf参数化
    crf_lrate = getattr(args, "crf_lrate", args.lrate)
    optim = torch.optim.AdamW(
        [
            {"params": net.parameters(), "lr": args.lrate},
            {"params": crf_net.parameters(), "lr": crf_lrate},
        ],
        weight_decay=args.weight_decay
    )

    models["net"] = net
    models["crf_net"] = crf_net
    models["optim"] = optim



    # optim = torch.optim.AdamW(net.parameters(), lr=args.lrate, weight_decay=args.weight_decay)
    if use_lr_scheduler:
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lambda it: it/4000 if it < 4000 else 0.95**(it/10000))
    else:
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lambda it: 1.0)

    models['lr_scheduler'] = lr_scheduler
    models['camera_mgr'] = camera_mgr.to(rank)

    start = -1

    ###### load pretrained weights; each process should do this
    # if (args.ckpt_path is not None) and (os.path.isfile(args.ckpt_path)):
    #     ckpts = [args.ckpt_path]
    # else:
    if args.force_ckpt_path and args.force_ckpt_path != 'None':  # todo: figure out why it is sometimes 'None' in the string form
        logger.warning(f'Forcing this checkpoint: {args.force_ckpt_path}')
        ckpts = [args.force_ckpt_path]
    else:
        if args.init_ckpt_path is not None and len(args.init_ckpt_path) > 0:
            ckpts = [args.init_ckpt_path]
        else:
            ckpts = []


        new_ckpts = [os.path.join(args.basedir, args.expname, f)
                     for f in sorted(os.listdir(os.path.join(args.basedir, args.expname))) if f.endswith('.pth')]

        if len(new_ckpts) > 0:
            # newer checkpoints of this particular experiment already exist and should be used
            ckpts = new_ckpts
        else:
            # this is the first time we train the model, we should use init_ckpt_path
            load_camera_mgr = False
            load_optimizer = False
            logger.info(f'Initializing the network using: {args.init_ckpt_path}')
            logger.info('NOT loading optimizer and camera manager parameters')


    def path2iter(path):
        tmp = os.path.basename(path)[:-4]
        idx = tmp.rfind('_')
        return int(tmp[idx + 1:])


    # 从checkpoint中加载模型参数时，按照迭代次数排序，优先加载最新的checkpoint
    ckpts = sorted(ckpts, key=path2iter)
    logger.info('Found ckpts: {}'.format(ckpts))
    # 只加载最新的checkpoint
    if len(ckpts) > 0 and not args.no_reload:
        fpath = ckpts[-1]
        logger.info('Reloading from: {}'.format(fpath))
        logger.info(f'load_camera_mgr={load_camera_mgr} load_optimizer={load_optimizer}')
        if (args.force_ckpt_path and args.force_ckpt_path != 'None') or len(new_ckpts) > 0:
            # only if it's reloading newer checkpoints of this particular experiment
            # otherwise, the iteration value is useless
            start = path2iter(fpath)

        # 注意 map_location 的设置，确保在正确的设备上加载模型参数
        map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
        # 从 checkpoint 文件中读出来参数字典
        to_load = torch.load(fpath, map_location=map_location)

        names = ['net']
        if 'crf_net' in to_load:
            names.append('crf_net')
        if load_optimizer:
            names.append('optim')
            names.append('lr_scheduler')

        optim_loaded = True
        for name in names:
            # 加载主网络net, 并且如果 checkpoint 中没有 bound_transform.translation 参数，则手动把当前模型中获取并添加到 checkpoint 的参数字典中
            '''
                bound_transform通常可以理解为对“场景边界”做一个可学习的变换，来适应不同场景的尺度和位置。这个变换通常包含一个平移（translation）和一个缩放（scale）部分。
                三维重建使用原始sfm/colmap坐标通常会有尺度不确定、原点随机、场景偏移的问题，算法通常假设在一个有限的标准立方体内进行重建，因此需要对输入的坐标进行变换，使其适应这个标准立方体。这个变换就是通过bound_transform来实现的。
                没有这个假设的话，位置编码会数值爆炸、mlp训练不稳定，高斯初始化会崩溃，hashgrid采样不均匀
                bound_transform允许模型在训练过程中自动对齐场景中心
            '''
            if name == 'net':
                # print(name) # bound_transform.translation
                if 'bound_transform.translation' not in to_load[name]:
                    to_load[name]['bound_transform.translation'] = models[name].bound_transform.translation
            '''
                lr_scheduler = learning rate scheduler, 用于动态调整学习率以提高训练效率和稳定性。加载 checkpoint 时，如果 lr_scheduler 的参数不在 checkpoint 中，则手动把当前模型中获取并添加到 checkpoint 的参数字典中。
                在 NeRF / 3DGS / 隐式场景优化中，学习率如果不变，通常会出现：前期震荡、中期卡住、后期无法细化
                根据epoch或step动态调整学习率，通常在前期使用较大的学习率以快速收敛，在后期使用较小的学习率以细化结果。
            '''
            # if name == 'lr_scheduler' and name not in to_load:
            if name == 'lr_scheduler':
                if name not in to_load:
                    models[name].last_epoch = start
                    continue
                if not optim_loaded:
                    logger.warning('Skipping lr_scheduler state load because optimizer state was not loaded.')
                    models[name].last_epoch = start
                    continue
            if name not in to_load:
                logger.warning(f'{name} not found in checkpoint, skipping load for this module.')
                # models[name].last_epoch = start
                continue
            # models[name].load_state_dict(to_load[name])  # todo: remove strict
            state_dict = to_load[name]

            # # ===== 关键：去掉 DDP 的 module. 前缀 =====
            # if list(state_dict.keys())[0].startswith('module.'):
            # ===== 关键：去掉 DDP 的 module. 前缀（仅模型参数字典） =====
            if isinstance(state_dict, dict) and len(state_dict) > 0 and list(state_dict.keys())[0].startswith('module.'):
                state_dict = {
                    k.replace('module.', '', 1): v
                    for k, v in state_dict.items()
                }

            # models[name].load_state_dict(state_dict, strict=True)
            if name in ['net', 'crf_net']:
                models[name].load_state_dict(state_dict, strict=True)
            elif name == 'optim':
                try:
                    models[name].load_state_dict(state_dict)
                except ValueError as e:
                    if 'different number of parameter groups' in str(e):
                        optim_loaded = False
                        logger.warning('Optimizer state has different parameter groups from current model; skipping optimizer reload.')
                    else:
                        raise
            else:
                models[name].load_state_dict(state_dict)


        if load_camera_mgr:
            name = 'camera_mgr'
            models[name].load_state_dict(to_load[name])

    logger.info(f'start={start}')
    return start, models
