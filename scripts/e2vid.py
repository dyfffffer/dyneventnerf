import numpy as np
import cv2
import torch

import torch
import numpy as np
import cv2

def events_to_image(events, H, W, t0=None, t1=None):
    data = torch.load(events, map_location="cpu")
    fid = 1

    img = data[fid].sum(0).reshape(H, W)
    
    # 白色背景
    rgb = torch.ones((H, W, 3), dtype=torch.float32)
    
    max_abs = img.abs().max() + 1e-6
    
    # 正事件强度
    pos = torch.clamp(img, 0, None) / max_abs
    # 负事件强度
    neg = torch.clamp(-img, 0, None) / max_abs
    
    # 红色通道：正事件保持1，负事件减少
    rgb[..., 0] = 1.0
    rgb[..., 0] = torch.where(img < 0, 1 - neg, rgb[..., 0])
    
    # 绿色通道：正负事件都减少
    rgb[..., 1] = 1.0
    rgb[..., 1] = torch.where(img > 0, 1 - pos, rgb[..., 1])
    rgb[..., 1] = torch.where(img < 0, 1 - neg, rgb[..., 1])
    
    # 蓝色通道：负事件保持1，正事件减少
    rgb[..., 2] = 1.0
    rgb[..., 2] = torch.where(img > 0, 1 - pos, rgb[..., 2])
    
    rgb = torch.clamp(rgb, 0, 1)
    return (rgb.numpy() * 255).astype(np.uint8)



img = events_to_image("/data/dyf/DATA/events.pt", 400, 600)
cv2.imwrite("accumulated_events.png", img)
