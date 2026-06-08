import torch
from torch.autograd import Variable
import os
import argparse
import sys
import importlib
import tempfile

# 强制重新加载模块以确保使用最新版本

if 'utils.dataPro' in sys.modules:
    importlib.reload(sys.modules['utils.dataPro'])

from net.Bgnet_self_attention import Net
from net.unetpp import UNetPlusPlus
from net.unet import UNet
from net.swinunet_tiny import SwinUNetTiny
from net.vit_seg_modeling import VisionTransformer, CONFIGS
from net.DATransUNet import DA_Transformer as DATransUNet, CONFIGS as CONFIGS_DATransUNet
from net.MySeg_Model import MySegNet
from net.S2Net.S2Net import S2NetBraTS
from net.S2Net.unet_CN import UNet as UNetCN
from net.segformer import SegFormerBraTS
from net.MISSFormer import MISSFormerBraTS
from net.nnunet import initialize_network

# AMIN-CNN 文件名含破折号，需用 importlib 加载
import importlib.util as _ilu
_amin_spec = _ilu.spec_from_file_location("amin_cnn", os.path.join(os.path.dirname(__file__), "net", "AMIN-CNN.py"))
_amin_mod = _ilu.module_from_spec(_amin_spec)
_amin_spec.loader.exec_module(_amin_mod)
EfficientAttentionUNet2D = _amin_mod.EfficientAttentionUNet2D
from net.CFATrans import CFATrans2D, CFATransCompositeLoss

# from net.Bgnet import Net

from utils.utils import clip_gradient, AvgMeter, poly_lr
from utils.dataPro import get_data_loaders
import torch.nn.functional as F
import numpy as np
import imageio
from tqdm import tqdm
from sklearn.metrics import f1_score
import torch.nn as nn
import cv2

# 导入图像处理相关库

import numpy as np
import cv2

from torch.optim import AdamW
import random
from scipy.ndimage import rotate, map_coordinates, gaussian_filter
import time
from datetime import datetime
from collections import defaultdict
from torch.cuda.amp import autocast, GradScaler

# 导入BraTS专用稳定组件

from brats_stable_loss import create_brats_stable_training_components, stable_training_step, TransUNetBceDiceLoss, MySegBceLoss, DATransUNetCeDiceLoss, S2NetDiceBceLoss, SegFormerCeDiceLoss, MISSFormerCeDiceLoss, BCEDiceFocalLoss


def _atomic_torch_save(obj, path: str):
    dir_name = os.path.dirname(path) or '.'
    os.makedirs(dir_name, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=os.path.basename(path) + '.', suffix='.tmp', dir=dir_name)
    os.close(fd)
    try:
        with open(tmp_path, 'wb') as f:
            torch.save(obj, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    finally:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass
from fix_training_issues import improved_validate
from utils.visualization import visualize_validation_batch

# 在导入部分添加

# 新增指标计算函数

# 启用极速模式配置

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')
torch.cuda.set_per_process_memory_fraction(0.95)

def seed_everything(seed: int, deterministic: bool = False):
    seed = int(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if bool(deterministic):
        try:
            os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
        except Exception:
            pass
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision('highest')
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:
            try:
                torch.use_deterministic_algorithms(True)
            except Exception:
                pass
    else:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.set_float32_matmul_precision('high')

def sanitize_bn_stats(model: nn.Module):
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            if getattr(m, 'running_mean', None) is not None:
                m.running_mean.data = torch.nan_to_num(m.running_mean.data, nan=0.0, posinf=0.0, neginf=0.0)
            if getattr(m, 'running_var', None) is not None:
                m.running_var.data = torch.nan_to_num(m.running_var.data, nan=1.0, posinf=1.0, neginf=1.0).clamp_min_(1e-6)

def _sanitize_bn_stats_(model: nn.Module):
    return sanitize_bn_stats(model)

class DATransUNetWrapper(nn.Module):
    def __init__(self, backbone: nn.Module, use_adapter: bool = True):
        super().__init__()
        self.backbone = backbone
        self.use_adapter = bool(use_adapter)
        self.adapter = None
        if self.use_adapter:
            self.adapter = nn.Conv2d(4, 3, kernel_size=1, bias=False)
            with torch.no_grad():
                self.adapter.weight.zero_()
                self.adapter.weight[0, 0, 0, 0] = 1.0
                self.adapter.weight[1, 1, 0, 0] = 1.0
                self.adapter.weight[2, 2, 0, 0] = 1.0

    def forward(self, x: torch.Tensor):
        if x.size(1) == 4:
            if self.adapter is not None:
                x = self.adapter(x)
            else:
                x = x[:, :3]
        logits_mc = self.backbone(x)
        if not isinstance(logits_mc, torch.Tensor) or logits_mc.dim() != 4 or logits_mc.size(1) < 4:
            raise RuntimeError(f"DATransUNet backbone must return [B,4,H,W] logits, got: {type(logits_mc)} {getattr(logits_mc, 'shape', None)}")
        prob = torch.softmax(logits_mc, dim=1)
        wt_p = (1.0 - prob[:, 0]).clamp(1e-6, 1.0 - 1e-6)
        tc_p = (prob[:, 2] + prob[:, 3]).clamp(1e-6, 1.0 - 1e-6)
        et_p = (prob[:, 3]).clamp(1e-6, 1.0 - 1e-6)
        prob_3 = torch.stack([wt_p, tc_p, et_p], dim=1)
        logits_3 = torch.log(prob_3 / (1.0 - prob_3))
        return {'stage4': logits_3, 'logits': logits_mc}

class ModelEMA:
    def __init__(self, model, decay=0.9995, warmup_steps=1500):
        # 纯EMA：维护一份模型副本并用 EMA 规则更新，不使用 SWA AveragedModel 的内部逻辑
        import copy
        self.ema_model = copy.deepcopy(model).eval()
        for p in self.ema_model.parameters():
            p.requires_grad_(False)
        self.decay = float(decay)
        self.warmup_steps = int(warmup_steps)
        self.step_count = 0

    def _get_decay(self):
        if self.step_count < self.warmup_steps:
            return self.decay * (self.step_count / max(1, self.warmup_steps))
        return self.decay

    def update(self, model):
        self.step_count += 1
        d = self._get_decay()
        with torch.no_grad():
            ema_params = list(self.ema_model.state_dict().items())
            model_params = list(model.state_dict().items())
            for (k_e, v_e), (k_m, v_m) in zip(ema_params, model_params):
                if v_e.dtype.is_floating_point:
                    v_m_safe = torch.nan_to_num(v_m, nan=0.0, posinf=0.0, neginf=0.0)
                    if not torch.isfinite(v_e).all():
                        v_e.copy_(v_m_safe)
                    else:
                        v_e.copy_(v_e * d + v_m_safe * (1.0 - d))
                else:
                    v_e.copy_(v_m)

def _count_params_m(model: nn.Module):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total / 1e6, trainable / 1e6

def compute_dice(pred, target):
    pred = pred.float()
    target = target.float()
    intersection = torch.sum(pred * target, dim=(-2, -1))
    union = torch.sum(pred, dim=(-2, -1)) + torch.sum(target, dim=(-2, -1))
    return (2. * intersection + 1e-5) / (union + 1e-5)

def compute_hd95(pred, target):
    """计算HD95；遇到空预测/空标签或无边界点时返回None（不计入平均）。"""
    pred = pred.cpu().numpy()
    target = target.cpu().numpy()
    # 空预测或空标签：跳过
    if np.all(pred == 0) or np.all(target == 0):
        return None

    def get_edge_points(img):
        # 使用形态学操作获取边界点
        from scipy.ndimage import binary_erosion
        kernel = np.ones((3,3))
        eroded = binary_erosion(img > 0, kernel)
        boundary = (img > 0) & (~eroded)
        return np.array(np.where(boundary)).T

    pred_points = get_edge_points(pred)
    target_points = get_edge_points(target)
    # 无边界点：跳过
    if len(pred_points) == 0 or len(target_points) == 0:
        return None

    d1 = np.array([np.min(np.linalg.norm(p - target_points, axis=1)) for p in pred_points])
    d2 = np.array([np.min(np.linalg.norm(t - pred_points, axis=1)) for t in target_points])

    # 计算95百分位数
    hd1 = np.percentile(d1, 95)
    hd2 = np.percentile(d2, 95)
    return max(hd1, hd2)

def et_specific_augmentation(image, target):
    # 仅当ET存在时应用增强
    if np.any(target[2] > 0):
        # 随机旋转 - 减少旋转角度
        if random.random() > 0.8:  # 降低概率
            angle = random.uniform(-8, 8)  # 减少角度范围
            image = rotate(image, angle, axes=(1, 2), reshape=False, order=1)
            target = rotate(target, angle, axes=(1, 2), reshape=False, order=0)

        # 弹性变形 - 大幅减少强度
        if random.random() > 0.9:  # 大幅降低概率
            alpha = random.uniform(30, 60)  # 大幅减少变形强度
            sigma = random.uniform(2, 4)    # 减少平滑参数
            shape = image.shape[1:]
            
            # 生成更温和的位移场
            dx = gaussian_filter((random.random() * 2 - 1) * alpha, sigma)
            dy = gaussian_filter((random.random() * 2 - 1) * alpha, sigma)
            
            # 限制位移幅度
            dx = np.clip(dx, -shape[1]*0.05, shape[1]*0.05)
            dy = np.clip(dy, -shape[0]*0.05, shape[0]*0.05)
            
            # 创建网格
            x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
            
            # 计算新坐标并确保在有效范围内
            indices_x = np.clip(x + dx, 0, shape[1]-1).reshape(-1)
            indices_y = np.clip(y + dy, 0, shape[0]-1).reshape(-1)
            
            # 应用变形
            new_image = np.zeros_like(image)
            for c in range(image.shape[0]):
                mapped_result = map_coordinates(
                    image[c], 
                    [indices_y, indices_x],
                    order=1,
                    mode='nearest'  # 使用最近邻避免边界问题
                )
                new_image[c] = mapped_result.reshape(shape)
            
            new_target = np.zeros_like(target)
            for c in range(target.shape[0]):
                mapped_result = map_coordinates(
                    target[c], 
                    [indices_y, indices_x],
                    order=0,
                    mode='nearest'
                )
                new_target[c] = mapped_result.reshape(shape)
            
            image, target = new_image, new_target

    return image, target

def dice_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    multiclass: bool = False,
    from_logits: bool = False,
    smooth: float = 1.0,
    eps: float = 1e-7,
) -> torch.Tensor:
    """Dice loss（1 - Dice系数）"""
    # 统一形状
    if pred.dim() == 3:
        pred = pred.unsqueeze(1)  # [B,1,H,W]
    if target.dim() == 3:
        target = target.unsqueeze(1)

    if multiclass:
        if from_logits:
            pred = F.softmax(pred, dim=1)
        else:
            pred = torch.clamp(pred, 0, 1)
        B, C, H, W = pred.shape
        if target.shape == (B, 1, H, W):  # 索引形式
            target_idx = target.squeeze(1).long()
            target = F.one_hot(target_idx, num_classes=C).permute(0, 3, 1, 2).float()
        elif target.shape == (B, C, H, W):
            target = target.float()
        else:
            raise ValueError("For multiclass, target must be [B,1,H,W] or [B,C,H,W]")

        dims = (0, 2, 3)
        inter = (pred * target).sum(dim=dims)
        card  = (pred + target).sum(dim=dims)
        dice  = (2. * inter + smooth) / (card + smooth + eps)
        return 1. - dice.mean()

    # binary
    if from_logits:
        pred = torch.sigmoid(pred)
    else:
        pred = torch.clamp(pred, 0, 1)

    target = target.float()
    if target.shape != pred.shape:
        if target.dim() == 3 and pred.dim() == 4 and pred.size(1) == 1:
            target = target.unsqueeze(1)
        else:
            raise ValueError("Shapes for binary dice must match")

    dims = (0, 2, 3)
    inter = (pred * target).sum(dim=dims)
    card  = (pred + target).sum(dim=dims)
    dice  = (2. * inter + smooth) / (card + smooth + eps)
    return 1. - dice.mean()

@torch.no_grad()
def dice_coeff(
    input: torch.Tensor,
    target: torch.Tensor,
    reduce_batch_first: bool = True,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    # 对齐 [B,1,H,W]
    if input.dim() == 3:
        input = input.unsqueeze(1)
    if target.dim() == 3:
        target = target.unsqueeze(1)
    input  = input.float()
    target = target.float()

    if reduce_batch_first:
        B = input.size(0)
        dices = []
        for b in range(B):
            x, y = input[b], target[b]
            inter = (x * y).sum()
            sets  = x.sum() + y.sum()
            if sets.item() == 0:
                sets = 2 * inter
            dices.append((2.0 * inter + epsilon) / (sets + epsilon))
        return torch.stack(dices).mean()
    else:
        inter = (input * target).sum()
        sets  = input.sum() + target.sum()
        if sets.item() == 0:
            sets = 2 * inter
        return (2.0 * inter + epsilon) / (sets + epsilon)

class EnhancedAvgMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.valid_count = 0

    def update(self, val, n=1):
        self.val = val
        if val != -1:  # 过滤特殊值
            self.sum += val * n
            self.valid_count += n
            self.count += n
            self.avg = self.sum / self.valid_count if self.valid_count > 0 else 0
        else:
            self.count += n

    def avg_filtered(self, filter_value=-1):
        if self.valid_count > 0:
            return self.sum / self.valid_count
        return 0

def validate(ema_model, val_loader, device='cuda'):
    """改进的验证函数：只在肿瘤存在时计算指标"""
    ema_model.eval()
    dice_meters = {'WT': EnhancedAvgMeter(), 'TC': EnhancedAvgMeter(), 'ET': EnhancedAvgMeter()}
    hd95_meters = {'WT': EnhancedAvgMeter(), 'TC': EnhancedAvgMeter(), 'ET': EnhancedAvgMeter()}

    # 统计信息
    tumor_counts = {'WT': 0, 'TC': 0, 'ET': 0}  # 存在肿瘤的切片数
    total_slices = 0

    # 添加预测值调试统计
    pred_stats = {'WT': [], 'TC': [], 'ET': []}
    debug_count = 0

    with torch.no_grad(), torch.amp.autocast(device_type="cuda", enabled=True):
        for images, targets, _ in val_loader:
            images = images.to(device, non_blocking=True)
            outputs = ema_model(images)
            targets = targets.to(device, non_blocking=True)
            if isinstance(outputs, dict):
                if 'logits' in outputs:
                    main_logits = outputs['logits']
                elif 'stage4' in outputs:
                    main_logits = outputs['stage4']
                else:
                    main_logits = list(outputs.values())[-1]
            else:
                main_logits = outputs
            preds = torch.sigmoid(main_logits)
            
            # 调试前几个batch的预测值分布
            if debug_count < 5:
                for c, name in enumerate(['WT', 'TC', 'ET']):
                    pred_slice = preds[0, c]  # 取第一个样本
                    pred_max = pred_slice.max().item()
                    pred_mean = pred_slice.mean().item()
                    above_05 = (pred_slice > 0.5).sum().item()
                    pred_stats[name].append((pred_max, pred_mean, above_05))
                debug_count += 1
            
            preds_bin = (preds > 0.5).float()
            if preds_bin.dim() == 4 and preds_bin.size(1) >= 3:
                preds_bin[:, 1] = torch.maximum(preds_bin[:, 1], preds_bin[:, 2])
                preds_bin[:, 0] = torch.maximum(preds_bin[:, 0], preds_bin[:, 1])
            
            batch_size = preds.size(0)
            total_slices += batch_size
            
            for i in range(batch_size):
                for c, name in enumerate(['WT', 'TC', 'ET']):
                    target_slice = targets[i, c]
                    
                    # 检查目标中是否存在肿瘤（至少有1个像素>0）
                    if torch.any(target_slice > 0):
                        tumor_counts[name] += 1
                        pred_slice = preds_bin[i, c]
                        
                        # 计算Dice系数
                        intersection = (pred_slice * target_slice).sum()
                        union = pred_slice.sum() + target_slice.sum()
                        
                        if union > 0:
                            dice = (2. * intersection) / (union + 1e-8)
                        else:
                            # 如果预测和目标都为0，Dice=1（完美匹配）
                            dice = 1.0
                        
                        dice_meters[name].update(dice.item())
                        
                        # 计算HD95（只在有肿瘤时计算）
                        try:
                            hd = compute_hd95(pred_slice, target_slice)
                            if hd is not None:
                                hd95_meters[name].update(hd)
                        except Exception as e:
                            # HD95计算失败时跳过
                            print(f"HD95计算失败 {name}: {e}")
                            continue
                    # 注意：当目标不存在肿瘤时，我们不记录任何值（既不记录-1也不记录其他值）
                    # 这样avg_filtered()只会计算真正有肿瘤的切片的平均值

    # 计算最终指标
    metrics = {
        'dice': {k: v.avg_filtered() for k, v in dice_meters.items()},
        'hd95': {k: v.avg_filtered() for k, v in hd95_meters.items()}
    }

    # 打印统计信息
    print(f"\n验证统计信息:")
    print(f"总切片数: {total_slices}")
    for name in ['WT', 'TC', 'ET']:
        print(f"{name}肿瘤存在的切片数: {tumor_counts[name]} ({tumor_counts[name]/total_slices*100:.1f}%)")
        print(f"{name} Dice: {metrics['dice'][name]:.4f}, HD95: {metrics['hd95'][name]:.2f}")

    # 打印预测值调试信息
    print(f"\n预测值调试信息:")
    for name in ['WT', 'TC', 'ET']:
        if pred_stats[name]:
            max_vals = [stat[0] for stat in pred_stats[name]]
            mean_vals = [stat[1] for stat in pred_stats[name]]
            above_05_vals = [stat[2] for stat in pred_stats[name]]
            print(f"{name}: 最大值={max(max_vals):.4f}, 平均值={sum(mean_vals)/len(mean_vals):.4f}, 超0.5像素数={max(above_05_vals)}")
            if max(max_vals) < 0.1:
                print(f"  ⚠️ {name}预测值过低，可能需要调整阈值或检查模型")

    return metrics

class ETEarlyStopper:
    def __init__(self, patience=20, min_epochs=40):
        self.patience = patience
        self.min_epochs = min_epochs
        self.best_avg_dice = 0.0
        self.counter = 0
        self.epoch_count = 0

    def __call__(self, current_metrics, epoch=None):
        if epoch is not None:
            self.epoch_count = epoch + 1
        else:
            self.epoch_count += 1
            
        # 在最小训练轮数之前不触发早停
        if self.epoch_count < self.min_epochs:
            return False
            
        # 使用平均Dice作为早停指标，但增加容忍度
        current_avg_dice = current_metrics.get('avg_dice', 0.0)
        
        # 增加改进阈值，避免微小波动触发早停
        improvement_threshold = 0.001
        if current_avg_dice > self.best_avg_dice + improvement_threshold:
            self.best_avg_dice = current_avg_dice
            self.counter = 0
            return False
        else:
            self.counter += 1
            # 增加额外的稳定性检查
            if self.counter >= self.patience:
                print(f"早停触发：连续{self.patience}个epoch无显著改进")
                print(f"当前最佳Dice: {self.best_avg_dice:.4f}, 当前Dice: {current_avg_dice:.4f}")
                return True
            return False

class GradientMonitor:
    """增强的梯度监控和分析类"""
    def __init__(self, explosion_threshold=100.0, clip_threshold=10.0):
        self.explosion_threshold = explosion_threshold
        self.clip_threshold = clip_threshold
        self.explosion_count = 0
        self.severe_explosion_count = 0

        # 新增：详细梯度分析
        self.grad_history = []
        self.problematic_layers = {}
        self.layer_grad_stats = {}
        
    def analyze_gradients_detailed(self, model):
        """详细分析模型梯度的分布和问题层"""
        grad_norms = []
        layer_info = []
        
        total_norm = 0.0
        max_grad = 0.0
        max_grad_layer = ""
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.data.norm(2).item()
                grad_norms.append(grad_norm)
                total_norm += grad_norm ** 2
                
                if grad_norm > max_grad:
                    max_grad = grad_norm
                    max_grad_layer = name
                
                # 记录问题层（梯度范数>50的层）
                if grad_norm > 50.0:
                    layer_info.append({
                        'name': name,
                        'grad_norm': grad_norm,
                        'shape': list(param.shape),
                        'mean': param.grad.data.mean().item(),
                        'std': param.grad.data.std().item(),
                        'max': param.grad.data.max().item(),
                        'min': param.grad.data.min().item()
                    })
                    
                    # 累积问题层统计
                    if name not in self.problematic_layers:
                        self.problematic_layers[name] = []
                    self.problematic_layers[name].append(grad_norm)
                
                # 记录所有层的梯度统计
                if name not in self.layer_grad_stats:
                    self.layer_grad_stats[name] = []
                self.layer_grad_stats[name].append(grad_norm)
        
        total_norm = total_norm ** 0.5
        
        # 记录梯度历史
        grad_analysis = {
            'total_norm': total_norm,
            'max_grad': max_grad,
            'max_grad_layer': max_grad_layer,
            'problematic_layers': layer_info,
            'grad_distribution': {
                'mean': np.mean(grad_norms) if grad_norms else 0,
                'std': np.std(grad_norms) if grad_norms else 0,
                'max': np.max(grad_norms) if grad_norms else 0,
                'min': np.min(grad_norms) if grad_norms else 0
            }
        }
        
        self.grad_history.append(grad_analysis)
        
        # 保持历史记录在合理范围内
        if len(self.grad_history) > 100:
            self.grad_history = self.grad_history[-50:]
            
        return grad_analysis
        
    def check_and_clip_gradients(self, model, optimizer, scaler):
        """检查梯度爆炸并应用梯度裁剪"""
        # 首先进行详细梯度分析
        grad_analysis = self.analyze_gradients_detailed(model)
        
        max_grad_norm = grad_analysis['total_norm']
        max_grad_layer = grad_analysis['max_grad_layer']
        explosion_detected = False
        skip_update = False
        
        # 检测梯度爆炸
        if max_grad_norm > self.explosion_threshold:
            explosion_detected = True
            self.explosion_count += 1
            
            # 严重梯度爆炸：超过阈值10倍
            if max_grad_norm > self.explosion_threshold * 10:
                self.severe_explosion_count += 1
                skip_update = True
                print(f"🚫 严重梯度爆炸 ({max_grad_norm:.2f})，跳过此次更新")
                print(f"   最大梯度层: {max_grad_layer}")
                
                # 打印问题层详情
                if grad_analysis['problematic_layers']:
                    print(f"   问题层数量: {len(grad_analysis['problematic_layers'])}")
                    for layer in grad_analysis['problematic_layers'][:3]:  # 只显示前3个
                        print(f"     {layer['name']}: 范数={layer['grad_norm']:.2f}, 均值={layer['mean']:.4f}")
                
                optimizer.zero_grad()
                return False, max_grad_norm, max_grad_layer
        
        # 应用梯度裁剪 - 总是进行裁剪以确保稳定性
        if explosion_detected:
            # 梯度爆炸时使用更激进的裁剪
            clip_value = min(self.clip_threshold * 0.1, 0.5)
            print(f"⚠️ 检测到梯度爆炸 ({max_grad_norm:.2f})，应用激进裁剪 (clip={clip_value:.2f})")
            print(f"   最大梯度层: {max_grad_layer}")
        else:
            # 正常情况下也进行更严格的裁剪
            clip_value = self.clip_threshold * 0.5
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_value)
        
        return True, max_grad_norm, max_grad_layer

    def get_gradient_report(self):
        """生成详细的梯度分析报告"""
        if not self.grad_history:
            return "无梯度历史数据"
        
        recent_history = self.grad_history[-20:]  # 最近20次的记录
        total_norms = [h['total_norm'] for h in recent_history]
        
        report = []
        report.append(f"\n{'='*60}")
        report.append(f"梯度分析详细报告 (最近{len(recent_history)}次)")
        report.append(f"{'='*60}")
        
        # 基本统计
        report.append(f"平均梯度范数: {np.mean(total_norms):.2f} (±{np.std(total_norms):.2f})")
        report.append(f"最大梯度范数: {np.max(total_norms):.2f}")
        report.append(f"最小梯度范数: {np.min(total_norms):.2f}")
        
        # 爆炸统计
        explosion_rate = self.explosion_count / len(self.grad_history) * 100
        severe_rate = self.severe_explosion_count / len(self.grad_history) * 100
        report.append(f"梯度爆炸率: {explosion_rate:.1f}% ({self.explosion_count}/{len(self.grad_history)})")
        report.append(f"严重爆炸率: {severe_rate:.1f}% ({self.severe_explosion_count}/{len(self.grad_history)})")
        
        # 问题层分析
        if self.problematic_layers:
            report.append(f"\n问题层分析:")
            for layer_name, grad_norms in self.problematic_layers.items():
                avg_norm = np.mean(grad_norms)
                max_norm = np.max(grad_norms)
                freq = len(grad_norms)
                report.append(f"  {layer_name}: 平均={avg_norm:.2f}, 最大={max_norm:.2f}, 出现次数={freq}")
        
        # 趋势分析
        if len(total_norms) >= 10:
            recent_10 = total_norms[-10:]
            earlier_10 = total_norms[-20:-10] if len(total_norms) >= 20 else total_norms[:-10]
            
            if earlier_10:
                recent_avg = np.mean(recent_10)
                earlier_avg = np.mean(earlier_10)
                trend = "上升" if recent_avg > earlier_avg else "下降"
                change = abs(recent_avg - earlier_avg) / earlier_avg * 100
                report.append(f"\n梯度趋势: {trend} ({change:.1f}%)")
        
        return "\n".join(report)

def train(train_loader, model, optimizer, epoch):
    model.train()
    try:
        setattr(model, 'current_epoch', int(epoch))
    except Exception:
        pass
    loss_meter = AvgMeter()
    dice_meters = [None, None, None]  # WT/TC/ET
    dice_meters_bin = [None, None, None]  # 阈值化Dice（0.5）WT/TC/ET
    # 训练集HD95统计（仅在目标存在时计入）
    train_hd95_meters = [EnhancedAvgMeter(), EnhancedAvgMeter(), EnhancedAvgMeter()]
    # 与验证口径对齐的训练Dice（阈值化且仅统计阳性切片）
    train_dice_pos_meters = [EnhancedAvgMeter(), EnhancedAvgMeter(), EnhancedAvgMeter()]

    # 使用BraTS稳定训练组件，优化梯度处理

    # ✅ 修复预热策略：与 scheduler 协同工作
    warmup_epochs = 5
    if epoch < warmup_epochs:
        # 线性预热
        progress = (epoch + 1) / warmup_epochs
        warmup_factor = 0.1 + 0.9 * progress
        for param_group in optimizer.optimizer.param_groups:
            base_lr = param_group.get('initial_lr', param_group['lr'])
            param_group['lr'] = base_lr * warmup_factor
        print(f"预热阶段 ({epoch+1}/{warmup_epochs}) - 当前学习率: {optimizer.optimizer.param_groups[0]['lr']:.2e}")
    # ✅ 预热结束后，不再手动设置，完全交给 scheduler
    # 注意：scheduler.step() 应该在 epoch 结束后调用

    # ✅ 修复梯度累积策略：减少累积步数，提高泛化能力
    # 原来：accumulation_steps=4 等效 batch_size=72，过大
    # 现在：accumulation_steps=2 等效 batch_size=36，更合理
    accumulation_steps = 2  # ✅ 固定为 2，不再动态调整

    with tqdm(train_loader, desc=f"Epoch {epoch+1}/{opt.epoch}", unit="batch") as pbar:
        for i, pack in enumerate(pbar):
            # optimizer.zero_grad() 已在stable_training_step中调用，避免重复
            
            # ✅ 定期清理显存缓存，防止碎片化导致 OOM
            # 增加清理频率：每 30 个 batch（从 50 降低）
            if i > 0 and i % 30 == 0:
                torch.cuda.empty_cache()
            
            if isinstance(pack, (tuple, list)):
                images, targets, edges = pack[:3]
            else:
                images, targets, edges = pack
            images = images.cuda(non_blocking=True).to(memory_format=torch.channels_last)
            targets = targets.cuda(non_blocking=True).to(memory_format=torch.channels_last)
            edges = edges.cuda(non_blocking=True).to(memory_format=torch.channels_last)

            # 已移除 ROI 裁剪放大（禁用所有 ROI 相关逻辑）

            # 优化的渐进式数据增强策略（仅训练循环中，针对ET样本温和增强）
            # 通过开关控制：默认关闭训练期ET专属增强
            if getattr(opt, 'enable_train_et_aug', False) and epoch < 25:
                augment_strength = max(0.1, 0.6 * np.exp(-epoch / 12.0))
                if random.random() < augment_strength:
                    augment_indices = random.sample(range(images.size(0)), max(1, int(images.size(0) * 0.3)))
                    for j in augment_indices:
                        img_np = images[j].detach().cpu().numpy()
                        target_np = targets[j].detach().cpu().numpy()
                        img_np, target_np = et_specific_augmentation(img_np, target_np)
                        images[j] = torch.from_numpy(img_np).to(images.device)
                        targets[j] = torch.from_numpy(target_np).to(targets.device)

            # 使用稳定训练步骤进行前向与反向过程（包含AMP、梯度裁剪与优化器更新）
            # 消融实验结论：仅保留主损失 + 边界加权Dice（+0.10%提升）
            loss_value, outputs = stable_training_step(
                model=model,
                criterion=criterion,
                optimizer=optimizer,
                scaler=scaler,
                images=images,
                targets=targets,
                device='cuda',
                edges=edges,
                aux_criterion=aux_criterion,
                epoch=epoch,
                boundary_weighted_dice_criterion=boundary_weighted_dice_criterion,  # ✅ 消融实验证明最有效
                deep_supervision_weight=(
                    float(getattr(opt, 'unet_deep_supervision_weight', 0.0))
                    if bool(getattr(opt, 'enable_unet_deep_supervision', False))
                    else 0.0
                ),
                cmee_edge_loss_weight=float(getattr(opt, 'cmee_edge_loss_weight', 0.0)),
            )

            # 若该批次被稳定策略跳过（loss异常），则继续下一个batch
            if loss_value is None or outputs is None:
                continue

            if i == 0:
                try:
                    root = model.module if hasattr(model, 'module') else model
                    if bool(getattr(root, 'enable_mbam', False)) or (getattr(root, 'mbam', None) is not None):
                        mbam = getattr(root, 'mbam', None)
                        mbam_edge_logits = outputs.get('mbam_edge_logits', None) if isinstance(outputs, dict) else None
                        mbam_blend = getattr(root, 'mbam_blend', None)
                        mbam_skip_blend = getattr(root, 'mbam_skip_blend', None)
                        mbam_dec3_blend = getattr(root, 'mbam_dec3_blend', None)
                        mbam_dec2_blend = getattr(root, 'mbam_dec2_blend', None)
                        mbam_dec1_blend = getattr(root, 'mbam_dec1_blend', None)
                        print(f"[MBAM][E{epoch+1}][B{i}] mbam={'ON' if mbam is not None else 'OFF'} edge_logits={'YES' if isinstance(mbam_edge_logits, torch.Tensor) else 'NO'}")
                        if mbam_blend is not None:
                            print(f"[MBAM][E{epoch+1}] mbam_blend_raw={float(mbam_blend.detach().float().cpu()):.6f}")
                            if getattr(mbam_blend, 'grad', None) is not None:
                                print(f"[MBAM][E{epoch+1}] mbam_blend_grad={float(mbam_blend.grad.detach().float().cpu()):.6e}")
                        if mbam_skip_blend is not None:
                            print(f"[MBAM][E{epoch+1}] mbam_skip_blend_raw={float(mbam_skip_blend.detach().float().cpu()):.6f}")
                            if getattr(mbam_skip_blend, 'grad', None) is not None:
                                print(f"[MBAM][E{epoch+1}] mbam_skip_blend_grad={float(mbam_skip_blend.grad.detach().float().cpu()):.6e}")
                        if mbam_dec3_blend is not None:
                            print(f"[MBAM][E{epoch+1}] mbam_dec3_blend_raw={float(mbam_dec3_blend.detach().float().cpu()):.6f}")
                            if getattr(mbam_dec3_blend, 'grad', None) is not None:
                                print(f"[MBAM][E{epoch+1}] mbam_dec3_blend_grad={float(mbam_dec3_blend.grad.detach().float().cpu()):.6e}")
                        if mbam_dec2_blend is not None:
                            print(f"[MBAM][E{epoch+1}] mbam_dec2_blend_raw={float(mbam_dec2_blend.detach().float().cpu()):.6f}")
                            if getattr(mbam_dec2_blend, 'grad', None) is not None:
                                print(f"[MBAM][E{epoch+1}] mbam_dec2_blend_grad={float(mbam_dec2_blend.grad.detach().float().cpu()):.6e}")
                        if mbam_dec1_blend is not None:
                            print(f"[MBAM][E{epoch+1}] mbam_dec1_blend_raw={float(mbam_dec1_blend.detach().float().cpu()):.6f}")
                            if getattr(mbam_dec1_blend, 'grad', None) is not None:
                                print(f"[MBAM][E{epoch+1}] mbam_dec1_blend_grad={float(mbam_dec1_blend.grad.detach().float().cpu()):.6e}")

                        if isinstance(mbam_edge_logits, torch.Tensor):
                            with torch.no_grad():
                                z = mbam_edge_logits.detach().float()
                                z_min = float(torch.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0).min().cpu())
                                z_max = float(torch.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0).max().cpu())
                                z_mean = float(torch.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0).mean().cpu())
                                print(f"[MBAM][E{epoch+1}] mbam_edge_logits: shape={tuple(z.shape)} mean={z_mean:.4f} min={z_min:.4f} max={z_max:.4f}")

                                if (mbam is not None) and hasattr(mbam, 'gate_class_logits'):
                                    gate_w = torch.softmax(mbam.gate_class_logits.detach().float(), dim=0).view(1, -1, 1, 1)
                                    gate = (torch.sigmoid(z) * gate_w).sum(dim=1, keepdim=True)
                                    g_mean = float(gate.mean().cpu())
                                    g_max = float(gate.max().cpu())
                                    g_min = float(gate.min().cpu())
                                    gw = gate_w.view(-1).cpu().numpy().tolist()
                                    print(f"[MBAM][E{epoch+1}] gate: mean={g_mean:.4f} min={g_min:.4f} max={g_max:.4f} gate_w={['%.3f'%x for x in gw]}")

                        gn_sum = 0.0
                        gn_max = 0.0
                        gn_cnt = 0
                        for n, p in root.named_parameters():
                            if (('mbam' in n) or ('mbam_' in n)) and (p.grad is not None):
                                g = p.grad.detach()
                                if g.numel() == 0:
                                    continue
                                v = float(g.float().norm().cpu())
                                gn_sum += v
                                gn_max = max(gn_max, v)
                                gn_cnt += 1
                        if gn_cnt > 0:
                            print(f"[MBAM][E{epoch+1}] grad_norm: params={gn_cnt} avg={gn_sum/gn_cnt:.4e} max={gn_max:.4e}")
                        else:
                            print(f"[MBAM][E{epoch+1}] grad_norm: params=0 (no grads found)")

                    if bool(getattr(opt, 'debug_cmee', False)):
                        cmee_mod = getattr(root, 'cmee', None)
                        enable_cmee_flag = bool(getattr(root, 'enable_cmee', False)) or (cmee_mod is not None)
                        cmee_edge_logits = outputs.get('cmee_edge_logits', None) if isinstance(outputs, dict) else None
                        edge_logits = outputs.get('edge_logits', None) if isinstance(outputs, dict) else None
                        cmee_requires_grad = bool(getattr(root, 'cmee_requires_grad', False))
                        print(f"[CMEE][E{epoch+1}][B{i}] cmee={'ON' if enable_cmee_flag else 'OFF'} trainable={'YES' if cmee_requires_grad else 'NO'} cmee_edge_logits={'YES' if isinstance(cmee_edge_logits, torch.Tensor) else 'NO'} edge_logits={'YES' if isinstance(edge_logits, torch.Tensor) else 'NO'}")
                        if isinstance(outputs, dict):
                            try:
                                print(f"[CMEE][E{epoch+1}] outputs_keys={sorted(list(outputs.keys()))}")
                            except Exception:
                                pass
                        if isinstance(cmee_edge_logits, torch.Tensor):
                            with torch.no_grad():
                                z = cmee_edge_logits.detach().float()
                                z_min = float(torch.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0).min().cpu())
                                z_max = float(torch.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0).max().cpu())
                                z_mean = float(torch.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0).mean().cpu())
                                print(f"[CMEE][E{epoch+1}] cmee_edge_logits: shape={tuple(z.shape)} mean={z_mean:.4f} min={z_min:.4f} max={z_max:.4f}")
                        if cmee_mod is not None:
                            try:
                                gn_sum = 0.0
                                gn_max = 0.0
                                gn_cnt = 0
                                for _p in cmee_mod.parameters():
                                    g = getattr(_p, 'grad', None)
                                    if not isinstance(g, torch.Tensor):
                                        continue
                                    if g.numel() == 0:
                                        continue
                                    v = float(torch.nan_to_num(g.detach().abs().mean().float(), nan=0.0, posinf=0.0, neginf=0.0).cpu())
                                    gn_sum += v
                                    gn_max = max(gn_max, v)
                                    gn_cnt += 1
                                if gn_cnt > 0:
                                    print(f"[CMEE][E{epoch+1}] grad_abs_mean: params={gn_cnt} avg={gn_sum/gn_cnt:.6e} max={gn_max:.6e}")
                                else:
                                    print(f"[CMEE][E{epoch+1}] grad_abs_mean: params=0 (no grads found)")
                            except Exception as _e:
                                print(f"[CMEE][E{epoch+1}] grad_abs_mean debug failed: {_e}")
                except Exception as _e:
                    print(f"[MBAM][E{epoch+1}] debug print failed: {_e}")

            loss_meter.update(loss_value, images.size(0))
            
            ema.update(model)
            
            with torch.no_grad():
                # 获取主要预测输出
                if isinstance(outputs, tuple) and len(outputs) == 4:
                    main_pred = outputs[2]  # o1是最高分辨率输出
                elif isinstance(outputs, dict) and 'logits' in outputs:
                    main_pred = outputs['logits']
                elif isinstance(outputs, dict) and 'stage4' in outputs:
                    main_pred = outputs['stage4']
                else:
                    main_pred = outputs
                    
                preds = torch.sigmoid(main_pred)
                # 使用可配置阈值进行二值化（默认0.45，与验证动态阈值接近）
                thr_val = getattr(opt, 'val_threshold', 0.45)
                preds_bin = (preds > thr_val).float()

                # 与验证一致：仅在目标存在时统计阈值化Dice
                batch_size = preds.size(0)
                for i in range(batch_size):
                    for c in range(3):
                        tgt_slice = targets[i, c]
                        if torch.any(tgt_slice > 0):
                            pred_slice = preds_bin[i, c]
                            # 直接使用 dice_coeff 计算该样本该类的Dice（二值预测）
                            dice_pos = dice_coeff(
                                pred_slice.unsqueeze(0),  # [1,H,W] → [1,1,H,W]
                                tgt_slice.unsqueeze(0),
                                reduce_batch_first=True
                            )
                            train_dice_pos_meters[c].update(dice_pos.item())

                # 训练集HD95（可选，默认关闭以加速）
                if getattr(opt, 'compute_train_hd95', False):
                    for i in range(batch_size):
                        for c in range(3):
                            tgt_slice = targets[i, c]
                            if torch.any(tgt_slice > 0):
                                pred_slice = preds_bin[i, c]
                                try:
                                    hd = compute_hd95(pred_slice, tgt_slice)
                                    if hd is not None:
                                        train_hd95_meters[c].update(hd)
                                except Exception:
                                    pass
            
            # 更新进度条显示
            pbar.set_postfix({
                'loss': f'{loss_meter.avg:.4f}',
                'WT(pos)': f'{train_dice_pos_meters[0].avg_filtered():.3f}',
                'TC(pos)': f'{train_dice_pos_meters[1].avg_filtered():.3f}',
                'ET(pos)': f'{train_dice_pos_meters[2].avg_filtered():.3f}'
            })

    # Epoch结束时的简要报告
    print(f"\n{'='*50}")
    print(f"Epoch {epoch+1} 训练完成")
    print(f"平均损失: {loss_meter.avg:.4f}")
    print(f"当前学习率: {optimizer.optimizer.param_groups[0]['lr']:.2e}")
    print(f"Train Dice(pos-only) - WT: {train_dice_pos_meters[0].avg_filtered():.3f}, TC: {train_dice_pos_meters[1].avg_filtered():.3f}, ET: {train_dice_pos_meters[2].avg_filtered():.3f}")
    if getattr(opt, 'compute_train_hd95', False):
        print(f"Train HD95 - WT: {train_hd95_meters[0].avg_filtered():.2f}, TC: {train_hd95_meters[1].avg_filtered():.2f}, ET: {train_hd95_meters[2].avg_filtered():.2f}")
    else:
        print("Train HD95: skipped (compute_train_hd95=False)")

    return (
        loss_meter.avg,
        train_dice_pos_meters[0].avg_filtered(),
        train_dice_pos_meters[1].avg_filtered(),
        train_dice_pos_meters[2].avg_filtered(),
        (train_hd95_meters[0].avg_filtered() if getattr(opt, 'compute_train_hd95', False) else float('nan')),
        (train_hd95_meters[1].avg_filtered() if getattr(opt, 'compute_train_hd95', False) else float('nan')),
        (train_hd95_meters[2].avg_filtered() if getattr(opt, 'compute_train_hd95', False) else float('nan')),
        optimizer.optimizer.param_groups[0]['lr']
    )

# 主程序入口

# 提前定义 ROI 放大函数，避免在训练过程中出现未定义错误

# 原位置定义保留在文件末尾，这里提前插入一份以确保先被解释执行

# 如需精简，可后续移除文件末尾的重复定义

if __name__ == '__main__':
    # 启用异常检测来找到原地操作
    torch.autograd.set_detect_anomaly(False)
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=80, help='epoch 数量')
    parser.add_argument('--lr', type=float, default=2.0e-4, help='学习率 (平衡速度与稳定性)')  # ✅ 1.5e-4 → 2.0e-4
    parser.add_argument('--batch_size', type=int, default=24, help='批量大小 (Swin Attention 显存优化)')
    parser.add_argument('--seed', type=int, default=2021, help='随机种子（用于可复现实验）')
    parser.add_argument('--deterministic', action='store_true', default=False, help='启用确定性模式（关闭TF32/benchmark并启用确定性算法；可能更慢）')
    parser.add_argument('--train_num_workers', type=int, default=8, help='训练 DataLoader workers 数（可复现实验可设0）')
    parser.add_argument('--train_prefetch_factor', type=int, default=2, help='训练 DataLoader 预取因子（num_workers>0时生效）')
    parser.add_argument('--train_persistent_workers', action='store_true', default=True, help='训练 DataLoader 是否持久化 workers（num_workers>0时生效）')
    parser.add_argument('--resume', type=str, default='', help='从checkpoint恢复训练的路径 (例如: checkpoints/xxx/epoch_020.pth)')
    parser.add_argument('--eval_only', action='store_true', default=False, help='仅加载checkpoint并执行一次验证后退出')
    # 仅加速验证:不改变训练
    parser.add_argument('--val_batch_size', type=int, default=4, help='验证batch size (>=1)')
    parser.add_argument('--val_num_workers', type=int, default=8, help='验证DataLoader工作进程数')
    parser.add_argument('--val_prefetch_factor', type=int, default=2, help='验证预取比例(>0)')
    parser.add_argument('--val_persistent_workers', action='store_true', default=False, help='验证持久化worker')
    parser.add_argument('--trainsize', type=int, default=128, help='训练数据大小')
    parser.add_argument('--clip', type=float, default=1.0, help='梯度裁剪阈值')
    parser.add_argument('--brats_root', type=str, default='', help='BraTS 数据根目录（包含 BraTS19_xxx/BraTS20_xxx 病例文件夹）。设置后会自动覆盖 t1/t2/t1ce/flair/seg root')
    parser.add_argument('--t1_root', type=str, default='./data/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData', help='T1 数据路径')
    parser.add_argument('--t2_root', type=str, default='./data/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData', help='T2 数据路径')
    parser.add_argument('--t1ce_root', type=str, default='./data/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData', help='T1CE 数据路径')
    parser.add_argument('--flair_root', type=str, default='./data/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData', help='FLAIR 数据路径')
    parser.add_argument('--seg_root', type=str, default='./data/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData', help='分割标签路径')
    parser.add_argument('--slice_range', type=int, nargs=2, default=(48, 107), help='切片范围 [start end]')
    parser.add_argument('--min_tumor_pixels', type=int, default=80, help='筛掉肿瘤像素少于该阈值的切片(合并WT/TC/ET)')
    parser.add_argument('--et_min_pixels', type=int, default=20, help='当存在ET时，要求ET像素数>该阈值才计为有效切片')
    parser.add_argument('--tc_min_pixels', type=int, default=35, help='当存在TC时，要求TC像素数>该阈值才计为有效切片')
    parser.add_argument('--train_save', type=str, default='BGNet_NoROI_BS30', help='模型保存路径 - 禁用ROI, BatchSize=30')
    # EMA 控制 (大batch下缩短warmup,加快EMA更新)
    parser.add_argument('--ema_decay', type=float, default=0.999, help='EMA 衰减系数')  # 降低以加快EMA更新速度
    parser.add_argument('--ema_warmup_steps', type=int, default=800, help='EMA 预热步数')  # ✅ 从1500缩短到800
    # 模型结构开关
    parser.add_argument('--eam_mode', type=str, default='all', choices=['all', 'x1x4'], help='EAM 输入模式')
    parser.add_argument('--layer3_stride', type=int, default=1, choices=[1, 2], help='Res2Net layer3 的步幅')
    # 🆕 全局注意力开关（默认启用，提升全局特征提取能力）
    parser.add_argument('--disable_swin_attention', action='store_true', default=False, help='禁用全局注意力模块（默认启用EfficientNonLocalBlock）')
    # 分肿瘤-模态选择开关（保留为默认开启）
    parser.add_argument('--enable_class_modality_selection', action='store_true', default=True, help='启用按类别的后期模态选择融合')
    # 新增：可禁用边界/细节/小目标增强模块（默认开启，可通过命令行关闭）
    # 已移除边界细化模块
    # 已移除小目标增强模块
    # 🆕 EFM 消融开关（默认启用 EFM，可通过 --disable_efm 关闭以做消融）
    parser.add_argument('--disable_efm', action='store_true', default=False, help='禁用 EFM 模块（用于消融实验）')
    parser.add_argument('--unet_cn_efm_layers', type=str, default='up2,up3,up4', help='UNet_CN EFM 注入层级（逗号分隔）：up2,up3,up4；用于消融实验')
    parser.add_argument('--unet_cn_bp_input_levels', type=str, default='0,1,2', help='UNet_CN BoundaryPredictor 输入层级（逗号分隔）：0=x1,1=x2,2=x3；用于消融实验')
    parser.add_argument('--disable_deep_supervision', action='store_true', default=False, help='禁用深度监督模块')

    # 增强损失函数选项
    parser.add_argument('--enable_dynamic_weights', action='store_true', default=True, help='启用动态权重调整')
    parser.add_argument('--disable_dynamic_weights', action='store_true', default=False, help='禁用动态权重调整')
    parser.add_argument('--training_stage', type=str, default='full', choices=['early', 'mid', 'full'], help='训练阶段')
    parser.add_argument('--enable_focal_tversky', action='store_true', default=True, help='仅对TC/ET启用Focal Tversky')
    parser.add_argument('--disable_tversky', action='store_true', default=False, help='TC/ET使用Dice而非Tversky')
    parser.add_argument('--ft_gamma', type=float, default=1.33, help='✅ 原始优秀配置')
    # 损失函数消融实验参数（仅保留有效的）
    parser.add_argument('--disable_ce_loss', action='store_true', default=False, help='禁用CE Loss')
    parser.add_argument('--disable_focal_loss', action='store_true', default=False, help='禁用Focal Loss')
    parser.add_argument('--disable_boundary_weighted_dice', action='store_true', default=False, help='禁用边界加权Dice（消融实验证明有效+0.10%）')
    parser.add_argument('--disable_late_modality_selection', action='store_true', default=False, help='禁用Late Modality Selection（节省4GB显存但性能下降9%）')
    parser.add_argument('--enable_unet_deep_supervision', action='store_true', default=False, help='UNet 启用深度监督：输出 stage1/stage2/stage3 辅助 logits 并参与训练（默认关闭）')
    parser.add_argument('--unet_deep_supervision_weight', type=float, default=0.0, help='UNet 深度监督 loss 权重（0 表示关闭；建议 0.1~0.3 起步）')
    parser.add_argument('--enable_unet_hierarchical_head', action='store_true', default=False, help='UNet 启用层级结构头（保证 ET⊆TC⊆WT；更偏向提升Dice稳定性）')
    parser.add_argument('--unet_hierarchical_warmup_epochs', type=int, default=0, help='UNet 层级结构头 warmup epoch 数（0表示直接启用；>0按epoch线性渐进启用）')
    parser.add_argument('--unet_hierarchical_mode', type=str, default='chain', choices=['chain', 'clamp'], help='UNet 层级结构头模式：chain(乘法链式，约束强但可能压低TC/ET) / clamp(仅裁剪到父类，更温和)')
    parser.add_argument('--cmee_edge_loss_weight', type=float, default=0.0, help='CMEE 边界监督 loss 权重（0表示关闭；建议 0.05~0.15 起步，内部带 warmup）')
    parser.add_argument('--debug_cmee', action='store_true', default=False, help='打印 CMEE 生效诊断信息（每个 epoch 首个 batch 输出一次）')
    parser.add_argument('--enable_unet_mbam', action='store_true', default=False, help='UNet 启用 MBAM 边界引导（输出 edge_logits 并可选门控特征）')
    parser.add_argument('--disable_unet_mbam_guidance', action='store_true', default=False, help='禁用 UNet MBAM 对特征的门控，仅保留 edge_logits 边界监督')
    parser.add_argument('--unet_mbam_guidance_position', type=str, default='dec0', choices=['dec0', 'dec3', 'both'], help='UNet MBAM 边界引导注入位置：dec0(默认，最高分辨率)、dec3(最早解码层)、both(多尺度)')
    parser.add_argument('--unet_mbam_guidance_layers', type=str, default='', help='MBAM guidance 注入层列表（逗号分隔）：dec3,dec2,dec1,dec0,skip；非空时覆盖 unet_mbam_guidance_position')
    parser.add_argument('--unet_mbam_guidance_gain', type=float, default=1.0, help='MBAM guidance 注入增益（乘到 gate 上），用于增强注入强度')
    parser.add_argument('--unet_mbam_guidance_signed', action='store_true', default=False, help='MBAM guidance 使用 signed gate：m=2*gate-1，实现边界外抑制、边界内增强（更偏向HD95）')
    parser.add_argument('--unet_mbam_guidance_gate_source', type=str, default='auto', choices=['auto', 'mbam', 'cmee'], help='MBAM guidance 的 gate 来源：auto(默认按原逻辑)、mbam(强制用MBAM gate)、cmee(强制用cmee edge_logits 生成 gate)')
    parser.add_argument('--unet_mbam_guidance_detach_gate', action='store_true', default=False, help='MBAM guidance 使用 detach 的 gate（减少与 seg loss 的梯度冲突；更稳）')
    parser.add_argument('--unet_mbam_guidance_modulation_clip', type=float, default=0.0, help='MBAM guidance 乘性注入的调制幅度裁剪（0=关闭；例如0.3表示限制到[0.7,1.3]）')
    parser.add_argument('--unet_mbam_gate_from_mbam', action='store_true', default=False, help='UNet MBAM 边界引导 gate 使用 MBAM 自身输出（当同时启用 cmee 时，可实现“cmee 监督 + MBAM 引导”）')
    parser.add_argument('--unet_mbam_input_levels', type=str, default='0123', help='MBAM 输入 encoder 特征层选择（字符串包含0/1/2/3）：0=x0,1=x1,2=x2,3=x3；例如"023"表示去掉x1')
    parser.add_argument('--unet_mbam_input_from_esab', type=str, default='none', choices=['none', 'replace', 'add'], help='MBAM 输入是否使用 ESAB 边缘增强：none(默认，不用) / replace(用ESAB输出替换MBAM输入) / add(将ESAB输出残差加到MBAM输入上，更温和)')
    parser.add_argument('--unet_mbam_gate_mode', type=str, default='weighted', choices=['weighted', 'union', 'max'], help='MBAM gate 聚合方式：weighted(默认，按gate_class_logits加权)、union(类别并集)、max(取最大类)')
    parser.add_argument('--unet_mbam_gate_dilate', type=int, default=0, help='MBAM gate 膨胀半径（0=关闭；1=3x3；2=5x5 maxpool 膨胀）')
    parser.add_argument('--unet_mbam_gate_power', type=float, default=1.0, help='MBAM gate 幂次增强（>1更尖锐，<1更扩散；默认1不变）')
    parser.add_argument('--unet_edge_logits_source', type=str, default='auto', choices=['auto', 'mbam', 'cmee'], help='UNet 边界监督用的 edge_logits 来源：auto(默认：优先cmee，否则MBAM)、mbam(强制MBAM三通道)、cmee(强制cmee)')
    parser.add_argument('--enable_unet_cmee', action='store_true', default=False, help='UNet 启用 cmee（会输出 edge_logits 供解码融合/边界监督）')
    parser.add_argument('--unet_cmee_single_channel', action='store_true', default=False, help='UNet cmee 输出单通道边缘图（WT/TC/ET 合并；仍会输出 edge_logits 供监督）')
    parser.add_argument('--unet_cmee_trainable', action='store_true', default=False, help='UNet cmee 允许训练（解除 no_grad/冻结），使 edge_logits 可通过边界损失学习')
    parser.add_argument('--freeze_unet_cmee', action='store_true', default=False, help='冻结 UNet cmee 参数（用于消融；默认启用 cmee 时为可训练）')
    parser.add_argument('--unet_cmee_modalities', type=str, default='3,1', help='UNet cmee 输入的两个模态（索引或名称），默认 3,1 即 flair+t1ce；例如 3,2(flair+t2) / 1,2(t1ce+t2) / flair,t1ce')
    parser.add_argument('--unet_cmee_scale', type=int, default=2, help='UNet cmee 下采样倍数（默认2；更快但更粗）')
    parser.add_argument('--unet_cmee_base_channels', type=int, default=32, help='UNet cmee base channels（默认32；可调大增强表达但更慢）')
    parser.add_argument('--enable_unet_cmee_fusion', action='store_true', default=False, help='UNet 启用 cmee 特征融合到主干（非gate）：需要 cmee 返回 fused edge feature')
    parser.add_argument('--unet_cmee_fusion_layers', type=str, default='', help='cmee 组合融合位置（逗号分隔）：例如 dec2,dec0；非空则覆盖 unet_cmee_fusion_layer')
    parser.add_argument('--unet_cmee_fusion_layer', type=str, default='dec2', choices=['dec3', 'dec2', 'dec1', 'dec0', 'logits'], help='cmee 融合位置（单点）：dec3/dec2/dec1/dec0/logits')
    parser.add_argument('--unet_cmee_fusion_source', type=str, default='feat', choices=['feat', 'edge_logits'], help='cmee 融合源：feat(默认，融合 cmee fused feature) / edge_logits(融合 cmee 的边界logits，更直接但更强)')
    parser.add_argument('--unet_cmee_fusion_max_gain', type=float, default=0.5, help='cmee 融合最大增益（sigmoid(alpha)*max_gain），建议 0.3~0.8')
    parser.add_argument('--unet_cmee_fusion_weight', type=float, default=1.0, help='cmee 融合额外权重（乘到增益上），用于快速调强度')
    parser.add_argument('--disable_unet_cmee_fusion_detach', action='store_true', default=False, help='cmee 融合不detach特征（允许主损失梯度回传到cmee，可能更强但不稳）')
    parser.add_argument('--unet_cmee_fusion_init_logit', type=float, default=-4.0, help='cmee 融合初始logit（默认-4使初始注入很弱，更稳）')
    parser.add_argument('--enable_unet_fmg', action='store_true', default=False, help='UNet 启用 fmg：使用fg_prob对多尺度skip轻量门控，偏提升平均Dice/TC/ET稳定性')
    parser.add_argument('--unet_fmg_max_gain', type=float, default=0.6, help='fmg 最大增益（建议 0.4~0.8）')
    parser.add_argument('--disable_unet_fmg_detach', action='store_true', default=False, help='fmg 不对 fg_prob detach（允许门控分支反向影响 fg_head，可能更激进但不稳定）')
    parser.add_argument('--unet_fmg_init_logit', type=float, default=-4.0, help='fmg 初始logit（默认-4使初始增益很弱；例如-2/-1可让fmg更早生效）')
    parser.add_argument('--unet_fmg_gate_source', type=str, default='fg', choices=['fg', 'cmee_edge'], help='fmg gate 来源：fg(默认，使用fg_head前景概率) / cmee_edge(使用cmee_edge_logits聚合得到的边界gate)')
    parser.add_argument('--enable_unet_basf', action='store_true', default=False, help='UNet 启用 BASF：boundary-aware skip fusion（在高分辨率 skip 上用边界gate轻量调制，偏改善HD95/边界外点）')
    parser.add_argument('--unet_basf_gate_source', type=str, default='auto', choices=['auto', 'cmee_edge', 'mbam_edge', 'edge_logits'], help='BASF gate 来源：auto(优先cmee，否则mbam) / cmee_edge / mbam_edge / edge_logits')
    parser.add_argument('--unet_basf_max_gain', type=float, default=0.4, help='BASF 最大增益（sigmoid(alpha)*max_gain）；建议 0.2~0.6')
    parser.add_argument('--disable_unet_basf_detach_gate', action='store_true', default=False, help='BASF 不 detach gate（允许 seg loss 反向影响边界分支；可能更强但不稳）')
    parser.add_argument('--unet_basf_gate_dilate', type=int, default=0, help='BASF gate 膨胀半径（0=关闭；1=3x3；2=5x5 maxpool 膨胀）')
    parser.add_argument('--unet_basf_gate_blur', type=int, default=0, help='BASF gate 平滑半径（0=关闭；1=3x3 avgpool；2=5x5）')
    parser.add_argument('--unet_basf_mode', type=str, default='enhance', choices=['enhance', 'suppress'], help='BASF 模式：enhance(边界附近增强skip) / suppress(边界附近抑制skip，偏去毛刺)')
    parser.add_argument('--unet_basf_init_logit', type=float, default=-4.0, help='BASF 初始logit（默认-4初始很弱；-2/-1/0 更早生效）')
    # 训练期ET专属增强开关（默认关闭，几何增强由数据集的 _augment 负责）
    parser.add_argument('--enable_train_et_aug', action='store_true', default=False, help='训练期启用ET专属增强（默认关闭）')

    # 已移除所有 ROI 相关命令行选项

    # 类别权重（Dice/Tversky中的alpha，顺序：WT, TC, ET），适度提高TC/ET有助于小目标
    parser.add_argument('--alpha', type=float, nargs=3, default=[4.5, 4.0, 3.0], help='✅ 原始优秀配置')

    # 新增：类均衡采样权重（控制切片抽样概率）- ✅ 原始配置
    parser.add_argument('--sampler_weight_wt', type=float, default=1.0, help='加权采样-仅WT切片权重')
    parser.add_argument('--sampler_weight_tc', type=float, default=6.0, help='✅ 原始优秀配置')
    parser.add_argument('--sampler_weight_et', type=float, default=8.0, help='✅ 原始优秀配置')

    # 验证相关参数（恢复动态阈值）
    parser.add_argument('--val_threshold', type=float, default=0.5, help='验证时的固定阈值（当启用动态阈值时，该值仅作为起点/参考）')
    parser.add_argument('--compute_train_hd95', action='store_true', default=False, help='是否在训练阶段计算HD95（默认关闭以提升速度）')
    parser.add_argument('--use_dynamic_threshold', action='store_true', default=False, help='是否使用动态阈值（自动搜索每类最优阈值）')
    parser.add_argument('--disable_dynamic_threshold', action='store_true', default=False, help='禁用动态阈值（用于对照实验；与 --use_dynamic_threshold 配合使用）')
    parser.add_argument('--dynamic_threshold_mode', type=str, default='epoch', choices=['batch', 'epoch'], help='动态阈值模式：batch(每个batch重选，默认) / epoch(先校准一次后全程固定，波动更小)')
    parser.add_argument('--dynamic_threshold_calib_batches', type=int, default=8, help='dynamic_threshold_mode=epoch 时，用于校准阈值的前N个batch（建议 6~12）')
    parser.add_argument('--max_val_batches', type=int, default=None, help='验证时最大批次数')
    parser.add_argument('--max_val_tumor_slices', type=int, default=None, help='验证时最大肿瘤切片数（加速）')
    parser.add_argument('--include_empty_in_metrics', action='store_true', default=False, help='验证时是否将空GT样本计入指标（默认不计入）')
    # 为PVT本地权重添加一个健壮的默认绝对路径
    default_pvt_ckpt = os.path.join(os.path.dirname(__file__), 'dataset_cache', 'pvt_v2_b2.pth')
    parser.add_argument('--pvt_local_ckpt', type=str, default=default_pvt_ckpt, help='本地PVTv2-B2预训练权重路径')
    # 可学习输入适配
    parser.add_argument('--use_learnable_adapter', action='store_true', default=True, help='启用1x1 Conv将4通道映射到3通道以保护预训练空间')
    parser.add_argument('--disable_learnable_adapter', action='store_true', default=False, help='禁用可学习输入适配器（将回退为静态4通道适配）')

    default_swin_ckpt = os.path.join(os.path.dirname(__file__), 'pretrained', 'swin_tiny_patch4_window7_224.pth')
    parser.add_argument('--swin_pretrained', type=str, default=default_swin_ckpt, help='SwinUNet 预训练权重路径（默认自动加载；置空则不加载）')
    parser.add_argument('--swin_drop_path_rate', type=float, default=0.2, help='SwinUNet drop_path_rate')
    parser.add_argument('--swin_use_checkpoint', action='store_true', default=False, help='SwinUNet 使用 checkpoint 以节省显存')

    default_transunet_npz = os.path.join(os.path.dirname(__file__), 'pretrained', 'R50+ViT-B_16.npz')
    parser.add_argument('--transunet_variant', type=str, default='R50-ViT-B_16', help='TransUNet variant key in CONFIGS')
    parser.add_argument('--transunet_pretrained_npz', type=str, default=default_transunet_npz, help='TransUNet npz pretrained weights path (optional)')

    default_datransunet_npz = os.path.join(os.path.dirname(__file__), 'pretrained', 'R50+ViT-B_16.npz')
    parser.add_argument('--datransunet_variant', type=str, default='R50-ViT-B_16', help='DATransUNet variant key in CONFIGS_DATransUNet')
    parser.add_argument('--datransunet_pretrained_npz', type=str, default=default_datransunet_npz, help='DATransUNet npz pretrained weights path (default: auto load)')

    default_myseg_ckpt = os.path.join(os.path.dirname(__file__), 'pretrained', 'best_model.pth')
    parser.add_argument('--myseg_pretrained', type=str, default=default_myseg_ckpt, help='MySegNet pretrained weights path (optional)')

    parser.add_argument('--segformer_variant', type=str, default='B0', choices=['B0', 'B1', 'B2', 'B3', 'B4', 'B5'], help='SegFormer variant')

    parser.add_argument('--missformer_token_mlp_mode', type=str, default='mix_skip', choices=['mix', 'mix_skip', 'mlp'], help='MISSFormer token mlp mode')

    parser.add_argument('--bce_dice_focal_class_weights', type=float, nargs=3, default=[1.0, 1.0, 1.0], help='bce_dice_focal: WT/TC/ET 类别权重 (3个浮点数)')

    parser.add_argument('--main_loss', type=str, default='auto', choices=['auto', 'brats_stable', 'transunet_ce_dice', 'myseg_bce', 'datransunet_ce_dice', 's2net_dice_bce', 'segformer_ce_dice', 'missformer_ce_dice', 'bce_dice_focal', 'cfatrans_composite'], help='选择主损失函数')

    parser.add_argument('--model_name', type=str, default='bgnet', choices=['bgnet', 'unetpp', 'unet', 'unet_cn', 'unet_cn_efm', 'unet_cn_msefm', 'unet_cn_msefm_noedge', 'swinunet', 'transunet', 'mysegnet', 'datransunet', 's2net', 'segformer', 'missformer', 'nnunet2d', 'amin_cnn', 'cfatrans'], help='选择模型架构')
    # 🔧 默认通道减半，无需手动指定

    # 🆕 双编码器参数
    parser.add_argument('--enable_dual_encoder', action='store_true', default=False, help='启用双编码器 (Res2Net + Swin Transformer)')
    parser.add_argument('--swin_fusion_weight', type=float, default=0.3, help='Swin Transformer 融合权重 (0.0-1.0)')
    # parser.add_argument('--width_multiplier', type=float, default=0.5, help='🔧 Backbone通道缩减倍数 (1.0=原始, 0.5=减半)')
    opt = parser.parse_args()

    default_brats2019_root = './data/BraTS2019_TrainingData/MICCAI_BraTS_2019_Data_Training'
    brats_root = str(getattr(opt, 'brats_root', '')).strip()
    if (not brats_root) and (not os.path.exists(str(getattr(opt, 't1_root', '')))) and os.path.exists(default_brats2019_root):
        brats_root = default_brats2019_root
    if brats_root:
        opt.t1_root = brats_root
        opt.t2_root = brats_root
        opt.t1ce_root = brats_root
        opt.flair_root = brats_root
        opt.seg_root = brats_root

    seed_everything(int(getattr(opt, 'seed', 2021)), deterministic=bool(getattr(opt, 'deterministic', False)))
    try:
        torch.cuda.set_per_process_memory_fraction(0.9)
    except Exception:
        pass

    # 🔧 通道数已在模型中固定为 0.5（减半），无需命令行参数

    # 确保检查点目录存在
    os.makedirs(os.path.join('checkpoints', opt.train_save), exist_ok=True)

    # 计算可学习适配器标志
    use_adapter_flag = opt.use_learnable_adapter and not opt.disable_learnable_adapter

    if opt.model_name == 'unetpp':
        model = UNetPlusPlus(num_classes=3, in_channels=4, base_channels=64).cuda()
    elif opt.model_name == 'unet':
        enable_cmee = bool(getattr(opt, 'enable_unet_cmee', False))
        freeze_cmee = bool(getattr(opt, 'freeze_unet_cmee', False))
        cmee_trainable = (enable_cmee and (not freeze_cmee)) or bool(getattr(opt, 'unet_cmee_trainable', False))
        model = UNet(
            num_classes=3,
            in_channels=4,
            base_channels=64,
            enable_deep_supervision=bool(getattr(opt, 'enable_unet_deep_supervision', False)),
            enable_mbam=bool(getattr(opt, 'enable_unet_mbam', False)),
            enable_mbam_guidance=(not getattr(opt, 'disable_unet_mbam_guidance', False)),
            mbam_guidance_position=str(getattr(opt, 'unet_mbam_guidance_position', 'dec0')),
            mbam_guidance_layers=str(getattr(opt, 'unet_mbam_guidance_layers', '')),
            mbam_guidance_gain=float(getattr(opt, 'unet_mbam_guidance_gain', 1.0)),
            mbam_guidance_signed=bool(getattr(opt, 'unet_mbam_guidance_signed', False)),
            mbam_guidance_gate_source=str(getattr(opt, 'unet_mbam_guidance_gate_source', 'auto')),
            mbam_guidance_detach_gate=bool(getattr(opt, 'unet_mbam_guidance_detach_gate', False)),
            mbam_guidance_modulation_clip=float(getattr(opt, 'unet_mbam_guidance_modulation_clip', 0.0)),
            mbam_gate_from_mbam=bool(getattr(opt, 'unet_mbam_gate_from_mbam', False)),
            mbam_input_levels=str(getattr(opt, 'unet_mbam_input_levels', '0123')),
            mbam_input_from_esab=str(getattr(opt, 'unet_mbam_input_from_esab', 'none')),
            mbam_gate_mode=str(getattr(opt, 'unet_mbam_gate_mode', 'weighted')),
            mbam_gate_dilate=int(getattr(opt, 'unet_mbam_gate_dilate', 0)),
            mbam_gate_power=float(getattr(opt, 'unet_mbam_gate_power', 1.0)),
            enable_cmee=enable_cmee,
            cmee_single_channel=bool(getattr(opt, 'unet_cmee_single_channel', False)),
            cmee_trainable=cmee_trainable,
            cmee_modalities=str(getattr(opt, 'unet_cmee_modalities', '3,1')),
            cmee_scale=int(getattr(opt, 'unet_cmee_scale', 2)),
            cmee_base_channels=int(getattr(opt, 'unet_cmee_base_channels', 32)),
            enable_cmee_fusion=bool(getattr(opt, 'enable_unet_cmee_fusion', False)),
            cmee_fusion_layers=str(getattr(opt, 'unet_cmee_fusion_layers', '')),
            cmee_fusion_layer=str(getattr(opt, 'unet_cmee_fusion_layer', 'dec2')),
            cmee_fusion_source=str(getattr(opt, 'unet_cmee_fusion_source', 'feat')),
            cmee_fusion_max_gain=float(getattr(opt, 'unet_cmee_fusion_max_gain', 0.5)),
            cmee_fusion_weight=float(getattr(opt, 'unet_cmee_fusion_weight', 1.0)),
            cmee_fusion_detach=(not bool(getattr(opt, 'disable_unet_cmee_fusion_detach', False))),
            cmee_fusion_init_logit=float(getattr(opt, 'unet_cmee_fusion_init_logit', -4.0)),
            edge_logits_source=str(getattr(opt, 'unet_edge_logits_source', 'auto')),
            enable_hierarchical_head=bool(getattr(opt, 'enable_unet_hierarchical_head', False)),
            hierarchical_warmup_epochs=int(getattr(opt, 'unet_hierarchical_warmup_epochs', 0)),
            hierarchical_mode=str(getattr(opt, 'unet_hierarchical_mode', 'chain')),
            enable_fmg=False,
            fmg_max_gain=float(getattr(opt, 'unet_fmg_max_gain', 0.6)),
            fmg_detach=(not bool(getattr(opt, 'disable_unet_fmg_detach', False))),
            fmg_init_logit=float(getattr(opt, 'unet_fmg_init_logit', -4.0)),
            fmg_gate_source=str(getattr(opt, 'unet_fmg_gate_source', 'fg')),
            enable_basf=bool(getattr(opt, 'enable_unet_basf', False)),
            basf_gate_source=str(getattr(opt, 'unet_basf_gate_source', 'auto')),
            basf_max_gain=float(getattr(opt, 'unet_basf_max_gain', 0.4)),
            basf_detach_gate=(not bool(getattr(opt, 'disable_unet_basf_detach_gate', False))),
            basf_gate_dilate=int(getattr(opt, 'unet_basf_gate_dilate', 0)),
            basf_gate_blur=int(getattr(opt, 'unet_basf_gate_blur', 0)),
            basf_mode=str(getattr(opt, 'unet_basf_mode', 'enhance')),
            basf_init_logit=float(getattr(opt, 'unet_basf_init_logit', -4.0)),
        ).cuda()
    elif opt.model_name == 'unet_cn':
        model = UNetCN(in_channels=4, num_classes=3, base_channels=64).cuda()
    elif opt.model_name == 'unet_cn_efm':
        # 单尺度 EFM：仅在最后一层使用 GT 边界进行特征增强
        model = UNetCN(in_channels=4, num_classes=3, base_channels=64, enable_efm=True).cuda()
    elif opt.model_name == 'unet_cn_msefm':
        # M2: 多尺度 EFM + BoundaryPredictor（完整方案）
        efm_layers = str(getattr(opt, 'unet_cn_efm_layers', 'up2,up3,up4'))
        bp_input_levels = str(getattr(opt, 'unet_cn_bp_input_levels', '0,1,2'))
        model = UNetCN(in_channels=4, num_classes=3, base_channels=64, enable_msefm=True, use_edge_pred=True, efm_layers=efm_layers, bp_input_levels=bp_input_levels).cuda()
    elif opt.model_name == 'unet_cn_msefm_noedge':
        # M1: 多尺度 EFM（训练用GT边界，推理无边界）
        efm_layers = str(getattr(opt, 'unet_cn_efm_layers', 'up2,up3,up4'))
        model = UNetCN(in_channels=4, num_classes=3, base_channels=64, enable_msefm=True, use_edge_pred=False, efm_layers=efm_layers).cuda()
    elif opt.model_name == 'swinunet':
        swin_pretrained = getattr(opt, 'swin_pretrained', None)
        if swin_pretrained is not None and str(swin_pretrained).strip() == "":
            swin_pretrained = None
        model = SwinUNetTiny(
            num_classes=3,
            in_channels=4,
            img_size=int(opt.trainsize),
            patch_size=4,
            window_size=None,
            drop_path_rate=float(getattr(opt, 'swin_drop_path_rate', 0.2)),
            use_checkpoint=bool(getattr(opt, 'swin_use_checkpoint', False)),
            pretrained=swin_pretrained,
        ).cuda()
    elif opt.model_name == 'transunet':
        variant = getattr(opt, 'transunet_variant', 'ViT-B_16')
        if variant not in CONFIGS:
            raise ValueError(f"Unknown transunet_variant: {variant}. Available: {list(CONFIGS.keys())}")
        config = CONFIGS[variant]
        if config.patches.get('grid') is not None:
            gs = int(opt.trainsize) // 16
            if gs < 1:
                raise ValueError(f"trainsize too small for R50-hybrid TransUNet. Got trainsize={opt.trainsize}")
            config.patches.grid = (gs, gs)
        config.n_classes = 3
        config.activation = 'sigmoid'
        if not hasattr(config, 'n_skip'):
            config.n_skip = 0
        if not hasattr(config, 'skip_channels'):
            config.skip_channels = [0, 0, 0, 0]
        if int(opt.trainsize) % int(config.patches['size'][0]) != 0:
            raise ValueError(f"trainsize must be divisible by patch size {config.patches['size'][0]}. Got trainsize={opt.trainsize}")
        model = VisionTransformer(config, img_size=int(opt.trainsize), num_classes=3, vis=False, in_channels=4).cuda()
        npz_path = getattr(opt, 'transunet_pretrained_npz', '')
        if npz_path is None:
            npz_path = ''
        if str(npz_path).strip() != '' and os.path.exists(str(npz_path)):
            weights = np.load(str(npz_path))
            model.load_from(weights)
    elif opt.model_name == 'datransunet':
        variant = getattr(opt, 'datransunet_variant', 'R50-ViT-B_16')
        if variant not in CONFIGS_DATransUNet:
            raise ValueError(f"Unknown datransunet_variant: {variant}. Available: {list(CONFIGS_DATransUNet.keys())}")
        config = CONFIGS_DATransUNet[variant]
        # 让 grid 与当前输入分辨率匹配，保证 patch_size=16
        if config.patches.get('grid') is not None:
            gs = int(opt.trainsize) // 16
            if gs < 1:
                raise ValueError(f"trainsize too small for R50-hybrid DATransUNet. Got trainsize={opt.trainsize}")
            config.patches.grid = (gs, gs)
        config.n_classes = 4
        if not hasattr(config, 'n_skip'):
            config.n_skip = 0
        if not hasattr(config, 'skip_channels'):
            config.skip_channels = [0, 0, 0, 0]
        base_model = DATransUNet(config, img_size=int(opt.trainsize), num_classes=4, vis=False)
        npz_path = getattr(opt, 'datransunet_pretrained_npz', '')
        if npz_path is None:
            npz_path = ''
        if str(npz_path).strip() != '' and os.path.exists(str(npz_path)):
            base_model.load_from(weights=np.load(str(npz_path)))
        model = DATransUNetWrapper(base_model, use_adapter=use_adapter_flag).cuda()
    elif opt.model_name == 'mysegnet':
        model = MySegNet(in_ch=4, out_ch=3, img_size=int(opt.trainsize)).cuda()
        ckpt_path = getattr(opt, 'myseg_pretrained', '')
        if ckpt_path is None:
            ckpt_path = ''
        if str(ckpt_path).strip() != '' and os.path.exists(str(ckpt_path)):
            pretrained_dict = torch.load(str(ckpt_path), map_location='cpu', weights_only=False)
            if isinstance(pretrained_dict, dict) and not all(isinstance(k, str) for k in pretrained_dict.keys()):
                pretrained_dict = dict(pretrained_dict)
            model_dict = model.state_dict()

            k_in = 'c1.layer.0.weight'
            if k_in in pretrained_dict and k_in in model_dict:
                w = pretrained_dict[k_in]
                if isinstance(w, torch.Tensor) and w.ndim == 4 and w.shape[1] == 1 and model_dict[k_in].shape[1] == 4:
                    w4 = torch.zeros_like(model_dict[k_in])
                    w4[:, :1] = w
                    w4[:, 1:] = w.repeat(1, 3, 1, 1)
                    pretrained_dict[k_in] = w4

            k_out_w, k_out_b = 'out.weight', 'out.bias'
            if k_out_w in pretrained_dict and k_out_w in model_dict:
                w = pretrained_dict[k_out_w]
                if isinstance(w, torch.Tensor) and w.ndim == 4 and w.shape[0] == 1 and model_dict[k_out_w].shape[0] == 3:
                    pretrained_dict[k_out_w] = w.repeat(3, 1, 1, 1)
            if k_out_b in pretrained_dict and k_out_b in model_dict:
                b = pretrained_dict[k_out_b]
                if isinstance(b, torch.Tensor) and b.ndim == 1 and b.shape[0] == 1 and model_dict[k_out_b].shape[0] == 3:
                    pretrained_dict[k_out_b] = b.repeat(3)

            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and np.shape(model_dict[k]) == np.shape(v)}
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)
    elif opt.model_name == 's2net':
        model = S2NetBraTS(in_channels=4, num_classes=3).cuda()
    elif opt.model_name == 'segformer':
        model = SegFormerBraTS(in_channels=4, num_classes=3, model_name=str(getattr(opt, 'segformer_variant', 'B0')), image_size=int(opt.trainsize)).cuda()
    elif opt.model_name == 'missformer':
        if int(opt.trainsize) != 224:
            raise ValueError(f"MISSFormer requires --trainsize 224. Got trainsize={opt.trainsize}")
        model = MISSFormerBraTS(in_channels=4, num_classes=3, token_mlp_mode=str(getattr(opt, 'missformer_token_mlp_mode', 'mix_skip')), image_size=int(opt.trainsize)).cuda()
    elif opt.model_name == 'nnunet2d':
        model = initialize_network(threeD=False, num_classes=3, num_input_channels=4).cuda()
    elif opt.model_name == 'amin_cnn':
        # AMIN-CNN: EfficientNetB4 + Multi-Scale Attention U-Net（对比模型）
        try:
            model = EfficientAttentionUNet2D(in_channels=4, out_channels=3, pretrained=True).cuda()
            print("✅ AMIN-CNN: 已加载 EfficientNetB4 预训练权重")
        except Exception as e:
            print(f"⚠️  AMIN-CNN: 预训练权重加载失败 ({e})，使用随机初始化")
            model = EfficientAttentionUNet2D(in_channels=4, out_channels=3, pretrained=False).cuda()
    elif opt.model_name == 'cfatrans':
        # CFATrans: CNN-Transformer 混合窄金字塔（对比模型）
        model = CFATrans2D(in_channels=4, num_classes=3).cuda()
    else:
        model = Net(
            num_classes=3,
            eam_mode='all',
            layer3_stride=1,
            enable_swin_attention=not getattr(opt, 'disable_swin_attention', False),  # ✅ 默认启用，可通过--disable_swin_attention禁用
            # 小目标增强模块已移除
            enable_paper_edge=True,
            paper_edge_scale=2,
            paper_edge_base_channels=32,
            paper_edge_requires_grad=False,
            enable_late_modality_selection=not getattr(opt, 'disable_late_modality_selection', False),
            # width_multiplier 使用模型默认值 0.5（已在 Bgnet_self_attention.py 中固定）
            # 🆕 双编码器参数
            enable_dual_encoder=opt.enable_dual_encoder,
            swin_fusion_weight=opt.swin_fusion_weight,
            # 🆕 EFM 消融：根据命令行开关决定是否启用 EFM
            enable_efm=not getattr(opt, 'disable_efm', False)
        ).cuda()

    total_m, trainable_m = _count_params_m(model)
    print(f"Model Params: total={total_m:.2f}M | trainable={trainable_m:.2f}M")

    ema = ModelEMA(model, decay=opt.ema_decay, warmup_steps=opt.ema_warmup_steps)

    # 创建BraTS专用稳定训练组件 - 增强版（方案D + 边界加权 Dice Loss）
    # 处理损失函数消融参数
    enable_dynamic = opt.enable_dynamic_weights and not getattr(opt, 'disable_dynamic_weights', False)
    enable_tversky = opt.enable_focal_tversky and not getattr(opt, 'disable_tversky', False)
    enable_boundary = not getattr(opt, 'disable_boundary_weighted_dice', False)  # 消融实验证明：边界加权Dice最有效

    if str(getattr(opt, 'model_name', '')).lower().strip() == 's2net':
        enable_boundary = False
    if str(getattr(opt, 'model_name', '')).lower().strip() == 'swinunet':
        enable_boundary = False
    if str(getattr(opt, 'model_name', '')).lower().strip() == 'unetpp':
        enable_boundary = False
    if str(getattr(opt, 'model_name', '')).lower().strip() == 'segformer':
        enable_boundary = False
    if str(getattr(opt, 'model_name', '')).lower().strip() == 'missformer':
        enable_boundary = False
    if str(getattr(opt, 'model_name', '')).lower().strip() == 'amin_cnn':
        enable_boundary = False
    if str(getattr(opt, 'model_name', '')).lower().strip() == 'cfatrans':
        enable_boundary = False

    # 消融实验结论：仅保留主损失 + 边界加权Dice
    criterion, optimizer, scaler, scheduler, topology_criterion, boundary_weighted_dice_criterion, boundary_tversky_criterion, progressive_edge_criterion = create_brats_stable_training_components(
        model=model, 
        base_lr=opt.lr,
        enable_dynamic_weights=enable_dynamic,
        training_stage=opt.training_stage,
        class_alpha=opt.alpha,
        enable_focal_tversky=enable_tversky,
        ft_gamma=getattr(opt, 'ft_gamma', 1.33),
        enable_boundary_weighted_dice=enable_boundary,  # ✅ 消融实验证明最有效 (+0.10%)
        enable_boundary_tversky=False,  # ❌ 消融实验移除
        enable_progressive_edge_supervision=False,  # ❌ 消融实验证明有害 (-0.16%)
        disable_ce_loss=getattr(opt, 'disable_ce_loss', False),
        disable_focal_loss=getattr(opt, 'disable_focal_loss', False),
    )

    main_loss = getattr(opt, 'main_loss', 'auto')
    if main_loss == 'auto':
        if opt.model_name == 'transunet':
            main_loss = 'transunet_ce_dice'
        elif opt.model_name == 'datransunet':
            main_loss = 'datransunet_ce_dice'
        elif opt.model_name == 'mysegnet':
            main_loss = 'myseg_bce'
        elif opt.model_name == 's2net':
            main_loss = 's2net_dice_bce'
        elif opt.model_name == 'segformer':
            main_loss = 'segformer_ce_dice'
        elif opt.model_name == 'missformer':
            main_loss = 'missformer_ce_dice'
        elif opt.model_name == 'amin_cnn':
            main_loss = 'transunet_ce_dice'
        elif opt.model_name == 'cfatrans':
            main_loss = 'cfatrans_composite'
        elif opt.model_name in (
            'unet_cn', 'unet_cn_efm', 'unet_cn_msefm', 'unet_cn_msefm_noedge',
        ):
            main_loss = 'bce_dice_focal'
        else:
            main_loss = 'brats_stable'
    if main_loss == 'transunet_ce_dice':
        criterion = TransUNetBceDiceLoss()
        boundary_weighted_dice_criterion = None
    if main_loss == 'myseg_bce':
        criterion = MySegBceLoss()
        boundary_weighted_dice_criterion = None
    if main_loss == 'datransunet_ce_dice':
        criterion = DATransUNetCeDiceLoss(num_classes=4, ce_weight=0.5, dice_weight=0.5)
        boundary_weighted_dice_criterion = None
    if main_loss == 's2net_dice_bce':
        criterion = S2NetDiceBceLoss()
        boundary_weighted_dice_criterion = None
    if main_loss == 'segformer_ce_dice':
        criterion = SegFormerCeDiceLoss(num_classes=4, ce_weight=0.4, dice_weight=0.6)
        boundary_weighted_dice_criterion = None
    if main_loss == 'missformer_ce_dice':
        criterion = MISSFormerCeDiceLoss(num_classes=4, ce_weight=0.4, dice_weight=0.6)
        boundary_weighted_dice_criterion = None
    if main_loss == 'bce_dice_focal':
        criterion = BCEDiceFocalLoss(bce_weight=0.4, dice_weight=0.4, focal_weight=0.2, focal_gamma=2.0, edge_weight=0.01, class_weights=getattr(opt, 'bce_dice_focal_class_weights', [1.0, 1.0, 1.0]))
        boundary_weighted_dice_criterion = None
    if main_loss == 'cfatrans_composite':
        criterion = CFATransCompositeLoss()
        boundary_weighted_dice_criterion = None

    # CFATrans 使用论文指定的 SGD 优化器 (lr=1e-3, momentum=0.9, weight_decay=1e-4)
    if opt.model_name == 'cfatrans':
        sgd_lr = 1e-3
        optimizer.optimizer = torch.optim.SGD(
            model.parameters(), lr=sgd_lr, momentum=0.9, weight_decay=1e-4
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer.optimizer, T_max=opt.epoch - 5, eta_min=sgd_lr * 0.01
        )
        print(f"✅ CFATrans: 使用 SGD 优化器 (lr={sgd_lr}, momentum=0.9)")

    # 保持默认：浅层与深监督沿用主损失；如需启用Fusion辅助可再加入
    aux_criterion = None

    # 初始化早停器和最佳指标跟踪
    early_stopper = ETEarlyStopper(patience=20, min_epochs=40)
    best_et_dice = 0.0  # 用于跟踪最佳平均Dice分数
    slice_range_tuple = tuple(opt.slice_range)

    # 加载数据
    edge_mode = 'otus' if opt.model_name == 'mysegnet' else 'gt'
    train_loader, val_loader, test_loader = get_data_loaders(
        t1_root=opt.t1_root,
        t2_root=opt.t2_root,
        t1ce_root=opt.t1ce_root,
        flair_root=opt.flair_root,
        seg_root=opt.seg_root,
        batch_size=opt.batch_size,
        trainsize=opt.trainsize,
        slice_range=slice_range_tuple,
        sampler_weight_wt=opt.sampler_weight_wt,
        sampler_weight_tc=opt.sampler_weight_tc,
        sampler_weight_et=opt.sampler_weight_et,
        min_tumor_pixels=opt.min_tumor_pixels,
        et_min_pixels=opt.et_min_pixels,
        tc_min_pixels=opt.tc_min_pixels,
        val_batch_size=opt.val_batch_size,
        val_num_workers=opt.val_num_workers,
        val_prefetch_factor=opt.val_prefetch_factor,
        persistent_workers=opt.val_persistent_workers,
        edge_mode=edge_mode,
        seed=int(getattr(opt, 'seed', 2021)),
        train_num_workers=int(getattr(opt, 'train_num_workers', 8)),
        train_prefetch_factor=int(getattr(opt, 'train_prefetch_factor', 2)),
        train_persistent_workers=bool(getattr(opt, 'train_persistent_workers', True)),
    )

    print("开始训练")
    print(f"Total steps per epoch: {len(train_loader)}")
    print(f"✅ 当前 Batch Size: {opt.batch_size}")
    print(f"✅ 训练集样本总数: {len(train_loader) * opt.batch_size} (约)")

    # ✅ 从 checkpoint 恢复训练
    start_epoch = 0
    resume_epoch_raw = None
    ema_loaded = False
    resume_missing_keys = []
    resume_unexpected_keys = []
    resume_ema_missing_keys = []
    resume_ema_unexpected_keys = []
    if opt.resume and os.path.exists(opt.resume):
        print(f"========================================")
        print(f"🔄 从 checkpoint 恢复训练: {opt.resume}")
        # PyTorch 2.6+ 需要设置 weights_only=False 来加载完整的 checkpoint
        checkpoint = torch.load(opt.resume, map_location='cuda', weights_only=False)
        
        # 加载模型权重
        state_to_load = checkpoint.get('raw_model_state_dict', None)
        if state_to_load is None:
            state_to_load = checkpoint.get('model_state_dict', None)
        if state_to_load is not None:
            missing_keys, unexpected_keys = model.load_state_dict(state_to_load, strict=False)
            resume_missing_keys = list(missing_keys) if missing_keys is not None else []
            resume_unexpected_keys = list(unexpected_keys) if unexpected_keys is not None else []
            if len(missing_keys) > 0:
                print(f"ℹ️  恢复模型权重: missing_keys={len(missing_keys)}")
            if len(unexpected_keys) > 0:
                print(f"ℹ️  恢复模型权重: unexpected_keys={len(unexpected_keys)}")
            print(f"✅ 模型权重已加载")
        
        # 加载优化器状态
        if 'optimizer_state_dict' in checkpoint:
            try:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                print(f"✅ 优化器状态已加载")
            except Exception as e:
                print(f"⚠️  优化器状态加载失败（可能由于参数结构变更），将使用新优化器状态: {e}")
        
        # 加载调度器状态
        if 'scheduler_state_dict' in checkpoint and scheduler is not None:
            try:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                print(f"✅ 学习率调度器已加载")
            except Exception as e:
                print(f"⚠️  学习率调度器状态加载失败（可能由于配置变更），将使用新调度器状态: {e}")

        # 加载 GradScaler 状态
        if 'scaler_state_dict' in checkpoint and scaler is not None:
            try:
                scaler.load_state_dict(checkpoint['scaler_state_dict'])
                print(f"✅ GradScaler 状态已加载")
            except Exception as e:
                print(f"⚠️  GradScaler 状态加载失败（将使用新 scaler 状态）: {e}")
        
        # 加载 EMA 权重
        if 'ema_state_dict' in checkpoint and ema is not None and getattr(ema, 'ema_model', None) is not None:
            try:
                ema_missing, ema_unexpected = ema.ema_model.load_state_dict(checkpoint['ema_state_dict'], strict=False)
                resume_ema_missing_keys = list(ema_missing) if ema_missing is not None else []
                resume_ema_unexpected_keys = list(ema_unexpected) if ema_unexpected is not None else []
                ema_loaded = True
                print(f"✅ EMA 权重已加载")
            except Exception as e:
                print(f"⚠️  EMA 权重加载失败（可能由于参数结构变更），将使用模型权重同步 EMA: {e}")

        if (ema is not None) and (getattr(ema, 'ema_model', None) is not None) and (not ema_loaded):
            try:
                ema_missing, ema_unexpected = ema.ema_model.load_state_dict(model.state_dict(), strict=False)
                resume_ema_missing_keys = list(ema_missing) if ema_missing is not None else []
                resume_ema_unexpected_keys = list(ema_unexpected) if ema_unexpected is not None else []
                ema_loaded = True
                print(f"ℹ️  Checkpoint 未包含 EMA 权重，已使用模型权重同步 EMA")
            except Exception as e:
                print(f"⚠️  EMA 同步失败（评估将回退到模型权重）: {e}")
        
        # 恢复 epoch
        if 'epoch' in checkpoint:
            resume_epoch_raw = int(checkpoint['epoch'])
            start_epoch = int(checkpoint['epoch']) + 1
            print(f"✅ 从 Epoch {start_epoch + 1} 继续训练")
        
        # 打印 checkpoint 中的 batch size（如果有）
        if 'batch_size' in checkpoint:
            saved_bs = checkpoint['batch_size']
            print(f"ℹ️  Checkpoint 中的 Batch Size: {saved_bs}")
            if saved_bs != opt.batch_size:
                print(f"⚠️  注意: 当前训练使用 Batch Size: {opt.batch_size}")
                print(f"⚠️  优化器状态是基于 BS={saved_bs} 保存的，但会自动适配")
        
        print(f"========================================")
        print()
    elif opt.resume:
        print(f"⚠️  警告: checkpoint 文件不存在: {opt.resume}")
        if bool(getattr(opt, 'eval_only', False)):
            raise SystemExit(2)
        print(f"⚠️  将从头开始训练")
        print()

    if bool(getattr(opt, 'eval_only', False)):
        if (len(resume_missing_keys) > 0) or (len(resume_unexpected_keys) > 0) or (len(resume_ema_missing_keys) > 0) or (len(resume_ema_unexpected_keys) > 0):
            print("\n[EvalOnly] ❌ 检测到 checkpoint 与当前构建的模型结构不匹配，因此本次评估结果将是无意义的。")
            print("[EvalOnly] 请使用训练该 checkpoint 时完全一致的模型开关/参数（例如 enable_unet_eam / enable_unet_paper_edge 等），再重新运行 --eval_only。")
            print(f"[EvalOnly] model_state_dict: missing={len(resume_missing_keys)}, unexpected={len(resume_unexpected_keys)}")
            print(f"[EvalOnly] ema_state_dict:   missing={len(resume_ema_missing_keys)}, unexpected={len(resume_ema_unexpected_keys)}")
            raise SystemExit(2)

        model_to_eval = (getattr(ema, 'ema_model', None) if bool(ema_loaded) else None) or model
        _sanitize_bn_stats_(model_to_eval)
        print(f"[EvalOnly] 使用{'EMA' if bool(ema_loaded) else '原始'}权重进行评估")
        eval_epoch = int(resume_epoch_raw) if resume_epoch_raw is not None else int(max(0, start_epoch - 1))
        if hasattr(model_to_eval, 'set_epoch'):
            try:
                model_to_eval.set_epoch(int(eval_epoch))
            except Exception:
                pass
        eval_run_name = f"{str(opt.train_save)}__evalonly"
        use_dyn_thr = bool(opt.use_dynamic_threshold) and (not bool(getattr(opt, 'disable_dynamic_threshold', False)))
        val_metrics = improved_validate(
            model_to_eval,
            val_loader,
            threshold=opt.val_threshold,
            save_metrics=True,
            epoch=eval_epoch,
            use_dynamic_threshold=use_dyn_thr,
            max_val_batches=opt.max_val_batches,
            max_val_tumor_slices=opt.max_val_tumor_slices,
            include_empty_in_metrics=opt.include_empty_in_metrics,
            thresholds_bounds=None,
            per_class_temps=None,
            et_min_pixels=opt.et_min_pixels,
            run_name=eval_run_name,
            dynamic_threshold_mode=str(getattr(opt, 'dynamic_threshold_mode', 'batch')),
            dynamic_threshold_calib_batches=int(getattr(opt, 'dynamic_threshold_calib_batches', 8)),
        )
        print(f"\nEvalOnly Results:")
        print(f"Dice - WT: {val_metrics['dice_WT']:.4f} | TC: {val_metrics['dice_TC']:.4f} | ET: {val_metrics['dice_ET']:.4f}")
        print(f"HD95 - WT: {val_metrics['hd95_WT']:.2f} | TC: {val_metrics['hd95_TC']:.2f} | ET: {val_metrics['hd95_ET']:.2f}")
        print(f"Average Dice: {val_metrics['avg_dice']:.4f} | Average HD95: {val_metrics['avg_hd95']:.2f}")
        raise SystemExit(0)

    # 初始化日志系统
    log_dir = "training_logs"
    os.makedirs(log_dir, exist_ok=True)

    # 从train_save路径中提取基础名称（处理路径中的目录）
    train_save_basename = os.path.basename(opt.train_save)
    log_path = os.path.join(log_dir, f"{train_save_basename}.csv")
    # 额外：紧凑对比日志（训练Dice在前，验证HD95在后）
    compare_log_path = os.path.join(log_dir, f"{train_save_basename}_overfit.csv")

    if not os.path.exists(log_path):
        with open(log_path, "w") as f:
            f.write("epoch,loss,WT,TC,ET,lr,val_WT_dice,val_TC_dice,val_ET_dice,val_avg_dice,val_WT_hd95,val_TC_hd95,val_ET_hd95,val_avg_hd95\n")
    if not os.path.exists(compare_log_path):
        with open(compare_log_path, "w") as f:
            f.write("epoch,compare\n")

    # 在主循环中
    for epoch in range(start_epoch, opt.epoch):
        # 设置 criterion 的 epoch（用于渐进式边界监督）
        if hasattr(criterion, 'set_epoch'):
            criterion.set_epoch(epoch)
        # 禁用冻结策略：不冻结任何层
        print(f"[Epoch {epoch+1}] 冻结策略已禁用（不冻结任何层）")
        # 已移除 ROI 概率调度与日志

        # 训练阶段
        loss, wt, tc, et, tr_hd95_wt, tr_hd95_tc, tr_hd95_et, lr = train(train_loader, model, optimizer, epoch)
        
        # 每个epoch都进行验证（保持原有策略）
        model_to_eval = getattr(ema, 'ema_model', None) or model
        _sanitize_bn_stats_(model_to_eval)
        if hasattr(model_to_eval, 'set_epoch'):
            try:
                model_to_eval.set_epoch(int(epoch))
            except Exception:
                pass
        print(f"[Validation] 使用{'EMA' if getattr(ema, 'ema_model', None) is not None else '原始'}权重进行评估")
        # 使用动态阈值进行验证（自动搜索最优阈值）
        use_dyn_thr = bool(opt.use_dynamic_threshold) and (not bool(getattr(opt, 'disable_dynamic_threshold', False)))
        val_metrics = improved_validate(
            model_to_eval,
            val_loader,
            threshold=opt.val_threshold,
            save_metrics=True,
            epoch=epoch,
            use_dynamic_threshold=use_dyn_thr,
            max_val_batches=opt.max_val_batches,
            max_val_tumor_slices=opt.max_val_tumor_slices,
            include_empty_in_metrics=opt.include_empty_in_metrics,
            thresholds_bounds=None,
            per_class_temps=None,
            et_min_pixels=opt.et_min_pixels,
            run_name=opt.train_save,
            dynamic_threshold_mode=str(getattr(opt, 'dynamic_threshold_mode', 'batch')),
            dynamic_threshold_calib_batches=int(getattr(opt, 'dynamic_threshold_calib_batches', 8)),
        )
        
        # ✅ 修复学习率调度：预热期后才开始调用 scheduler.step()
        warmup_epochs = 5
        if epoch >= warmup_epochs:
            # ✅ 使用 epoch - warmup_epochs 作为 scheduler 的输入
            # 这样 Epoch 5 对应 scheduler 的 Epoch 0
            scheduler.step()
            print(f"调度器更新 (scheduler epoch {epoch - warmup_epochs}) - 学习率: {optimizer.optimizer.param_groups[0]['lr']:.2e}")
        else:
            print(f"预热阶段 ({epoch+1}/{warmup_epochs}) - 学习率: {optimizer.optimizer.param_groups[0]['lr']:.2e}")
        avg_dice = (val_metrics['dice_WT'] + val_metrics['dice_TC'] + val_metrics['dice_ET']) / 3
        
        # 更新最佳模型
        current_et_dice = val_metrics['dice_ET']
        current_avg_dice = (val_metrics['dice_WT'] + val_metrics['dice_TC'] + val_metrics['dice_ET']) / 3
        
        # 使用平均Dice作为主要指标
        if current_avg_dice > best_et_dice:
            best_et_dice = current_avg_dice
            model_to_save = getattr(ema, 'ema_model', None) or model
            try:
                ckpt = {
                    'epoch': epoch,
                    'opt': vars(opt),
                    'model_state_dict': model_to_save.state_dict(),
                    'raw_model_state_dict': model.state_dict(),
                    'ema_state_dict': getattr(ema, 'ema_model', None).state_dict() if getattr(ema, 'ema_model', None) is not None else None,
                    'optimizer_state_dict': optimizer.state_dict() if optimizer is not None else None,
                    'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
                    'scaler_state_dict': scaler.state_dict() if scaler is not None else None,
                    'best_avg_dice': best_et_dice,
                    'val_metrics': val_metrics,
                    'rng_state': {
                        'python': random.getstate(),
                        'numpy': np.random.get_state(),
                        'torch': torch.get_rng_state(),
                        'cuda_all': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                    },
                }
                _atomic_torch_save(ckpt, f'checkpoints/{opt.train_save}/best_model.pth')
                print(f"保存最佳模型，平均Dice: {best_et_dice:.4f}, ET Dice: {current_et_dice:.4f}")
            except Exception as e:
                print(f"⚠️  保存 best_model.pth 失败: {e}")
        
        # 打印验证结果
        print(f"\nValidation @ Epoch {epoch+1}:")
        print(f"Dice - WT: {val_metrics['dice_WT']:.4f} | TC: {val_metrics['dice_TC']:.4f} | ET: {val_metrics['dice_ET']:.4f}")
        print(f"HD95 - WT: {val_metrics['hd95_WT']:.2f} | TC: {val_metrics['hd95_TC']:.2f} | ET: {val_metrics['hd95_ET']:.2f}")
        print(f"Average Dice: {val_metrics['avg_dice']:.4f} | Average HD95: {val_metrics['avg_hd95']:.2f}")
        
        # 早停判断
        if early_stopper(val_metrics, epoch):
            print(f"Early stopping triggered at epoch {epoch+1}")
            break
            
        # 记录到CSV - 每个epoch都记录完整信息
        with open(log_path, "a") as f:
            log_str = (f"{epoch+1},{loss:.4f},{wt:.4f},{tc:.4f},{et:.4f},{lr:.6f},"
                      f"{val_metrics['dice_WT']:.4f},{val_metrics['dice_TC']:.4f},"
                      f"{val_metrics['dice_ET']:.4f},{val_metrics['avg_dice']:.4f},"
                      f"{val_metrics['hd95_WT']:.2f},{val_metrics['hd95_TC']:.2f},"
                      f"{val_metrics['hd95_ET']:.2f},{val_metrics['avg_hd95']:.2f}")
            f.write(log_str + "\n")
        # 写入紧凑对比日志：WT/TC/ET Dice（训练） + WT/TC/ET HD95（验证）
        compare_str = (
            f"WT_Dice={wt:.4f}, TC_Dice={tc:.4f}, ET_Dice={et:.4f}, "
            f"WT_HD95={tr_hd95_wt:.2f}, TC_HD95={tr_hd95_tc:.2f}, ET_HD95={tr_hd95_et:.2f}, "
            f"Val_WT_Dice={val_metrics['dice_WT']:.4f}, Val_TC_Dice={val_metrics['dice_TC']:.4f}, Val_ET_Dice={val_metrics['dice_ET']:.4f}, "
            f"Val_WT_HD95={val_metrics['hd95_WT']:.2f}, Val_TC_HD95={val_metrics['hd95_TC']:.2f}, Val_ET_HD95={val_metrics['hd95_ET']:.2f}"
        )
        with open(compare_log_path, "a") as f2:
            f2.write(f"{epoch+1},{compare_str}\n")
        
    print("训练完成！")
    print(f"  • 最佳模型: checkpoints/{opt.train_save}/best_model.pth (Dice={best_et_dice:.4f})")

    try:
        best_ckpt_path = f"checkpoints/{opt.train_save}/best_model.pth"
        if os.path.exists(best_ckpt_path):
            best_ckpt = torch.load(best_ckpt_path, map_location='cpu', weights_only=False)
            best_state = best_ckpt.get('model_state_dict', None)
            if best_state is not None:
                best_epoch = int(best_ckpt.get('epoch', opt.epoch - 1)) + 1
                model_to_vis = getattr(ema, 'ema_model', None) or model
                model_to_vis.load_state_dict(best_state, strict=False)
                _sanitize_bn_stats_(model_to_vis)
                model_to_vis.eval()
                vis_save_dir = os.path.join('./visualizations', opt.train_save)
                visualize_validation_batch(
                    model=model_to_vis,
                    val_loader=val_loader,
                    device='cuda',
                    epoch=best_epoch,
                    save_dir=vis_save_dir,
                    num_samples=100,
                )
                print(f"✅ 训练结束可视化已保存到: {vis_save_dir}/epoch_{best_epoch:03d}")
    except Exception as e:
        print(f"⚠️  训练结束可视化失败: {str(e)}")

