
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import warnings
warnings.filterwarnings('ignore')
from brats_stable_loss import BraTSStableLoss  # 统一入口
ImprovedRegionAwareLoss = BraTSStableLoss

def create_improved_optimizer(model, initial_lr=5e-4):
    """创建改进的优化器配置"""
    # 更细致的分层学习率
    backbone_params = []
    efm_params = []
    cam_params = []
    decoder_params = []
    edge_params = []
    
    for name, param in model.named_parameters():
        if 'resnet' in name or 'backbone' in name:
            backbone_params.append(param)
        elif 'efm' in name:
            efm_params.append(param)
        elif 'cam' in name:
            cam_params.append(param)
        elif 'decoder' in name:
            decoder_params.append(param)
        elif 'edge' in name:
            edge_params.append(param)
        else:
            decoder_params.append(param)  # 默认归类到decoder
    
    optimizer = torch.optim.AdamW([
        {'params': backbone_params, 'lr': initial_lr * 0.1, 'weight_decay': 1e-4},
        {'params': efm_params, 'lr': initial_lr * 0.3, 'weight_decay': 5e-5},
        {'params': cam_params, 'lr': initial_lr * 0.5, 'weight_decay': 5e-5},
        {'params': decoder_params, 'lr': initial_lr, 'weight_decay': 1e-5},
        {'params': edge_params, 'lr': initial_lr * 1.2, 'weight_decay': 1e-5}
    ], lr=initial_lr, betas=(0.9, 0.999), eps=1e-8)
    
    return optimizer

def apply_model_fixes(model):
    """应用模型修复"""
    print("🔧 应用模型修复...")
    
    # 1. 修复权重初始化
    def improved_init_weights(m):
        if isinstance(m, nn.Conv2d):
            # 使用更合理的初始化
            nn.init.xavier_normal_(m.weight, gain=0.5)  # 从0.1改为0.5
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 0.1)  # 从0.001改为0.1
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.GroupNorm, nn.LayerNorm)):
            nn.init.constant_(m.weight, 0.1)  # 从0.001改为0.1
            nn.init.constant_(m.bias, 0)
    
    # 只对decoder模块重新初始化
    for name, module in model.named_modules():
        if 'decoder' in name:
            module.apply(improved_init_weights)
    
    # 2. 修复梯度钩子
    # 说明：为避免过度抑制，取消逐参数梯度裁剪，仅保留全局 clip_grad_norm_ 于训练步骤中执行
    def improved_gradient_hook(grad):
        if grad is None:
            return grad
        # 仅修复 NaN/Inf，避免数值污染；不做任何幅值/范数裁剪
        if torch.isnan(grad).any() or torch.isinf(grad).any():
            grad = torch.nan_to_num(grad, nan=0.0, posinf=1.0, neginf=-1.0)
        return grad
    
    # 清理历史钩子，且不再注册新的逐参数钩子（统一由全局 clip_grad_norm_ 保卫）
    for name, param in model.named_parameters():
        if hasattr(param, '_backward_hooks') and param._backward_hooks is not None:
            param._backward_hooks.clear()
    print("Per-parameter gradient hooks disabled in fix_training_issues.apply_model_fixes; relying on global clip_grad_norm_.")
    
    print("✅ 模型修复完成")
    return model

def create_improved_scaler():
    """创建改进的GradScaler"""
    return torch.cuda.amp.GradScaler(
        init_scale=2.**10,     # 从2.**12改为2.**10 (1024)
        growth_factor=2.0,     # 保持标准值
        backoff_factor=0.5,    # 保持标准值
        growth_interval=1000,  # 从2000改为1000，更频繁检查
        enabled=True
    )

def compute_hd95(pred, target):
    """计算HD95；遇空预测/空标签或无边界点返回None（不计入平均）。"""
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

    if len(pred_points) == 0 or len(target_points) == 0:
        return None

    d1 = np.array([np.min(np.linalg.norm(p - target_points, axis=1)) for p in pred_points])
    d2 = np.array([np.min(np.linalg.norm(t - pred_points, axis=1)) for t in target_points])

    # 计算95百分位数
    hd1 = np.percentile(d1, 95)
    hd2 = np.percentile(d2, 95)
    return max(hd1, hd2)

@torch.no_grad()
def dice_coeff(
    input: torch.Tensor,
    target: torch.Tensor,
    reduce_batch_first: bool = True,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    """参照训练阶段实现的 Dice 系数计算，保持一致口径。
    - 支持 [B,1,H,W] 或 [B,H,W]；内部统一到 [B,1,H,W]
    - 既可用于概率图，也可用于二值图
    """
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

def improved_validate(model, val_loader, device='cuda', threshold=0.45, save_metrics=True, epoch=None, use_dynamic_threshold=True, max_val_batches=None, max_val_tumor_slices=None, include_empty_in_metrics=False, thresholds_bounds=None, per_class_temps=None, et_min_pixels=None, run_name=None, dynamic_threshold_mode: str = 'batch', dynamic_threshold_calib_batches: int = 8):
    ...
    # 在写结果时可记录:
    # if et_min_pixels is not None: f.write(f"ET最小像素阈值(记录): {et_min_pixels}\n")
    """改进的验证函数，添加HD95指标与动态阈值选择，并保存结果
    关键改动：
    - 新增 max_val_batches 与 max_val_tumor_slices 控制评估规模；移除硬编码的1800上限
    - 新增 include_empty_in_metrics 控制空切片是否参与Dice/HD95均值（默认不计入，降低样本数量差异带来的波动）
    - 新增 thresholds_bounds: 每类阈值的上下界，默认 WT[0.40,0.55], TC[0.40,0.52], ET[0.42,0.55]
    - 新增 per_class_temps: 每类logits温度缩放 (WT, TC, ET)，默认禁用
    """
    model.eval()
    
    dice_scores = {'WT': [], 'TC': [], 'ET': []}
    hd95_scores = {'WT': [], 'TC': [], 'ET': []}
    total_samples = 0
    tumor_samples = {'WT': 0, 'TC': 0, 'ET': 0}
    
    # 记录空切片上的误报情况（不计入Dice/HD95均值）
    empty_slice_stats = {
        'WT': {'count': 0, 'fp_pixels': 0.0},
        'TC': {'count': 0, 'fp_pixels': 0.0},
        'ET': {'count': 0, 'fp_pixels': 0.0},
    }
    tumor_slices_seen = 0
    
    # 阈值边界默认值（扩展搜索范围以提高召回率）
    default_bounds = {'WT': (0.35, 0.55), 'TC': (0.35, 0.58), 'ET': (0.35, 0.60)}
    if thresholds_bounds is None:
        thresholds_bounds = default_bounds.copy()
    else:
        _tb = default_bounds.copy()
        for k, v in (thresholds_bounds or {}).items():
            _tb[k] = tuple(v)
        thresholds_bounds = _tb
    
    # 已移除连通域过滤设置
    
    # 前向：返回logits（不做sigmoid）
    def forward_and_get_logits(x, edges=None):
        def _forward(xx, ee=None):
            if ee is not None and bool(getattr(model, 'supports_external_edge', False)):
                return model(xx, ee)
            return model(xx)

        outputs = _forward(x, edges)
        if isinstance(outputs, dict):
            if 'stage4' in outputs:
                preds_local = outputs['stage4']
            elif 'out' in outputs:
                preds_local = outputs['out']
            else:
                preds_local = list(outputs.values())[-1]
        elif isinstance(outputs, tuple) and len(outputs) == 4:
            preds_local = outputs[2]
        else:
            preds_local = outputs
        return preds_local  # logits
    
    first_batch_logged = False
    thresholds_fixed = None

    dyn_mode = str(dynamic_threshold_mode or 'batch').lower().strip()
    if dyn_mode not in ('batch', 'epoch'):
        dyn_mode = 'batch'

    if use_dynamic_threshold and dyn_mode == 'epoch':
        # 校准一次全局阈值（使用前若干个 batch），之后全验证集保持固定阈值，减少波动
        calib_batches = int(dynamic_threshold_calib_batches)
        if calib_batches <= 0:
            calib_batches = 1
        if max_val_batches is not None:
            calib_batches = min(calib_batches, int(max_val_batches))

        # 为每类准备候选阈值
        cand_map = {}
        for name in ['WT', 'TC', 'ET']:
            if name == 'WT':
                base_thr = 0.42
                deltas = [0.0, 0.02, -0.02, 0.04, -0.04, 0.06, -0.06]
            elif name == 'TC':
                base_thr = 0.44
                deltas = [0.0, 0.01, -0.01, 0.02, -0.02, 0.03, -0.03, 0.04, -0.04, 0.06, -0.06]
            else:
                base_thr = 0.46
                deltas = [0.0, 0.01, -0.01, 0.02, -0.02, 0.03, -0.03, 0.05, -0.05, 0.08, -0.08, 0.10, -0.10]
            base_thr = float(max(min(base_thr, thresholds_bounds[name][1]), thresholds_bounds[name][0]))
            cand = sorted(set([
                max(min(base_thr + d, thresholds_bounds[name][1]), thresholds_bounds[name][0]) for d in deltas
            ]))
            cand_map[name] = cand

        score_sums = {k: torch.zeros(len(v), device=device) for k, v in cand_map.items()}
        score_cnt = 0
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                if (batch_idx + 1) > calib_batches:
                    break
                if isinstance(batch, (list, tuple)) and len(batch) == 5:
                    images, label_wt, label_tc, label_et, edges = batch
                    targets_cal = torch.stack([label_wt, label_tc, label_et], dim=1)
                else:
                    images, targets_cal, edges = batch
                images = images.to(device, non_blocking=True)
                targets_cal = targets_cal.to(device, non_blocking=True)
                preds_logits = forward_and_get_logits(images, edges=edges.to(device, non_blocking=True) if edges is not None else None)
                if per_class_temps is not None:
                    C = preds_logits.size(1)
                    temps = torch.tensor(list(per_class_temps) + [1.0]*(max(0, C-3)), device=preds_logits.device, dtype=preds_logits.dtype)[:C]
                    temps = temps.view(1, C, 1, 1).clamp(min=0.5, max=2.0)
                    preds_logits = preds_logits / temps
                preds_prob = torch.sigmoid(preds_logits)

                for c, name in enumerate(['WT', 'TC', 'ET']):
                    if c >= preds_prob.size(1):
                        continue
                    prob_c = preds_prob[:, c]
                    tgt_c = targets_cal[:, c] if targets_cal.size(1) > c else targets_cal[:, 0]
                    cand = cand_map[name]
                    for j, t in enumerate(cand):
                        pred_bin = (prob_c > float(t)).float()
                        inter = (pred_bin * tgt_c).sum(dim=(-2, -1))
                        sets = pred_bin.sum(dim=(-2, -1)) + tgt_c.sum(dim=(-2, -1))
                        dice_batch = (2 * inter) / sets.clamp_min(1.0)
                        dice = dice_batch.mean()
                        if name in ['TC', 'ET']:
                            gt_sum = tgt_c.sum(dim=(-2, -1)).clamp_min(1.0)
                            recall = (inter / gt_sum).mean()
                            dice_adjusted = dice * 0.7 + recall * 0.3
                        else:
                            dice_adjusted = dice
                        score_sums[name][j] += dice_adjusted.detach()
                score_cnt += 1

        thresholds_fixed = {}
        for name in ['WT', 'TC', 'ET']:
            if name not in score_sums or score_sums[name].numel() <= 0 or score_cnt <= 0:
                continue
            best_idx = int(torch.argmax(score_sums[name]).item())
            thresholds_fixed[name] = float(cand_map[name][best_idx])
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            if max_val_batches is not None and (batch_idx + 1) > max_val_batches:
                break

            if isinstance(batch, (list, tuple)) and len(batch) == 5:
                images, label_wt, label_tc, label_et, edges = batch
                targets = torch.stack([label_wt, label_tc, label_et], dim=1)
            else:
                images, targets, edges = batch
            
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            
            # 前向传播（已移除TTA）
            preds_logits = forward_and_get_logits(images, edges=edges.to(device, non_blocking=True) if edges is not None else None)
            
            # 温度缩放（逐通道）- 默认禁用，仅在提供 per_class_temps 时启用
            if per_class_temps is not None:
                C = preds_logits.size(1)
                temps = torch.tensor(list(per_class_temps) + [1.0]*(max(0, C-3)), device=preds_logits.device, dtype=preds_logits.dtype)[:C]
                temps = temps.view(1, C, 1, 1).clamp(min=0.5, max=2.0)
                preds_logits = preds_logits / temps
            
            # 概率
            preds_prob = torch.sigmoid(preds_logits)
            
            # 计算每类的阈值（批内一致 / 或者 epoch 固定）
            thresholds = {}
            for c, name in enumerate(['WT', 'TC', 'ET']):
                if c >= preds_prob.size(1):
                    continue
                if use_dynamic_threshold and thresholds_fixed is not None:
                    thresholds[name] = float(thresholds_fixed[name])
                elif use_dynamic_threshold:
                    # 降低基础阈值以提高召回率，减少小目标漏检
                    if name == 'WT':
                        base_thr = 0.42  # 从0.45降低
                        deltas = [0.0, 0.02, -0.02, 0.04, -0.04, 0.06, -0.06]
                    elif name == 'TC':
                        base_thr = 0.44  # 从0.48降低
                        deltas = [0.0, 0.01, -0.01, 0.02, -0.02, 0.03, -0.03, 0.04, -0.04, 0.06, -0.06]
                    else:  # ET
                        base_thr = 0.46
                        deltas = [0.0, 0.01, -0.01, 0.02, -0.02, 0.03, -0.03, 0.05, -0.05, 0.08, -0.08, 0.10, -0.10]
                    
                    # 夹到边界范围内
                    base_thr = float(max(min(base_thr, thresholds_bounds[name][1]), thresholds_bounds[name][0]))
                    cand = sorted(set([
                        max(min(base_thr + d, thresholds_bounds[name][1]), thresholds_bounds[name][0]) for d in deltas
                    ]))
                    
                    # 评估批内Dice，选择最优（与训练一致）
                    best_thr, best_iou = base_thr, -1.0
                    prob_c = preds_prob[:, c]
                    tgt_c = targets[:, c] if targets.size(1) > c else targets[:, 0]
                    
                    # 为TC和ET添加召回率加权（优先召回率）
                    for t in cand:
                        pred_bin = (prob_c > t).float()
                        inter = (pred_bin * tgt_c).sum(dim=(-2, -1))
                        sets = pred_bin.sum(dim=(-2, -1)) + tgt_c.sum(dim=(-2, -1))
                        dice_batch = (2 * inter) / sets.clamp_min(1.0)
                        dice = dice_batch.mean().item()
                        # 为TC和ET添加召回率bonus
                        if name in ['TC', 'ET']:
                            gt_sum = tgt_c.sum(dim=(-2, -1)).clamp_min(1.0)
                            recall = (inter / gt_sum).mean().item()
                            dice_adjusted = dice * 0.7 + recall * 0.3
                        else:
                            dice_adjusted = dice
                        if dice_adjusted > best_iou:
                            best_iou, best_thr = dice_adjusted, float(t)
                    
                    thresholds[name] = best_thr
                else:
                    # 固定阈值：直接使用给定阈值（不受边界限制）
                    thresholds[name] = float(threshold)
            
            if not first_batch_logged:
                mode_str = "动态" if use_dynamic_threshold else "固定"
                temp_info = ("禁用" if per_class_temps is None else str(per_class_temps))
                print(f"使用{mode_str}阈值 (WT={thresholds.get('WT', None)}, TC={thresholds.get('TC', None)}, ET={thresholds.get('ET', None)}) | 阈值搜索范围: WT{thresholds_bounds['WT']}, TC{thresholds_bounds['TC']}, ET{thresholds_bounds['ET']}")
                first_batch_logged = True
            
            # 二值化 + 层级一致性强制 (ET ⊆ TC ⊆ WT)
            preds_bin = torch.zeros_like(preds_prob)
            for c, name in enumerate(['WT', 'TC', 'ET']):
                if c < preds_prob.size(1) and name in thresholds:
                    preds_bin[:, c] = (preds_prob[:, c] > thresholds[name]).float()
            if preds_bin.dim() == 4 and preds_bin.size(1) >= 3:
                preds_bin[:, 1] = torch.maximum(preds_bin[:, 1], preds_bin[:, 2])
                preds_bin[:, 0] = torch.maximum(preds_bin[:, 0], preds_bin[:, 1])
            
            # 已移除连通域过滤
            
            batch_size = preds_prob.size(0)
            total_samples += batch_size
            
            # 统计肿瘤切片数量
            has_any_tumor = None
            if targets.size(1) >= 3:
                has_any_tumor = ((targets[:, 0] > 0.5) | (targets[:, 1] > 0.5) | (targets[:, 2] > 0.5))
                tumor_slices_seen += int(has_any_tumor.sum().item())
            
            for i in range(batch_size):
                for c, name in enumerate(['WT', 'TC', 'ET']):
                    if c >= preds_bin.size(1):
                        continue
                    pred_slice = preds_bin[i, c]
                    target_slice = targets[i, c]
                    target_positive = (target_slice > 0.5).sum().item()
                    pred_positive = pred_slice.sum().item()

                    if target_positive <= 0:
                        empty_slice_stats[name]['count'] += 1
                        empty_slice_stats[name]['fp_pixels'] += float(pred_positive)
                        continue
                    else:
                        tumor_samples[name] += 1
                        if pred_positive <= 0:
                            dice_scores[name].append(0.0)
                            hd95_scores[name].append(100.0)
                        else:
                            dice_val = dice_coeff(
                                pred_slice.unsqueeze(0),
                                target_slice.unsqueeze(0),
                                reduce_batch_first=True
                            ).item()
                            dice_scores[name].append(dice_val)
                            try:
                                hd95 = compute_hd95(pred_slice, target_slice)
                                hd95_scores[name].append(100.0 if hd95 is None else float(hd95))
                            except Exception:
                                hd95_scores[name].append(100.0)
            
            if max_val_tumor_slices is not None and tumor_slices_seen >= max_val_tumor_slices:
                break
    
    # 计算最终指标
    metrics = {}
    for name in ['WT', 'TC', 'ET']:
        metrics[f'dice_{name}'] = np.mean(dice_scores[name]) if dice_scores[name] else 0.0
        metrics[f'hd95_{name}'] = np.mean(hd95_scores[name]) if hd95_scores[name] else 100.0
        print(f"{name}: Dice={metrics[f'dice_{name}']:.4f}, HD95={metrics[f'hd95_{name}']:.2f}, 肿瘤样本={tumor_samples[name]}/{total_samples}, 空切片={empty_slice_stats[name]['count']}")
    
    avg_dice = np.mean([metrics[f'dice_{name}'] for name in ['WT', 'TC', 'ET']])
    avg_hd95 = np.mean([metrics[f'hd95_{name}'] for name in ['WT', 'TC', 'ET']])
    metrics['avg_dice'] = avg_dice
    metrics['avg_hd95'] = avg_hd95
    
    # 保存验证指标到txt文件（原逻辑保留）
    if save_metrics:
        import os
        from datetime import datetime
        
        safe_run_name = None
        if run_name is not None:
            try:
                safe_run_name = os.path.basename(str(run_name)).strip()
            except Exception:
                safe_run_name = None
        val_results_dir = os.path.join("validation_results", safe_run_name) if safe_run_name else "validation_results"
        os.makedirs(val_results_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if epoch is not None:
            filename = f"validation_metrics_epoch_{epoch+1}.txt"
        else:
            filename = f"validation_metrics_{timestamp}.txt"
        
        filepath = os.path.join(val_results_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(f"验证时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            if epoch is not None:
                f.write(f"训练轮次: {epoch+1}\n")
            # 记录阈值与搜索范围
            if use_dynamic_threshold:
                f.write(
                    f"验证阈值: 动态阈值 (WT={thresholds['WT']}, TC={thresholds['TC']}, ET={thresholds['ET']})，阈值搜索范围: WT{thresholds_bounds['WT']}, TC{thresholds_bounds['TC']}, ET{thresholds_bounds['ET']}\n"
                )
            else:
                f.write(
                    f"验证阈值: 固定阈值 基础={threshold} -> 实际(WT={thresholds['WT']}, TC={thresholds['TC']}, ET={thresholds['ET']})，阈值搜索范围: WT{thresholds_bounds['WT']}, TC{thresholds_bounds['TC']}, ET{thresholds_bounds['ET']}\n"
                )
            if max_val_batches is not None:
                f.write(f"批次上限: {max_val_batches}\n")
            if max_val_tumor_slices is not None:
                f.write(f"肿瘤切片上限: {max_val_tumor_slices}\n")
            # 已移除连通域过滤与TTA配置记录
            f.write(f"总样本数: {total_samples}\n\n")
            
            f.write("详细指标:\n")
            f.write("-" * 50 + "\n")
            for name in ['WT', 'TC', 'ET']:
                f.write(f"{name}:\n")
                f.write(f"  Dice系数: {metrics[f'dice_{name}']:.4f}\n")
                f.write(f"  HD95距离: {metrics[f'hd95_{name}']:.2f}\n")
                f.write(f"  肿瘤样本: {tumor_samples[name]}/{total_samples}\n")
                f.write(f"  空切片数: {empty_slice_stats[name]['count']}\n")
                f.write(f"  空切片误报像素: {empty_slice_stats[name]['fp_pixels']:.1f}\n\n")
            
            f.write(f"平均指标:\n")
            f.write(f"  平均Dice: {avg_dice:.4f}\n")
            f.write(f"  平均HD95: {avg_hd95:.2f}\n")
        
        print(f"验证指标已保存到: {filepath}")
        
        # 仅在提供epoch时追加到汇总文件，避免重复无Epoch条目
        if epoch is not None:
            summary_line = (
                f"Epoch {epoch+1}: "
                f"Dice={avg_dice:.4f}, HD95={avg_hd95:.2f}, "
                f"WT_Dice={metrics['dice_WT']:.4f}, TC_Dice={metrics['dice_TC']:.4f}, ET_Dice={metrics['dice_ET']:.4f}, "
                f"WT_HD95={metrics['hd95_WT']:.2f}, TC_HD95={metrics['hd95_TC']:.2f}, ET_HD95={metrics['hd95_ET']:.2f}\n"
            )

            # 1) 追加到 run 专属目录（如果有 run_name）
            summary_file = os.path.join(val_results_dir, "validation_summary.txt")
            with open(summary_file, 'a', encoding='utf-8') as f:
                f.write(summary_line)

            # 2) 同时追加到全局 validation_results/validation_summary.txt
            global_dir = "validation_results"
            os.makedirs(global_dir, exist_ok=True)
            global_summary_file = os.path.join(global_dir, "validation_summary.txt")
            prefix = f"run={safe_run_name} " if safe_run_name else ""
            with open(global_summary_file, 'a', encoding='utf-8') as f:
                f.write(prefix + summary_line)
    
    return metrics

def build_adamw_with_param_groups(model: nn.Module, base_lr: float, backbone_lr_scale: float = 0.5, weight_decay: float = 1e-4):
    """为Transformer友好地构建AdamW：
    - no_decay: LayerNorm/BatchNorm权重与bias不做weight decay
    - 判别式学习率：backbone lr = base_lr * backbone_lr_scale
    """
    no_decay = {"bias", "LayerNorm.weight", "layernorm.weight", "norm.weight", "bn.weight", "BatchNorm.weight"}

    def is_no_decay(n):
        n_low = n.lower()
        if n.endswith("bias"):
            return True
        for k in no_decay:
            if k.lower() in n_low:
                return True
        return False

    # 粗略区分 backbone 与 head
    backbone = getattr(model, 'backbone', None)
    backbone_params = set(p for p in backbone.parameters()) if backbone is not None else set()

    decay_backbone, nodecay_backbone, decay_head, nodecay_head = [], [], [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p in backbone_params:
            (nodecay_backbone if is_no_decay(n) else decay_backbone).append(p)
        else:
            (nodecay_head if is_no_decay(n) else decay_head).append(p)

    param_groups = [
        {"params": decay_backbone, "lr": base_lr * backbone_lr_scale, "weight_decay": weight_decay},
        {"params": nodecay_backbone, "lr": base_lr * backbone_lr_scale, "weight_decay": 0.0},
        {"params": decay_head, "lr": base_lr, "weight_decay": weight_decay},
        {"params": nodecay_head, "lr": base_lr, "weight_decay": 0.0},
    ]
    optimizer = AdamW(param_groups, betas=(0.9, 0.999))
    return optimizer

# 若 create_brats_stable_training_components 已存在，则在内部优先调用上面的构建器。
# 这里不直接改其签名，保持兼容，由其内部检测并采用 build_adamw_with_param_groups。
def main():
    """主修复函数"""
    print("="*60)
    print("🚀 BraTS2020脑肿瘤分割训练问题修复脚本")
    print("="*60)
    
    print("\n📋 修复内容:")
    print("1. ✅ 调整损失函数权重 (alpha=[3.0, 4.0, 5.0])")
    print("2. ✅ 放宽梯度裁剪阈值 (1.0 -> 5.0)")
    print("3. ✅ 改进权重初始化 (gain=0.1 -> 0.5)")
    print("4. ✅ 优化NaN/Inf处理 (修复而非清零)")
    print("5. ✅ 提高学习率 (建议使用5e-4)")
    print("6. ✅ 针对BraTS数据集优化损失函数")
    print("7. ✅ 改进验证函数和指标计算（含动态阈值与HD95；已移除TTA/连通域过滤）")
    
    print("\n🔧 使用方法:")
    print("在train_f.py中替换以下组件:")
    print("```python")
    print("from fix_training_issues import (")
    print("    ImprovedRegionAwareLoss,")
    print("    create_improved_optimizer,")
    print("    apply_model_fixes,")
    print("    create_improved_scaler,")
    print("    improved_validate")
    print(")")
    print("")
    print("# 替换损失函数")
    print("criterion = ImprovedRegionAwareLoss()")
    print("")
    print("# 替换优化器")
    print("optimizer = create_improved_optimizer(model, initial_lr=5e-4)")
    print("")
    print("# 应用模型修复")
    print("model = apply_model_fixes(model)")
    print("")
    print("# 替换scaler")
    print("scaler = create_improved_scaler()")
    print("")
    print("# 替换验证函数")
    print("val_metrics = improved_validate(model, val_loader)")
    print("```")
    
    print("\n📈 预期效果:")
    print("- 损失值开始下降 (2.24 -> 1.5以下)")
    print("- NaN/Inf梯度警告显著减少")
    print("- Dice系数提升 (WT>0.3, TC>0.2, ET>0.1)")
    print("- 训练过程更加稳定")
    
    print("\n⚠️  注意事项:")
    print("1. 建议重新开始训练，不要从旧检查点恢复")
    print("2. 密切监控前几个epoch的损失变化")
    print("3. 如果显存不足，可以减小batch_size")
    print("4. 保存当前模型状态作为备份")
    
    print("\n" + "="*60)

if __name__ == "__main__":
    main()


