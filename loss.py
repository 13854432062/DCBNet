import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast


class TransUNetBceDiceLoss(nn.Module):
    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5, smooth: float = 1e-5):
        super().__init__()
        self.bce_weight = float(bce_weight)
        self.dice_weight = float(dice_weight)
        self.smooth = float(smooth)

    def _extract_logits(self, outputs):
        if isinstance(outputs, dict):
            if 'logits' in outputs:
                return outputs['logits']
            if 'stage4' in outputs:
                return outputs['stage4']
        if isinstance(outputs, (tuple, list)) and len(outputs) > 0:
            return outputs[0]
        return outputs

    def forward(self, outputs, targets, edges=None):
        logits = self._extract_logits(outputs)
        if not isinstance(logits, torch.Tensor):
            raise TypeError(f"Unsupported outputs type for TransUNetBceDiceLoss: {type(outputs)}")

        if logits.shape[-2:] != targets.shape[-2:]:
            logits = F.interpolate(logits, size=targets.shape[-2:], mode='bilinear', align_corners=False)

        targets_f = targets.float()
        bce = F.binary_cross_entropy_with_logits(logits, targets_f, reduction='mean')

        prob = torch.sigmoid(logits)
        dims = (0, 2, 3)
        inter = (prob * targets_f).sum(dim=dims)
        denom = prob.sum(dim=dims) + targets_f.sum(dim=dims)
        dice = (2.0 * inter + self.smooth) / (denom + self.smooth)
        dice_loss = 1.0 - dice.mean()

        return self.bce_weight * bce + self.dice_weight * dice_loss


class MySegBceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, outputs, targets, edges=None):
        logits = outputs
        if isinstance(outputs, dict):
            if 'logits' in outputs:
                logits = outputs['logits']
            elif 'stage4' in outputs:
                logits = outputs['stage4']
        if isinstance(outputs, (tuple, list)) and len(outputs) > 0:
            logits = outputs[0]
        if not isinstance(logits, torch.Tensor):
            raise TypeError(f"Unsupported outputs type for MySegBceLoss: {type(outputs)}")

        if logits.shape[-2:] != targets.shape[-2:]:
            logits = F.interpolate(logits, size=targets.shape[-2:], mode='bilinear', align_corners=False)

        return F.binary_cross_entropy_with_logits(logits, targets.float(), reduction='mean')


class S2NetDiceBceLoss(nn.Module):
    def __init__(self, smooth: float = 1e-5, eps: float = 1e-6):
        super().__init__()
        self.smooth = float(smooth)
        self.eps = float(eps)

    def _extract_logits(self, outputs):
        if isinstance(outputs, dict):
            if 'logits' in outputs:
                return outputs['logits']
            if 'stage4' in outputs:
                return outputs['stage4']
        if isinstance(outputs, (tuple, list)) and len(outputs) > 0:
            return outputs[0]
        return outputs

    def forward(self, outputs, targets, edges=None):
        logits = self._extract_logits(outputs)
        if not isinstance(logits, torch.Tensor):
            raise TypeError(f"Unsupported outputs type for S2NetDiceBceLoss: {type(outputs)}")

        if logits.shape[-2:] != targets.shape[-2:]:
            logits = F.interpolate(logits, size=targets.shape[-2:], mode='bilinear', align_corners=False)

        tgt = targets.float()
        prob = torch.sigmoid(logits).clamp(self.eps, 1.0 - self.eps)

        bce = F.binary_cross_entropy_with_logits(logits, tgt, reduction='mean')

        dims = (0, 2, 3)
        inter = (prob * tgt).sum(dim=dims)
        denom = prob.sum(dim=dims) + tgt.sum(dim=dims)
        dice = (2.0 * inter + self.smooth) / (denom + self.smooth)
        dice_loss = 1.0 - dice.mean()

        return dice_loss + bce


class BCEDiceFocalLoss(nn.Module):
    def __init__(
        self,
        bce_weight: float = 0.4,
        dice_weight: float = 0.4,
        focal_weight: float = 0.2,
        focal_gamma: float = 2.0,
        edge_weight: float = 0.05,
        class_weights=None,
        smooth: float = 1e-5,
        eps: float = 1e-6,
        edge_warmup_epochs: int = 6,
        edge_rampup_epochs: int = 10,
    ):
        super().__init__()
        self.bce_weight = float(bce_weight)
        self.dice_weight = float(dice_weight)
        self.focal_weight = float(focal_weight)
        self.focal_gamma = float(focal_gamma)
        self.edge_weight_max = float(edge_weight)
        self.edge_warmup_epochs = int(edge_warmup_epochs)
        self.edge_rampup_epochs = int(edge_rampup_epochs)
        self._current_epoch = 0
        if class_weights is None:
            self.register_buffer('class_weights', torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32))
        else:
            cw = torch.as_tensor(class_weights, dtype=torch.float32)
            if cw.numel() != 3:
                raise ValueError(f"BCEDiceFocalLoss expects class_weights with 3 elements (WT/TC/ET), got {int(cw.numel())}")
            self.register_buffer('class_weights', cw.view(3))
        self.smooth = float(smooth)
        self.eps = float(eps)

    def set_epoch(self, epoch: int):
        self._current_epoch = int(epoch)

    def _get_edge_weight(self) -> float:
        epoch = self._current_epoch
        if epoch < self.edge_warmup_epochs:
            return 0.0
        ramp_epoch = epoch - self.edge_warmup_epochs
        if ramp_epoch >= self.edge_rampup_epochs:
            return self.edge_weight_max
        t = float(ramp_epoch + 1) / float(self.edge_rampup_epochs)
        return self.edge_weight_max * t

    def _extract_logits(self, outputs):
        if isinstance(outputs, dict):
            if 'logits' in outputs:
                return outputs['logits']
            if 'stage4' in outputs:
                return outputs['stage4']
        if isinstance(outputs, (tuple, list)) and len(outputs) > 0:
            return outputs[0]
        return outputs

    def _dice_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        prob = torch.sigmoid(logits).clamp(self.eps, 1.0 - self.eps)
        tgt = targets.float()
        dims = (0, 2, 3)
        inter = (prob * tgt).sum(dim=dims)
        denom = prob.sum(dim=dims) + tgt.sum(dim=dims)
        dice = (2.0 * inter + self.smooth) / (denom + self.smooth)
        dice_loss_per_c = 1.0 - dice
        w = self.class_weights.to(device=logits.device, dtype=logits.dtype)
        w = w / (w.sum() + self.eps)
        return (dice_loss_per_c * w).sum()

    def _focal_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        tgt = targets.float()
        bce = F.binary_cross_entropy_with_logits(logits, tgt, reduction='none')
        p = torch.sigmoid(logits)
        pt = p * tgt + (1.0 - p) * (1.0 - tgt)
        focal = ((1.0 - pt).clamp_min(self.eps) ** self.focal_gamma) * bce
        focal_per_c = focal.mean(dim=(0, 2, 3))
        w = self.class_weights.to(device=logits.device, dtype=focal_per_c.dtype)
        w = w / (w.sum() + self.eps)
        return (focal_per_c * w).sum()

    def forward(self, outputs, targets, edges=None):
        logits = self._extract_logits(outputs)
        if not isinstance(logits, torch.Tensor):
            raise TypeError(f"Unsupported outputs type for BCEDiceFocalLoss: {type(outputs)}")

        if logits.shape[-2:] != targets.shape[-2:]:
            logits = F.interpolate(logits, size=targets.shape[-2:], mode='bilinear', align_corners=False)

        bce_per = F.binary_cross_entropy_with_logits(logits, targets.float(), reduction='none')
        bce_per_c = bce_per.mean(dim=(0, 2, 3))
        w = self.class_weights.to(device=logits.device, dtype=bce_per_c.dtype)
        w = w / (w.sum() + self.eps)
        bce = (bce_per_c * w).sum()
        dice = self._dice_loss(logits, targets)
        focal = self._focal_loss(logits, targets)

        loss = self.bce_weight * bce + self.dice_weight * dice + self.focal_weight * focal

        current_edge_weight = self._get_edge_weight()
        if (
            current_edge_weight > 0
            and edges is not None
            and isinstance(outputs, dict)
            and isinstance(outputs.get('edge', None), torch.Tensor)
        ):
            edge_logits = outputs['edge']
            edge_targets = edges.float()
            if edge_logits.shape[-2:] != edge_targets.shape[-2:]:
                edge_logits = F.interpolate(edge_logits, size=edge_targets.shape[-2:], mode='bilinear', align_corners=False)
            if edge_targets.dim() == 3:
                edge_targets = edge_targets.unsqueeze(1)
            if edge_targets.size(1) == 1 and edge_logits.size(1) > 1:
                edge_targets = edge_targets.repeat(1, edge_logits.size(1), 1, 1)
            if edge_targets.size(1) != edge_logits.size(1):
                edge_loss = None
            else:
                edge_loss = F.binary_cross_entropy_with_logits(edge_logits, edge_targets, reduction='mean')
            if edge_loss is not None and torch.isfinite(edge_loss).item():
                loss = loss + current_edge_weight * edge_loss

        return loss


class SegFormerCeDiceLoss(nn.Module):
    def __init__(self, num_classes: int = 4, ce_weight: float = 0.4, dice_weight: float = 0.6, smooth: float = 1e-5):
        super().__init__()
        self.num_classes = int(num_classes)
        if self.num_classes < 2:
            raise ValueError(f"SegFormerCeDiceLoss expects num_classes>=2, got {self.num_classes}")
        self.ce_weight = float(ce_weight)
        self.dice_weight = float(dice_weight)
        self.smooth = float(smooth)
        self.ce = nn.CrossEntropyLoss()

    def _extract_logits_mc(self, outputs):
        if isinstance(outputs, dict) and 'logits_mc' in outputs:
            return outputs['logits_mc']
        if isinstance(outputs, dict):
            if 'logits' in outputs:
                return outputs['logits']
            if 'stage4' in outputs:
                return outputs['stage4']
        if isinstance(outputs, (tuple, list)) and len(outputs) > 0:
            return outputs[0]
        return outputs

    def _targets_to_index(self, targets: torch.Tensor) -> torch.Tensor:
        if not (isinstance(targets, torch.Tensor) and targets.dim() == 4 and targets.size(1) >= 3):
            raise ValueError(f"SegFormerCeDiceLoss expects targets [B,3,H,W] (WT/TC/ET). Got: {getattr(targets, 'shape', None)}")
        wt = targets[:, 0] > 0.5
        tc = targets[:, 1] > 0.5
        et = targets[:, 2] > 0.5
        y = torch.zeros_like(targets[:, 0], dtype=torch.long)
        y = torch.where(wt, torch.ones_like(y), y)
        y = torch.where(tc, torch.full_like(y, 2), y)
        y = torch.where(et, torch.full_like(y, 3), y)
        return y

    def _softmax_dice_foreground(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        prob = torch.softmax(logits, dim=1)
        y_oh = F.one_hot(y.clamp(0, self.num_classes - 1), num_classes=self.num_classes).permute(0, 3, 1, 2).float()

        dims = (0, 2, 3)
        inter = (prob * y_oh).sum(dim=dims)
        denom = prob.sum(dim=dims) + y_oh.sum(dim=dims)
        dice = (2.0 * inter + self.smooth) / (denom + self.smooth)
        if dice.numel() <= 1:
            return 1.0 - dice.mean()
        return 1.0 - dice[1:].mean()

    def forward(self, outputs, targets, edges=None):
        logits = self._extract_logits_mc(outputs)
        if not isinstance(logits, torch.Tensor):
            raise TypeError(f"Unsupported outputs type for SegFormerCeDiceLoss: {type(outputs)}")

        y = self._targets_to_index(targets)
        if logits.shape[-2:] != y.shape[-2:]:
            logits = F.interpolate(logits, size=y.shape[-2:], mode='bilinear', align_corners=False)

        if logits.size(1) != self.num_classes:
            raise ValueError(f"SegFormerCeDiceLoss expects logits [B,{self.num_classes},H,W], got {tuple(logits.shape)}")

        loss_ce = self.ce(logits, y)
        loss_dice = self._softmax_dice_foreground(logits, y)
        return self.ce_weight * loss_ce + self.dice_weight * loss_dice


class MISSFormerCeDiceLoss(SegFormerCeDiceLoss):
    def __init__(self, num_classes: int = 4, ce_weight: float = 0.4, dice_weight: float = 0.6, smooth: float = 1e-5):
        super().__init__(num_classes=int(num_classes), ce_weight=float(ce_weight), dice_weight=float(dice_weight), smooth=float(smooth))


class DATransUNetDiceLoss(nn.Module):
    def __init__(self, n_classes: int):
        super().__init__()
        self.n_classes = int(n_classes)

    def _one_hot_encoder(self, input_tensor: torch.Tensor) -> torch.Tensor:
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs: torch.Tensor, target: torch.Tensor, weight=None, softmax: bool = False) -> torch.Tensor:
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            loss += dice * weight[i]
        return loss / self.n_classes


class DATransUNetCeDiceLoss(nn.Module):
    def __init__(self, num_classes: int = 4, ce_weight: float = 0.5, dice_weight: float = 0.5):
        super().__init__()
        self.num_classes = int(num_classes)
        self.ce_weight = float(ce_weight)
        self.dice_weight = float(dice_weight)
        self.ce = nn.CrossEntropyLoss()
        self.dice = DATransUNetDiceLoss(self.num_classes)

    def _extract_logits(self, outputs):
        if isinstance(outputs, dict):
            if 'logits' in outputs:
                return outputs['logits']
            if 'stage4' in outputs:
                return outputs['stage4']
        if isinstance(outputs, (tuple, list)) and len(outputs) > 0:
            return outputs[0]
        return outputs

    def _targets_to_index(self, targets: torch.Tensor) -> torch.Tensor:
        if not (isinstance(targets, torch.Tensor) and targets.dim() == 4 and targets.size(1) >= 3):
            raise ValueError(f"DATransUNetCeDiceLoss expects targets [B,3,H,W] (WT/TC/ET). Got: {getattr(targets, 'shape', None)}")
        wt = targets[:, 0] > 0.5
        tc = targets[:, 1] > 0.5
        et = targets[:, 2] > 0.5
        y = torch.zeros_like(targets[:, 0], dtype=torch.long)
        y = torch.where(wt, torch.ones_like(y), y)
        y = torch.where(tc, torch.full_like(y, 2), y)
        y = torch.where(et, torch.full_like(y, 3), y)
        return y

    def forward(self, outputs, targets, edges=None):
        logits = self._extract_logits(outputs)
        if not isinstance(logits, torch.Tensor):
            raise TypeError(f"Unsupported outputs type for DATransUNetCeDiceLoss: {type(outputs)}")

        if logits.shape[-2:] != targets.shape[-2:]:
            logits = F.interpolate(logits, size=targets.shape[-2:], mode='bilinear', align_corners=False)

        y = self._targets_to_index(targets)
        loss_ce = self.ce(logits, y)
        loss_dice = self.dice(logits, y, softmax=True)
        return self.ce_weight * loss_ce + self.dice_weight * loss_dice

# 🆕 方案D: 拓扑损失采用惰性导入以降低耦合（在使用处再导入）



def tversky_loss(logits: torch.Tensor, target: torch.Tensor, alpha: float = 0.5, beta: float = 0.5, smooth: float = 1e-3) -> torch.Tensor:
    """计算 Tversky 损失，返回 1-Index。适用于小目标召回。

    logits: Tensor[B,H,W] 原始预测
    target: Tensor[B,H,W] 二值标签
    alpha, beta: FP 与 FN 的权重系数，alpha 小→偏召回，beta 小→偏精度
    """
    prob = torch.sigmoid(logits).clamp(smooth, 1 - smooth)
    target = target.float()
    tp = (prob * target).sum(dim=(-2, -1))
    fp = ((1 - target) * prob).sum(dim=(-2, -1))
    fn = (target * (1 - prob)).sum(dim=(-2, -1))
    tversky_idx = (tp + smooth) / (tp + alpha * fp + beta * fn + smooth)
    return 1 - tversky_idx


class BoundaryWeightedDiceLoss(nn.Module):
    """🎯 边界加权 Dice Loss - 直接提升边界分割精度
    
    核心思想：
      - 使用 GT 边界作为像素权重
      - 边界区域的错误会产生更大的损失
      - 不需要训练 EAM，不改变模型结构
      - 类别差异化权重：WT < TC < ET
    
    优势：
      - ✅ 零风险（不改模型）
      - ✅ GT 边界 100% 准确
      - ✅ 直接提升 TC/ET Dice
      - ✅ 预期 TC +0.6%, ET +0.8%
    
    使用：
      loss = boundary_weighted_dice(pred, target, gt_edges)
      # pred: [B, 3, H, W] - sigmoid后的概率
      # target: [B, 3, H, W] - GT分割标签
      # gt_edges: [B, 3, H, W] - GT边界（0/1）
    """
    
    def __init__(self, class_weights=None, smooth=1e-5):
        """
        Args:
            class_weights: [WT, TC, ET] 边界权重，默认 [1.0, 1.5, 2.0]
            smooth: 数值稳定性平滑项
        """
        super().__init__()
        if class_weights is None:
            # WT: 大区域，边界清晰 → 权重 1.0
            # TC: 中等难度 → 权重 1.5
            # ET: 小区域，最难 → 权重 2.0
            class_weights = [1.0, 1.5, 2.0]
        self.register_buffer('class_weights', torch.tensor(class_weights, dtype=torch.float32))
        self.smooth = smooth
    
    def forward(self, pred, target, gt_edges):
        """
        Args:
            pred: [B, 3, H, W] - 模型预测（sigmoid后的概率，范围 0-1）
            target: [B, 3, H, W] - GT 分割标签（0/1）
            gt_edges: [B, 3, H, W] - GT 边界（0/1）
        
        Returns:
            loss: 边界加权 Dice Loss（标量）
        """
        # 确保输入为 float
        pred = pred.float()
        target = target.float()
        gt_edges = gt_edges.float()
        
        # 计算像素权重
        # 边界区域: pixel_weights = 1.0 + class_weights = [2.0, 2.5, 3.0]
        # 非边界:   pixel_weights = 1.0
        class_weights = self.class_weights.view(1, 3, 1, 1).to(pred.device)
        pixel_weights = 1.0 + gt_edges * class_weights  # [B, 3, H, W]
        
        # 加权交集和并集
        intersection = (pred * target * pixel_weights).sum(dim=(2, 3))  # [B, 3]
        pred_sum = (pred * pixel_weights).sum(dim=(2, 3))              # [B, 3]
        target_sum = (target * pixel_weights).sum(dim=(2, 3))          # [B, 3]
        
        # Dice 系数
        dice = (2 * intersection + self.smooth) / (pred_sum + target_sum + self.smooth)  # [B, 3]
        
        # 返回损失（1 - Dice）
        return 1.0 - dice.mean()


class BoundaryAwareTverskyLoss(nn.Module):
    def __init__(self, boundary_alpha=0.3, boundary_beta=0.7, non_boundary_alpha=0.5, non_boundary_beta=0.5):
        super().__init__()
        self.boundary_alpha = boundary_alpha
        self.boundary_beta = boundary_beta
        self.non_boundary_alpha = non_boundary_alpha
        self.non_boundary_beta = non_boundary_beta
        self.smooth = 1e-5

    def forward(self, pred, target, boundary_mask):
        prob = torch.sigmoid(pred)
        boundary_mask = boundary_mask.float()
        target = target.float()

        boundary_pred = prob * boundary_mask
        boundary_target = target * boundary_mask

        tp_b = (boundary_pred * boundary_target).sum(dim=(2, 3))
        fp_b = ((1 - boundary_target) * boundary_pred).sum(dim=(2, 3))
        fn_b = (boundary_target * (1 - boundary_pred)).sum(dim=(2, 3))

        tversky_b = (tp_b + self.smooth) / (tp_b + self.boundary_alpha * fp_b + self.boundary_beta * fn_b + self.smooth)
        loss_b = 1 - tversky_b

        non_boundary_mask = 1.0 - boundary_mask
        nb_pred = prob * non_boundary_mask
        nb_target = target * non_boundary_mask

        tp_nb = (nb_pred * nb_target).sum(dim=(2, 3))
        fp_nb = ((1 - nb_target) * nb_pred).sum(dim=(2, 3))
        fn_nb = (nb_target * (1 - nb_pred)).sum(dim=(2, 3))

        tversky_nb = (tp_nb + self.smooth) / (tp_nb + self.non_boundary_alpha * fp_nb + self.non_boundary_beta * fn_nb + self.smooth)
        loss_nb = 1 - tversky_nb

        total_loss = 0.6 * loss_b + 0.4 * loss_nb
        return total_loss.mean()




 
class ProgressiveEdgeSupervision(nn.Module):
    def __init__(self, class_weights=None, base_weight=0.2):
        super().__init__()
        if class_weights is None:
            class_weights = [1.0, 1.5, 2.0]
        self.register_buffer('class_weights', torch.tensor(class_weights, dtype=torch.float32))
        self.base_weight = float(base_weight)

    def _sched(self, epoch: int):
        if epoch is None:
            return 1.0
        e = int(epoch)
        if e <= 8:
            t = e / 8.0
            return 0.5 + 0.5 * t
        if e <= 25:
            return 1.0
        t = min((e - 25) / 20.0, 1.0)
        return 1.0 - 0.4 * t

    def forward(self, outputs, gt_edges, epoch=None):
        if not isinstance(outputs, dict):
            return None
        keys = [k for k in outputs.keys() if 'edge_logits' in k]
        if len(keys) == 0:
            if 'edge' in outputs and outputs['edge'].dim() == 4 and outputs['edge'].shape[1] == 3:
                keys = ['edge']
            else:
                return None
        if gt_edges.dim() == 3:
            gt_edges = gt_edges.unsqueeze(1)
        gt = gt_edges.float()
        loss_total = None
        for k in keys:
            logits = outputs[k]
            if logits.shape[-2:] != gt.shape[-2:]:
                logits = F.interpolate(logits, size=gt.shape[-2:], mode='bilinear', align_corners=False)
            loss_map = F.binary_cross_entropy_with_logits(logits, gt, reduction='none')
            cw = self.class_weights.view(1, -1, 1, 1).to(loss_map.device)
            loss = (loss_map * cw).mean()
            loss_total = loss if loss_total is None else loss_total + loss
        scale = self.base_weight * self._sched(epoch)
        return loss_total * scale if loss_total is not None else None


def mbam_boundary_loss(edge_logits: torch.Tensor,
                      gt_edges: torch.Tensor,
                      class_weights=None,
                      smooth: float = 1e-5) -> torch.Tensor:
    """显式 MBAM 边界监督：逐类 BCE + Dice，ET 权重更高。

    Args:
        edge_logits: [B, 3, H, W]，MBAM 输出的三类边界 logits（未过 sigmoid）
        gt_edges   : [B, 3, H, W] 或 [B, 1, H, W] / [B, H, W]，GT 边界
        class_weights: [WT, TC, ET] 的类别权重，默认 [1.0, 1.7, 2.5]
    Returns:
        标量边界损失
    """
    if class_weights is None:
        class_weights = [1.0, 1.7, 2.5]

    # 规范 GT 形状
    if gt_edges.dim() == 3:
        gt_edges = gt_edges.unsqueeze(1)

    try:
        if isinstance(edge_logits, torch.Tensor) and edge_logits.dim() == 4 and edge_logits.shape[1] == 1:
            if isinstance(gt_edges, torch.Tensor) and gt_edges.dim() == 4 and gt_edges.shape[1] == 3:
                gt_edges = (gt_edges > 0.5).float().sum(dim=1, keepdim=True).clamp(0.0, 1.0)
    except Exception:
        pass
    # [B,1,H,W]
    gt_edges = gt_edges.float()

    # 多通道 logits：GT 需要是 [B,3,H,W]
    if gt_edges.shape[1] == 1 and edge_logits.shape[1] == 3:
        gt_edges = gt_edges.repeat(1, 3, 1, 1)

    # 尺寸对齐
    if edge_logits.shape[-2:] != gt_edges.shape[-2:]:
        edge_logits = F.interpolate(edge_logits, size=gt_edges.shape[-2:], mode='bilinear', align_corners=False)

    # BCE with logits（逐类，带正负不均衡修正）
    edge_logits_f = edge_logits.float()
    gt_edges_f = gt_edges.float()
    try:
        total = float(gt_edges_f.shape[0] * gt_edges_f.shape[2] * gt_edges_f.shape[3])
        pos = gt_edges_f.sum(dim=(0, 2, 3))
        neg = pos.new_full(pos.shape, total) - pos
        pos_weight = neg / (pos + 1.0)
        pos_weight = torch.where(pos > 0.0, pos_weight, torch.ones_like(pos_weight))
        pos_weight = pos_weight.clamp(1.0, 30.0).detach().view(1, -1, 1, 1)
    except Exception:
        pos_weight = None

    bce_map = F.binary_cross_entropy_with_logits(
        edge_logits_f,
        gt_edges_f,
        reduction='none',
        pos_weight=pos_weight,
    )  # [B,3,H,W]
    bce_per_class = bce_map.mean(dim=(0, 2, 3))

    # Dice loss（使用 sigmoid 概率）
    prob = torch.sigmoid(edge_logits_f).clamp(smooth, 1 - smooth)
    inter = (prob * gt_edges).sum(dim=(0, 2, 3))
    pred_sum = prob.sum(dim=(0, 2, 3))
    target_sum = gt_edges.sum(dim=(0, 2, 3))
    dice = (2 * inter + smooth) / (pred_sum + target_sum + smooth)
    dice_loss = 1.0 - dice

    per_class_loss = bce_per_class + dice_loss            # [3]

    # 类别权重（ET 边界权重最高）
    if edge_logits.shape[1] == 1:
        cw = torch.ones(1, dtype=torch.float32, device=edge_logits.device)
    else:
        cw = torch.tensor(class_weights, dtype=torch.float32, device=edge_logits.device)
    cw = cw / (cw.sum() + 1e-8)
    loss = (per_class_loss * cw).sum()
    return loss


def tumor_foreground_loss(fg_logits: torch.Tensor,
                          fg_targets: torch.Tensor,
                          smooth: float = 1e-5) -> torch.Tensor:
    """肿瘤前景监督：前景掩码的 BCE + Dice 组合损失。

    Args:
        fg_logits : [B, 1, H, W]，前景 logits（未过 sigmoid）
        fg_targets: [B, 1, H, W] 或 [B, H, W]，前景 GT（二值，WT/TC/ET 的并集）
    """
    if fg_targets.dim() == 3:
        fg_targets = fg_targets.unsqueeze(1)

    fg_targets = fg_targets.float()

    # 尺寸对齐
    if fg_logits.shape[-2:] != fg_targets.shape[-2:]:
        fg_logits = F.interpolate(fg_logits, size=fg_targets.shape[-2:], mode='bilinear', align_corners=False)

    # BCE
    bce = F.binary_cross_entropy_with_logits(fg_logits, fg_targets, reduction='mean')

    # Dice
    prob = torch.sigmoid(fg_logits).clamp(smooth, 1 - smooth)
    inter = (prob * fg_targets).sum(dim=(0, 2, 3))
    pred_sum = prob.sum(dim=(0, 2, 3))
    target_sum = fg_targets.sum(dim=(0, 2, 3))
    dice = (2 * inter + smooth) / (pred_sum + target_sum + smooth)
    dice_loss = 1.0 - dice.mean()

    return 0.5 * bce + 0.5 * dice_loss


class BraTSStableLoss(nn.Module):
    def __init__(self, alpha=[2.7, 3.2, 3.8], smooth=1e-3,
                enable_dynamic_weights: bool = True,
                training_stage: str = 'full',
                class_weights=None,
                enable_focal_tversky: bool = True,
                ft_gamma: float = 1.33,
                disable_ce_loss: bool = False,
                disable_focal_loss: bool = False):
        super().__init__()
        self.alpha = torch.tensor(alpha, dtype=torch.float32)
        self.smooth = smooth
        default_cw = [1.0, 1.40, 1.75]
        cw = default_cw if class_weights is None else class_weights
        self.register_buffer('class_weights', torch.tensor(cw, dtype=torch.float32))
        self.register_buffer('focal_gamma_by_class', torch.tensor([1.5, 1.2, 1.2], dtype=torch.float32))
        self.register_buffer('ce_class_multipliers', torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32))
        self.enable_dynamic_weights = enable_dynamic_weights
        self.training_stage = training_stage
        self.enable_focal_tversky = bool(enable_focal_tversky)
        self.ft_gamma = float(ft_gamma)
        self.ft_gamma_final = 1.0
        self.ft_gamma_anneal_steps = 6000
        # 损失函数消融开关
        self.disable_ce_loss = bool(disable_ce_loss)
        self.disable_focal_loss = bool(disable_focal_loss)
        # 阶段权重
        if training_stage == 'early':
            self.dice_weight, self.ce_weight, self.focal_weight = 0.50, 0.25, 0.25
        elif training_stage == 'mid':
            self.dice_weight, self.ce_weight, self.focal_weight = 0.40, 0.25, 0.35
        else:
            self.dice_weight, self.ce_weight, self.focal_weight = 0.35, 0.25, 0.40
        self.eps = 1e-6
        self.register_buffer('loss_idx', torch.tensor(0, dtype=torch.long))
        self.loss_threshold = 6.5
        self.max_loss_value = 15.0
        self.skip_batches = 0
        self.aux_use_light_loss = False
        self.register_buffer('loss_history', torch.zeros(100, dtype=torch.float32))

    def _get_dynamic_weights(self):
        if not self.enable_dynamic_weights:
            return self.dice_weight, self.ce_weight, self.focal_weight
        step = int(self.loss_idx.item())
        r = min(step / 2000.0, 1.0)
        dice_w = self.dice_weight * (1.0 - 0.2 * r)
        ce_w = self.ce_weight * (1.0 - 0.2 * r)
        focal_w = self.focal_weight * (1.0 + 0.3 * r)
        total = dice_w + ce_w + focal_w + 1e-8
        dice_w, ce_w, focal_w = [w * (1.0 / total) for w in (dice_w, ce_w, focal_w)]
        return dice_w, ce_w, focal_w

    def stable_dice_loss(self, pred, target, class_idx):
        prob = torch.sigmoid(pred).clamp(self.eps, 1 - self.eps)
        target = target.float()
        inter = (prob * target).sum(dim=(-2, -1))
        pred_sum = prob.sum(dim=(-2, -1))
        target_sum = target.sum(dim=(-2, -1))
        if int(class_idx) in (1, 2):
            alpha, beta = 0.25, 0.75
            denom = inter + alpha * (pred_sum - inter) + beta * (target_sum - inter)
            tversky = (inter + self.smooth) / (denom + self.smooth)
            loss_raw = 1 - tversky
            if bool(getattr(self, 'enable_focal_tversky', False)):
                step = int(self.loss_idx.item())
                if float(getattr(self, 'ft_gamma_anneal_steps', 0)) > 0:
                    a = min(max(step / float(max(1, int(self.ft_gamma_anneal_steps))), 0.0), 1.0)
                else:
                    a = 1.0
                gamma_now = (1.0 - a) * float(getattr(self, 'ft_gamma', 1.33)) + a * float(getattr(self, 'ft_gamma_final', 1.0))
                loss_raw = torch.clamp(loss_raw, 0.0, 1.0) ** gamma_now
        else:
            dice = (2 * inter + self.smooth) / (pred_sum + target_sum + self.smooth)
            loss_raw = 1 - dice
        empty_tgt = (target_sum < 1e-3)
        loss_raw = torch.where(empty_tgt, torch.clamp(1 - pred_sum * 0.1, 0.8, 1.0), loss_raw)
        return loss_raw.mean()

    def stable_ce_loss(self, pred, target):
        target = target.float()
        return F.binary_cross_entropy_with_logits(pred, target, reduction='mean')

    def weighted_ohem_bce(self, pred, target, class_idx, ohem_ratio: float = 0.25):
        target = target.float()
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        if isinstance(self.class_weights, torch.Tensor) and int(class_idx) < int(self.class_weights.numel()):
            bce = bce * float(self.class_weights[int(class_idx)].detach().cpu())
        b = bce.reshape(bce.size(0), -1)
        k = max(1, int(b.size(1) * float(ohem_ratio)))
        topk, _ = torch.topk(b, k=k, dim=1, largest=True, sorted=False)
        return topk.mean()

    def stable_focal_loss(self, pred, target, class_idx):
        target = target.float()
        gamma = 2.0
        if isinstance(getattr(self, 'focal_gamma_by_class', None), torch.Tensor) and int(class_idx) < int(self.focal_gamma_by_class.numel()):
            gamma = float(self.focal_gamma_by_class[int(class_idx)].detach().cpu())
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        p = torch.sigmoid(pred)
        pt = p * target + (1.0 - p) * (1.0 - target)
        loss = bce * ((1.0 - pt) ** gamma)
        return loss.mean()

    def check_loss_stability(self, loss, class_idx):
        if loss is None:
            return False, 'none'
        if not torch.isfinite(loss).all():
            return False, 'nan/inf'
        v = float(loss.detach().cpu())
        if v > float(getattr(self, 'max_loss_value', 15.0)):
            return False, 'too_large'
        return True, 'ok'

    def update_loss_history(self, total_loss: torch.Tensor):
        try:
            idx = int(self.loss_idx.item())
        except Exception:
            idx = 0
        pos = idx % int(self.loss_history.numel())
        v = float(total_loss.detach().cpu()) if isinstance(total_loss, torch.Tensor) else float(total_loss)
        self.loss_history[pos] = float(v)
        self.loss_idx += 1

    def get_adaptive_threshold(self):
        h = self.loss_history.detach()
        valid = h[h > 0]
        if valid.numel() < 8:
            return float(getattr(self, 'loss_threshold', 6.5))
        m = float(valid.mean().cpu())
        s = float(valid.std(unbiased=False).cpu())
        return max(float(getattr(self, 'loss_threshold', 6.5)), m + 2.5 * s)

    def forward(self, outputs, targets, edges=None, epoch=None):
        """前向传播
        参数：
            outputs: 模型输出（dict or Tensor[B,C,H,W]）
            targets: 分割标签 Tensor[B,C,H,W] 或 Tensor[B,1,H,W]
            edges: 可选边界标签 Tensor[B,C,H,W] / Tensor[B,1,H,W] / Tensor[B,H,W]
        """
        # 判断是否为辅助分支调用（Tensor 多为 stage1/2/3 或 ds_*）
        is_aux_call = not isinstance(outputs, dict)
        # 获取预测结果
        if isinstance(outputs, dict):
            if 'logits' in outputs:
                pred = outputs['logits']
            elif 'stage4' in outputs:
                pred = outputs['stage4']
            else:
                pred = list(outputs.values())[-1]
        else:
            pred = outputs

        # 确保目标格式正确
        if targets.dim() == 3:  # [B, H, W]
            targets = targets.unsqueeze(1)  # [B, 1, H, W]

        batch_size, num_classes = pred.shape[0], pred.shape[1]

        total_loss = 0.0
        valid_samples = 0
        class_losses = []

        # 逐类别计算损失
        per_class_probs = []  # 用于GDL与自适应加权
        per_class_targets = []
        for c in range(num_classes):
            pred_c = pred[:, c]
            target_c = targets[:, c] if targets.shape[1] > c else targets[:, 0]

            dice_loss = self.stable_dice_loss(pred_c, target_c, c)
            if is_aux_call and self.aux_use_light_loss:
                # 辅助分支走轻量路径：Dice + 平滑CE；跳过OHEM/Focal/边界/Lovasz/HD
                ce_loss = self.stable_ce_loss(pred_c, target_c) if not self.disable_ce_loss else pred_c.new_tensor(0.0)
                focal_loss = pred_c.new_tensor(0.0)
            else:
                # 主分支使用完整路径
                if not self.disable_ce_loss:
                    ce_loss = self.weighted_ohem_bce(pred_c, target_c, c)
                    ce_loss = ce_loss * self.ce_class_multipliers[c].to(ce_loss.device)
                else:
                    ce_loss = pred_c.new_tensor(0.0)
                
                if not self.disable_focal_loss:
                    focal_loss = self.stable_focal_loss(pred_c, target_c, c)
                else:
                    focal_loss = pred_c.new_tensor(0.0)

            
            # 动态/阶段化权重
            dice_w, ce_w, focal_w = self._get_dynamic_weights()
            
            # 如果禁用了某个损失，重新归一化权重
            if self.disable_ce_loss or self.disable_focal_loss:
                active_weights = []
                if not self.disable_ce_loss:
                    active_weights.append(('ce', ce_w))
                if not self.disable_focal_loss:
                    active_weights.append(('focal', focal_w))
                active_weights.append(('dice', dice_w))
                
                total_w = sum(w for _, w in active_weights)
                if total_w > 0:
                    dice_w = dice_w / total_w
                    ce_w = ce_w / total_w if not self.disable_ce_loss else 0.0
                    focal_w = focal_w / total_w if not self.disable_focal_loss else 0.0

            # 组合损失
            class_loss = (
                dice_w * dice_loss +
                ce_w * ce_loss +
                focal_w * focal_loss
            )
            # 移除 Lovasz/边界辅助/HD 代理等叠加项

            # 新增：应用最终类权重（更关注 TC/ET）
            class_loss = class_loss * self.class_weights[c].to(class_loss.device)

            # 检查稳定性
            is_stable, msg = self.check_loss_stability(class_loss, c)

            if is_stable:
                class_losses.append(class_loss)
                valid_samples += 1
            else:
                print(f"Warning: Class {c} loss unstable ({msg}), applying fallback")
                ce_fallback = self.stable_ce_loss(pred_c, target_c)
                fallback_loss = (
                    dice_w * dice_loss +
                    ce_w * ce_fallback
                )
                fallback_loss = fallback_loss * self.class_weights[c].to(class_loss.device)
                class_losses.append(fallback_loss)
                valid_samples += 1

            # 收集用于GDL/ACW的概率与标签
            with torch.no_grad():
                per_class_probs.append(torch.sigmoid(pred_c).clamp(self.eps, 1 - self.eps))
                per_class_targets.append(target_c.float())

        # 计算总损失
        if valid_samples > 0:
            total_loss = torch.stack(class_losses).mean()

            if num_classes >= 3:
                prob_all = torch.sigmoid(pred).clamp(self.eps, 1 - self.eps)
                p_wt = prob_all[:, 0]
                p_tc = prob_all[:, 1]
                p_et = prob_all[:, 2]
                hier_violation = F.relu(p_et - p_tc) + F.relu(p_tc - p_wt)
                L_hier = hier_violation.mean()
                step = int(self.loss_idx.item())
                warmup_steps = 2000
                peak_steps = 6000
                cooldown_steps = 4000
                w_start, w_peak, w_end = 0.15, 0.25, 0.18
                if step <= warmup_steps:
                    t = step / max(1, warmup_steps)
                    w_hier = (1.0 - t) * w_start + t * w_peak
                elif step <= peak_steps:
                    w_hier = w_peak
                else:
                    t = min((step - peak_steps) / max(1, cooldown_steps), 1.0)
                    w_hier = (1.0 - t) * w_peak + t * w_end
                total_loss = total_loss + w_hier * L_hier

            # 移除 GDL/ACB 叠加

            # 移除 EdgeAlign 逻辑

            # 最终稳定性检查
            total_loss = torch.clamp(total_loss, 0.0, self.max_loss_value)

            # 更新损失历史
            self.update_loss_history(total_loss)

            # 自适应跳过过大损失batch
            adaptive_threshold = self.get_adaptive_threshold()
            if total_loss > adaptive_threshold and total_loss > 8.0:
                print(f"⚠️ 损失值过大: {total_loss:.4f}，跳过此batch")
                self.skip_batches += 1
                return None
            elif total_loss > adaptive_threshold:
                print(f"📊 损失值较高: {total_loss:.4f}，但继续训练")

            return total_loss
        else:
            print("Warning: All classes unstable, using fallback loss")
            self.skip_batches += 1
            return None


def create_brats_stable_training_components(model,
                                            base_lr: float = 3e-4,
                                            enable_dynamic_weights: bool = True,
                                            training_stage: str = 'full',
                                            class_alpha = [2.7, 3.2, 3.8],
                                            class_weights = None,
                                            enable_focal_tversky: bool = True,
                                            ft_gamma: float = 1.33,
                                            enable_boundary_weighted_dice: bool = True,
                                            enable_boundary_tversky: bool = False,
                                            enable_progressive_edge_supervision: bool = False,
                                            disable_ce_loss: bool = False,
                                            disable_focal_loss: bool = False):
    """创建BraTS稳定训练组件（支持动态权重/边界损失/训练阶段设置/类别权重)
    
    新增参数：
        enable_topology_loss: 是否启用方案D的拓扑和边界损失
        enable_boundary_weighted_dice: 是否启用边界加权 Dice Loss（推荐）
        disable_ce_loss: 是否禁用CE Loss（损失函数消融）
    """

    # 处理类别权重（顺序：[WT, TC, ET]）
    if class_alpha is not None:
        try:
            assert len(class_alpha) == 3
            alpha_list = [float(class_alpha[0]), float(class_alpha[1]), float(class_alpha[2])]
        except Exception:
            alpha_list = [2.7, 3.2, 3.8]
    else:
        alpha_list = [2.7, 3.2, 3.8]

    # 选择损失函数（固定为稳定主损失）
    criterion = BraTSStableLoss(
        alpha=class_alpha,
        smooth=1e-3,
        enable_dynamic_weights=enable_dynamic_weights,
        training_stage=training_stage,
        class_weights=class_weights,
        enable_focal_tversky=enable_focal_tversky,
        ft_gamma=ft_gamma,
        disable_ce_loss=disable_ce_loss,
        disable_focal_loss=disable_focal_loss,
    )
    
    # 方案D: 拓扑和边界损失（惰性导入，避免在未启用时增加依赖和开销）
    topology_criterion = None
    
    # 边界加权 Dice Loss（推荐启用）
    boundary_weighted_dice_criterion = None
    boundary_tversky_criterion = None
    if enable_boundary_weighted_dice:
        boundary_weighted_dice_criterion = BoundaryWeightedDiceLoss(
            class_weights=[1.0, 1.7, 2.5],
            smooth=1e-5
        )
        print("✅ 边界加权 Dice Loss 已启用 (class_weights=[1.0, 1.7, 2.5])")
    if enable_boundary_tversky:
        boundary_tversky_criterion = BoundaryAwareTverskyLoss(
            boundary_alpha=0.3, boundary_beta=0.7,
            non_boundary_alpha=0.5, non_boundary_beta=0.5
        )
        print("✅ 边界感知 Tversky 已启用")

    # 稳定化优化器
    optimizer = BraTSStableOptimizer(model, base_lr)

    # 保守的GradScaler设置
    scaler = torch.cuda.amp.GradScaler(
        init_scale=2**6,
        growth_factor=1.2,
        backoff_factor=0.5,
        growth_interval=2000
    )

    # 优化学习率调度器：温和下降，防止过拟合
    # 训练 50 epoch，前 5 epoch 预热，后 40 epoch 余弦退火，最后 5 epoch 保持最小学习率
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer.optimizer,
        T_max=40,  # 30 → 40（温和的学习率下降，防止过拟合）
        eta_min=base_lr * 0.05  # 1% → 5%（保持更高的最小学习率）
    )

    progressive_edge_criterion = ProgressiveEdgeSupervision() if enable_progressive_edge_supervision else None
    return criterion, optimizer, scaler, scheduler, topology_criterion, boundary_weighted_dice_criterion, boundary_tversky_criterion, progressive_edge_criterion


class BraTSStableOptimizer:
    def __init__(self, model: nn.Module, base_lr: float = 3e-4):
        base_lr = float(base_lr)
        root = model.module if hasattr(model, 'module') else model
        pe_params = []
        if hasattr(root, 'cmee') and getattr(root, 'cmee', None) is not None:
            pe_params = [p for p in root.cmee.parameters() if p.requires_grad]
        pe_ids = set(id(p) for p in pe_params)

        blend_params = []
        for _name in (
            'mbam_blend',
            'mbam_skip_blend',
            'mbam_dec3_blend',
            'mbam_dec2_blend',
            'mbam_dec1_blend',
            'cmee_fuse_alpha',
            'basf_x0_alpha',
        ):
            p = getattr(root, _name, None)
            if isinstance(p, torch.nn.Parameter) and p.requires_grad:
                blend_params.append(p)
        blend_ids = set(id(p) for p in blend_params)

        main_params = [p for p in root.parameters() if p.requires_grad and id(p) not in pe_ids and id(p) not in blend_ids]

        param_groups = []
        if len(main_params) > 0:
            param_groups.append({'params': main_params, 'lr': base_lr})
        if len(pe_params) > 0:
            param_groups.append({'params': pe_params, 'lr': base_lr * 0.5})
        if len(blend_params) > 0:
            param_groups.append({'params': blend_params, 'lr': base_lr, 'weight_decay': 0.0})

        self.optimizer = torch.optim.AdamW(
            param_groups,
            lr=base_lr,
            betas=(0.9, 0.999),
            weight_decay=1e-4,
        )

    def zero_grad(self):
        self.optimizer.zero_grad(set_to_none=True)

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict):
        return self.optimizer.load_state_dict(state_dict)

    @property
    def param_groups(self):
        return self.optimizer.param_groups

    def __getattr__(self, name: str):
        # fallback to underlying torch optimizer attributes when needed
        if name in ('optimizer',):
            return super().__getattribute__(name)
        return getattr(self.optimizer, name)


def stable_training_step(model, criterion, optimizer, scaler, images, targets, device, edges=None,
                         aux_criterion: nn.Module = None, epoch: int = None,
                         boundary_weighted_dice_criterion=None,
                         deep_supervision_weight: float = 0.0,
                         cmee_edge_loss_weight: float = 0.0):
    """稳定化训练步骤（主损失 + 边界加权Dice）
    
    消融实验结论：
        - 保留: 主损失(Dice+CE+Focal+Tversky) + 边界加权Dice (+0.10%)
        - 移除: 多尺度监督(-0.02%), 边界监督(-0.16%), 边界 Tversky
    """

    # 数据预处理
    images = images.to(device, non_blocking=True)
    targets = targets.to(device, non_blocking=True)
    edges = edges.to(device, non_blocking=True) if edges is not None else None
    # 输入稳定化
    images = torch.clamp(images, -5, 5)

    # 清零梯度
    optimizer.zero_grad()

    def _sanitize_bn_stats_(_model: nn.Module):
        for m in _model.modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                if getattr(m, 'running_mean', None) is not None:
                    m.running_mean.data = torch.nan_to_num(m.running_mean.data, nan=0.0, posinf=0.0, neginf=0.0)
                if getattr(m, 'running_var', None) is not None:
                    m.running_var.data = torch.nan_to_num(m.running_var.data, nan=1.0, posinf=1.0, neginf=1.0).clamp_min_(1e-6)

    # 前向传播（加入浅层监督）
    with autocast():
        _force_fp32 = bool(getattr(model, 'force_fp32_forward', False))
        if epoch is not None and hasattr(model, 'set_epoch'):
            try:
                model.set_epoch(int(epoch))
            except Exception:
                pass
        if edges is not None and bool(getattr(model, 'supports_external_edge', False)):
            if _force_fp32:
                with autocast(enabled=False):
                    outputs = model(images, edges)
            else:
                outputs = model(images, edges)
        else:
            if _force_fp32:
                with autocast(enabled=False):
                    outputs = model(images)
            else:
                outputs = model(images)

        outputs_for_loss = outputs
        if isinstance(outputs, tuple) and len(outputs) == 4:
            outputs_for_loss = outputs[2]
            if outputs_for_loss.shape[-2:] != targets.shape[-2:]:
                outputs_for_loss = F.interpolate(outputs_for_loss, size=targets.shape[-2:], mode='bilinear', align_corners=False)
        elif isinstance(outputs, dict):
            key = 'logits' if 'logits' in outputs else ('stage4' if 'stage4' in outputs else None)
            if key is not None and isinstance(outputs.get(key, None), torch.Tensor):
                main = outputs[key]
                if main.shape[-2:] != targets.shape[-2:]:
                    main = F.interpolate(main, size=targets.shape[-2:], mode='bilinear', align_corners=False)
                outputs_for_loss = dict(outputs)
                outputs_for_loss[key] = main
            else:
                outputs_for_loss = outputs
        elif isinstance(outputs, torch.Tensor):
            if outputs.shape[-2:] != targets.shape[-2:]:
                outputs_for_loss = F.interpolate(outputs, size=targets.shape[-2:], mode='bilinear', align_corners=False)
            else:
                outputs_for_loss = outputs

        # 主损失（Dice + CE + Focal + Tversky）
        loss_main = criterion(outputs_for_loss, targets, edges=edges)

        if loss_main is None or torch.isnan(loss_main) or torch.isinf(loss_main) or loss_main > 20.0:
            print(f"跳过异常主损失: {float(loss_main) if loss_main is not None else 'None'}")
            try:
                def _stats(name: str, t: torch.Tensor):
                    if t is None:
                        print(f"  - {name}: None")
                        return
                    if not isinstance(t, torch.Tensor):
                        print(f"  - {name}: type={type(t)}")
                        return
                    finite = torch.isfinite(t)
                    finite_ratio = float(finite.float().mean().detach().cpu()) if t.numel() > 0 else 1.0
                    t_det = t.detach()
                    t_min = float(torch.nan_to_num(t_det, nan=0.0, posinf=0.0, neginf=0.0).min().cpu()) if t_det.numel() > 0 else 0.0
                    t_max = float(torch.nan_to_num(t_det, nan=0.0, posinf=0.0, neginf=0.0).max().cpu()) if t_det.numel() > 0 else 0.0
                    print(f"  - {name}: shape={tuple(t.shape)}, dtype={t.dtype}, device={t.device}, finite={finite_ratio:.4f}, min={t_min:.4f}, max={t_max:.4f}")

                _stats('images', images)
                _stats('targets', targets)
                _stats('edges', edges)

                logits_dbg = None
                if isinstance(outputs_for_loss, torch.Tensor):
                    logits_dbg = outputs_for_loss
                elif isinstance(outputs_for_loss, dict) and 'logits' in outputs_for_loss:
                    logits_dbg = outputs_for_loss['logits']
                elif isinstance(outputs_for_loss, dict) and 'stage4' in outputs_for_loss:
                    logits_dbg = outputs_for_loss['stage4']
                elif isinstance(outputs_for_loss, (tuple, list)) and len(outputs_for_loss) > 0 and isinstance(outputs_for_loss[0], torch.Tensor):
                    logits_dbg = outputs_for_loss[0]
                _stats('logits', logits_dbg)
            except Exception as _e:
                print(f"  (诊断信息打印失败: {_e})")
            try:
                _sanitize_bn_stats_(model)
            except Exception:
                pass
            optimizer.zero_grad()
            return None, None

        loss = loss_main

        if float(deep_supervision_weight) > 0.0 and isinstance(outputs, dict):
            try:
                w_ds = float(deep_supervision_weight)
                if epoch is not None:
                    warmup_ep = 6
                    t = min(max(int(epoch) / max(1, warmup_ep), 0.0), 1.0)
                    w_ds = w_ds * t

                if w_ds > 0.0:
                    cw = None
                    if hasattr(criterion, 'class_weights'):
                        cw = getattr(criterion, 'class_weights', None)
                        if isinstance(cw, torch.Tensor) and cw.numel() > 0:
                            cw = cw.to(device=targets.device, dtype=targets.dtype)
                            if cw.numel() != targets.shape[1]:
                                cw = None

                    eps = float(getattr(criterion, 'eps', 1e-6))
                    smooth = float(getattr(criterion, 'smooth', 1e-3))

                    def _aux_bce_dice(_logits: torch.Tensor) -> torch.Tensor:
                        if _logits.shape[-2:] != targets.shape[-2:]:
                            _logits = F.interpolate(_logits, size=targets.shape[-2:], mode='bilinear', align_corners=False)
                        _tgt = targets.float()
                        bce = F.binary_cross_entropy_with_logits(_logits, _tgt, reduction='none')
                        if cw is not None:
                            bce = (bce * cw.view(1, -1, 1, 1)).mean()
                        else:
                            bce = bce.mean()

                        prob = torch.sigmoid(_logits).clamp(eps, 1.0 - eps)
                        inter = (prob * _tgt).sum(dim=(-2, -1))
                        denom = prob.sum(dim=(-2, -1)) + _tgt.sum(dim=(-2, -1))
                        dice = (2.0 * inter + smooth) / (denom + smooth)
                        dice_loss = 1.0 - dice
                        if cw is not None:
                            dice_loss = (dice_loss * cw.view(1, -1)).mean()
                        else:
                            dice_loss = dice_loss.mean()
                        return 0.5 * bce + 0.5 * dice_loss

                    loss_ds = None
                    if 'stage1' in outputs and isinstance(outputs['stage1'], torch.Tensor):
                        loss_ds = _aux_bce_dice(outputs['stage1']) * 1.0
                    if 'stage2' in outputs and isinstance(outputs['stage2'], torch.Tensor):
                        l2 = _aux_bce_dice(outputs['stage2']) * 0.5
                        loss_ds = l2 if loss_ds is None else (loss_ds + l2)
                    if 'stage3' in outputs and isinstance(outputs['stage3'], torch.Tensor):
                        l3 = _aux_bce_dice(outputs['stage3']) * 0.25
                        loss_ds = l3 if loss_ds is None else (loss_ds + l3)

                    if loss_ds is not None and (not torch.isnan(loss_ds)) and (not torch.isinf(loss_ds)):
                        loss = loss + w_ds * loss_ds
            except Exception as e:
                print(f"⚠️  deep supervision loss 计算失败: {e}")
                pass
        
        # 边界加权 Dice Loss（消融实验证明最有效: +0.10%）
        if boundary_weighted_dice_criterion is not None and edges is not None:
            try:
                # 获取主输出的sigmoid概率
                main_logits = None
                if isinstance(outputs_for_loss, dict):
                    if 'logits' in outputs_for_loss:
                        main_logits = outputs_for_loss['logits']
                    elif 'stage4' in outputs_for_loss:
                        main_logits = outputs_for_loss['stage4']
                if main_logits is None and isinstance(outputs_for_loss, torch.Tensor):
                    main_logits = outputs_for_loss
                if main_logits is not None:
                    # 避免 logit 过大导致数值问题
                    main_logits = torch.nan_to_num(main_logits, nan=0.0, posinf=10.0, neginf=-10.0)
                    main_logits = torch.clamp(main_logits, -10.0, 10.0)
                    boundary_weighted_dice_loss = boundary_weighted_dice_criterion(main_logits, targets, edges)
                    if boundary_weighted_dice_loss is not None and torch.isfinite(boundary_weighted_dice_loss).item():
                        loss = loss + boundary_weighted_dice_loss
            except Exception as e:
                print(f"⚠️  边界加权 Dice Loss 计算失败: {e}")
                pass

        if float(cmee_edge_loss_weight) > 0.0 and edges is not None and isinstance(outputs, dict) and ('cmee_edge_logits' in outputs):
            try:
                cmee_logits = outputs['cmee_edge_logits']
                cmee_loss = mbam_boundary_loss(cmee_logits, edges, class_weights=[1.0, 1.7, 2.5])
                if not torch.isnan(cmee_loss) and not torch.isinf(cmee_loss):
                    w = float(cmee_edge_loss_weight)
                    if epoch is not None:
                        warmup_ep = 6
                        t = min(max(int(epoch) / max(1, warmup_ep), 0.0), 1.0)
                        w = w * t
                    if w > 0.0:
                        loss = loss + w * cmee_loss
            except Exception as e:
                print(f"⚠️  CMEE 边界监督损失计算失败: {e}")
                pass

        # 🆕 MultiScaleEFM 边界预测监督
        if edges is not None and isinstance(outputs, dict) and ('edge_pred' in outputs):
            try:
                edge_pred_logits = outputs['edge_pred']
                # 调整尺寸
                if edge_pred_logits.shape[-2:] != edges.shape[-2:]:
                    edge_pred_logits = F.interpolate(edge_pred_logits, size=edges.shape[-2:], mode='bilinear', align_corners=False)
                # BCE 损失
                edge_pred_loss = F.binary_cross_entropy_with_logits(edge_pred_logits, edges, reduction='mean')
                if torch.isfinite(edge_pred_loss).item():
                    # 边界预测权重：渐进式启用
                    w = 0.1
                    if epoch is not None:
                        warmup_ep = 3
                        t = min(max(int(epoch) / max(1, warmup_ep), 0.0), 1.0)
                        w = w * t
                    if w > 0.0:
                        loss = loss + w * edge_pred_loss
            except Exception as e:
                print(f"⚠️  MultiScaleEFM 边界预测损失计算失败: {e}")
                pass

        # 🆕 显式 MBAM 边界监督：使用 MBAM 的三通道边界 logits 与 GT 边界对齐
        if edges is not None and isinstance(outputs, dict) and ('mbam_edge_logits' in outputs):
            try:
                edge_logits = outputs['mbam_edge_logits']
                mbam_loss = mbam_boundary_loss(edge_logits, edges, class_weights=[1.0, 1.7, 2.5])

                if not torch.isnan(mbam_loss) and not torch.isinf(mbam_loss):
                    # 边界监督权重：早期较强，后期适度衰减
                    if epoch is None:
                        ew = 0.20
                    else:
                        warmup_ep = 6
                        peak_ep = 20
                        cooldown_span = 20
                        w_start, w_peak, w_end = 0.18, 0.30, 0.18
                        e = int(epoch)
                        if e <= warmup_ep:
                            t = e / max(1, warmup_ep)
                            ew = (1.0 - t) * w_start + t * w_peak
                        elif e <= peak_ep:
                            ew = w_peak
                        else:
                            t = min((e - peak_ep) / max(1, cooldown_span), 1.0)
                            ew = (1.0 - t) * w_peak + t * w_end

                    loss = loss + ew * mbam_loss
            except Exception as e:
                print(f"⚠️  MBAM 边界监督损失计算失败: {e}")
                pass

        # 🆕 肿瘤前景监督：使用 fg_logits 与 WT/TC/ET 并集的前景掩码对齐
        if isinstance(outputs, dict) and 'fg_logits' in outputs:
            try:
                fg_logits = outputs['fg_logits']  # [B,1,H,W]

                # 构建前景 GT：WT/TC/ET 的并集
                fg_targets = targets
                if fg_targets.dim() == 4:
                    # [B,C,H,W]，当 C>=3 时按通道求并集；C==1 时直接使用
                    if fg_targets.shape[1] > 1:
                        fg_bin = (fg_targets > 0.5).float()
                        fg_targets = (fg_bin.sum(dim=1, keepdim=True) > 0.5).float()  # [B,1,H,W]
                    else:
                        fg_targets = (fg_targets > 0.5).float()
                elif fg_targets.dim() == 3:
                    fg_targets = (fg_targets > 0.5).float().unsqueeze(1)

                fg_loss = tumor_foreground_loss(fg_logits, fg_targets)

                if not torch.isnan(fg_loss) and not torch.isinf(fg_loss):
                    # 前景监督权重：比主损失弱，随 epoch 温和调度
                    if epoch is None:
                        fw = 0.15
                    else:
                        warmup_ep = 6
                        peak_ep = 20
                        cooldown_span = 20
                        w_start, w_peak, w_end = 0.12, 0.25, 0.15
                        e = int(epoch)
                        if e <= warmup_ep:
                            t = e / max(1, warmup_ep)
                            fw = (1.0 - t) * w_start + t * w_peak
                        elif e <= peak_ep:
                            fw = w_peak
                        else:
                            t = min((e - peak_ep) / max(1, cooldown_span), 1.0)
                            fw = (1.0 - t) * w_peak + t * w_end

                    loss = loss + fw * fg_loss
            except Exception as e:
                print(f"⚠️  Tumor foreground loss 计算失败: {e}")
                pass

    # 反向传播与优化（防止单批次异常中断）
    scaled_loss = scaler.scale(loss)
    try:
        scaled_loss.backward()
    except Exception as e:
        print(f"Backward异常，跳过此batch: {e}")
        optimizer.zero_grad()
        return None, None

    # 梯度处理
    scaler.unscale_(optimizer.optimizer)

    # 更温和的梯度裁剪
    try:
        root = model.module if hasattr(model, 'module') else model
        pe_params = []
        if hasattr(root, 'cmee') and getattr(root, 'cmee', None) is not None:
            pe_params = [p for p in root.cmee.parameters() if p.requires_grad]
        pe_ids = set(id(p) for p in pe_params)
        eam_params = []
        if hasattr(root, 'eam') and getattr(root, 'eam', None) is not None:
            eam_params = [p for p in root.eam.parameters() if p.requires_grad]
        eam_ids = set(id(p) for p in eam_params)
        main_params = [p for p in root.parameters() if p.requires_grad and id(p) not in pe_ids and id(p) not in eam_ids]
        if len(main_params) > 0:
            torch.nn.utils.clip_grad_norm_(main_params, max_norm=1.0)
        if len(eam_params) > 0:
            torch.nn.utils.clip_grad_norm_(eam_params, max_norm=0.5)
        if len(pe_params) > 0:
            torch.nn.utils.clip_grad_norm_(pe_params, max_norm=0.5)
    except Exception:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    # 优化步骤
    scaler.step(optimizer.optimizer)
    scaler.update()

    return loss.item(), outputs
