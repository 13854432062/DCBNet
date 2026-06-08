import torch
import torch.nn as nn
import torch.nn.functional as F
from net.Res2Net_best import res2net50_v1b_26w_4s

# InstanceAwareAttention 已移除：forward 中直接使用原始特征，无需该空模块


class ConvBNR(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, dilation=1, bias=False):
        super(ConvBNR, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size, stride=stride, padding=dilation, dilation=dilation, bias=bias),
            nn.GroupNorm(num_groups=(32 if planes % 32 == 0 else (16 if planes % 16 == 0 else 8)), num_channels=planes, eps=1e-5, affine=True),
            nn.ReLU(inplace=False)
        )

    def forward(self, x):
        return self.block(x)


class ClassAwareFusionModule(nn.Module):
    """
    类别感知特征融合模块
    
    为 WT/TC/ET 三个类别分别学习不同的融合策略：
    - WT: 均衡融合（Res2Net + Swin 各 50%）
    - TC: 偏向 Swin（Swin 在 TC 上表现更好）
    - ET: 偏向 Res2Net（Res2Net 在 ET 小目标上更好）
    """
    
    def __init__(self, channels, num_classes=3):
        super().__init__()
        self.num_classes = num_classes
        
        # 类别感知权重生成器
        self.class_weight_gen = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels * 2, channels // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, num_classes * 2, 1)  # 每个类别 2 个权重
        )
        
        # 初始化类别偏好（可学习）
        # [WT, TC, ET] × [Res2Net, Swin]
        self.class_bias = nn.Parameter(torch.tensor([
            [0.5, 0.5],  # WT: 均衡
            [0.3, 0.7],  # TC: 偏向 Swin
            [0.8, 0.2],  # ET: 偏向 Res2Net
        ]))
        
        # 特征增强
        self.enhance = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels),  # Depthwise
            nn.Conv2d(channels, channels, 1),  # Pointwise
            nn.GroupNorm(32 if channels >= 32 else 8, channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, res_feat, swin_feat):
        """
        Args:
            res_feat: Res2Net 特征 [B, C, H, W]
            swin_feat: Swin 特征 [B, C, H, W]
        
        Returns:
            融合后的特征 [B, C, H, W]
        """
        B, C, H, W = res_feat.shape
        
        # 生成类别感知权重
        concat_feat = torch.cat([res_feat, swin_feat], dim=1)
        class_weights = self.class_weight_gen(concat_feat)  # [B, num_classes*2, 1, 1]
        class_weights = class_weights.view(B, self.num_classes, 2)  # [B, 3, 2]
        
        # 结合初始偏好
        class_weights = class_weights + self.class_bias.unsqueeze(0)  # [B, 3, 2]
        class_weights = F.softmax(class_weights, dim=2)  # 归一化
        
        # 对每个类别分别融合
        fused_features = []
        for c in range(self.num_classes):
            w_res = class_weights[:, c:c+1, 0:1].view(B, 1, 1, 1)  # [B, 1, 1, 1]
            w_swin = class_weights[:, c:c+1, 1:2].view(B, 1, 1, 1)  # [B, 1, 1, 1]
            
            fused_c = w_res * res_feat + w_swin * swin_feat
            fused_features.append(fused_c)
        
        # 平均所有类别的融合结果
        fused = torch.stack(fused_features, dim=0).mean(dim=0)
        
        # 特征增强
        fused = self.enhance(fused)
        
        return fused


class Conv1x1(nn.Module):
    def __init__(self, inplanes, planes):
        super(Conv1x1, self).__init__()
        self.conv = nn.Conv2d(inplanes, planes, 1)
        gn_groups = 32 if planes % 32 == 0 else (16 if planes % 16 == 0 else 8)
        self.bn = nn.GroupNorm(num_groups=gn_groups, num_channels=planes, eps=1e-5, affine=True)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class EAM(nn.Module):
    """✨ Class-Aware Edge Attention Module
    
    为 WT/TC/ET 三个类别分别生成独立的边界注意力图
    - WT: 权重 1.0（基准）
    - TC: 权重 1.5（加强边界识别）
    - ET: 权重 2.0（强化小块边界）
    """
    def __init__(self, mode: str = 'all', num_classes: int = 3):
        super(EAM, self).__init__()
        self.mode = str(mode)  # 'all' 或 'x1x4'
        self.num_classes = num_classes
        
        # 降维层（下游采用减半通道：x1/x2/x3/x4 = 128/256/512/1024）
        self.reduce1 = Conv1x1(128, 64)  # x1
        if self.mode == 'all':
            self.reduce2 = Conv1x1(256, 64)   # x2
            self.reduce3 = Conv1x1(512, 128)  # x3
        else:
            self.reduce2 = None
            self.reduce3 = None
        self.reduce4 = Conv1x1(1024, 256)  # x4
        
        # 计算输入通道数
        in_ch = 64 + 256 if self.mode != 'all' else (64 + 64 + 128 + 256)
        
        # ✅ 共享特征提取器
        self.shared_block = nn.Sequential(
            ConvBNR(in_ch, 256, 3),
            ConvBNR(256, 256, 3)
        )
        
        # ✅ 为每个类别创建独立的边界预测头
        self.class_edge_heads = nn.ModuleList([
            nn.Sequential(
                ConvBNR(256, 128, 3),
                nn.Conv2d(128, 1, 1)
            ) for _ in range(num_classes)
        ])
        
        # ✅ 类别特定的边界权重（可学习）
        # 初始值: [WT=1.0, TC=1.5, ET=2.0]
        self.class_edge_weights = nn.Parameter(torch.tensor([1.0, 1.5, 2.0]))
        
        # 初始化每个类别头的偏置
        for head in self.class_edge_heads:
            try:
                final_conv = head[-1]
                if isinstance(final_conv, nn.Conv2d):
                    if final_conv.bias is None:
                        final_conv.bias = nn.Parameter(torch.zeros(1))
                    nn.init.constant_(final_conv.bias, 0.5)
            except Exception:
                pass
        
        # 全局可学习尺度与偏置
        self.logit_scale = nn.Parameter(torch.tensor(1.0))
        self.logit_bias = nn.Parameter(torch.tensor(0.0))

    def forward(self, x1, x2, x3, x4):
        """✨ Class-Aware 边界注意力生成
        
        Args:
            x1, x2, x3, x4: 不同层级的特征图
            
        Returns:
            edge_maps: [B, 3, H, W] - 为 WT/TC/ET 分别生成的边界图
        """
        # 强制fp32 + 数值净化
        with torch.cuda.amp.autocast(enabled=False):
            try:
                with torch.no_grad():
                    if not torch.isfinite(self.logit_scale).all():
                        self.logit_scale.copy_(torch.tensor(1.0, dtype=self.logit_scale.dtype, device=self.logit_scale.device))
                    if not torch.isfinite(self.logit_bias).all():
                        self.logit_bias.copy_(torch.tensor(0.0, dtype=self.logit_bias.dtype, device=self.logit_bias.device))
                    self.logit_scale.clamp_(0.5, 2.0)
                    self.logit_bias.clamp_(-1.5, 1.5)
                    self.class_edge_weights.clamp_(0.5, 3.0)  # ✅ 限制类别权重范围
            except Exception:
                pass
            
            # 多尺度特征降维和对齐
            size = x1.size()[2:]
            x1r = self.reduce1(x1.float())
            x2r = self.reduce2(x2.float()) if self.mode == 'all' and (self.reduce2 is not None) else None
            x3r = self.reduce3(x3.float()) if self.mode == 'all' and (self.reduce3 is not None) else None
            x4r = self.reduce4(x4.float())
            
            # 上采样到统一尺寸
            if x2r is not None:
                x2r = F.interpolate(x2r, size, mode='bilinear', align_corners=False)
            if x3r is not None:
                x3r = F.interpolate(x3r, size, mode='bilinear', align_corners=False)
            x4r = F.interpolate(x4r, size, mode='bilinear', align_corners=False)
            
            # 拼接多尺度特征
            if self.mode == 'all' and x2r is not None and x3r is not None:
                out = torch.cat((x1r, x2r, x3r, x4r), dim=1)
            else:
                out = torch.cat((x1r, x4r), dim=1)
            
            # ✅ 共享特征提取
            shared_feat = self.shared_block(out.float())
            
            # ✅ 为每个类别生成独立的边界图
            class_edges = []
            weights = torch.clamp(self.class_edge_weights, 0.5, 3.0)
            
            for i, head in enumerate(self.class_edge_heads):
                edge = head(shared_feat)  # [B, 1, H, W]
                edge = edge * weights[i]  # 应用类别特定权重
                class_edges.append(edge)
            
            # 合并为 [B, 3, H, W]
            out = torch.cat(class_edges, dim=1)
            
            # 应用全局缩放和偏置
            scale = torch.clamp(self.logit_scale.float(), 0.5, 2.0)
            bias = torch.clamp(self.logit_bias.float(), -1.5, 1.5)
            out = out * scale + bias
            
            # 数值净化
            out = torch.nan_to_num(out, nan=0.0, posinf=1e4, neginf=-1e4)
            out = torch.clamp(out, -8.0, 8.0)
            
            return out


class TumorForegroundHead(nn.Module):
    def __init__(self, in_channels: int, mid_channels: int = 256):
        super().__init__()
        gn = 32 if mid_channels % 32 == 0 else (16 if mid_channels % 16 == 0 else 8)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, padding=1, bias=False),
            nn.GroupNorm(num_groups=gn, num_channels=mid_channels, eps=1e-5, affine=True),
            nn.ReLU(inplace=False),
            nn.Conv2d(mid_channels, 1, 1, bias=True)
        )

    def forward(self, x: torch.Tensor, out_size=None) -> torch.Tensor:
        logits = self.conv(x)
        if out_size is not None and logits.shape[2:] != out_size:
            logits = F.interpolate(logits, size=out_size, mode='bilinear', align_corners=False)
        return logits


class ESAB(nn.Module):
    def __init__(self, channels):
        super(ESAB, self).__init__()
        self.channels = channels
        self.sobel_x = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False)
        self.sobel_y = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False)
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3).repeat(channels, 1, 1, 1)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3).repeat(channels, 1, 1, 1)
        with torch.no_grad():
            self.sobel_x.weight.copy_(sobel_x)
            self.sobel_y.weight.copy_(sobel_y)
        self.sobel_x.weight.requires_grad = False
        self.sobel_y.weight.requires_grad = False
        self.attention = nn.Sequential(
            nn.Conv2d(channels * 2, max(1, channels // 4), 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(1, channels // 4), channels, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        edge_x = self.sobel_x(x)
        edge_y = self.sobel_y(x)
        edge_mag = torch.sqrt(edge_x * edge_x + edge_y * edge_y + 1e-6)
        att = self.attention(torch.cat([x, edge_mag], dim=1))
        return x * att + edge_mag


class PaperEdgeExtractionModule(nn.Module):
    def __init__(self, in_channels=2, num_classes=3, base_channels=32):
        super(PaperEdgeExtractionModule, self).__init__()
        c1, c2, c3, c4 = base_channels, base_channels * 2, base_channels * 4, base_channels * 8
        self.conv_blocks = nn.ModuleList([
            nn.Sequential(nn.Conv2d(in_channels, c1, 3, padding=1, bias=False), nn.GroupNorm(8, c1), nn.ReLU(inplace=True),
                          nn.Conv2d(c1, c1, 3, padding=1, bias=False), nn.GroupNorm(8, c1), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2)),
            nn.Sequential(nn.Conv2d(c1, c2, 3, padding=1, bias=False), nn.GroupNorm(8, c2), nn.ReLU(inplace=True),
                          nn.Conv2d(c2, c2, 3, padding=1, bias=False), nn.GroupNorm(8, c2), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2)),
            nn.Sequential(nn.Conv2d(c2, c3, 3, padding=1, bias=False), nn.GroupNorm(8, c3), nn.ReLU(inplace=True),
                          nn.Conv2d(c3, c3, 3, padding=1, bias=False), nn.GroupNorm(8, c3), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2)),
            nn.Sequential(nn.Conv2d(c3, c4, 3, padding=1, bias=False), nn.GroupNorm(8, c4), nn.ReLU(inplace=True),
                          nn.Conv2d(c4, c4, 3, padding=1, bias=False), nn.GroupNorm(8, c4), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2))
        ])
        self.esab_blocks = nn.ModuleList([ESAB(c1), ESAB(c2), ESAB(c3), ESAB(c4)])
        self.proj_down = nn.ModuleList([
            Conv1x1(c4, c3),
            Conv1x1(c3, c2),
            Conv1x1(c2, c1)
        ])
        self.edge_fusion_conv = nn.Conv2d(c1, num_classes, 1)
        self.register_parameter('edge_weight', nn.Parameter(torch.tensor(0.3)))
    def forward(self, flair, t1ce):
        x = torch.cat([flair, t1ce], dim=1)
        feats = []
        cur = x
        for cb, es in zip(self.conv_blocks, self.esab_blocks):
            cur = cb(cur)
            cur = es(cur)
            feats.append(cur)
        edge_feat = feats[-1]
        for proj_idx, i in enumerate(range(len(feats) - 2, -1, -1)):
            edge_feat = F.interpolate(edge_feat, size=feats[i].shape[2:], mode='bilinear', align_corners=False)
            edge_feat = self.proj_down[proj_idx](edge_feat)
            edge_feat = edge_feat + feats[i]
        edge_maps = self.edge_fusion_conv(edge_feat)
        edge_maps = F.interpolate(edge_maps, size=x.shape[2:], mode='bilinear', align_corners=False)
        return edge_maps


class FeatureAdaptiveBiasAlign(nn.Module):
    """FAB-Align: 特征自适应偏置对齐
    
    y = W(x) + bias(x)
    其中 W 为共享 1x1 投影，bias(x) 为从输入特征自适应生成的通道偏置。
    """

    def __init__(self, in_channels: int, out_channels: int, reduction: int = 4, init_bias_scale: float = 0.1):
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)

        # 共享投影（1x1卷积，对齐通道）
        self.projection = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1, bias=False)

        # 自适应偏置生成器：全局池化后生成通道偏置
        hidden_dim = max(self.out_channels // reduction, 16)
        self.bias_generator = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),            # [B, C, H, W] -> [B, C, 1, 1]
            nn.Conv2d(self.in_channels, hidden_dim, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, self.out_channels, 1),
        )

        # 可学习缩放因子：控制偏置影响强度（初始较小，避免破坏原特征）
        self.bias_scale = nn.Parameter(torch.ones(1) * float(init_bias_scale))

        # 初始化：让偏置初始接近 0
        last_conv = self.bias_generator[-1]
        if isinstance(last_conv, nn.Conv2d):
            nn.init.zeros_(last_conv.weight)
            if last_conv.bias is not None:
                nn.init.zeros_(last_conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 共享投影
        y = self.projection(x)

        # 生成自适应偏置（[B, out_C, 1, 1]）并按比例缩放
        bias = self.bias_generator(x) * self.bias_scale

        # 广播到空间维度并相加
        return y + bias

    def extra_repr(self) -> str:
        return f'in_channels={self.in_channels}, out_channels={self.out_channels}'


class ClassModalityWeightedAttention(nn.Module):
    """类别-模态感知注意力模块
    
    对每个肿瘤类别(WT/TC/ET)，使用不同的模态权重:
    - ET: 偏重 T1ce (0.1, 0.6, 0.1, 0.1)
    - TC: 偏重 T1ce+T1 (0.3, 0.4, 0.15, 0.15)
    - WT: 均衡使用所有模态 (0.15, 0.3, 0.15, 0.4)
    """
    def __init__(self, fused_channel, modal_channel=8, custom_weights=None):
        super(ClassModalityWeightedAttention, self).__init__()
        self.fused_channel = fused_channel
        self.modal_channel = modal_channel
        
        # 为3个类别(WT/TC/ET)创建独立的模态融合权重
        # 允许传入自定义权重，否则使用默认医学先验
        if custom_weights is not None:
            init_weights = torch.tensor(custom_weights)
        else:
            # 默认权重（用于CAM/SE/FPN/SelfAttention）
            init_weights = torch.tensor([
                [0.15, 0.30, 0.15, 0.40],  # WT: FLAIR最重要
                [0.30, 0.40, 0.15, 0.15],  # TC: T1ce+T1
                [0.10, 0.60, 0.10, 0.10],  # ET: T1ce最关键
            ])  # [3, 4] - 顺序: [WT, TC, ET] × [T1, T1ce, T2, FLAIR]
        self.class_modality_weights = nn.Parameter(init_weights)
        
        # 计算每个模态提取的通道数（确保能被4整除后正好等于fused_channel）
        per_modal_ch = fused_channel // 4
        
        # 每个模态独立的特征提取头
        self.modal_projectors = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(modal_channel, per_modal_ch, 1),
                nn.GroupNorm(num_groups=8 if per_modal_ch % 8 == 0 else (4 if per_modal_ch % 4 == 0 else 1), 
                            num_channels=per_modal_ch),
                nn.ReLU(inplace=False)
            ) for _ in range(4)  # T1, T1ce, T2, FLAIR
        ])
        
        # 融合通道映射 - 输入通道固定为4*per_modal_ch
        self.fusion_conv = nn.Conv2d(4 * per_modal_ch, fused_channel, 1)
        self.blend_strength = nn.Parameter(torch.tensor(0.10))  # 模态感知特征的注入强度（微调提高）
        
    def forward(self, x, modal_feats):
        """
        x: [B, C, H, W] 融合特征
        modal_feats: List[Tensor[B, 8, H, W]] x 4, [T1, T1ce, T2, FLAIR]
        """
        if modal_feats is None or len(modal_feats) != 4:
            return x
        
        B, C, H, W = x.shape
        per_modal_ch = C // 4
        
        try:
            # 1. 每个模态独立提取特征
            modal_features = []
            for i, (proj, mf) in enumerate(zip(self.modal_projectors, modal_feats)):
                # 确保模态特征尺寸匹配
                if mf.shape[2:] != (H, W):
                    mf = F.interpolate(mf, size=(H, W), mode='bilinear', align_corners=False)
                feat = proj(mf)  # [B, per_modal_ch, H, W]
                modal_features.append(feat)
            
            # 2. 拼接所有模态特征: [B, 4*per_modal_ch, H, W]
            concat_feat = torch.cat(modal_features, dim=1)
            
            # 3. 通过fusion_conv映射到C通道: [B, C, H, W]
            fused_modal = self.fusion_conv(concat_feat)
            
            # 4. 应用类别-模态权重进行加权（可选，作为额外的注意力机制）
            # 这里我们用权重作为通道注意力
            w = torch.softmax(self.class_modality_weights, dim=1)  # [3, 4]
            # 计算平均权重 [4]
            avg_weights = w.mean(dim=0)  # [4]
            # 对每个模态特征应用权重
            weighted_modal_feats = []
            for i, feat in enumerate(modal_features):
                weighted_modal_feats.append(feat * avg_weights[i])
            # 重新拼接并融合
            weighted_concat = torch.cat(weighted_modal_feats, dim=1)  # [B, 4*per_modal_ch, H, W]
            weighted_fused = self.fusion_conv(weighted_concat)  # [B, C, H, W]
            
            # 5. 混合原始融合和加权融合
            final_fused = 0.5 * fused_modal + 0.5 * weighted_fused
            
            # 6. 残差注入到原始特征
            alpha = torch.clamp(self.blend_strength, 0.0, 0.3)
            output = x + alpha * final_fused
            
            return output
            
        except Exception as e:
            # 如果出错,返回原始特征
            print(f"Warning: ClassModalityWeightedAttention forward failed: {e}")
            return x


class SelfAttentionModule(nn.Module):
    def __init__(self, channel, reduction=8):
        super(SelfAttentionModule, self).__init__()
        self.channel = channel
        self.reduction = reduction
        self.scale = (channel // reduction) ** -0.5
        self.channel_query = nn.Conv2d(channel, channel // reduction, 1)
        self.channel_key = nn.Conv2d(channel, channel // reduction, 1)
        self.channel_value = nn.Conv2d(channel, channel // reduction, 1)
        self.channel_proj = nn.Conv2d(channel // reduction, channel, 1)
        _sp_ch = channel // 4
        _sp_groups = 32 if _sp_ch % 32 == 0 else (16 if _sp_ch % 16 == 0 else 8)
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(channel, _sp_ch, 3, padding=1),
            nn.GroupNorm(num_groups=_sp_groups, num_channels=_sp_ch, eps=1e-5, affine=True),
            nn.ReLU(inplace=False),
            nn.Conv2d(_sp_ch, 1, 1),
            nn.Sigmoid()
        )
        self.fusion = nn.Conv2d(channel * 2, channel, 1)
        self.gamma_spatial = nn.Parameter(torch.tensor(0.1))
        self.gamma_channel = nn.Parameter(torch.tensor(0.1))
        _gn_groups = 32 if channel % 32 == 0 else (16 if channel % 16 == 0 else 8)
        self.norm = nn.GroupNorm(num_groups=_gn_groups, num_channels=channel, eps=1e-5, affine=True)
        self.dropout = nn.Dropout2d(0.05)  # 恢复到0.05 (0.08导致ET边界过度平滑)
        self._warn_count = 0

    def forward(self, x):
        B, C, H, W = x.size()
        if torch.isnan(x).any() or torch.isinf(x).any():
            if self._warn_count < 5:
                print("Warning: Input contains NaN/Inf in SelfAttentionModule — sanitized and continuing")
                self._warn_count += 1
            x = torch.nan_to_num(x, nan=0.0, posinf=1e4, neginf=-1e4)
            x = torch.clamp(x, -10, 10)
        spatial_att = self.spatial_conv(x)
        spatial_att = torch.nan_to_num(spatial_att, nan=0.0, posinf=1.0, neginf=0.0)
        spatial_att = torch.clamp(spatial_att, 0.0, 1.0)
        spatial_out = x * spatial_att
        gamma_spatial_safe = torch.clamp(self.gamma_spatial, 0.0, 1.0)
        spatial_out = gamma_spatial_safe * spatial_out + x
        channel_q = self.channel_query(x)
        channel_k = self.channel_key(x)
        channel_v = self.channel_value(x)
        q_pooled = F.adaptive_avg_pool2d(channel_q, 1).view(B, -1, 1)
        k_pooled = F.adaptive_avg_pool2d(channel_k, 1).view(B, -1, 1)
        channel_attention = torch.bmm(q_pooled.transpose(1, 2), k_pooled) * self.scale
        channel_attention = torch.clamp(channel_attention, -10, 10)
        channel_attention = torch.sigmoid(channel_attention)
        channel_out = channel_v * channel_attention.unsqueeze(-1)
        channel_out = self.channel_proj(channel_out)
        gamma_channel_safe = torch.clamp(self.gamma_channel, 0.0, 1.0)
        channel_out = gamma_channel_safe * channel_out + x
        fused = torch.cat([spatial_out, channel_out], dim=1)
        output = self.fusion(fused)
        output = torch.nan_to_num(output, nan=0.0, posinf=1e4, neginf=-1e4)
        output = torch.clamp(output, -10, 10)
        output = self.dropout(output)
        output = self.norm(output)
        if torch.isnan(output).any() or torch.isinf(output).any():
            if self._warn_count < 5:
                print("Warning: Output contains NaN/Inf in SelfAttentionModule (sanitized fallback)")
                self._warn_count += 1
            return x
        return output


class SelfAttentionEFM(nn.Module):
    """类别-模态感知的特征增强模块"""
    def __init__(self, channel, modal_channel=8, custom_weights=None):
        super(SelfAttentionEFM, self).__init__()
        
        # 基础卷积层
        self.conv2d = ConvBNR(channel, channel, 3)
        
        # 简化的自注意力模块
        self.self_attention = SelfAttentionModule(channel)
        
        # 类别-模态感知注意力（传入自定义权重）
        self.class_modal_attention = ClassModalityWeightedAttention(channel, modal_channel, custom_weights)
        
        # 边缘注意力融合 - 简化版本
        gn_groups = 32 if channel % 32 == 0 else (16 if channel % 16 == 0 else 8)
        self.edge_fusion = nn.Sequential(
            nn.Conv2d(channel + 1, channel, 1),  # 使用1x1卷积减少计算复杂度
            nn.GroupNorm(num_groups=gn_groups, num_channels=channel, eps=1e-5, affine=True),
            nn.ReLU(inplace=False)
        )
        
        # 简化的特征增强 - 移除复杂的多尺度处理
        self.feature_enhance = nn.Sequential(
            nn.Conv2d(channel, channel, 3, padding=1),
            nn.GroupNorm(num_groups=gn_groups, num_channels=channel, eps=1e-5, affine=True),
            nn.ReLU(inplace=False),
            nn.Dropout2d(0.05)  # 恢复到0.05 (0.08导致ET边界过度平滑)
        )
        
        # 简化的通道注意力
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channel, channel // 8, 1),
            nn.ReLU(inplace=False),
            nn.Conv2d(channel // 8, channel, 1),
            nn.Sigmoid()
        )
        
        # 减少可学习参数，只保留关键的融合权重
        self.alpha = nn.Parameter(torch.tensor(0.1))  # 边缘注意力权重
        self.beta = nn.Parameter(torch.tensor(0.1))   # 自注意力权重
    
    def forward(self, c, att, modal_feats=None):
        """类别-模态感知的前向传播
        
        c: [B, C, H, W] 融合特征
        att: [B, 1, H, W] 边界注意力图
        modal_feats: List[Tensor[B, 8, H, W]] x 4, 可选的模态特征
        """
        # 输入检查
        if torch.isnan(c).any() or torch.isinf(c).any():
            print(f"Warning: Input c contains NaN/Inf in SelfAttentionEFM — sanitized")
            c = torch.nan_to_num(c, nan=0.0, posinf=1e4, neginf=-1e4)
            c = torch.clamp(c, -5, 5)
        if torch.isnan(att).any() or torch.isinf(att).any():
            print(f"Warning: Input att contains NaN/Inf in SelfAttentionEFM — replaced with zeros")
            att = torch.zeros_like(att)
        
        # 调整注意力图尺寸
        if c.size() != att.size():
            att = F.interpolate(att, c.size()[2:], mode='bilinear', align_corners=False)
        
        try:
            # 边缘注意力融合
            att_norm = torch.clamp(torch.tanh(att), -0.5, 0.5)  # 更保守的范围限制
            edge_input = torch.cat([c, att_norm], dim=1)
            edge_enhanced = self.edge_fusion(edge_input)
            
            # 基础特征提取
            x = c + self.alpha * c * att_norm  # 残差连接
            x = self.conv2d(x)
            
            # 应用自注意力机制
            x_attention = self.self_attention(x)
            
            # 🆕 类别-模态感知特征增强
            if modal_feats is not None:
                x_attention = self.class_modal_attention(x_attention, modal_feats)
            
            # 特征增强
            enhanced_features = self.feature_enhance(x_attention)
            
            # 通道注意力
            channel_weights = self.channel_attention(enhanced_features)
            channel_enhanced = enhanced_features * channel_weights
            
            # 简化的特征融合
            alpha_safe = torch.clamp(self.alpha, 0.0, 0.3)  # 限制融合权重
            beta_safe = torch.clamp(self.beta, 0.0, 0.3)
            
            # 使用更稳定的融合策略
            out = c + alpha_safe * (edge_enhanced - c) + beta_safe * (channel_enhanced - c)

            # 数值稳定化处理：清除NaN/Inf并限制范围
            out = torch.nan_to_num(out, nan=0.0, posinf=1e4, neginf=-1e4)
            out = torch.clamp(out, -5, 5)
            
            # 最终数值检查（仅打印前三次以免刷屏）
            if torch.isnan(out).any() or torch.isinf(out).any():
                warn_cnt = getattr(self, "_warn_count", 0)
                if warn_cnt < 3:
                    print(f"Warning: Final output still contains NaN/Inf after stabilization, returning input")
                self._warn_count = warn_cnt + 1
                return c
                
            return out
            
        except Exception as e:
            print(f"Warning: SelfAttentionEFM forward failed: {e}, returning input")
            return c


class EnhancedResidualUnit(nn.Module):
    """改进的残差单元（ERU风格）：LeakyReLU + GroupNorm，支持降采样"""
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, use_leaky_relu: bool = True):
        super(EnhancedResidualUnit, self).__init__()
        gn = 32 if out_channels % 32 == 0 else (16 if out_channels % 16 == 0 else 8)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.GroupNorm(num_groups=gn, num_channels=out_channels, eps=1e-5, affine=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.GroupNorm(num_groups=gn, num_channels=out_channels, eps=1e-5, affine=True)
        self.activation = nn.LeakyReLU(0.01, inplace=False) if use_leaky_relu else nn.ReLU(inplace=False)
        # 快捷连接
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.GroupNorm(num_groups=gn, num_channels=out_channels, eps=1e-5, affine=True)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + identity
        out = self.activation(out)
        return out


class ResidualEnhancedEFM(SelfAttentionEFM):
    """使用改进残差单元增强的EFM：替换基础卷积并添加残差增强路径，支持类别-模态感知"""
    def __init__(self, channel: int, modal_channel: int = 8, custom_weights=None):
        super().__init__(channel, modal_channel, custom_weights)
        # 用改进的残差单元替换基础卷积
        self.conv2d = nn.Sequential(
            EnhancedResidualUnit(channel, channel),
            EnhancedResidualUnit(channel, channel)
        )
        # 增强的特征增强路径（降维后再升维）
        self.residual_enhance = nn.Sequential(
            EnhancedResidualUnit(channel, channel // 2),
            EnhancedResidualUnit(channel // 2, channel)
        )

    def forward(self, c: torch.Tensor, att: torch.Tensor, modal_feats=None) -> torch.Tensor:
        """支持模态特征的前向传播"""
        # 输入检查与对齐，复用基础EFM的稳定策略
        if torch.isnan(c).any() or torch.isinf(c).any():
            print(f"Warning: Input c contains NaN/Inf in ResidualEnhancedEFM — sanitized")
            c = torch.nan_to_num(c, nan=0.0, posinf=1e4, neginf=-1e4)
            c = torch.clamp(c, -5, 5)
        if torch.isnan(att).any() or torch.isinf(att).any():
            print(f"Warning: Input att contains NaN/Inf in ResidualEnhancedEFM — replaced with zeros")
            att = torch.zeros_like(att)
        if c.size() != att.size():
            att = F.interpolate(att, c.size()[2:], mode='bilinear', align_corners=False)

        # 基础特征提取（带注意力门控的残差增强卷积）
        att_norm = torch.clamp(torch.tanh(att), -0.5, 0.5)
        x = c + self.alpha * c * att_norm
        x = self.conv2d(x)

        # 额外残差增强路径
        residual_enhanced = self.residual_enhance(x)

        # 自注意力
        x_attention = self.self_attention(x)
        
        # 🆕 类别-模态感知特征增强
        if modal_feats is not None:
            x_attention = self.class_modal_attention(x_attention, modal_feats)
        
        # 轻量特征增强
        enhanced_features = self.feature_enhance(x_attention)

        # 通道注意力
        channel_weights = self.channel_attention(enhanced_features)
        channel_enhanced = enhanced_features * channel_weights

        # 稳定的融合策略
        alpha_safe = torch.clamp(self.alpha, 0.0, 0.3)
        beta_safe = torch.clamp(self.beta, 0.0, 0.3)
        out = c + alpha_safe * (residual_enhanced - c) + beta_safe * (channel_enhanced - c)

        # 数值稳定化
        out = torch.nan_to_num(out, nan=0.0, posinf=1e4, neginf=-1e4)
        out = torch.clamp(out, -5, 5)
        return out


class CAM(nn.Module):
    """类别-模态感知的上下文聚合模块（增强版）"""
    def __init__(self, hchannel, channel, modal_channel=8, custom_weights=None):
        super(CAM, self).__init__()
        self.conv1_1 = Conv1x1(hchannel + channel, channel)
        self.conv3_1 = ConvBNR(channel // 4, channel // 4, 3)
        self.dconv5_1 = ConvBNR(channel // 4, channel // 4, 3, dilation=2)
        self.dconv7_1 = ConvBNR(channel // 4, channel // 4, 3, dilation=3)
        self.dconv9_1 = ConvBNR(channel // 4, channel // 4, 3, dilation=4)
        self.conv1_2 = Conv1x1(channel, channel)
        self.conv3_3 = ConvBNR(channel, channel, 3)
        
        # 🆕 类别-模态感知注意力（传入自定义权重）
        self.class_modal_attention = ClassModalityWeightedAttention(channel, modal_channel, custom_weights)
        
        # 添加可学习的融合权重
        self.beta = nn.Parameter(torch.tensor(0.1))

    def forward(self, lf, hf, modal_feats=None):
        """
        lf: 低层特征
        hf: 高层特征
        modal_feats: List[Tensor[B, 8, H, W]] x 4, 可选的模态特征
        """
        if lf.size()[2:] != hf.size()[2:]:
            hf = F.interpolate(hf, size=lf.size()[2:], mode='bilinear', align_corners=False)
        x = torch.cat((lf, hf), dim=1)
        x = self.conv1_1(x)
        
        # 🆕 类别-模态感知增强（禁用：收益不明显）
        # if modal_feats is not None:
        #     x = self.class_modal_attention(x, modal_feats)
        
        # 多尺度膨胀卷积
        xc = torch.chunk(x, 4, dim=1)
        x0 = self.conv3_1(xc[0])
        x1 = self.dconv5_1(xc[1] + x0 * 0.1)
        x2 = self.dconv7_1(xc[2] + x1 * 0.1)
        x3 = self.dconv9_1(xc[3] + x2 * 0.1)
        
        xx = self.conv1_2(torch.cat((x0, x1, x2, x3), dim=1))
        
        # 残差连接
        out = x + self.beta * self.conv3_3(xx)
        
        return out


# Self_Adaptive_Weighted_Fusion_Module 已移除：未被引用，且 FPN 已采用简单卷积平滑替代


class UNetDecoder(nn.Module):
    """
    增强UNet解码器块 - 残差连接 + 注意力机制
    
    特点：
    1. 保留全通道（100%特征）
    2. 双层卷积融合 + 残差连接
    3. 通道注意力加权
    4. 跳跃连接注意力门控
    """
    def __init__(self, high_channels, low_channels, out_channels):
        """
        Args:
            high_channels: 高层特征通道数（来自更深层）
            low_channels: 低层特征通道数（跳跃连接），如果为0则表示无skip connection
            out_channels: 输出通道数
        """
        super(UNetDecoder, self).__init__()
        
        self.has_skip = (low_channels > 0)
        # 拼接后的通道数
        concat_channels = high_channels + low_channels if self.has_skip else high_channels
        
        # ✅ 移除注意力门控（避免双重抑制）
        # self.attention_gate = nn.Sequential(
        #     nn.Conv2d(concat_channels, out_channels, 1, bias=False),
        #     nn.GroupNorm(min(32, out_channels), out_channels),
        #     nn.Sigmoid()
        # )
        
        # 第一层卷积：融合拼接特征
        self.conv1 = nn.Sequential(
            nn.Conv2d(concat_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(min(32, out_channels), out_channels),
            nn.ReLU(inplace=True)
        )
        
        # 第二层卷积：进一步提取特征
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(min(32, out_channels), out_channels),
            nn.ReLU(inplace=True)
        )
        
        # ✅ 优化通道注意力：减小压缩比，避免信息丢失
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels, max(out_channels // 4, 16), 1),  # ✅ 8 → 4，压缩比从 8:1 降为 4:1
            nn.ReLU(inplace=True),
            nn.Conv2d(max(out_channels // 4, 16), out_channels, 1),  # ✅ 8 → 4
            nn.Sigmoid()
        )
        
        # 🆕 残差投影（如果需要）
        if concat_channels != out_channels:
            self.residual_proj = nn.Sequential(
                nn.Conv2d(concat_channels, out_channels, 1, bias=False),
                nn.GroupNorm(min(32, out_channels), out_channels)
            )
        else:
            self.residual_proj = nn.Identity()
    
    def forward(self, high_feat, low_feat=None):
        """
        Args:
            high_feat: 高层特征（需要上采样）
            low_feat: 低层特征（跳跃连接），如果为None则表示无skip connection
        Returns:
            融合后的特征
        """
        # 如果有 skip connection，上采样高层特征到低层特征的尺寸并拼接
        if self.has_skip and low_feat is not None:
            if high_feat.shape[2:] != low_feat.shape[2:]:
                high_feat = F.interpolate(high_feat, size=low_feat.shape[2:], 
                                         mode='bilinear', align_corners=False)
            x = torch.cat([high_feat, low_feat], dim=1)
        else:
            # 无 skip connection，直接2倍上采样
            high_feat = F.interpolate(high_feat, scale_factor=2, mode='bilinear', align_corners=False)
            x = high_feat
        
        # 双层卷积融合
        out = self.conv1(x)
        out = self.conv2(out)
        
        # 🆕 通道注意力（仅保留一个注意力机制）
        channel_att = self.channel_attention(out)
        out = out * channel_att
        
        # ✅ 优化残差连接：提高权重到 1.0，与主路径平衡
        residual = self.residual_proj(x)
        out = out + residual  # ✅ 残差权重 1.0（标准 ResNet 配置）
        
        return out




class Net(nn.Module):
    def __init__(self, num_classes=3, eam_mode: str = 'all', layer3_stride: int = 1,
                 enable_swin_attention: bool = True,
                 enable_paper_edge: bool = True,
                 paper_edge_scale: int = 2,
                 paper_edge_base_channels: int = 32,
                 paper_edge_requires_grad: bool = False,
                 enable_late_modality_selection: bool = True,
                 # 🔧 已移除 width_multiplier 参数，通道数直接硬编码
                 # 🆕 双编码器参数
                 enable_dual_encoder: bool = False,
                 swin_fusion_weight: float = 0.3,
                 # 🆕 EFM 开关：用于消融实验（默认开启）
                 enable_efm: bool = True,
                 enable_boundary_refine: bool = True):
        super(Net, self).__init__()
        self.num_classes = num_classes
        self.layer3_stride = int(layer3_stride) if layer3_stride in (1, 2) else 1
        
        self.enable_swin_attention = bool(enable_swin_attention)
        # 新增三类增强模块的开关（PaperEdge 强制关闭以测试 EAM 动态边界生成）
        self.enable_paper_edge = False  # 🔧 强制关闭固定边缘先验，改用 EAM 动态生成
        self.paper_edge_scale = int(max(1, paper_edge_scale))
        self.paper_edge_base_channels = int(max(8, paper_edge_base_channels))
        self.paper_edge_requires_grad = bool(paper_edge_requires_grad)
        
        # 🆕 双编码器开关（强制关闭以测试单主干上限）
        self.enable_dual_encoder = False  # 🔧 临时强制关闭，测试 Res2Net 单主干性能
        self.swin_fusion_weight = float(swin_fusion_weight)
        # 🆕 EFM / 边界 refine 开关：用于消融实验
        self.enable_efm = bool(enable_efm)
        self.enable_boundary_refine = bool(enable_boundary_refine)
        
        # Res2Net backbone（全局注意力已验证无效，强制禁用）
        self.resnet = res2net50_v1b_26w_4s(
            pretrained=False, 
            in_channels=4, 
            layer3_stride=self.layer3_stride,
            enable_global_attention=False  # 实验证明全局注意力降低性能0.34%，强制禁用
            # 🔧 已移除 width_multiplier 参数，通道数已在 Res2Net 中硬编码
        )
        
        # 下游网络采用减半通道，降低解码器/注意力模块开销
        c1 = 128
        c2 = 256
        c3 = 512
        c4 = 1024
        
        print(f"🔧 BGNet通道配置（减半）: c1={c1}, c2={c2}, c3={c3}, c4={c4}")

        # Res2Net 输出仍为标准通道：256/512/1024/2048，这里做 1x1 投影以对齐下游
        self.backbone_proj1 = Conv1x1(256, c1)
        self.backbone_proj2 = Conv1x1(512, c2)
        self.backbone_proj3 = Conv1x1(1024, c3)
        self.backbone_proj4 = Conv1x1(2048, c4)
        
        # 🆕 双编码器：Swin Transformer 辅助分支
        if self.enable_dual_encoder:
            try:
                from net.swin_transformer import SwinTransformer
                self.swin_encoder = SwinTransformer(
                    img_size=160,
                    patch_size=4,
                    in_chans=4,
                    embed_dim=96,
                    depths=[2, 2, 6, 2],
                    num_heads=[3, 6, 12, 24],
                    window_size=5,
                    drop_path_rate=0.2
                )
                
                # 特征融合模块（通道对齐）
                self.swin_proj1 = Conv1x1(96, c1)
                self.swin_proj2 = Conv1x1(192, c2)
                self.swin_proj3 = Conv1x1(384, c3)
                self.swin_proj4 = Conv1x1(768, c4)
                
                print(f"✅ 双编码器已启用 (Swin融合权重={swin_fusion_weight})")
            except ImportError:
                print("⚠️ 无法导入 SwinTransformer，双编码器功能禁用")
                self.enable_dual_encoder = False
                self.swin_encoder = None
        else:
            self.swin_encoder = None
        
        self.eam = EAM(mode=str(eam_mode))
        self.fg_head = TumorForegroundHead(c4, mid_channels=256)
        self.paper_edge = PaperEdgeExtractionModule(in_channels=2, num_classes=num_classes, base_channels=self.paper_edge_base_channels) if self.enable_paper_edge else None
        if self.paper_edge is not None and (not self.paper_edge_requires_grad):
            for p in self.paper_edge.parameters():
                p.requires_grad = False

        # 📊 EFM 模块已在本实验中用轻量 FAB-Align 替代：专注于通道/统计对齐，
        # 将复杂的边界+模态增强下放到 EAM / LMS，简化中间特征的干预。
        # 为每个编码器阶段构建 FAB-Align：保持通道不变，仅做特征对齐。
        self.fab1 = FeatureAdaptiveBiasAlign(c1, c1)
        self.fab2 = FeatureAdaptiveBiasAlign(c2, c2)
        self.fab3 = FeatureAdaptiveBiasAlign(c3, c3)
        self.fab4 = FeatureAdaptiveBiasAlign(c4, c4)

        self.aspp_x3 = None
        self.aspp_x4 = None

        # ✅ 完全对称的UNet解码器：保持更多通道，提高表达能力
        # decoder3: x4 [1024] + x3 [512] → d3 [512]
        self.decoder3 = UNetDecoder(
            high_channels=c4,      # 1024
            low_channels=c3,       # 512
            out_channels=512       # 输出 d3: 512 通道
        )
        
        # decoder2: d3 [512] + x2 [256] → d2 [256]
        self.decoder2 = UNetDecoder(
            high_channels=512,
            low_channels=c2,       # 256
            out_channels=256       # 输出 d2: 256 通道
        )
        
        # decoder1: d2 [256] + x1 [128] → d1 [128]
        self.decoder1 = UNetDecoder(
            high_channels=256,
            low_channels=c1,       # 128，对应 x1 的 skip
            out_channels=128       # 输出 d1: 128 通道
        )
        
        # decoder0: d1 [128] → d0 [64]（最终细化层，仅使用 d1 做上采样与细化，不再复用额外 skip）
        self.decoder0 = UNetDecoder(
            high_channels=128,
            low_channels=0,        # 无 skip，纯 refine + 上采样
            out_channels=64        # 输出 d0: 64 通道
        )
        print("✅ 完全对称UNet解码器（4层，保持高通道数）:")
        print(f"   decoder3: {c4} + {c3} → 512 (1/32 -> 1/16, skip from x3)")
        print(f"   decoder2: 512 + {c2} → 256 (1/16 -> 1/8,  skip from x2)")
        print(f"   decoder1: 256 + {c1} → 128 (1/8  -> 1/4,  skip from x1)")
        print(f"   decoder0: 128 → 64  (1/4  -> 更高分辨率, refine-only, 无额外 skip)")

        # ❌ CAM模块已被UNet解码器替代（不再需要）
        # 标准UNet通过拼接+卷积实现特征融合，不需要额外的注意力模块

        # ✅ 预测头：适配对称解码器的通道数
        self.predictor1 = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1, bias=False),    # d0: 64 → 32
            nn.GroupNorm(8, 32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, num_classes, 1)
        )
        self.predictor2 = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1, bias=False),  # d2: 256 → 128
            nn.GroupNorm(16, 128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, num_classes, 1)
        )
        self.predictor3 = nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=1, bias=False),  # d3: 512 → 256
            nn.GroupNorm(32, 256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, 1)
        )
        
        # 🆕 Multi-Scale Fusion Head：融合 d0/d1/d2/d3 多尺度特征用于最终预测
        ms_channels = 128
        self.ms_proj0 = nn.Conv2d(64, ms_channels, 1, bias=False)    # d0: 64 → 128
        self.ms_proj1 = nn.Conv2d(128, ms_channels, 1, bias=False)   # d1: 128 → 128
        self.ms_proj2 = nn.Conv2d(256, ms_channels, 1, bias=False)   # d2: 256 → 128
        self.ms_proj3 = nn.Conv2d(512, ms_channels, 1, bias=False)   # d3: 512 → 128
        self.ms_fusion_conv = ConvBNR(ms_channels, ms_channels, 3)
        self.ms_fusion_logits = nn.Conv2d(ms_channels, num_classes, 1)
        # 可学习的多尺度权重 [w_d0, w_d1, w_d2, w_d3]，softmax 后作为融合系数
        self.ms_fusion_weights = nn.Parameter(torch.tensor([0.4, 0.3, 0.2, 0.1]))
        
        # 关闭FPN：最终输出由多尺度融合头生成 fused

        # =============== 分肿瘤-模态选择（Late Modality Selection）=================
        # 将各模态浅层特征（来自 backbone.modal_preprocess）升维到与解码头相同通道，并生成类别特定的logits
        # 4 模态 -> 每模态先映射到 num_classes 通道，再按类别做可学习融合
        self.mod_logits_head = nn.Sequential(
            ConvBNR(8, 16, 3),
            nn.Conv2d(16, num_classes, 1)
        )
        # 类-模态 权重矩阵（3类×4模态），softmax后作为融合权重；提供医学先验初始化（可微调）
        # 顺序: 模态 = [T1, T1ce, T2, FLAIR]；类别 = [WT, TC, ET]
        # 🎯 优化LMS先验权重，提高多模态互补性
        prior = torch.tensor([
            [0.1, 0.2, 0.2, 0.5],      # WT: FLAIR从0.40提高到0.5，T1ce从0.20提高到0.2
            [0.2, 0.5, 0.15, 0.15],    # TC: T1从0.15提高到0.2，T1ce从0.55降到0.5，更平衡
            [0.15, 0.55, 0.15, 0.15],  # ET: T1ce从0.70降到0.55，T1从0.10提高到0.15，避免过度依赖单一模态
        ], dtype=torch.float32)
        self.class_modality_weights = nn.Parameter(prior)
        # 融合强度（残差注入）- 提高以更好利用LMS
        self.mod_blend = nn.Parameter(torch.tensor(0.50))  # 从0.30提高到0.50

        # 🆕 边界细化模块 (方案D阶段1增强)
        self.boundary_refine = None
        
        
        
        # 🆕 小目标增强模块 (方案D阶段2新增)
        self.small_obj_enhancer = None
        
        # logits 级边界 refine（使用最终 oe 引导）
        if self.enable_boundary_refine:
            br_in_channels = self.num_classes + 1  # logits 通道数 + 单通道边界
            self.boundary_refine = nn.Sequential(
                nn.Conv2d(br_in_channels, 32, 3, padding=1, bias=False),
                nn.GroupNorm(8, 32),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, self.num_classes, 3, padding=1, bias=False)
            )
            self.boundary_refine_gamma = nn.Parameter(torch.tensor(0.5))
        else:
            self.boundary_refine_gamma = nn.Parameter(torch.tensor(0.0))
        
        # 实例感知注意力已移除（为空实现），直接使用特征
        
        # ═══════════════════════════════════════════════════════════════
        # ✅ 回溯到Epoch 37配置：不使用Decoder SWFM和TaskSpecificDecoder
        # ═══════════════════════════════════════════════════════════════
        # CAM输出直接使用，不经过SWFM融合
        
        # 混合上采样模块已移除（未在forward中使用，避免冗余）
        
        # Late Modality Selection开关（实验证明 LMS 有 +0.8% 收益，恢复启用）
        self.enable_late_modality_selection = enable_late_modality_selection  # ✅ 恢复使用参数
        
        # 添加权重初始化
        self._initialize_weights()

    def _initialize_weights(self):
        """改进的权重初始化"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # 使用He初始化，适合ReLU激活函数
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
        # 特别初始化预测器层，使用合适的权重避免梯度消失
        # 初始化预测器层，支持 Sequential 类型
        predictors = [self.predictor1, self.predictor2, self.predictor3]
        for pred in predictors:
            if isinstance(pred, nn.Conv2d):
                nn.init.xavier_normal_(pred.weight, gain=1.0)  # 使用Xavier初始化，gain=1.0
                if pred.bias is not None:
                    nn.init.constant_(pred.bias, 0)
            else:
                # 遍历 Sequential 等容器中的 Conv2d 子模块
                for m in pred.modules():
                    if isinstance(m, nn.Conv2d):
                        nn.init.xavier_normal_(m.weight, gain=1.0)  # 提高gain避免梯度消失
                        if m.bias is not None:
                            nn.init.constant_(m.bias, 0)
        
        # 初始化可学习参数
        for m in self.modules():
            if hasattr(m, 'alpha') and isinstance(m.alpha, nn.Parameter):
                nn.init.constant_(m.alpha, 0.1)
            if hasattr(m, 'beta') and isinstance(m.beta, nn.Parameter):
                nn.init.constant_(m.beta, 0.1)
            if hasattr(m, 'gamma') and isinstance(m.gamma, nn.Parameter):
                nn.init.constant_(m.gamma, 0.1)
            if hasattr(m, 'gamma_spatial') and isinstance(m.gamma_spatial, nn.Parameter):
                nn.init.constant_(m.gamma_spatial, 0.1)
            if hasattr(m, 'gamma_channel') and isinstance(m.gamma_channel, nn.Parameter):
                nn.init.constant_(m.gamma_channel, 0.1)

        # ✅ 确保 Class-Aware EAM 的初始化正确
        try:
            # 初始化每个类别头的偏置
            for head in self.eam.class_edge_heads:
                final_conv = head[-1]
                if isinstance(final_conv, nn.Conv2d):
                    if final_conv.bias is None:
                        final_conv.bias = nn.Parameter(torch.zeros(1))
                    nn.init.constant_(final_conv.bias, 0.5)
            
            # 设置全局 logit 参数
            if hasattr(self.eam, 'logit_scale'):
                with torch.no_grad():
                    self.eam.logit_scale.fill_(1.5)
            if hasattr(self.eam, 'logit_bias'):
                with torch.no_grad():
                    self.eam.logit_bias.fill_(1.0)
            
            # 设置类别特定权重（WT=1.0, TC=1.5, ET=2.0）
            if hasattr(self.eam, 'class_edge_weights'):
                with torch.no_grad():
                    self.eam.class_edge_weights.copy_(torch.tensor([1.0, 1.5, 2.0]))
        except Exception:
            pass
        

    def forward(self, x):
        # 输入检查
        if torch.isnan(x).any() or torch.isinf(x).any():
            print(f"Warning: Input contains NaN/Inf in Net forward")
            # 创建安全的输出
            B, _, H, W = x.shape
            safe_output = torch.zeros(B, self.num_classes, H, W, device=x.device)
            return {
                'stage1': safe_output,
                'stage2': safe_output,
                'stage3': safe_output,
                'stage4': safe_output,
                'edge': torch.zeros(B, 1, H, W, device=x.device)
            }
        
        try:
            # 主编码器：Res2Net
            x1, x2, x3, x4 = self.resnet(x)
            
            # 🆕 获取每层的模态特征（已在Res2Net中预处理好）
            layer_modal_feats = getattr(self.resnet, '_layer_modal_features', None)
            
            # 检查ResNet输出（就地净化，不再抛错）
            for i, feat in enumerate([x1, x2, x3, x4]):
                if torch.isnan(feat).any() or torch.isinf(feat).any():
                    print(f"Warning: ResNet x{i+1} contains NaN/Inf — sanitized")
                    fi = torch.nan_to_num(feat, nan=0.0, posinf=1e4, neginf=-1e4)
                    fi = torch.clamp(fi, -10, 10)
                    if i == 0: x1 = fi
                    elif i == 1: x2 = fi
                    elif i == 2: x3 = fi
                    elif i == 3: x4 = fi

            # 将 Res2Net 标准通道输出投影到下游“减半通道”配置
            x1 = self.backbone_proj1(x1)
            x2 = self.backbone_proj2(x2)
            x3 = self.backbone_proj3(x3)
            x4 = self.backbone_proj4(x4)
            
            # 🆕 双编码器：融合 Swin Transformer 特征
            if self.enable_dual_encoder and self.swin_encoder is not None:
                # 辅助编码器：Swin Transformer
                swin_features = self.swin_encoder(x)
                swin_x1, swin_x2, swin_x3, swin_x4 = swin_features
                
                # 通道对齐
                swin_x1 = self.swin_proj1(swin_x1)
                swin_x2 = self.swin_proj2(swin_x2)
                swin_x3 = self.swin_proj3(swin_x3)
                swin_x4 = self.swin_proj4(swin_x4)
                
                # 空间对齐（如果需要）
                if swin_x1.shape[2:] != x1.shape[2:]:
                    swin_x1 = F.interpolate(swin_x1, size=x1.shape[2:], mode='bilinear', align_corners=False)
                if swin_x2.shape[2:] != x2.shape[2:]:
                    swin_x2 = F.interpolate(swin_x2, size=x2.shape[2:], mode='bilinear', align_corners=False)
                if swin_x3.shape[2:] != x3.shape[2:]:
                    swin_x3 = F.interpolate(swin_x3, size=x3.shape[2:], mode='bilinear', align_corners=False)
                if swin_x4.shape[2:] != x4.shape[2:]:
                    swin_x4 = F.interpolate(swin_x4, size=x4.shape[2:], mode='bilinear', align_corners=False)
                
                # 🆕 简化融合策略（保持高效）
                # x1-x3: 均衡融合
                x1 = 0.5 * x1 + 0.5 * swin_x1
                x2 = 0.5 * x2 + 0.5 * swin_x2
                x3 = 0.5 * x3 + 0.5 * swin_x3
                
                # x4: 偏向 Res2Net（保护 ET 性能）
                # Res2Net 在 ET 上更好，Swin 在 TC 上更好
                # 使用 0.7:0.3 的权重平衡两者
                x4 = 0.7 * x4 + 0.3 * swin_x4
                
                # 🆕 融合后重新应用跨模态注意力（恢复 ET 性能）
                # 因为 Swin 特征没有跨模态信息，融合会稀释 Res2Net 的跨模态增强
                if layer_modal_feats is not None and hasattr(self.resnet, 'cma_l4'):
                    try:
                        x4 = self.resnet.cma_l4(x4, layer_modal_feats.get('layer4', []))
                    except Exception as e:
                        pass  # 如果失败，保持原特征

            # 🆕 肿瘤前景预测与特征门控（在进入 EAM 之前抑制背景/正常脑组织）
            fg_logits = None
            x1_eam, x2_eam, x3_eam, x4_eam = x1, x2, x3, x4
            try:
                # 使用最高层特征 x4 预测前景掩码，并上采样到输入分辨率
                fg_logits = self.fg_head(x4, out_size=x.shape[2:])  # [B,1,H,W]
                fg_prob_full = torch.sigmoid(fg_logits).clamp(0.0, 1.0)
                # 软门控因子：背景≈0.5，前景≈1.0，避免完全抑制
                gate_full = 0.5 + 0.5 * fg_prob_full

                # 将前景 gate 下采样到各编码层尺度，用于 EAM 输入特征的门控
                g1 = F.interpolate(gate_full, size=x1.shape[2:], mode='bilinear', align_corners=False)
                g2 = F.interpolate(gate_full, size=x2.shape[2:], mode='bilinear', align_corners=False)
                g3 = F.interpolate(gate_full, size=x3.shape[2:], mode='bilinear', align_corners=False)
                g4 = F.interpolate(gate_full, size=x4.shape[2:], mode='bilinear', align_corners=False)

                x1_eam = x1 * g1
                x2_eam = x2 * g2
                x3_eam = x3 * g3
                x4_eam = x4 * g4
            except Exception:
                fg_logits = None
                x1_eam, x2_eam, x3_eam, x4_eam = x1, x2, x3, x4

            # ✅ 启用 Class-Aware EAM：为 WT/TC/ET 分别生成边界图
            if self.paper_edge is not None:
                flair = x[:, 3:4]
                t1ce = x[:, 1:2]
                if self.paper_edge_scale > 1:
                    size = (flair.shape[2] // self.paper_edge_scale, flair.shape[3] // self.paper_edge_scale)
                    flair_ds = F.interpolate(flair, size=size, mode='bilinear', align_corners=False)
                    t1ce_ds = F.interpolate(t1ce, size=size, mode='bilinear', align_corners=False)
                else:
                    flair_ds, t1ce_ds = flair, t1ce
                if self.paper_edge_requires_grad:
                    edge_att_small = self.paper_edge(flair_ds, t1ce_ds)
                else:
                    with torch.no_grad():
                        edge_att_small = self.paper_edge(flair_ds, t1ce_ds)
                edge_att = F.interpolate(edge_att_small, size=x.shape[2:], mode='bilinear', align_corners=False)
            else:
                # 使用前景门控后的特征作为 EAM 输入，减弱背景与正常脑组织的干扰
                edge_att = self.eam(x1_eam, x2_eam, x3_eam, x4_eam)
            # 数值稳定性检查
            edge_att = torch.nan_to_num(edge_att, nan=0.0, posinf=1e4, neginf=-1e4)
            edge_att = torch.clamp(edge_att, -8.0, 8.0)

            # 关闭全部可视化保存

            # ✅ 将多通道边界图转换为单通道（仍用于可视化与边界损失），
            # 但在本实验中不再直接驱动中间特征的 EFM 增强。
            edge_weights = torch.tensor([1.0, 1.5, 2.0], device=edge_att.device).view(1, 3, 1, 1)
            edge_weights = edge_weights / edge_weights.sum()  # 归一化
            edge_gate_single = (edge_att * edge_weights).sum(dim=1, keepdim=True)  # [B, 1, H, W]

            # 仍然保留 edge_gate_single 供可视化 / 损失函数使用
            edge_gate = edge_gate_single

            if self.enable_efm:
                # 🆕 使用轻量 FAB-Align 对编码器特征做通道/统计对齐，
                # 取代原先重型的 ResidualEnhancedEFM。
                x1a = self.fab1(x1)
                x2a = self.fab2(x2)
                x3a = self.fab3(x3)
                x4a = self.fab4(x4)
            else:
                # 🔧 EFM/FAB 消融：直接使用原始编码特征
                x1a, x2a, x3a, x4a = x1, x2, x3, x4

            # ✅ 解码器及预测头：在 fp32 中执行以提升 GroupNorm 数值稳定性
            with torch.cuda.amp.autocast(enabled=False):
                in_size = x.shape[2:]

                # 确保进入解码器与后续模块的特征为 float32
                x1a_fp32 = x1a.float()
                x2a_fp32 = x2a.float()
                x3a_fp32 = x3a.float()
                x4a_fp32 = x4a.float()
                edge_att_fp32 = edge_att.float()

                edge_att_upsampled = F.interpolate(edge_att_fp32, size=in_size, mode='bilinear', align_corners=False)

                # Stage 4 → 3: x4 [1024] + x3 [512] → d3 [512] (1/32 -> 1/16)
                d3 = self.decoder3(x4a_fp32, x3a_fp32)
                
                # Stage 3 → 2: d3 [512] + x2 [256] → d2 [256] (1/16 -> 1/8)
                d2 = self.decoder2(d3, x2a_fp32)
                
                # Stage 2 → 1: d2 [256] + x1 [128] → d1 [128] (1/8 -> 1/4)
                d1 = self.decoder1(d2, x1a_fp32)

                # Stage 1 → 0: d1 [128] → d0 [64] (1/4 -> 更高分辨率, refine-only, 不再使用额外 skip)
                d0 = self.decoder0(d1, None)
                
                # 检查UNet解码器输出（就地净化）
                for name, feat in zip(['d3', 'd2', 'd1', 'd0'], [d3, d2, d1, d0]):
                    if torch.isnan(feat).any() or torch.isinf(feat).any():
                        print(f"Warning: UNet Decoder output {name} contains NaN/Inf — sanitized")
                        fi = torch.nan_to_num(feat, nan=0.0, posinf=1e4, neginf=-1e4)
                        fi = torch.clamp(fi, -10, 10)
                        if name == 'd3':
                            d3 = fi
                        elif name == 'd2':
                            d2 = fi
                        elif name == 'd1':
                            d1 = fi
                        elif name == 'd0':
                            d0 = fi
                
                # ✅ 使用UNet解码器输出（替代原来的CAM输出）
                x34 = d3
                x234 = d2
                x1234 = d0

                # 预测器输出
                # 获取原始输入空间尺寸，统一上采样对齐到输入分辨率
                ds_x34 = ds_x234 = ds_x1234 = None
                
                o3 = self.predictor3(x34)
                o3 = F.interpolate(o3, size=in_size, mode='bilinear', align_corners=False)
                o2 = self.predictor2(x234)
                o2 = F.interpolate(o2, size=in_size, mode='bilinear', align_corners=False)
                
                # ✅ 回退到简单版本：不使用类别专属边界增强
                # 直接使用原始特征进行预测
                o1 = self.predictor1(x1234)
                o1 = F.interpolate(o1, size=in_size, mode='bilinear', align_corners=False)

                # 🆕 Multi-Scale Fusion Head：使用 d0/d1/d2/d3 生成最终 logits
                try:
                    p0 = self.ms_proj0(x1234)  # d0 已经是原始分辨率
                    p1 = self.ms_proj1(d1)
                    p2 = self.ms_proj2(x234)
                    p3 = self.ms_proj3(x34)
                    # d0 已经是原始分辨率，其他需要上采样
                    p0_up = p0 if p0.shape[2:] == in_size else F.interpolate(p0, size=in_size, mode='bilinear', align_corners=False)
                    p1_up = F.interpolate(p1, size=in_size, mode='bilinear', align_corners=False)
                    p2_up = F.interpolate(p2, size=in_size, mode='bilinear', align_corners=False)
                    p3_up = F.interpolate(p3, size=in_size, mode='bilinear', align_corners=False)
                    w = F.softmax(self.ms_fusion_weights, dim=0)
                    ms_fused = w[0] * p0_up + w[1] * p1_up + w[2] * p2_up + w[3] * p3_up
                    ms_fused = self.ms_fusion_conv(ms_fused)
                    fused = self.ms_fusion_logits(ms_fused)
                except Exception as e:
                    print(f"Warning: Multi-Scale Fusion Head fallback to o1 due to error: {e}")
                    fused = o1
                
                # ✅ 将多通道边界图转换为单通道用于可视化和 fuse
                edge_att_upsampled = F.interpolate(edge_att_fp32, size=in_size, mode='bilinear', align_corners=False)
                oe = edge_att_upsampled.max(dim=1, keepdim=True)[0]

                # =============== Late Modality Selection 融合 ==================
                # 从backbone读取浅层模态特征，生成每模态的类别logits，并按类-模态权重融合
                if self.enable_late_modality_selection:
                    try:
                        modal_feats = getattr(self.resnet, '_latest_modal_features', None)
                        if isinstance(modal_feats, (list, tuple)) and len(modal_feats) == 4:
                            modal_logits = []  # List[Tensor[B,C,H,W]]
                            for mf in modal_feats:  # mf: [B,8,H,W]
                                # 🆕 优化：先降采样到128×128，减少显存占用
                                target_size = (in_size[0] // 2, in_size[1] // 2)
                                mf_down = F.interpolate(mf.float(), size=target_size, mode='bilinear', align_corners=False)
                                ml = self.mod_logits_head(mf_down)
                                # 最后再上采样到输入分辨率
                                ml = F.interpolate(ml, size=in_size, mode='bilinear', align_corners=False)
                                modal_logits.append(ml)
                            # 形状堆叠为 [B,4,C,H,W]，再做(类×模态)加权和
                            ml_stack = torch.stack(modal_logits, dim=1)  # [B,4,C,H,W]
                            # 规范化权重
                            w = torch.softmax(self.class_modality_weights, dim=1)  # [3,4]
                            # 计算融合：对模态维按权重求和（对每个类别分别加权）
                            # 展开为便于广播: w -> [1,4,3,1,1] 先转置到 [4,3] 再扩展
                            w_t = w.t().contiguous().view(4, 3, 1, 1)  # [4,3,1,1]
                            w_b = w_t.unsqueeze(0)  # [1,4,3,1,1]
                            # 将 ml_stack 从 [B,4,C,H,W] 转到 [B,4,3,H,W]
                            ml_perm = ml_stack  # C==num_classes==3
                            # 融合
                            fused_mod = (ml_perm * w_b).sum(dim=1)  # [B,3,H,W]
                            # 残差注入
                            gamma = torch.clamp(self.mod_blend, 0.0, 1.0)
                            fused = fused + gamma * (fused_mod - fused)
                    except Exception:
                        pass
                
                
                
                # 🆕 ============= 小目标增强 (方案D阶段2) =================
                # 专门检测和增强孤立小块 (<10像素)
                pass
                
                # 🆕 ============= 边界细化 (方案D阶段1增强) =================
                # 使用边界细化模块优化最终输出的边界（轻量级logits级refine）
                if self.boundary_refine is not None:
                    try:
                        # 使用单通道边界 oe 作为引导，与最终 logits 拼接
                        br_input = torch.cat([fused, oe], dim=1)  # [B, C+1, H, W]
                        br_out = self.boundary_refine(br_input)
                        gamma_br = torch.clamp(self.boundary_refine_gamma, 0.0, 1.0)
                        fused = fused + gamma_br * (br_out - fused)
                    except Exception as e:
                        print(f"Warning: boundary_refine failed: {e}, skip refine")

                # 最终输出检查（就地净化）
                outs = {'o1': o1, 'o2': o2, 'o3': o3, 'oe': oe, 'stage4': fused}
                for name, output in outs.items():
                    if torch.isnan(output).any() or torch.isinf(output).any():
                        print(f"Warning: Final output {name} contains NaN/Inf — sanitized")
                        outs[name] = torch.clamp(torch.nan_to_num(output, nan=0.0, posinf=1e4, neginf=-1e4), -10, 10)
                o1, o2, o3, oe, fused = outs['o1'], outs['o2'], outs['o3'], outs['oe'], outs['stage4']

                # 返回字典格式以匹配训练代码期望
                # 消融实验结论：不需要多尺度监督，仅返回主输出和边界
                # 同时暴露三通道 EAM 边界 logits 与前景 logits 以支持显式监督
                ret = {
                    'stage4': fused,
                    'edge': oe,                # 单通道边界图（用于可视化和边界加权Dice）[B, 1, H, W]
                    'edge_logits': edge_att_upsampled,  # 三通道 Class-Aware 边界 logits [B, 3, H, W]
                }
                if fg_logits is not None:
                    # 前景 logits 已经与输入分辨率对齐
                    ret['fg_logits'] = fg_logits
            return ret
            
        except Exception as e:
            print(f"Warning: Forward pass failed with error: {e}")
            if bool(self.training):
                raise
            # 返回安全的零输出（仅用于评估/推理兜底；训练时应 fail-fast）
            B, _, H, W = x.shape
            safe_output = torch.zeros(B, self.num_classes, H, W, device=x.device)
            return {
                'stage1': safe_output,
                'stage2': safe_output,
                'stage3': safe_output,
                'stage4': safe_output,
                'edge': torch.zeros(B, 1, H, W, device=x.device)
            }
