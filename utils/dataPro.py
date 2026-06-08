import os
import numpy as np
import torch
import torch.utils.data as data
# import torchvision.transforms as transforms  # 注释掉以避免导入错误
import nibabel as nib
from PIL import Image
import cv2
from sklearn.model_selection import train_test_split
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
import random
import torch.nn.functional as F
import time
from tqdm import tqdm
from skimage import morphology  # 添加必要导入 ✅
import matplotlib.pyplot as plt
import math
from net.Double_Otus import Seg
# 设置随机种子保证可重复性
random.seed(2021)
torch.manual_seed(2021)
np.random.seed(2021)
class BraTSDataset(data.Dataset):
    def __init__(self, t1_root, t2_root, t1ce_root, flair_root, seg_root, 
                trainsize=256, slice_range=None, mode="train", min_tumor_pixels=0, et_min_pixels=0, tc_min_pixels=35,
                edge_mode: str = 'gt'):
        self.trainsize = trainsize
        self.mode = mode
        self.slice_range = slice_range
        self.edge_mode = str(edge_mode)
        self.min_tumor_pixels = int(min_tumor_pixels) if min_tumor_pixels is not None else 0
        self.et_min_pixels = int(et_min_pixels) if et_min_pixels is not None else 0
        self.tc_min_pixels = int(tc_min_pixels) if tc_min_pixels is not None else 35
        self.status_cache_dir = "./dataset_cache"
        self.cache_version = "v2"  # 版本控制
        slice_str = f"{slice_range[0]}_{slice_range[1]}" if slice_range else "full"
        self.norm_slices_dir = os.path.join(
            self.status_cache_dir,
            f"norm_slices_{self.trainsize}_{slice_str}_{self.cache_version}"
        )
        os.makedirs(self.norm_slices_dir, exist_ok=True)
        self._pos_index_cache = {}
        self._raw_memmaps = {}
        self._norm_pos_index_cache = {}
        self._norm_memmaps = {}
        # 初始化路径并验证数据
        self.t1_files, self.t2_files, self.t1ce_files, self.flair_files, self.seg_files = self._validate_and_load_paths(
            t1_root, t2_root, t1ce_root, flair_root, seg_root
        )
        self.num_cases = len(self.t1_files)
        self.stats_path = os.path.join(self.status_cache_dir, "robust_stats.npz")
        # 构建有效样本索引
        self.samples = self._build_valid_samples()
        self._split_dataset()
        # 自动生成/加载缓存
        if mode == "train":
            # 训练模式优先检查缓存
            if os.path.exists(self.stats_path):
                print(f"检测到训练统计量缓存: {self.stats_path}")
                data = np.load(self.stats_path)
                self.global_stats = {
                    'mean': data['mean'],
                    'std': data['std'],
                    'p2': data['p2'] if 'p2' in data.files else None,
                    'p50': data['p50'] if 'p50' in data.files else None,
                    'p98': data['p98'] if 'p98' in data.files else None
                }
                # 若缺少稳健百分位统计，则重新计算一次
                if self.global_stats['p2'] is None or self.global_stats['p50'] is None or self.global_stats['p98'] is None:
                    print("缓存缺少稳健百分位统计，重新计算...")
                    self.global_stats = self._compute_robust_stats()
            else:
                print("未找到统计量缓存，开始重新计算...")
                self.global_stats = self._compute_robust_stats()
        else:
            # 验证/测试模式保持原有逻辑
            print(f"加载训练统计量: {self.stats_path}")
            if os.path.exists(self.stats_path):
                data = np.load(self.stats_path)
                print(data['mean'], data['std'])
                self.global_stats = {
                    'mean': data['mean'],
                    'std': data['std'],
                    'p2': data['p2'] if 'p2' in data.files else None,
                    'p50': data['p50'] if 'p50' in data.files else None,
                    'p98': data['p98'] if 'p98' in data.files else None
                }
            else:
                raise FileNotFoundError("训练统计量未生成，请先运行训练模式")
            
        print(f"{mode}数据集初始化完成，共{len(self)}个有效样本 | min_tumor_pixels={self.min_tumor_pixels} | et_min_pixels={self.et_min_pixels} | tc_min_pixels={self.tc_min_pixels}")
        # 归一化与增强策略（先初始化归一化策略，再生成规范化缓存）
        self.use_per_case_norm = True  # 每病例3D均值/方差归一化
        self.use_robust_norm = True    # 全局稳健归一化（回退）
        self.use_slice_rescale = True  # 切片级重标定到全局p2/p98
        # 每病例统计缓存
        self.case_stats_dir = os.path.join(self.status_cache_dir, "case_stats_v1")
        os.makedirs(self.case_stats_dir, exist_ok=True)
        self._case_stats_cache = {}
        # 数据增强（仅训练）- 适度增强以防止过拟合
        self.enable_aug = (self.mode == "train")
        self.aug_prob = 0.70  # 降低到0.70 (0.80过强导致训练不稳定)
        self.noise_std = 0.08  # 从0.05提高到0.08 (适度增加)
        self.gamma_range = (0.8, 1.25)  # 从(0.9, 1.1)扩大到(0.8, 1.25)
        self.brightness = 0.15  # 从0.1提高到0.15 (适度增加)
        # 先生成/补齐规范化切片缓存
        self._precompute_norm_cache()
        # 在初始化时增加缓存完整性检查（规范化缓存）
        if not self._validate_cache_files():
            raise RuntimeError("缓存文件不完整，请重新生成")
        
    def __len__(self):
        return len(self.samples)
    def _validate_cache_files(self):
        """验证规范化病例缓存完整性"""
        for case_idx in set(s[0] for s in self.samples):
            path = os.path.join(self.norm_slices_dir, f"norm_case_{case_idx}.npy")
            if not os.path.exists(path):
                return False
            try:
                data = np.load(path)
                if len(data) == 0:
                    return False
            except Exception as e:
                print(f"缓存文件损坏: {path}, 错误: {str(e)}")
                return False
        return True
    def _precompute_cache(self):
        pass

    def _precompute_norm_cache(self):
        os.makedirs(self.norm_slices_dir, exist_ok=True)
        case_indices = list(set(s[0] for s in self.samples))
        for case_idx in case_indices:
            norm_case_path = os.path.join(self.norm_slices_dir, f"norm_case_{case_idx}.npy")
            if os.path.exists(norm_case_path):
                continue
            slice_indices = [s[1] for s in self.samples if s[0] == case_idx]
            norm_entries = []
            for slice_idx in slice_indices:
                img_t, masks_t, edges_t = self._process_slice(case_idx, slice_idx)
                norm_entries.append(
                    (
                        np.int32(slice_idx),
                        img_t.numpy().astype(np.float16),
                        (masks_t.numpy() * 1.0).astype(np.uint8),
                        (edges_t.numpy() * 1.0).astype(np.uint8),
                    )
                )
            norm_dtype = [
                ('slice_idx', np.int32),
                ('image', np.float16, (4, self.trainsize, self.trainsize)),
                ('mask', np.uint8, (3, self.trainsize, self.trainsize)),
                ('edge', np.uint8, (3, self.trainsize, self.trainsize)),
            ]
            norm_struct = np.array(norm_entries, dtype=norm_dtype)
            norm_struct = np.require(norm_struct, requirements=['W'])
            np.save(norm_case_path, norm_struct)

    def _save_case_cache(self, case_idx):
        pass

    def _get_case_stats(self, case_idx, t1=None, t2=None, t1ce=None, flair=None, seg=None):
        """获取或计算单病例四模态 μ/σ（整3D，基于脑区mask）。"""
        # 健壮性防护：确保缓存字典存在（兼容多线程/热重载场景）
        if not hasattr(self, '_case_stats_cache') or self._case_stats_cache is None:
            self._case_stats_cache = {}
        # 健壮性防护：确保统计目录存在（兼容并发/热重载场景）
        if not hasattr(self, 'case_stats_dir') or self.case_stats_dir is None:
            if not hasattr(self, 'status_cache_dir') or self.status_cache_dir is None:
                self.status_cache_dir = "./dataset_cache"
            self.case_stats_dir = os.path.join(self.status_cache_dir, "case_stats_v1")
            os.makedirs(self.case_stats_dir, exist_ok=True)
        if case_idx in self._case_stats_cache:
            return self._case_stats_cache[case_idx]
        stats_file = os.path.join(self.case_stats_dir, f"case_{case_idx}_stats.npz")
        if os.path.exists(stats_file):
            try:
                data = np.load(stats_file)
                stats = {
                    'mean': data['mean'].astype(np.float32),
                    'std': data['std'].astype(np.float32)
                }
                self._case_stats_cache[case_idx] = stats
                return stats
            except Exception:
                pass
        # 需要计算
        if t1 is None:
            t1, t2, t1ce, flair, seg = self._load_images(case_idx)
        try:
            t1_np = np.asarray(t1, dtype=np.float32)
            t2_np = np.asarray(t2, dtype=np.float32)
            t1ce_np = np.asarray(t1ce, dtype=np.float32)
            flair_np = np.asarray(flair, dtype=np.float32)
            seg_np = np.asarray(seg)
            brain_mask = (t1_np > 0) | (t2_np > 0) | (t1ce_np > 0) | (flair_np > 0) | (seg_np > 0)
            means = np.zeros(4, dtype=np.float32)
            stds = np.zeros(4, dtype=np.float32)
            for i, vol in enumerate([t1_np, t1ce_np, t2_np, flair_np]):
                vals = vol[brain_mask]
                if vals.size < 100:
                    means[i] = float(self.global_stats['mean'][i])
                    stds[i] = float(self.global_stats['std'][i])
                else:
                    means[i] = float(np.mean(vals))
                    stds[i] = float(np.std(vals) + 1e-6)
            stats = {'mean': means, 'std': stds}
            try:
                np.savez(stats_file, mean=means, std=stds)
            except Exception:
                pass
            self._case_stats_cache[case_idx] = stats
            return stats
        except Exception as e:
            print(f"病例 {case_idx} 统计量计算失败: {str(e)}")
            return None

    def _augment(self, img: torch.Tensor, masks: torch.Tensor, edges: torch.Tensor):
        """几何+强度数据增强（仅训练）。
        - 几何：单次仿射（旋转±12°、缩放0.80–1.20、随机水平/垂直翻转）
          图像使用 bilinear；标签与边缘使用 nearest；同一仿射同步作用。
        - 强度：Gamma/亮度抖动 + 高斯噪声。
        """
        if random.random() < self.aug_prob:
            # 采样仿射参数（单次采样，同步应用）
            angle_deg = random.uniform(-12.0, 12.0)
            angle_rad = angle_deg * math.pi / 180.0
            scale = random.uniform(0.80, 1.20)
            flip_x = -1.0 if random.random() < 0.5 else 1.0
            flip_y = -1.0 if random.random() < 0.5 else 1.0

            s_x = scale * flip_x
            s_y = scale * flip_y
            cos_a = math.cos(angle_rad)
            sin_a = math.sin(angle_rad)

            # 组合旋转与缩放（含翻转）: R @ S -> 2x3 仿射矩阵
            a = cos_a * s_x
            b = -sin_a * s_y
            c_ = sin_a * s_x
            d = cos_a * s_y
            theta = torch.tensor([[a, b, 0.0], [c_, d, 0.0]], dtype=img.dtype, device=img.device).unsqueeze(0)

            # 生成采样网格（对齐到角点关闭，与项目插值保持一致）
            _, H, W = img.shape
            grid = F.affine_grid(theta, size=(1, img.shape[0], H, W), align_corners=False)

            # 应用到图像（bilinear）与标签/边缘（nearest）
            img = F.grid_sample(img.unsqueeze(0), grid, mode='bilinear', padding_mode='zeros', align_corners=False).squeeze(0)
            masks = F.grid_sample(masks.unsqueeze(0), grid, mode='nearest', padding_mode='zeros', align_corners=False).squeeze(0)
            edges = F.grid_sample(edges.unsqueeze(0), grid, mode='nearest', padding_mode='zeros', align_corners=False).squeeze(0)

            # 强度Gamma/亮度抖动（逐通道）
            c = img.shape[0]
            for i in range(c):
                g = random.uniform(self.gamma_range[0], self.gamma_range[1])
                b = random.uniform(-self.brightness, self.brightness)
                ch = img[i]
                x = torch.clamp((ch + 3.0) / 6.0, 0.0, 1.0)
                x = torch.clamp(x ** g + b, 0.0, 1.0)
                img[i] = x * 6.0 - 3.0

            # 高斯噪声（轻度）
            if self.noise_std > 0:
                noise = torch.randn_like(img) * self.noise_std
                img = torch.clamp(img + noise, -3.5, 3.5)

        return img, masks, edges
    def _check_cache(self):
        return True

    def _build_cache_index(self):
        return []
    def __getitem__(self, index):
        max_retries = 3
        for _ in range(max_retries):
            try:
                case_idx, slice_idx = self.samples[index]
                img_tensor, masks, edges = self._process_slice(case_idx, slice_idx)
                if self.enable_aug:
                    img_tensor, masks, edges = self._augment(img_tensor, masks, edges)
                if self.edge_mode == 'otus':
                    otus_edge = Seg(img_tensor)  # [1,H,W]
                    if otus_edge.dim() == 2:
                        otus_edge = otus_edge.unsqueeze(0)
                    otus_edge = torch.clamp(otus_edge.float(), 0.0, 1.0)
                    edges = otus_edge
                return img_tensor, masks, edges
            except Exception as e:
                print(f"加载样本{index}失败: {str(e)}")
                index = random.randint(0, len(self)-1)  # 随机选择新索引
        
        # 多次重试失败后返回空数据（需在collate_fn中处理）
        return (
            torch.zeros(4, self.trainsize, self.trainsize),
            torch.zeros(3, self.trainsize, self.trainsize),
            torch.zeros(1 if self.edge_mode == 'otus' else 3, self.trainsize, self.trainsize)
        )

    def _validate_and_load_paths(self, *roots):
        """验证并加载有效病例路径（添加.gz后缀）"""
        t1_root, t2_root, t1ce_root, flair_root, seg_root = roots

        def _normalize_root(root, suffix):
            root = os.path.normpath(str(root))
            base = os.path.basename(root)
            if base.startswith(("BraTS19", "BraTS20")) and os.path.isdir(root):
                for ext in (".nii.gz", ".nii"):
                    if os.path.exists(os.path.join(root, f"{base}_{suffix}{ext}")):
                        return os.path.dirname(root)
            return root

        t1_root = _normalize_root(t1_root, "t1")
        t2_root = _normalize_root(t2_root, "t2")
        t1ce_root = _normalize_root(t1ce_root, "t1ce")
        flair_root = _normalize_root(flair_root, "flair")
        seg_root = _normalize_root(seg_root, "seg")
        
        case_dirs = [
            d for d in os.listdir(t1_root)
            if os.path.isdir(os.path.join(t1_root, d)) and (d.startswith("BraTS19") or d.startswith("BraTS20"))
        ]
        valid_cases = []
        for case_dir in case_dirs:
            case_id = case_dir

            def _pick_path(root, suffix):
                base = os.path.join(root, case_dir, f"{case_id}_{suffix}")
                for ext in (".nii.gz", ".nii"):
                    p = base + ext
                    if os.path.exists(p):
                        return p
                return base + ".nii"

            paths = {
                't1': _pick_path(t1_root, "t1"),
                't2': _pick_path(t2_root, "t2"),
                't1ce': _pick_path(t1ce_root, "t1ce"),
                'flair': _pick_path(flair_root, "flair"),
                'seg': _pick_path(seg_root, "seg")
            }

            if all(os.path.exists(p) for p in paths.values()):
                valid_cases.append(paths)
            else:
                missing = [k for k, v in paths.items() if not os.path.exists(v)]
                print(f"病例{case_id}缺失文件: {missing}")
        
        return (
            [c['t1'] for c in valid_cases],
            [c['t2'] for c in valid_cases],
            [c['t1ce'] for c in valid_cases],
            [c['flair'] for c in valid_cases],
            [c['seg'] for c in valid_cases]
        )

    @lru_cache(maxsize=32)  # 大幅增加缓存大小，减少数据分布不一致
    def _load_images(self, case_idx):
        """带内存缓存的数据加载"""
        try:
            t1 = nib.load(self.t1_files[case_idx], mmap=True).dataobj
            t2 = nib.load(self.t2_files[case_idx], mmap=True).dataobj
            t1ce = nib.load(self.t1ce_files[case_idx], mmap=True).dataobj
            flair = nib.load(self.flair_files[case_idx], mmap=True).dataobj
            seg = nib.load(self.seg_files[case_idx], mmap=True).dataobj
            return (t1, t2, t1ce, flair, seg)
        except Exception as e:
            print(f"加载病例{case_idx}失败: {str(e)}")
            raise
    def _load_seg_proxy(self, case_idx):
        """仅加载Seg代理，避免不必要的多模态占用。"""
        try:
            return nib.load(self.seg_files[case_idx], mmap=True).dataobj
        except Exception as e:
            print(f"加载分割{case_idx}失败: {str(e)}")
            raise
    def _compute_robust_stats(self):
        """稳健统计计算：全局 p2/p50/p98 + mean/std（仅脑区）。"""
        print("⚡ 开始计算统计量...")
        
        # 采样配置
        config = {
            'sample_ratio': 0.3,
            'slices_per_case': 10,
        }

        # 初始化统计量存储
        sum_ = np.zeros(4)
        sum_sq = np.zeros(4)
        count = np.zeros(4)
        all_vals = [list() for _ in range(4)]

        # 病例抽样
        case_indices = np.random.choice(
            len(self.t1_files),
            size=int(len(self.t1_files)*config['sample_ratio']),
            replace=False
        )
        
        for case_idx in tqdm(case_indices, desc="统计计算进度"):
            try:
                result = self._process_single_case(case_idx, config)
                if result is None:
                    continue

                if isinstance(result, tuple) and len(result) == 4:
                    local_sum, local_sum_sq, local_count, sample_vals = result
                else:
                    local_sum, local_sum_sq, local_count = result
                    sample_vals = None
                sum_ += local_sum
                sum_sq += local_sum_sq
                count += local_count
                # 累积采样值用于百分位估计
                if sample_vals is not None:
                    for i in range(4):
                        try:
                            sv = sample_vals[i]
                            if isinstance(sv, np.ndarray) and sv.size > 0:
                                all_vals[i].append(sv)
                        except Exception:
                            pass
            except Exception as e:
                print(f"病例{case_idx}处理失败: {str(e)}")
                continue

        # 数值稳定的统计量计算
        mean = np.zeros(4)
        std = np.zeros(4)
        p2 = np.zeros(4)
        p50 = np.zeros(4)
        p98 = np.zeros(4)
        
        for i in range(4):
            if count[i] > 0:
                mean[i] = sum_[i] / count[i]
                variance = max(0, sum_sq[i] / count[i] - mean[i]**2)
                std[i] = np.sqrt(variance)
            
            # 更严格的预设值规则
            if count[i] < 1000 or std[i] < 1e-6:
                # 使用更保守的经验值
                safe_stats = [
                    (400.0, 200.0),  # T1
                    (500.0, 300.0),  # T1ce  
                    (1000.0, 500.0), # T2
                    (600.0, 250.0)   # FLAIR
                ]
                mean[i], std[i] = safe_stats[i]
                print(f"模态{i}使用安全经验值: μ={mean[i]:.1f}±{std[i]:.1f}")
            
            # 确保std不会太小
            std[i] = max(std[i], 1e-6)
            
            # 额外的数值稳定性检查
            if np.isnan(mean[i]) or np.isnan(std[i]) or np.isinf(mean[i]) or np.isinf(std[i]):
                mean[i], std[i] = safe_stats[i]
                print(f"检测到NaN/Inf，模态{i}使用安全默认值: μ={mean[i]:.1f}±{std[i]:.1f}")
            # 百分位估计（使用采样）
            if len(all_vals[i]) > 0:
                vals = np.concatenate(all_vals[i])
                p2[i], p50[i], p98[i] = np.percentile(vals, [2, 50, 98])
            else:
                p2[i], p50[i], p98[i] = mean[i] - 2*std[i], mean[i], mean[i] + 2*std[i]

        print(f"最终统计量:\nμ: {np.round(mean)}\nσ: {np.round(std)}\nrobust p2/p50/p98: {np.round(p2)}, {np.round(p50)}, {np.round(p98)}")
        np.savez(self.stats_path, mean=mean, std=std, p2=p2, p50=p50, p98=p98)
        print(f"统计量已保存至 {self.stats_path}")
        return {'mean': mean.astype(np.float32), 'std': std.astype(np.float32), 'p2': p2.astype(np.float32), 'p50': p50.astype(np.float32), 'p98': p98.astype(np.float32)}

    def _process_single_case(self, case_idx, config):
        """简化的单个病例处理逻辑"""
        try:
            t1, t2, t1ce, flair, seg = self._load_images(case_idx)
            
            # 使用标签维度和 slice_range 限定候选切片，避免在Proxy上做全体比较
            min_slice, max_slice = self._get_valid_range(seg)
            candidate_slices = np.arange(min_slice, max_slice + 1)
            if candidate_slices.size == 0:
                return None

            # 简单抽样
            n_select = min(len(candidate_slices), config['slices_per_case'])
            selected_slices = np.random.choice(candidate_slices, n_select, replace=False)

            # 切片处理
            local_sum = np.zeros(4)
            local_sum_sq = np.zeros(4)
            local_count = np.zeros(4)
            sample_vals = [list() for _ in range(4)]
            
            for slice_idx in selected_slices:
                # 将当前切片转换为numpy数组
                t1_s = np.asarray(t1[..., slice_idx])
                t1ce_s = np.asarray(t1ce[..., slice_idx])
                t2_s = np.asarray(t2[..., slice_idx])
                flair_s = np.asarray(flair[..., slice_idx])
                seg_s = np.asarray(seg[..., slice_idx])

                modalities = [t1_s, t1ce_s, t2_s, flair_s]

                # 基于单切片的轻量脑区mask
                slice_mask = (t1_s > 50) | (t2_s > 300) | (flair_s > 200) | (seg_s > 0)
                if slice_mask.sum() <= 100:
                    continue

                for mod_idx in range(4):
                    data = modalities[mod_idx][slice_mask]
                    if data.size < 50: 
                        continue
                    
                    # 简单统计
                    local_sum[mod_idx] += data.sum()
                    local_sum_sq[mod_idx] += (data**2).sum()
                    local_count[mod_idx] += data.size
                    # 收集少量样本用于百分位估计
                    sample_n = min(2000, data.size)
                    if sample_n > 0:
                        choice = np.random.choice(data.size, sample_n, replace=False)
                        sample_vals[mod_idx].append(data[choice])

            # 合并为每模态一个数组，避免上层二次拼接过多小数组
            for i in range(4):
                if len(sample_vals[i]) > 0:
                    sample_vals[i] = np.concatenate(sample_vals[i])
                else:
                    sample_vals[i] = np.array([], dtype=np.float32)
            return (local_sum, local_sum_sq, local_count, sample_vals)
            
        except Exception as e:
            print(f"处理病例{case_idx}出错: {str(e)}")
            return None

    def _dynamic_clip(self, data, min_clip, max_clip):
        """动态截断方法（保留98%数据）"""
        p_low, p_high = np.percentile(data, [2, 98])
        return np.clip(data, 
                    max(min_clip, p_low), 
                    min(max_clip, p_high))

    def _robust_filter(self, data, mod_idx):
        """稳健数据过滤（模态自适应）"""
        q1, q3 = np.percentile(data, [25, 75])
        iqr = q3 - q1
        
        # 不同模态设置不同过滤范围
        thresholds = {
            0: 1.5,  # T1
            1: 2.0,  # T1ce（允许更宽范围）
            2: 2.5,  # T2（保留更多高强度）
            3: 1.8   # FLAIR
        }
        return data[(data > q1 - thresholds[mod_idx]*iqr) & 
                (data < q3 + thresholds[mod_idx]*iqr)]
        
    def _get_valid_range(self, seg):
        """获取有效切片范围"""
        min_slice = 0
        max_slice = seg.shape[2] - 1
        if self.slice_range:
            min_slice = max(self.slice_range[0], 0)
            max_slice = min(self.slice_range[1], max_slice)
        return min_slice, max_slice

    def _is_valid_slice_from_seg(self, seg_proxy, slice_idx):
        """基于分割代理判断切片是否有效（仅按标签统计，低内存）。"""
        seg_slice = np.asarray(seg_proxy[..., slice_idx])
        brain_area_ratio = np.sum(seg_slice > 0) / (seg_slice.size)
        if brain_area_ratio < 0.01:  
            return False

        tumor_pixels = np.count_nonzero((seg_slice == 1) | (seg_slice == 2) | (seg_slice == 4))
        if self.min_tumor_pixels > 0 and tumor_pixels < self.min_tumor_pixels:
            return False

        tc_pixels = np.count_nonzero((seg_slice == 1) | (seg_slice == 4))
        if tc_pixels > 0 and tc_pixels <= self.tc_min_pixels:
            return False

        et_pixels = np.count_nonzero(seg_slice == 4)
        if et_pixels > 0 and self.et_min_pixels > 0 and et_pixels <= self.et_min_pixels:
            return False
        return (
            (seg_slice.ndim == 2) and
            (seg_slice.min() >= 0) and (seg_slice.max() <= 4) and
            (np.any(seg_slice > 0))
        )

    def _build_valid_samples(self):
        """构建有效样本索引"""
        samples = []
        print(f"\n开始构建样本索引(slice_range: {self.slice_range})...")
        
        for case_idx in range(self.num_cases):
            seg_proxy = self._load_seg_proxy(case_idx)
            min_slice, max_slice = self._get_valid_range(seg_proxy)
            
            # 进度提示
            if case_idx % 10 == 0:
                print(f"处理病例 {case_idx+1}/{self.num_cases}，切片范围[{min_slice}-{max_slice}]")
            
            for slice_idx in range(min_slice, max_slice + 1):
                if self._is_valid_slice_from_seg(seg_proxy, slice_idx):
                    samples.append((case_idx, slice_idx))
        
        print(f"共找到 {len(samples)} 个有效切片")
        return samples

    def _split_dataset(self):
        """病例级数据集划分（训练:验证:测试 = 70%:15%:15%）"""
        case_ids = list(set([s[0] for s in self.samples]))
        
        # 第一次划分：训练+临时（验证+测试）
        train_ids, temp_ids = train_test_split(
            case_ids, 
            test_size=0.3, 
            random_state=42
        )
        
        # 第二次划分：验证和测试
        val_ids, test_ids = train_test_split(
            temp_ids, 
            test_size=0.5,  # 0.3 * 0.5=0.15
            random_state=42
        )
        
        # 根据模式筛选样本
        if self.mode == "train":
            keep_ids = train_ids
        elif self.mode == "val":
            keep_ids = val_ids
        elif self.mode == "test":
            keep_ids = test_ids
        else:
            raise ValueError(f"Invalid mode: {self.mode}")
        
        self.samples = [s for s in self.samples if s[0] in keep_ids]

    def _process_slice(self, case_idx, slice_idx):
        norm = self._load_from_norm_case(case_idx, slice_idx)
        if norm is not None:
            image_tensor, masks_raw, edges_raw = norm
            masks = masks_raw.float()
            edges = edges_raw.float()
            return (
                torch.clamp(image_tensor.float(), -3.0, 3.0),
                torch.clamp(masks, 0.0, 1.0),
                torch.clamp(edges, 0.0, 1.0)
            )
        # 规范化缓存缺失时，直接从原始 NIfTI 计算
        t1, t2, t1ce, flair, seg = self._load_images(case_idx)
        slices = {
            't1': np.asarray(t1[..., slice_idx], dtype=np.float32),
            't1ce': np.asarray(t1ce[..., slice_idx], dtype=np.float32),
            't2': np.asarray(t2[..., slice_idx], dtype=np.float32),
            'flair': np.asarray(flair[..., slice_idx], dtype=np.float32)
        }
        four_channel = np.stack([slices[k] for k in ['t1', 't1ce', 't2', 'flair']], axis=0)
        four_channel = self._resize_volume(four_channel, (self.trainsize, self.trainsize))
        seg_slice = np.asarray(seg[..., slice_idx])
        resized_seg = cv2.resize(seg_slice.astype(np.uint8), (self.trainsize, self.trainsize), interpolation=cv2.INTER_NEAREST)
        wt, tc, et = self._create_masks(resized_seg)
        edges_np = self._create_edges(wt, tc, et)
        # 归一化（与原逻辑一致）
        image_tensor = torch.from_numpy(four_channel).float()
        normalized_channels = []
        case_stats = self._get_case_stats(case_idx) if self.use_per_case_norm else None
        for i in range(4):
            if self.use_per_case_norm and (case_stats is not None):
                mean = float(case_stats['mean'][i])
                std = float(case_stats['std'][i])
            else:
                mean = float(self.global_stats['mean'][i])
                std = float(self.global_stats['std'][i])
            p2 = float(self.global_stats.get('p2', [np.nan]*4)[i]) if isinstance(self.global_stats.get('p2', None), np.ndarray) else np.nan
            p50 = float(self.global_stats.get('p50', [np.nan]*4)[i]) if isinstance(self.global_stats.get('p50', None), np.ndarray) else np.nan
            p98 = float(self.global_stats.get('p98', [np.nan]*4)[i]) if isinstance(self.global_stats.get('p98', None), np.ndarray) else np.nan
            std = max(std, 1e-6)
            ch = image_tensor[i]
            if self.use_per_case_norm and (std >= 1e-6) and (not np.isnan(mean)) and (not np.isnan(std)):
                normalized = (ch - mean) / std
            elif self.use_robust_norm and not (np.isnan(p2) or np.isnan(p50) or np.isnan(p98)):
                if self.use_slice_rescale:
                    ch_np = ch.cpu().numpy()
                    brain_mask = ch_np != 0
                    if brain_mask.any():
                        p2_l, p98_l = np.percentile(ch_np[brain_mask], [2, 98])
                        scale = (p98 - p2) / max(1e-6, (p98_l - p2_l))
                        shift = p50 - (p2_l + p98_l) / 2.0 * scale
                        ch = (ch * scale + shift)
                denom = max(1e-6, (p98 - p2) / 2.0)
                normalized = (ch - p50) / denom
            else:
                normalized = (ch - mean) / std
            normalized_channels.append(torch.clamp(normalized, -3.0, 3.0))
        image_tensor = torch.stack(normalized_channels, dim=0)
        masks = torch.from_numpy(np.stack([wt, tc, et], axis=0)).float()
        edges = torch.from_numpy(np.stack(edges_np, axis=0)).float()
        return (
            torch.clamp(image_tensor, -3.0, 3.0),
            torch.clamp(masks, 0.0, 1.0),
            torch.clamp(edges, 0.0, 1.0)
        )


    def visualize_pre_post_normalization(self, case_idx: int, slice_idx: int, save_dir: str = None):
        """将归一化前后的四模态切片可视化并保存为单张图片。

        - 归一化前: 使用缓存中的原始强度（已resize，但未做标准化）
        - 归一化后: 使用 _process_slice 的输出（[-3,3] clamp 后）
        保存路径: {BGNet/norm_visuals}/case{case_idx}_slice{slice_idx}.png（或自定义 save_dir）
        """
        if save_dir is None:
            repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
            save_dir = os.path.join(repo_root, "norm_visuals")
        os.makedirs(save_dir, exist_ok=True)
        # 原始（预归一化）
        raw = self._load_from_cache(case_idx, slice_idx)['image'].numpy()  # [4,H,W], float32
        # 归一化后
        norm_img, masks, edges = self._process_slice(case_idx, slice_idx)
        norm_img = norm_img.numpy()  # [4,H,W]

        # 可视化配置
        titles = ['T1', 'T1ce', 'T2', 'FLAIR']

        def scale_uint8(img2d: np.ndarray, method: str = 'percentile') -> np.ndarray:
            if method == 'percentile':
                p1, p99 = np.percentile(img2d, [1, 99])
                if p99 - p1 < 1e-6:
                    return np.zeros_like(img2d, dtype=np.uint8)
                x = np.clip((img2d - p1) / (p99 - p1), 0, 1)
                return (x * 255).astype(np.uint8)
            else:
                mn, mx = float(img2d.min()), float(img2d.max())
                if mx - mn < 1e-6:
                    return np.zeros_like(img2d, dtype=np.uint8)
                x = (img2d - mn) / (mx - mn)
                return (x * 255).astype(np.uint8)

        def scale_uint8_norm(img2d: np.ndarray) -> np.ndarray:
            # 归一化后范围约[-3,3]
            x = (np.clip(img2d, -3.0, 3.0) + 3.0) / 6.0
            return (x * 255).astype(np.uint8)

        # 画布: 2 行（pre/post）× 4 模态
        fig, axes = plt.subplots(2, 4, figsize=(12, 6))
        for i in range(4):
            pre_img = scale_uint8(raw[i], method='percentile')
            post_img = scale_uint8_norm(norm_img[i])
            axes[0, i].imshow(pre_img, cmap='gray')
            axes[0, i].set_title(f"{titles[i]} (Pre)")
            axes[0, i].axis('off')
            axes[1, i].imshow(post_img, cmap='gray')
            axes[1, i].set_title(f"{titles[i]} (Post)")
            axes[1, i].axis('off')
        fig.suptitle(f"Case {case_idx} | Slice {slice_idx}")
        plt.tight_layout()
        out_path = os.path.join(save_dir, f"case{case_idx}_slice{slice_idx}.png")
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        return out_path

    def visualize_pre_post_batch(self, save_dir: str = None, max_samples: int = 20):
        """批量保存前后归一化可视化图（从dataset.samples的前 max_samples 个）。"""
        if save_dir is None:
            repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
            save_dir = os.path.join(repo_root, "norm_visuals")
        os.makedirs(save_dir, exist_ok=True)
        num = min(max_samples, len(self.samples))
        for idx in range(num):
            case_idx, slice_idx = self.samples[idx]
            try:
                self.visualize_pre_post_normalization(case_idx, slice_idx, save_dir)
            except Exception as e:
                print(f"可视化失败 case={case_idx} slice={slice_idx}: {str(e)}")


    def _load_from_cache(self, case_idx, slice_idx):
        return None

    def _load_from_norm_case(self, case_idx, slice_idx):
        path = os.path.join(self.norm_slices_dir, f"norm_case_{case_idx}.npy")
        if not os.path.exists(path):
            return None
        try:
            arr = self._norm_memmaps.get(case_idx)
            if arr is None:
                arr = np.load(path, mmap_mode='r')
                self._norm_memmaps[case_idx] = arr
                self._norm_pos_index_cache[case_idx] = {int(s): i for i, s in enumerate(arr['slice_idx'])}
            pos_map = self._norm_pos_index_cache.get(case_idx)
            if pos_map is None:
                pos_map = {int(s): i for i, s in enumerate(arr['slice_idx'])}
                self._norm_pos_index_cache[case_idx] = pos_map
            pos = pos_map.get(int(slice_idx))
            if pos is None:
                idxs = np.where(arr['slice_idx'] == np.int32(slice_idx))[0]
                if idxs.size == 0:
                    return None
                pos = int(idxs[0])
                self._norm_pos_index_cache[case_idx][int(slice_idx)] = pos
            image = torch.as_tensor(arr['image'][pos].copy()).float()
            mask = torch.as_tensor(arr['mask'][pos].copy())
            edge = torch.as_tensor(arr['edge'][pos].copy())
            return image, mask, edge
        except Exception as e:
            print(f"读取规范化病例缓存失败 case={case_idx}: {e}")
            return None
    
    def _resize_edge(self, edge):
        """边缘图调整尺寸"""
        if isinstance(edge, torch.Tensor):
            edge = edge.numpy()
        
        resized = cv2.resize(
            edge.astype(np.float32),
            (self.trainsize, self.trainsize),
            interpolation=cv2.INTER_NEAREST
        )
        return torch.from_numpy(resized).float()
    
    def _normalize(self, data, modality):
        """增强鲁棒性的归一化"""
        idx = ['t1','t1ce','t2','flair'].index(modality)
        mean = self.global_stats['mean'][idx]
        std = self.global_stats['std'][idx]
        
        # 更严格的数值稳定性处理
        if np.isnan(mean) or np.isnan(std) or np.isinf(mean) or np.isinf(std) or std < 1e-6:
            # 使用模态特定的安全默认值
            safe_defaults = {
                't1': (400.0, 200.0),
                't1ce': (500.0, 300.0), 
                't2': (1000.0, 500.0),
                'flair': (600.0, 250.0)
            }
            mean, std = safe_defaults[modality]
            print(f"警告：{modality}使用安全默认统计量 μ={mean:.1f}, σ={std:.1f}")
        
        # 确保std不会太小
        std = max(std, 1e-6)
        
        # 输入数据的数值稳定性检查
        if np.any(np.isnan(data)) or np.any(np.isinf(data)):
            print(f"警告：{modality}数据包含NaN/Inf，进行清理")
            data = np.nan_to_num(data, nan=0.0, posinf=1000.0, neginf=-1000.0)
        
        # 数值稳定的归一化
        try:
            normalized = (data - mean) / std
        except (ZeroDivisionError, FloatingPointError):
            print(f"警告：{modality}归一化失败，使用安全除法")
            normalized = np.divide(data - mean, std, 
                                  out=np.zeros_like(data), 
                                  where=(std != 0))
        
        # 更保守的clamp范围
        normalized = np.clip(normalized, -3.0, 3.0)
        
        # 最终的数值稳定性检查
        if np.any(np.isnan(normalized)) or np.any(np.isinf(normalized)):
            print(f"警告：{modality}归一化结果异常，使用零填充")
            normalized = np.nan_to_num(normalized, nan=0.0, posinf=3.0, neginf=-3.0)
        
        return normalized.astype(np.float32)
    def _resize_volume(self, data, size):
        """体积数据调整尺寸"""
        # 确保为 float32，避免 Short 等整型导致的插值错误
        np_data = np.asarray(data, dtype=np.float32)
        tensor = torch.from_numpy(np_data).unsqueeze(0)  # [1, C, H, W]
        out = F.interpolate(
            tensor,
            size=size,
            mode='bilinear',
            align_corners=False
        ).squeeze(0)
        return out.detach().cpu().numpy().astype(np.float32)
    def _create_masks(self, seg):
        # 使用正确的标签定义
        wt = np.isin(seg, [1, 2, 4])  # 全肿瘤区域
        tc = np.isin(seg, [1, 4])     # 肿瘤核心
        et = (seg == 4)               # 增强肿瘤
        
        # 确保输出是二值标签
        return [
            wt.astype(np.float32),
            tc.astype(np.float32),
            et.astype(np.float32)
        ]
    
    def _resize_mask(self, mask):
        """保持标签完整性的调整尺寸"""
        if mask.size == 0:
            return torch.zeros((self.trainsize, self.trainsize), dtype=torch.long)
        
        # 确保输入是numpy数组
        if isinstance(mask, torch.Tensor):
            mask = mask.numpy()
        
        resized = cv2.resize(
            mask.astype(np.uint8), 
            (self.trainsize, self.trainsize),
            interpolation=cv2.INTER_NEAREST
        )
        return torch.from_numpy(resized).float()
    
    def _create_edges(self, *masks):
        """稳定的边缘生成"""
        edges = []
        for mask in masks:
            try:
                # 强制转换为合法OpenCV输入
                mask = np.ascontiguousarray(np.squeeze(mask).astype(np.uint8))
                
                # 空掩膜处理
                if mask.ndim != 2 or np.max(mask) == 0:
                    edges.append(np.zeros((self.trainsize, self.trainsize), dtype=np.float32))
                    continue
                    
                # 轮廓提取
                contours, _ = cv2.findContours(
                    mask,
                    cv2.RETR_TREE,
                    cv2.CHAIN_APPROX_SIMPLE
                )
                
                # 绘制轮廓
                edge = np.zeros_like(mask, dtype=np.uint8)
                if contours:
                    cv2.drawContours(edge, contours, -1, 255, 2)
                
                # 调整尺寸并归一化
                edge = cv2.resize(edge, (self.trainsize, self.trainsize),
                        interpolation=cv2.INTER_NEAREST)
                edges.append(edge)
            except Exception as e:
                print(f"边缘生成失败: {str(e)}")
                edges.append(np.zeros((self.trainsize, self.trainsize), dtype=np.float32))
        return edges

def visualize_cache_sample(case_idx=0, slice_idx=70, save_dir="./cache_visual"):
    """
    可视化缓存中的单个样本
    case_idx: 病例索引
    slice_idx: 切片索引
    save_dir: 可视化结果保存路径
    """
    os.makedirs(save_dir, exist_ok=True)
    
    path = os.path.join("./dataset_cache", f"norm_slices_256_full_v2", f"norm_case_{case_idx}.npy")
    if not os.path.exists(path):
        print(f"规范化缓存不存在: {path}")
        return
    arr = np.load(path, mmap_mode='r')
    idxs = np.where(arr['slice_idx'] == np.int32(slice_idx))[0]
    if idxs.size == 0:
        print(f"Case {case_idx} 中未找到切片 {slice_idx}")
        return
    pos = int(idxs[0])
    
    plt.figure(figsize=(18, 10))
    
    # 可视化四模态图像
    modalities = ['T1', 'T1ce', 'T2', 'FLAIR']
    for i in range(4):
        plt.subplot(3, 4, i+1)
        img = arr['image'][pos][i]
        plt.imshow(img, cmap='gray', vmax=np.percentile(img, 99))
        plt.title(f"{modalities[i]}\nRange: {img.min():.1f}-{img.max():.1f}")
        plt.axis('off')
    
    # 可视化分割标签
    masks = arr['mask'][pos]
    mask_titles = ['WT (1+2+4)', 'TC (1+4)', 'ET (4)']
    for i in range(3):
        plt.subplot(3, 4, 5+i)
        plt.imshow(masks[i], cmap='jet', vmin=0, vmax=1)
        plt.title(f"{mask_titles[i]}\nPos: {masks[i].sum()}px")
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.axis('off')

    # 可视化边缘图
    edges = arr['edge'][pos]
    for i in range(3):
        plt.subplot(3, 4, 9+i)
        plt.imshow(edges[i], cmap='gray')
        plt.title(f"Edge {mask_titles[i]}\nEdges: {edges[i].sum()/255:.0f}px")
        plt.axis('off')
    
    # 添加统计信息
    plt.subplot(3,4,12)
    plt.axis('off')
    stats_text = [
        f"Case: {case_idx} Slice: {slice_idx}",
        "Image Stats:",
        f"Mean: {arr['image'][pos].mean(axis=(1,2)).round(1)}",
        f"Std: {arr['image'][pos].std(axis=(1,2)).round(1)}",
        "\nMask Areas (%):",
        f"WT: {masks[0].mean()*100:.1f}%",
        f"TC: {masks[1].mean()*100:.1f}%", 
        f"ET: {masks[2].mean()*100:.1f}%"
    ]
    plt.text(0, 0.5, '\n'.join(stats_text), fontsize=10)
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/case{case_idx}_slice{slice_idx}.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"可视化结果已保存至 {save_dir}/case{case_idx}_slice{slice_idx}.png")
if __name__ == "__main__":
    for case in range(10):
        try:
            path = os.path.join("./dataset_cache", f"norm_slices_256_full_v2", f"norm_case_{case}.npy")
            arr = np.load(path)
            mid_slice = arr['slice_idx'][len(arr)//2]
            visualize_cache_sample(case, int(mid_slice))
        except Exception as e:
            print(f"可视化病例 {case} 失败: {str(e)}")
def get_data_loaders(t1_root, t2_root, t1ce_root, flair_root, seg_root, 
                    batch_size=4, trainsize=256, slice_range=None,
                    val_t1_root=None, val_t2_root=None, val_t1ce_root=None, 
                    val_flair_root=None, val_seg_root=None,
                    sampler_weight_wt=1.0, sampler_weight_tc=3.5, sampler_weight_et=5.0,
                    min_tumor_pixels=0, et_min_pixels=0, tc_min_pixels=35,
                    val_batch_size=1, val_num_workers=8, val_prefetch_factor=2, persistent_workers=False,
                    edge_mode: str = 'gt',
                    seed: int = 2021,
                    train_num_workers: int = 8,
                    train_prefetch_factor: int = 2,
                    train_persistent_workers: bool = True):
    def collate_fn(batch):
        filtered = [b for b in batch if 
                   not (torch.all(b[0] == 0).item() and 
                        torch.all(b[1] == 0).item())]
        if len(filtered) == 0:
            dummy = (
                torch.zeros(4, trainsize, trainsize),
                torch.zeros(3, trainsize, trainsize),
                torch.zeros(1 if str(edge_mode) == 'otus' else 3, trainsize, trainsize)
            )
            return torch.utils.data.default_collate([dummy])
        return torch.utils.data.default_collate(filtered)
    print("\n" + "="*30 + " 初始化数据加载器 " + "="*30)
    
    train_set = BraTSDataset(t1_root, t2_root, t1ce_root, flair_root, seg_root,
                            trainsize, slice_range, "train", min_tumor_pixels=min_tumor_pixels, et_min_pixels=et_min_pixels, tc_min_pixels=tc_min_pixels,
                            edge_mode=edge_mode)
    val_set = BraTSDataset(t1_root, t2_root, t1ce_root, flair_root, seg_root,
                            trainsize, slice_range, "val", min_tumor_pixels=min_tumor_pixels, et_min_pixels=et_min_pixels, tc_min_pixels=tc_min_pixels,
                            edge_mode=edge_mode)
    test_set = BraTSDataset(t1_root, t2_root, t1ce_root, flair_root, seg_root,
                           trainsize, slice_range, "test", min_tumor_pixels=min_tumor_pixels, et_min_pixels=et_min_pixels, tc_min_pixels=tc_min_pixels,
                           edge_mode=edge_mode)

    import math
    seed = int(seed) if seed is not None else 2021
    g = torch.Generator()
    try:
        g.manual_seed(seed)
    except Exception:
        pass

    def _seed_worker(worker_id: int):
        try:
            worker_seed = int(torch.initial_seed()) % (2**32)
        except Exception:
            worker_seed = seed + int(worker_id)
        np.random.seed(worker_seed)
        random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    weights_np = np.empty(len(train_set.samples), dtype=np.float32)
    case_memmaps = {}
    case_pos_maps = {}
    for idx, (case_idx, slice_idx) in enumerate(train_set.samples):
        try:
            if case_idx not in case_memmaps:
                case_path = os.path.join(train_set.norm_slices_dir, f"norm_case_{case_idx}.npy")
                arr = np.load(case_path, mmap_mode='r')
                case_memmaps[case_idx] = arr
                pos_map = {int(s): i for i, s in enumerate(arr['slice_idx'])}
                case_pos_maps[case_idx] = pos_map
            else:
                arr = case_memmaps[case_idx]
            pos = case_pos_maps[case_idx].get(int(slice_idx), None)
            if pos is None:
                weights_np[idx] = float(sampler_weight_wt)
                continue
            m_et = arr['mask'][pos, 2]
            if (m_et > 0).any():
                weights_np[idx] = float(sampler_weight_et)
                continue
            m_tc = arr['mask'][pos, 1]
            if (m_tc > 0).any():
                weights_np[idx] = float(sampler_weight_tc)
            else:
                weights_np[idx] = float(sampler_weight_wt)
        except Exception:
            weights_np[idx] = float(sampler_weight_wt)

    weights = torch.from_numpy(weights_np.astype(np.float64))
    try:
        sampler = data.WeightedRandomSampler(weights, num_samples=int(weights.shape[0]), replacement=True, generator=g)
    except TypeError:
        sampler = data.WeightedRandomSampler(weights, num_samples=int(weights.shape[0]), replacement=True)

    _train_num_workers = int(max(0, train_num_workers))
    train_loader_kwargs = dict(
        batch_size=batch_size,
        sampler=sampler,
        num_workers=_train_num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn,
        worker_init_fn=_seed_worker,
        persistent_workers=(bool(train_persistent_workers) if _train_num_workers > 0 else False),
    )
    if _train_num_workers > 0:
        train_loader_kwargs['prefetch_factor'] = int(max(2, train_prefetch_factor))
    try:
        train_loader_kwargs['generator'] = g
    except Exception:
        pass
    train_loader = data.DataLoader(train_set, **train_loader_kwargs)

    _val_num_workers = int(max(0, val_num_workers))
    val_loader_kwargs = dict(
        batch_size=int(max(1, val_batch_size)),
        shuffle=False,
        num_workers=_val_num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        worker_init_fn=_seed_worker,
        persistent_workers=(bool(persistent_workers) if _val_num_workers > 0 else False),
    )
    if _val_num_workers > 0:
        val_loader_kwargs['prefetch_factor'] = int(max(2, val_prefetch_factor))
    try:
        val_loader_kwargs['generator'] = g
    except Exception:
        pass
    val_loader = data.DataLoader(val_set, **val_loader_kwargs)
    
    test_loader_kwargs = dict(
        batch_size=int(max(1, val_batch_size)),
        shuffle=False,
        num_workers=_val_num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        worker_init_fn=_seed_worker,
        persistent_workers=(bool(persistent_workers) if _val_num_workers > 0 else False),
    )
    if _val_num_workers > 0:
        test_loader_kwargs['prefetch_factor'] = int(max(2, val_prefetch_factor))
    try:
        test_loader_kwargs['generator'] = g
    except Exception:
        pass
    test_loader = data.DataLoader(test_set, **test_loader_kwargs)
    
    return train_loader, val_loader, test_loader
