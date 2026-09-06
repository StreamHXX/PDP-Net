"""
多分支注意力网络 - RTX 4090D 优化版
严格按照模型图实现,针对24GB显存优化

模型架构:
- ResNet50 Backbone (提取C2, C4, C5特征)
- Low-level Color Branch (LAB色度/亮度对比解耦 + 独立颜色/对比度编码)
- Mid-level Direction Branch (水平/垂直方向卷积 + 空间注意力)
- High-level Semantic Branch (多尺度DW卷积 + 全局上下文注意力)
- Concat Fusion (直接拼接融合)

输出指标: Params, Model Size, FPS, FLOPs, Accuracy, Precision, Recall, F1
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from torchvision import transforms, models
from torchvision.models import ResNet50_Weights
from PIL import Image
import os
import math
import time
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

# ==================== CUDA优化设置 (RTX 4090D专属) ====================
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision('high')
    print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
    print(f"✓ 显存: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.1f} GB")

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


# ==================== RGB to LAB 转换 ====================
class RGB2LAB(nn.Module):
    """RGB到LAB颜色空间转换"""

    def __init__(self):
        super().__init__()
        self.register_buffer('rgb2xyz', torch.tensor([
            [0.412453, 0.357580, 0.180423],
            [0.212671, 0.715160, 0.072169],
            [0.019334, 0.119193, 0.950227]
        ]).float())
        self.register_buffer('white_point', torch.tensor([0.95047, 1.0, 1.08883]).float())

    def forward(self, rgb):
        rgb = torch.clamp(rgb.permute(0, 2, 3, 1), 0, 1)
        mask = rgb > 0.04045
        rgb_linear = torch.where(mask, ((rgb + 0.055) / 1.055) ** 2.4, rgb / 12.92)
        xyz = torch.matmul(rgb_linear, self.rgb2xyz.T) / self.white_point
        mask = xyz > 0.008856
        f_xyz = torch.where(mask, xyz ** (1 / 3), (903.3 * xyz + 16) / 116)
        L = 116 * f_xyz[..., 1] - 16
        a = 500 * (f_xyz[..., 0] - f_xyz[..., 1])
        b = 200 * (f_xyz[..., 1] - f_xyz[..., 2])
        return torch.stack([L, a, b], dim=-1).permute(0, 3, 1, 2)


# ==================== Low-level Color Branch ====================
class ColorEncoderBlock(nn.Module):
    """颜色编码块 - 对应模型图左侧的Conv2d+BN+ReLU"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class LowLevelColorBranch(nn.Module):
    """低层颜色分支 - LAB 解耦增强版

    与原始版本相比，本版本不再将完整 LAB 特征整体投影后与 C2 相加，
    而是将 LAB 显式拆成两类互补信息：

    1) 颜色子分支：使用 a、b、chroma，重点描述色偏、饱和度、颜色漂移等退化；
    2) 对比度子分支：使用 L、局部亮度差、局部标准差、亮度梯度，重点描述低照度、过曝、雾化、模糊等导致的明暗和边缘变化；
    3) 两个子分支分别拥有独立的投影层和卷积编码层，最后再拼接融合。
    """

    def __init__(self, in_channels=512, out_dim=256, num_classes=18):
        super().__init__()

        self.rgb_to_lab = RGB2LAB()

        # 用于反归一化（与你 transforms.Normalize 一致）
        self.register_buffer('img_mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('img_std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        # 固定 Sobel 卷积核：用于从 LAB 的 L 通道提取亮度梯度，不参与学习
        sobel_x = torch.tensor([[-1, 0, 1],
                                [-2, 0, 2],
                                [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3) / 8.0
        sobel_y = torch.tensor([[-1, -2, -1],
                                [0, 0, 0],
                                [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3) / 8.0
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)

        # ========== 颜色子分支：a / b / chroma ==========
        # LAB 色度特征：a_norm、b_norm、chroma，共 3 通道
        self.color_lab_proj = nn.Sequential(
            nn.Conv2d(3, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

        self.color_encoder = nn.ModuleList([
            ColorEncoderBlock(in_channels, in_channels),
            ColorEncoderBlock(in_channels, in_channels),
            ColorEncoderBlock(in_channels, in_channels),
            ColorEncoderBlock(in_channels, in_channels)
        ])

        # ========== 对比度子分支：L / local contrast / local std / gradient ==========
        contrast_channels = in_channels // 2

        # 将 C2 特征压缩到对比度分支通道数
        self.contrast_backbone_proj = nn.Sequential(
            nn.Conv2d(in_channels, contrast_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(contrast_channels),
            nn.ReLU(inplace=True)
        )

        # LAB 亮度对比度特征：L_norm、local_contrast、local_std、grad_mag，共 4 通道
        self.contrast_lab_proj = nn.Sequential(
            nn.Conv2d(4, contrast_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(contrast_channels),
            nn.ReLU(inplace=True)
        )

        self.contrast_encoder = nn.Sequential(
            ColorEncoderBlock(contrast_channels, contrast_channels),
            ColorEncoderBlock(contrast_channels, contrast_channels)
        )

        self.color_pool = nn.AdaptiveAvgPool2d(1)
        self.contrast_pool = nn.AdaptiveAvgPool2d(1)

        self.proj = nn.Sequential(
            nn.Linear(in_channels + contrast_channels, out_dim),
            nn.ReLU(inplace=True)
        )

        self.aux_classifier = nn.Linear(out_dim, num_classes)

    def _split_lab_features(self, lab):
        """将 LAB 显式拆分为色度特征和亮度对比度特征。

        参数:
            lab: (B, 3, H, W)，取值约为 L∈[0,100], a/b∈[-128,128]

        返回:
            color_lab:    (B, 3, H, W) = [a_norm, b_norm, chroma]
            contrast_lab: (B, 4, H, W) = [L_norm, local_contrast, local_std, grad_mag]
        """
        # LAB 通道归一化
        L = lab[:, 0:1] / 100.0
        a = lab[:, 1:2] / 128.0
        b = lab[:, 2:3] / 128.0

        # 色度强度，用于描述颜色饱和度/色偏强弱
        chroma = torch.sqrt(a * a + b * b + 1e-6)
        color_lab = torch.cat([a, b, chroma], dim=1)

        # 亮度局部统计，用于描述对比度变化
        local_mean = F.avg_pool2d(L, kernel_size=7, stride=1, padding=3)
        local_diff = L - local_mean
        local_contrast = torch.abs(local_diff)
        local_std = torch.sqrt(
            F.avg_pool2d(local_diff * local_diff, kernel_size=7, stride=1, padding=3) + 1e-6
        )

        # 亮度梯度，用于描述明暗边缘和纹理清晰度
        grad_x = F.conv2d(L, self.sobel_x, padding=1)
        grad_y = F.conv2d(L, self.sobel_y, padding=1)
        grad_mag = torch.sqrt(grad_x * grad_x + grad_y * grad_y + 1e-6)

        contrast_lab = torch.cat([L, local_contrast, local_std, grad_mag], dim=1)
        return color_lab, contrast_lab

    def forward(self, x, original_image):
        # 反归一化到 [0,1]
        rgb_01 = original_image * self.img_std + self.img_mean
        rgb_01 = torch.clamp(rgb_01, 0.0, 1.0)

        # RGB -> LAB，然后显式拆成“色度信息”和“亮度对比度信息”
        lab = self.rgb_to_lab(rgb_01)  # (B, 3, H, W)
        color_lab, contrast_lab = self._split_lab_features(lab)

        # 下采样到 C2 尺寸
        color_lab_ds = F.interpolate(color_lab, size=x.shape[-2:], mode='bilinear', align_corners=False)
        contrast_lab_ds = F.interpolate(contrast_lab, size=x.shape[-2:], mode='bilinear', align_corners=False)

        # ========== 颜色子分支 ==========
        # C2 + LAB 色度特征，只强化颜色/色偏/饱和度相关表达
        color_x = x + self.color_lab_proj(color_lab_ds)
        for encoder in self.color_encoder:
            color_x = encoder(color_x)
        color_feat = self.color_pool(color_x).flatten(1)  # (B, 512)

        # ========== 对比度子分支 ==========
        # C2 压缩特征 + LAB 亮度对比度特征，只强化亮度变化/局部对比/边缘清晰度表达
        contrast_x = self.contrast_backbone_proj(x) + self.contrast_lab_proj(contrast_lab_ds)
        contrast_x = self.contrast_encoder(contrast_x)
        contrast_feat = self.contrast_pool(contrast_x).flatten(1)  # (B, 256)

        # 拼接颜色特征和对比度特征，保持低层颜色分支最终输出维度不变
        combined = torch.cat([color_feat, contrast_feat], dim=1)  # (B, 512 + 256)
        out_feat = self.proj(combined)  # (B, out_dim)
        aux_out = self.aux_classifier(out_feat)
        return out_feat, aux_out


# ==================== Mid-level Direction Branch ====================
class DirectionConvBlock(nn.Module):
    """方向卷积块 - Horizontal/Vertical"""

    def __init__(self, in_channels, out_channels, kernel_size=7):
        super().__init__()
        padding = kernel_size // 2

        self.horizontal = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 2, (1, kernel_size),
                      padding=(0, padding), bias=False),
            nn.BatchNorm2d(out_channels // 2),
            nn.ReLU(inplace=True)
        )

        self.vertical = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 2, (kernel_size, 1),
                      padding=(padding, 0), bias=False),
            nn.BatchNorm2d(out_channels // 2),
            nn.ReLU(inplace=True)
        )

        self.pw_conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        h_feat = self.horizontal(x)
        v_feat = self.vertical(x)
        concat = torch.cat([h_feat, v_feat], dim=1)
        return self.pw_conv(concat)


class SpatialAttention(nn.Module):
    """空间注意力"""

    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        concat = torch.cat([avg_out, max_out], dim=1)
        attention = self.sigmoid(self.conv(concat))
        return x * attention


class MidLevelDirectionBranch(nn.Module):
    """中级方向分支"""

    def __init__(self, in_channels=1024, out_dim=256, num_classes=18):
        super().__init__()
        self.direction_block = DirectionConvBlock(in_channels, in_channels, kernel_size=7)

        self.feature_agg = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

        self.spatial_attention = SpatialAttention()
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        self.proj = nn.Sequential(
            nn.Linear(in_channels, out_dim),
            nn.ReLU(inplace=True)
        )

        self.aux_classifier = nn.Linear(out_dim, num_classes)

    def forward(self, x):
        x = self.direction_block(x)
        x = self.feature_agg(x)
        x = self.spatial_attention(x)
        x = self.global_pool(x).flatten(1)
        out_feat = self.proj(x)
        aux_out = self.aux_classifier(out_feat)
        return out_feat, aux_out


# ==================== High-level Semantic Branch (降参核心改动) ====================
class AttentionWeightGenerator(nn.Module):
    """注意力权重生成器"""

    def __init__(self, in_channels, num_scales=3):
        super().__init__()
        self.num_scales = num_scales
        self.weight_gen = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, num_scales),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.weight_gen(x)


class DepthwiseSeparableConv(nn.Module):
    """Depthwise Separable Conv: DW 3x3 + PW 1x1"""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, bias=False):
        super().__init__()
        self.dw = nn.Conv2d(
            in_channels, in_channels, kernel_size=kernel_size, stride=stride,
            padding=padding, dilation=dilation, groups=in_channels, bias=bias
        )
        self.pw = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.dw(x)
        x = self.pw(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class MultiScaleConv(nn.Module):
    """多尺度卷积（轻量版）：全部使用 DepthwiseSeparableConv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.small = DepthwiseSeparableConv(in_channels, out_channels, kernel_size=3, padding=1, dilation=1)
        self.medium = nn.Sequential(
            DepthwiseSeparableConv(in_channels, out_channels, kernel_size=3, padding=1, dilation=1),
            DepthwiseSeparableConv(out_channels, out_channels, kernel_size=3, padding=1, dilation=1),
        )
        self.large = DepthwiseSeparableConv(in_channels, out_channels, kernel_size=3, padding=2, dilation=2)

    def forward(self, x):
        return self.small(x), self.medium(x), self.large(x)


class HighLevelSemanticBranch(nn.Module):
    """高级语义分支 - 轻量化版本（大幅降参）"""

    def __init__(self, in_channels=2048, out_dim=512, num_classes=18, bottleneck=512):
        super().__init__()
        self.bottleneck = bottleneck

        # 2048 -> bottleneck
        self.adapter = nn.Sequential(
            nn.Conv2d(in_channels, bottleneck, kernel_size=1, bias=False),
            nn.BatchNorm2d(bottleneck),
            nn.ReLU(inplace=True)
        )

        self.multi_scale_conv = MultiScaleConv(bottleneck, bottleneck)
        self.attention_gen = AttentionWeightGenerator(bottleneck, num_scales=3)

        # DRF Conv (轻量版)
        self.drf_conv = DepthwiseSeparableConv(bottleneck, bottleneck, kernel_size=3, padding=1, dilation=1)

        # Global Context Attention（SE-like）
        hidden = max(bottleneck // 16, 16)
        self.global_context = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(bottleneck, hidden, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, bottleneck, 1),
            nn.Sigmoid()
        )

        self.global_pool = nn.AdaptiveAvgPool2d(1)

        self.proj = nn.Sequential(
            nn.Linear(bottleneck, out_dim),
            nn.ReLU(inplace=True)
        )

        self.aux_classifier = nn.Linear(out_dim, num_classes)

    def forward(self, x):
        B = x.shape[0]
        x = self.adapter(x)

        small_feat, medium_feat, large_feat = self.multi_scale_conv(x)
        weights = self.attention_gen(x)  # (B,3)

        fused = (weights[:, 0].view(B, 1, 1, 1) * small_feat +
                 weights[:, 1].view(B, 1, 1, 1) * medium_feat +
                 weights[:, 2].view(B, 1, 1, 1) * large_feat)

        x = self.drf_conv(fused)

        context = self.global_context(x)
        x = x * context

        x = self.global_pool(x).flatten(1)
        out_feat = self.proj(x)
        aux_out = self.aux_classifier(out_feat)
        return out_feat, aux_out


# ==================== Concat Fusion ====================
class ConcatFusion(nn.Module):
    """Concat 融合（替换自适应门控融合）

    直接将三路分支特征拼接后，通过两层 MLP 得到融合特征，再分类。
    输出接口保持与原 AdaptiveGatedFusion 一致：返回 (logits, fused_feat)
    """

    def __init__(self, color_dim, direction_dim, semantic_dim, fusion_dim, num_classes):
        super().__init__()
        total_dim = color_dim + direction_dim + semantic_dim

        self.fusion = nn.Sequential(
            nn.Linear(total_dim, fusion_dim),
            nn.BatchNorm1d(fusion_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.BatchNorm1d(fusion_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )

        self.classifier = nn.Linear(fusion_dim // 2, num_classes)

    def forward(self, color_feat, direction_feat, semantic_feat):
        fused = torch.cat([color_feat, direction_feat, semantic_feat], dim=1)
        fused_feat = self.fusion(fused)
        logits = self.classifier(fused_feat)
        return logits, fused_feat


# ==================== 主模型 ====================
class MultibranchAttentionNet(nn.Module):
    """多分支注意力网络"""

    def __init__(self, num_classes=18, pretrained_path=None,
                 color_dim=256, direction_dim=256, semantic_dim=512, fusion_dim=1024):
        super().__init__()

        self.num_classes = num_classes

        # ========== ResNet50 Backbone ==========
        if pretrained_path and os.path.exists(pretrained_path):
            try:
                backbone = models.resnet50(weights=None)
                backbone.load_state_dict(torch.load(pretrained_path, map_location='cpu'))
                print(f"✓ 加载本地预训练权重: {pretrained_path}")
            except:
                backbone = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
                print("✓ 使用在线预训练权重")
        else:
            backbone = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
            print("✓ 使用在线预训练权重")

        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2  # C2: 512
        self.layer3 = backbone.layer3  # C4: 1024
        self.layer4 = backbone.layer4  # C5: 2048

        self.color_branch = LowLevelColorBranch(
            in_channels=512,
            out_dim=color_dim,
            num_classes=num_classes
        )

        self.direction_branch = MidLevelDirectionBranch(
            in_channels=1024,
            out_dim=direction_dim,
            num_classes=num_classes
        )

        # 语义分支：内部 bottleneck=512（关键降参）
        self.semantic_branch = HighLevelSemanticBranch(
            in_channels=2048,
            out_dim=semantic_dim,
            num_classes=num_classes,
            bottleneck=512
        )

        self.fusion = ConcatFusion(
            color_dim=color_dim,
            direction_dim=direction_dim,
            semantic_dim=semantic_dim,
            fusion_dim=fusion_dim,
            num_classes=num_classes
        )

    def forward(self, x, return_aux=False):
        original_image = x.clone()

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        c2 = self.layer2(x)
        c4 = self.layer3(c2)
        c5 = self.layer4(c4)

        color_feat, aux_color = self.color_branch(c2, original_image)
        direction_feat, aux_direction = self.direction_branch(c4)
        semantic_feat, aux_semantic = self.semantic_branch(c5)

        final_logits, fused_feat = self.fusion(color_feat, direction_feat, semantic_feat)

        if return_aux:
            return final_logits, aux_color, aux_direction, aux_semantic, \
                color_feat, direction_feat, semantic_feat

        return final_logits


# ==================== 损失函数 ====================
class MultibranchLoss(nn.Module):
    """多分支监督损失"""

    def __init__(self, num_classes, label_smoothing=0.1,
                 aux_color_weight=0.3, aux_direction_weight=0.3, aux_semantic_weight=0.3):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.aux_color_weight = aux_color_weight
        self.aux_direction_weight = aux_direction_weight
        self.aux_semantic_weight = aux_semantic_weight

    def forward(self, final_logits, aux_color, aux_direction, aux_semantic, labels):
        main_loss = self.ce_loss(final_logits, labels)
        loss_color = self.ce_loss(aux_color, labels)
        loss_direction = self.ce_loss(aux_direction, labels)
        loss_semantic = self.ce_loss(aux_semantic, labels)

        total_loss = (main_loss +
                      self.aux_color_weight * loss_color +
                      self.aux_direction_weight * loss_direction +
                      self.aux_semantic_weight * loss_semantic)

        loss_dict = {
            'main': main_loss.item(),
            'color': loss_color.item(),
            'direction': loss_direction.item(),
            'semantic': loss_semantic.item(),
            'total': total_loss.item()
        }
        return total_loss, loss_dict


# ==================== 数据集 ====================
class FastImageDataset(Dataset):
    """快速图像数据集加载器"""

    def __init__(self, root_dir, transform=None, split='train', train_ratio=0.8, seed=42):
        self.root_dir = root_dir
        self.transform = transform

        self.classes = sorted([d for d in os.listdir(root_dir)
                               if os.path.isdir(os.path.join(root_dir, d)) and not d.startswith('.')])
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

        valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp')
        all_samples = []
        for cls in self.classes:
            cls_path = os.path.join(root_dir, cls)
            for img_name in os.listdir(cls_path):
                if img_name.lower().endswith(valid_extensions) and not img_name.startswith('.'):
                    all_samples.append((os.path.join(cls_path, img_name), self.class_to_idx[cls]))

        np.random.seed(seed)
        train_samples, val_samples = [], []
        for cls_idx in range(len(self.classes)):
            cls_samples = [s for s in all_samples if s[1] == cls_idx]
            np.random.shuffle(cls_samples)
            split_idx = int(len(cls_samples) * train_ratio)
            train_samples.extend(cls_samples[:split_idx])
            val_samples.extend(cls_samples[split_idx:])

        self.samples = train_samples if split == 'train' else val_samples
        np.random.shuffle(self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        try:
            image = Image.open(img_path).convert('RGB')
        except:
            image = Image.new('RGB', (224, 224), (128, 128, 128))

        if self.transform:
            image = self.transform(image)

        return image, label


# ==================== 工具函数 ====================
def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def get_model_size_mb(model):
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    return (param_size + buffer_size) / 1024 / 1024


def measure_fps(model, input_size=(1, 3, 224, 224), device='cuda', n_runs=100, warmup=10):
    model.eval()
    dummy_input = torch.randn(*input_size).to(device)

    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy_input)

    if device == 'cuda':
        torch.cuda.synchronize()

    start = time.time()
    with torch.no_grad():
        for _ in range(n_runs):
            _ = model(dummy_input)

    if device == 'cuda':
        torch.cuda.synchronize()

    elapsed = time.time() - start
    fps = n_runs / elapsed
    latency = elapsed / n_runs * 1000
    return fps, latency


def estimate_flops(model, input_size=(1, 3, 224, 224), device='cuda'):
    try:
        from thop import profile, clever_format
        dummy_input = torch.randn(*input_size).to(device)
        flops, params = profile(model, inputs=(dummy_input,), verbose=False)
        flops_str, params_str = clever_format([flops, params], "%.2f")
        return flops, flops_str
    except:
        total_params, _ = count_parameters(model)
        estimated_flops = total_params * 2 * input_size[2] * input_size[3] / 49
        return estimated_flops, f"{estimated_flops / 1e9:.2f}G (estimated)"


# ==================== 评估指标 ====================
class MetricsCalculator:
    def __init__(self, num_classes, class_names):
        self.num_classes = num_classes
        self.class_names = class_names
        self.reset()

    def reset(self):
        self.all_preds = []
        self.all_labels = []
        self.all_probs = []

    def update(self, preds, labels, probs=None):
        self.all_preds.extend(preds.cpu().numpy())
        self.all_labels.extend(labels.cpu().numpy())
        if probs is not None:
            self.all_probs.extend(probs.cpu().numpy())

    def compute_metrics(self):
        preds = np.array(self.all_preds)
        labels = np.array(self.all_labels)

        metrics = {
            'accuracy': accuracy_score(labels, preds),
            'precision_macro': precision_score(labels, preds, average='macro', zero_division=0),
            'recall_macro': recall_score(labels, preds, average='macro', zero_division=0),
            'f1_macro': f1_score(labels, preds, average='macro', zero_division=0),
            'precision_weighted': precision_score(labels, preds, average='weighted', zero_division=0),
            'recall_weighted': recall_score(labels, preds, average='weighted', zero_division=0),
            'f1_weighted': f1_score(labels, preds, average='weighted', zero_division=0),
            'precision_per_class': precision_score(labels, preds, average=None, zero_division=0),
            'recall_per_class': recall_score(labels, preds, average=None, zero_division=0),
            'f1_per_class': f1_score(labels, preds, average=None, zero_division=0),
            'confusion_matrix': confusion_matrix(labels, preds),
            'classification_report': classification_report(labels, preds, target_names=self.class_names,
                                                           zero_division=0)
        }
        return metrics


# ==================== 可视化 ====================
class MetricsVisualizer:
    def __init__(self, save_dir, class_names):
        self.save_dir = save_dir
        self.class_names = class_names
        os.makedirs(save_dir, exist_ok=True)

    def plot_training_history(self, history):
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        epochs = range(1, len(history['train_loss']) + 1)

        axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Train', linewidth=2)
        axes[0, 0].plot(epochs, history['val_loss'], 'r-', label='Val', linewidth=2)
        axes[0, 0].set_title('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        axes[0, 1].plot(epochs, history['train_acc'], 'b-', label='Train', linewidth=2)
        axes[0, 1].plot(epochs, history['val_acc'], 'r-', label='Val', linewidth=2)
        axes[0, 1].set_title('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        if 'lr' in history:
            axes[1, 0].plot(epochs, history['lr'], 'g-', linewidth=2)
            axes[1, 0].set_title('Learning Rate')
            axes[1, 0].set_yscale('log')
            axes[1, 0].grid(True, alpha=0.3)

        if 'val_f1' in history:
            axes[1, 1].plot(epochs, history['val_f1'], 'm-', linewidth=2)
            axes[1, 1].set_title('Val F1')
            axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'training_history.png'), dpi=150)
        plt.close()

    def plot_confusion_matrix(self, cm):
        fig, axes = plt.subplots(1, 2, figsize=(20, 8))

        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.class_names, yticklabels=self.class_names, ax=axes[0])
        axes[0].set_title('Confusion Matrix (Counts)')

        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        sns.heatmap(np.nan_to_num(cm_norm), annot=True, fmt='.1%', cmap='Blues',
                    xticklabels=self.class_names, yticklabels=self.class_names, ax=axes[1])
        axes[1].set_title('Confusion Matrix (Normalized)')

        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'confusion_matrix.png'), dpi=150)
        plt.close()

    def plot_per_class_metrics(self, metrics):
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        x = np.arange(len(self.class_names))

        for ax, metric, title, color in zip(
                axes,
                ['precision_per_class', 'recall_per_class', 'f1_per_class'],
                ['Precision', 'Recall', 'F1'],
                ['steelblue', 'forestgreen', 'darkorange']
        ):
            ax.bar(x, metrics[metric], color=color, alpha=0.8)
            ax.set_xticks(x)
            ax.set_xticklabels(self.class_names, rotation=45, ha='right', fontsize=8)
            ax.set_title(f'{title} per Class')
            ax.set_ylim(0, 1.1)

        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'per_class_metrics.png'), dpi=150)
        plt.close()

    def save_all_plots(self, metrics, history=None):
        print("\n生成可视化...")
        if history:
            self.plot_training_history(history)
        self.plot_confusion_matrix(metrics['confusion_matrix'])
        self.plot_per_class_metrics(metrics)
        print(f"✓ 图表已保存至: {self.save_dir}")


# ==================== 训练函数 ====================
def train_one_epoch(model, loader, criterion, optimizer, scaler, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc='Training', ncols=100)
    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast():
            final_logits, aux_color, aux_direction, aux_semantic, _, _, _ = model(images, return_aux=True)
            loss, loss_dict = criterion(final_logits, aux_color, aux_direction, aux_semantic, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()
        _, predicted = final_logits.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        pbar.set_postfix({'loss': f'{running_loss / (pbar.n + 1):.4f}',
                          'acc': f'{100. * correct / total:.2f}%'})

    return running_loss / len(loader), 100. * correct / total


def evaluate(model, loader, device, metrics_calc=None):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()

    if metrics_calc:
        metrics_calc.reset()

    with torch.no_grad():
        pbar = tqdm(loader, desc='Evaluating', ncols=100)
        for images, labels in pbar:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)

            probs = F.softmax(outputs.float(), dim=1)
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            if metrics_calc:
                metrics_calc.update(predicted, labels, probs)

            pbar.set_postfix({'loss': f'{running_loss / (pbar.n + 1):.4f}',
                              'acc': f'{100. * correct / total:.2f}%'})

    return running_loss / len(loader), 100. * correct / total


# ==================== 主函数 ====================
def main():
    # ========== RTX 4090D优化配置 (24GB显存) ==========
    config = {
        # 数据
        'data_root': './dataset',
        'train_ratio': 0.8,

        # 模型
        'pretrained_path': './pretrained/resnet50-11ad3fa6.pth',
        'color_dim': 256,
        'direction_dim': 256,
        'semantic_dim': 512,
        'fusion_dim': 1024,

        # 训练
        'img_size': 224,
        'batch_size': 64,
        'num_epochs': 10,
        'lr': 1e-4,
        'backbone_lr': 1e-5,
        'weight_decay': 0.01,
        'num_workers': 8,

        # 损失权重
        'aux_color_weight': 0.3,
        'aux_direction_weight': 0.3,
        'aux_semantic_weight': 0.3,

        # 优化
        'use_amp': True,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'save_path': './results_multibranch',
        'seed': 42,
    }

    os.makedirs('./pretrained', exist_ok=True)
    os.makedirs(config['save_path'], exist_ok=True)

    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config['seed'])

    print("=" * 80)
    print("多分支注意力网络 - RTX 4090D优化版")
    print("=" * 80)
    print(f"配置信息:")
    print(f"  - Batch Size: {config['batch_size']} (针对24GB显存优化)")
    print(f"  - Workers: {config['num_workers']}")
    print(f"  - Mixed Precision: {config['use_amp']}")
    print(f"  - Device: {config['device']}")
    print("=" * 80)

    # 数据加载
    train_transform = transforms.Compose([
        transforms.Resize((config['img_size'] + 32, config['img_size'] + 32)),
        transforms.RandomCrop(config['img_size']),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([transforms.ColorJitter(0.1, 0.1, 0.1, 0.05)], p=0.3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((config['img_size'], config['img_size'])),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    if not os.path.exists(config['data_root']):
        print(f"错误: 数据集路径不存在: {config['data_root']}")
        return

    print("\n加载数据...")
    train_dataset = FastImageDataset(config['data_root'], train_transform, 'train',
                                     config['train_ratio'], config['seed'])
    val_dataset = FastImageDataset(config['data_root'], val_transform, 'val',
                                   config['train_ratio'], config['seed'])

    num_classes = len(train_dataset.classes)
    class_names = train_dataset.classes

    print(f"类别数: {num_classes}")
    print(f"训练样本: {len(train_dataset)}, 验证样本: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, config['batch_size'], shuffle=True,
                              num_workers=config['num_workers'], pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, config['batch_size'] * 2, shuffle=False,
                            num_workers=config['num_workers'], pin_memory=True)

    # 创建模型
    print("\n创建模型...")
    model = MultibranchAttentionNet(
        num_classes=num_classes,
        pretrained_path=config['pretrained_path'],
        color_dim=config['color_dim'],
        direction_dim=config['direction_dim'],
        semantic_dim=config['semantic_dim'],
        fusion_dim=config['fusion_dim']
    ).to(config['device'])

    # 模型分析
    total_params, trainable_params = count_parameters(model)
    model_size_mb = get_model_size_mb(model)
    fps, latency = measure_fps(model, device=config['device'])
    flops, flops_str = estimate_flops(model, device=config['device'])

    print(f"\n【模型信息】")
    print(f"  总参数: {total_params:,} ({total_params / 1e6:.2f}M)")
    print(f"  可训练参数: {trainable_params:,} ({trainable_params / 1e6:.2f}M)")
    print(f"  模型大小: {model_size_mb:.2f} MB")
    print(f"  FPS: {fps:.2f}")
    print(f"  Latency: {latency:.2f} ms")
    print(f"  FLOPs: {flops_str}")

    # 损失和优化器
    criterion = MultibranchLoss(
        num_classes=num_classes,
        aux_color_weight=config['aux_color_weight'],
        aux_direction_weight=config['aux_direction_weight'],
        aux_semantic_weight=config['aux_semantic_weight']
    )

    backbone_params, other_params = [], []
    for name, param in model.named_parameters():
        if param.requires_grad:
            if 'layer' in name or 'conv1' in name or 'bn1' in name:
                backbone_params.append(param)
            else:
                other_params.append(param)

    optimizer = optim.AdamW([
        {'params': backbone_params, 'lr': config['backbone_lr']},
        {'params': other_params, 'lr': config['lr']}
    ], weight_decay=config['weight_decay'])

    warmup_epochs = 5

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        return 0.5 * (1 + math.cos(math.pi * (epoch - warmup_epochs) / (config['num_epochs'] - warmup_epochs)))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = GradScaler() if config['use_amp'] else GradScaler(enabled=False)

    metrics_calc = MetricsCalculator(num_classes, class_names)
    visualizer = MetricsVisualizer(config['save_path'], class_names)

    # 训练
    print("\n" + "=" * 80)
    print("开始训练")
    print("=" * 80)

    best_f1 = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'val_f1': [], 'lr': []}

    total_start = time.time()

    for epoch in range(config['num_epochs']):
        print(f"\nEpoch {epoch + 1}/{config['num_epochs']}")
        print("-" * 80)

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, scaler, config['device'])
        val_loss, val_acc = evaluate(model, val_loader, config['device'], metrics_calc)
        epoch_metrics = metrics_calc.compute_metrics()
        val_f1 = epoch_metrics['f1_macro']

        scheduler.step()
        current_lr = optimizer.param_groups[-1]['lr']

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)
        history['lr'].append(current_lr)

        print(f"训练 - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
        print(f"验证 - Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%, F1: {val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'best_f1': best_f1,
                'config': config,
            }, os.path.join(config['save_path'], 'best_model.pth'))
            print(f"✓ 保存最佳模型 (F1: {best_f1:.4f})")

    total_time = time.time() - total_start
    print(f"\n训练完成，耗时: {total_time / 60:.1f}分钟")

    # 最终评估
    print("\n" + "=" * 80)
    print("最终评估")
    print("=" * 80)

    checkpoint = torch.load(os.path.join(config['save_path'], 'best_model.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])

    metrics_calc.reset()
    evaluate(model, val_loader, config['device'], metrics_calc)
    final_metrics = metrics_calc.compute_metrics()

    print("\n" + "=" * 80)
    print("【完整评估指标】")
    print("=" * 80)

    print(f"\n【模型指标】")
    print(f"  Params:        {total_params:,} ({total_params / 1e6:.2f}M)")
    print(f"  Model Size:    {model_size_mb:.2f} MB")
    print(f"  FLOPs:         {flops_str}")
    print(f"  FPS:           {fps:.2f}")
    print(f"  Latency:       {latency:.2f} ms")

    print(f"\n【分类指标】")
    print(f"  Accuracy:      {final_metrics['accuracy']:.4f} ({final_metrics['accuracy'] * 100:.2f}%)")
    print(f"  Precision:     {final_metrics['precision_macro']:.4f}")
    print(f"  Recall:        {final_metrics['recall_macro']:.4f}")
    print(f"  F1:            {final_metrics['f1_macro']:.4f}")

    print(f"\n【各类别指标】")
    print(f"{'Class':<20} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print("-" * 52)
    for i, cls in enumerate(class_names):
        p = final_metrics['precision_per_class'][i]
        r = final_metrics['recall_per_class'][i]
        f = final_metrics['f1_per_class'][i]
        print(f"{cls:<20} {p:>10.4f} {r:>10.4f} {f:>10.4f}")

    visualizer.save_all_plots(final_metrics, history)

    # 保存报告
    report_path = os.path.join(config['save_path'], 'report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("多分支注意力网络评估报告 (RTX 4090D优化版)\n")
        f.write("=" * 80 + "\n\n")

        f.write("【模型指标】\n")
        f.write(f"  Params:        {total_params:,} ({total_params / 1e6:.2f}M)\n")
        f.write(f"  Model Size:    {model_size_mb:.2f} MB\n")
        f.write(f"  FLOPs:         {flops_str}\n")
        f.write(f"  FPS:           {fps:.2f}\n")
        f.write(f"  Latency:       {latency:.2f} ms\n\n")

        f.write("【分类指标】\n")
        f.write(f"  Accuracy:      {final_metrics['accuracy']:.4f}\n")
        f.write(f"  Precision:     {final_metrics['precision_macro']:.4f}\n")
        f.write(f"  Recall:        {final_metrics['recall_macro']:.4f}\n")
        f.write(f"  F1:            {final_metrics['f1_macro']:.4f}\n")
        f.write(f"\n训练时间: {total_time / 60:.1f}分钟\n\n")

        f.write("【分类报告】\n")
        f.write(final_metrics['classification_report'])

    print(f"\n✓ 报告已保存至: {report_path}")
    print(f"✓ 结果目录: {config['save_path']}")
    print("=" * 80)


if __name__ == '__main__':
    main()
