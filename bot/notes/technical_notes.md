# 技术笔记与实现要点

**最后更新**: 2025-10-17

---

## 📚 目录

1. [Wide ResNet 实现要点](#wide-resnet-实现要点)
2. [ConvNeXt 适配 CIFAR](#convnext-适配-cifar)
3. [数据增强技术](#数据增强技术)
4. [训练技巧](#训练技巧)
5. [超参数调优经验](#超参数调优经验)
6. [常见问题与解决方案](#常见问题与解决方案)

---

## 🏗️ Wide ResNet 实现要点

### 架构特点

- **核心思想**: 加宽网络（增加通道数），而不是加深
- **标记**: WRN-depth-widen_factor
  - WRN-28-10: 28 层，10× 基础通道数 (160, 320, 640)
  - WRN-40-10: 40 层，10× 基础通道数

### 关键实现细节

#### 1. 基础通道数计算

```python
# depth = 28, widen_factor = 10
base_channels = [16, 160, 320, 640]  # 初始16，然后 16*k*widen_factor
```

#### 2. Depth 配置

```python
# 总层数 = 1 (conv1) + 6*n (3 groups × 2n blocks × conv) + 1 (fc)
# depth = 6n + 4
# 因此: n = (depth - 4) / 6

# WRN-28-10: n = (28 - 4) / 6 = 4
# WRN-40-10: n = (40 - 4) / 6 = 6
```

#### 3. 每个 group 的 block 数量

```python
def _wide_resnet_layers(depth):
    assert (depth - 4) % 6 == 0, 'depth should be 6n+4'
    n = (depth - 4) // 6
    return [n, n, n]  # 3 groups, 每组 n 个 blocks

# WRN-28-10: [4, 4, 4]
# WRN-40-10: [6, 6, 6]
```

#### 4. Dropout 位置

```python
# Wide ResNet 原论文建议在 residual block 的 BN-ReLU-Conv 之间加 dropout
class BasicBlock(nn.Module):
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)  # ← 这里
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(x)
        return self.relu(out)
```

### 参数量对比

| 模型 | 参数量 | FLOPs (32×32) |
|-----|--------|---------------|
| ResNet-50 | ~23.5M | ~1.3G |
| WRN-28-10 | ~36.5M | ~5.2G |
| WRN-40-10 | ~55.8M | ~8.0G |
| WRN-28-12 | ~52.8M | ~7.5G |

### 推荐配置 (CIFAR-100)

```python
model = WideResNet(
    depth=28,
    widen_factor=10,
    dropout_rate=0.3,  # 原论文推荐 0.3
    num_classes=100
)

# 训练超参数
lr = 0.1  # SGD 推荐
optimizer = SGD(momentum=0.9, weight_decay=5e-4, nesterov=True)
scheduler = CosineAnnealingLR(T_max=200)
```

### 参考资源

- 论文: "Wide Residual Networks" (BMVC 2016)
- 官方实现: <https://github.com/szagoruyko/wide-residual-networks>
- PyTorch 实现: <https://github.com/meliketoy/wide-resnet.pytorch>

---

## 🔄 ConvNeXt 适配 CIFAR

### 为什么选择 ConvNeXt？

- 2022 年 SOTA CNN 架构
- 性能接近 ViT，但更容易训练
- 纯卷积，无需 position embedding 等复杂操作

### 适配要点

#### 1. 输入层修改

```python
# 原始 (ImageNet 224×224)
self.stem = nn.Sequential(
    nn.Conv2d(3, 96, kernel_size=4, stride=4),  # → 56×56
    LayerNorm(96, eps=1e-6, data_format="channels_first")
)

# 适配 CIFAR (32×32)
self.stem = nn.Sequential(
    nn.Conv2d(3, 96, kernel_size=3, stride=1, padding=1),  # → 32×32
    LayerNorm(96, eps=1e-6, data_format="channels_first")
)
```

#### 2. 下采样策略

```python
# 原始: 4 个 stage, 分辨率变化 56→28→14→7
# 适配: 4 个 stage, 分辨率变化 32→16→8→4

# 在每个 stage 之间的 downsample
downsample_layer = nn.Sequential(
    LayerNorm(in_channels, eps=1e-6, data_format="channels_first"),
    nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=2),
)
```

#### 3. Stage 深度配置

| 模型 | Stage深度 | 通道数 | 参数量 |
|-----|-----------|--------|--------|
| ConvNeXt-Tiny | [3,3,9,3] | [96,192,384,768] | ~28M |
| ConvNeXt-Small | [3,3,27,3] | [96,192,384,768] | ~50M |
| ConvNeXt-Base | [3,3,27,3] | [128,256,512,1024] | ~89M |

**CIFAR 推荐**: ConvNeXt-Tiny (参数量适中，训练稳定)

#### 4. 关键模块: ConvNeXt Block

```python
class ConvNeXtBlock(nn.Module):
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # 扩展 4x
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)  # 压缩回来
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                   requires_grad=True)
        self.drop_path = DropPath(drop_path)
    
    def forward(self, x):
        shortcut = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # NCHW → NHWC
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # NHWC → NCHW
        x = shortcut + self.drop_path(x)
        return x
```

### 训练建议

```python
# ConvNeXt 原论文推荐配置
lr = 4e-3  # AdamW, with linear warmup
weight_decay = 0.05
batch_size = 4096  # 大 batch (我们用 128 即可)
epochs = 300
warmup_epochs = 20
scheduler = CosineAnnealingLR

# Layer-wise LR decay (可选)
layer_decay = 0.75  # 不同层使用不同学习率
```

### 参考资源

- 论文: "A ConvNet for the 2020s" (CVPR 2022)
- 官方实现: <https://github.com/facebookresearch/ConvNeXt>
- timm 实现: `timm.models.convnext`

---

## 🎨 数据增强技术

### RandAugment

#### 原理

- 2 个超参数: N (操作数量), M (变换幅度)
- 从 14 种操作中随机选择 N 个，强度统一为 M

#### 14 种操作

```
AutoContrast, Equalize, Invert, Rotate, Posterize, Solarize,
SolarizeAdd, Color, Contrast, Brightness, Sharpness,
ShearX, ShearY, TranslateX, TranslateY
```

#### CIFAR 推荐配置

```python
# 原论文推荐
N = 2  # 每张图片应用 2 个操作
M = 9  # 幅度 9 (范围 0-10)

# 更激进 (如果模型过拟合)
N = 3
M = 12
```

#### Albumentations 实现

```python
# 遗憾: Albumentations 没有内置 RandAugment
# 需要手动实现或使用 torchvision
from torchvision.transforms import RandAugment

transform = RandAugment(num_ops=2, magnitude=9)
```

### GridMask

#### 原理

- 在图像上覆盖规则的网格遮挡
- 比 CutOut 更结构化，保留更多上下文

#### 关键参数

```python
d = 96       # 网格间距
ratio = 0.6  # 遮挡比例
```

#### 实现示例

```python
class GridMask(nn.Module):
    def __init__(self, d_range=(96, 224), ratio=0.6, p=0.5):
        self.d_range = d_range
        self.ratio = ratio
        self.p = p
    
    def forward(self, x):
        if random.random() > self.p:
            return x
        
        h, w = x.shape[-2:]
        d = random.randint(*self.d_range)
        
        # 生成网格 mask
        mask = torch.ones_like(x)
        for i in range(0, h, d):
            for j in range(0, w, d):
                mask[:, :, i:i+int(d*self.ratio), j:j+int(d*self.ratio)] = 0
        
        return x * mask
```

### FMix

#### 原理

- 在傅里叶域进行混合
- 生成更自然的混合边界

#### 推荐配置

```python
fmix = FMix(
    decay_power=3.0,  # 频率衰减速度
    alpha=1.0,        # Beta 分布参数
    size=(32, 32),
    max_soft=0.0,     # 软遮罩边界
    reformulate=False
)
```

---

## 🎓 训练技巧

### Stochastic Depth (DropPath)

#### 原理

- 训练时随机跳过整个 residual block
- 类似于 dropout，但作用在层级而非神经元

#### 实现

```python
class DropPath(nn.Module):
    def __init__(self, drop_prob=0.):
        super().__init__()
        self.drop_prob = drop_prob
    
    def forward(self, x):
        if not self.training or self.drop_prob == 0.:
            return x
        
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # 二值化
        output = x.div(keep_prob) * random_tensor
        return output

# 在 ResNet block 中使用
class BasicBlock(nn.Module):
    def __init__(self, ..., drop_path_rate=0.1):
        ...
        self.drop_path = DropPath(drop_path_rate)
    
    def forward(self, x):
        shortcut = x
        out = self.conv2(self.conv1(x))
        out = self.drop_path(out)  # ← 这里
        return F.relu(out + shortcut)
```

#### 推荐配置

- Wide ResNet: 0.1-0.2
- ConvNeXt: 0.1 (Tiny), 0.4 (Base)
- 使用 **线性递增** 策略: 浅层 0.0 → 深层 max_drop_rate

### SAM (Sharpness-Aware Minimization)

#### 原理

- 寻找平坦的 loss landscape，提高泛化
- 两步优化: 先找最坏扰动，再在扰动点梯度下降

#### 实现

```python
class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False):
        self.base_optimizer = base_optimizer(params)
        self.rho = rho
        self.adaptive = adaptive
    
    @torch.no_grad()
    def first_step(self):
        # Step 1: 计算最坏扰动方向
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = self.rho / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None: continue
                e_w = p.grad * scale
                p.add_(e_w)  # 添加扰动
                self.state[p]["e_w"] = e_w
    
    @torch.no_grad()
    def second_step(self):
        # Step 2: 在扰动点计算梯度并优化
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.sub_(self.state[p]["e_w"])  # 移除扰动
        
        self.base_optimizer.step()
    
    def _grad_norm(self):
        norm = torch.norm(
            torch.stack([
                p.grad.norm(p=2)
                for group in self.param_groups
                for p in group["params"]
                if p.grad is not None
            ]),
            p=2
        )
        return norm

# 训练循环
optimizer = SAM(model.parameters(), torch.optim.SGD, rho=0.05)

loss = criterion(model(inputs), labels)
loss.backward()
optimizer.first_step(zero_grad=True)

# 第二次前向传播
criterion(model(inputs), labels).backward()
optimizer.second_step(zero_grad=True)
```

#### 推荐配置

- rho: 0.05 (CIFAR), 0.1 (ImageNet)
- adaptive: True (更稳定)
- 注意: 训练时间增加 ~2x

### Test-Time Augmentation (TTA)

#### 实现

```python
def predict_with_tta(model, image, num_augmentations=5):
    predictions = []
    
    # 原始图片
    predictions.append(model(image))
    
    # 水平翻转
    predictions.append(model(torch.flip(image, dims=[-1])))
    
    # 多次随机增强
    for _ in range(num_augmentations - 2):
        augmented = augment(image)  # 应用数据增强
        predictions.append(model(augmented))
    
    # 平均预测
    return torch.stack(predictions).mean(dim=0)
```

---

## 🎛️ 超参数调优经验

### Learning Rate

#### 基础规则

```python
# Linear Scaling Rule (Facebook AI)
lr_scaled = lr_base * (batch_size / 256)

# 示例:
# batch_size=128 → lr = 0.001 * (128/256) = 0.0005
# batch_size=256 → lr = 0.001
```

#### 不同优化器的推荐 LR

| 优化器 | CIFAR-10/100 推荐 | ImageNet 推荐 |
|--------|------------------|---------------|
| SGD | 0.1 | 0.1-0.2 |
| Adam | 0.001 | 0.001 |
| AdamW | 0.001 | 0.001-0.003 |

#### Warmup 策略

```python
# Linear Warmup
warmup_epochs = 10
for epoch in range(warmup_epochs):
    lr = lr_min + (lr_max - lr_min) * (epoch / warmup_epochs)
```

### Weight Decay

#### 经验法则

- 过小 (< 1e-4): 可能过拟合
- 适中 (1e-4 ~ 5e-4): 通用配置
- 过大 (> 1e-3): 可能欠拟合

#### 模型相关

- ResNet: 1e-4 ~ 5e-4
- Wide ResNet: 5e-4 ~ 1e-3
- ConvNeXt: 0.05 (较大！)

### Batch Size

#### 影响分析

- **小 batch (32-64)**:
  - ✅ 更好的泛化 (noise 更大)
  - ❌ 训练时间长
  - ❌ BN 统计不稳定

- **大 batch (256-512)**:
  - ✅ 训练快
  - ✅ BN 统计稳定
  - ❌ 需要更大的 LR
  - ❌ 可能泛化差

#### CIFAR 推荐

- **128**: 平衡点 (推荐)
- **256**: 如果显存充足

### Dropout

#### 不同位置的 dropout

```python
# FC 层: 0.3-0.5 (标准)
# ResNet bottleneck: 0.2-0.3
# Wide ResNet: 0.3 (原论文推荐)
# Transformer: 0.1 (ViT 推荐)
```

---

## 🐛 常见问题与解决方案

### 问题 1: 训练 loss 下降，验证 loss 上升 (过拟合)

**解决方案**:

1. ✅ 增加数据增强强度
2. ✅ 增大 weight decay
3. ✅ 增大 dropout
4. ✅ 减小模型容量
5. ✅ 添加 Stochastic Depth
6. ✅ 早停 (early stopping)

### 问题 2: 训练和验证 loss 都很高 (欠拟合)

**解决方案**:

1. ✅ 增大模型容量
2. ✅ 减小 weight decay
3. ✅ 增大学习率
4. ✅ 训练更多 epochs
5. ✅ 减弱数据增强
6. ✅ 检查数据预处理 (normalization 是否正确)

### 问题 3: Loss 突然爆炸 (NaN)

**原因**:

- 学习率过大
- 梯度爆炸
- 数值不稳定 (混合精度训练)

**解决方案**:

1. ✅ 减小学习率
2. ✅ 梯度裁剪 (clip_grad_norm)
3. ✅ 检查数据中是否有异常值
4. ✅ 关闭 AMP 测试
5. ✅ 使用更稳定的初始化 (He/Xavier)

### 问题 4: 训练速度慢

**优化方案**:

1. ✅ 启用 AMP (混合精度)
2. ✅ 增大 batch size
3. ✅ 增加 num_workers (DataLoader)
4. ✅ pin_memory=True
5. ✅ 使用 torch.compile() (PyTorch 2.0+)
6. ✅ 减少日志频率

### 问题 5: CUDA OOM (显存不足)

**解决方案**:

1. ✅ 减小 batch size
2. ✅ 梯度累积 (gradient accumulation)
3. ✅ 使用更小的模型
4. ✅ 减少 num_workers
5. ✅ torch.cuda.empty_cache()

### 问题 6: 某些类别 F1 特别低

**分析**:

- 查看混淆矩阵: 是否被误分类为某个特定类
- 查看数据: 是否样本不平衡

**解决方案**:

1. ✅ 使用 Focal Loss
2. ✅ 类别加权 (class weights)
3. ✅ 针对困难类别增强数据
4. ✅ 调整 Mixup/CutMix 策略

---

## 📊 性能基准 (CIFAR-100)

### SOTA 结果 (截至 2024)

| 模型 | 参数量 | Test Acc | 备注 |
|-----|--------|----------|------|
| AutoFormer | 23M | 88.6% | Transformer |
| ViT-H/14 | 632M | 90.9% | 预训练 + fine-tune |
| GPipe | - | 84.3% | 大规模集成 |
| PyramidNet-272 | 26M | 86.0% | 从头训练 |
| Shake-Shake-26 2x96d | 26M | 85.1% | 从头训练 |
| WRN-40-10 | 55M | 82.0% | 从头训练 |

### 我们的目标

- **F1 ≥ 0.85** ≈ **Test Acc ≥ 85%**
- 这在从头训练的情况下是**非常有挑战性的**！

---

**维护人员**: [你的名字]  
**参考资源**: Papers with Code, PyTorch Forums, Reddit r/MachineLearning
