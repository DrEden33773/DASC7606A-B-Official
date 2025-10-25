# EfficientNet-B0 参数量不匹配诊断

## 问题发现

2025-10-25

### 症状

用户报告训练时：

- GPU 占用率 < 50%
- 显存占用仅 4GB
- 对比 WRN-28-12: GPU 占用率 90%+, 显存 8-9GB

### 验证结果

```
当前实现参数量: 4,126,308 (4.13M)
期望参数量:     5,300,000 (5.30M)
差异:           -1,173,692 (-22.1%)
```

**❌ 严重问题: 参数量少了 22.1%**

## 配置对比

### 我的实现配置

```python
blocks_args = [
    [1,  16,  1, 1, 3],  # Stage 1: MBConv1, k3, c16, n1
    [6,  24,  2, 2, 3],  # Stage 2: MBConv6, k3, c24, n2
    [6,  40,  2, 2, 5],  # Stage 3: MBConv6, k5, c40, n2
    [6,  80,  3, 2, 3],  # Stage 4: MBConv6, k3, c80, n3
    [6, 112,  3, 1, 5],  # Stage 5: MBConv6, k5, c112, n3
    [6, 192,  4, 2, 5],  # Stage 6: MBConv6, k5, c192, n4
    [6, 320,  1, 1, 3],  # Stage 7: MBConv6, k3, c320, n1
]
# [expand_ratio, channels, num_layers, stride, kernel_size]
```

### 论文 Table 1 配置 (EfficientNet-B0 Baseline)

| Stage | Operator      | Resolution | Channels | Layers | Stride |
|-------|---------------|------------|----------|--------|--------|
| 1     | MBConv1, k3x3 | 112×112    | 16       | 1      | 1      |
| 2     | MBConv6, k3x3 | 112×112    | 24       | 2      | 2      |
| 3     | MBConv6, k5x5 | 56×56      | 40       | 2      | 2      |
| 4     | MBConv6, k3x3 | 28×28      | 80       | 3      | 2      |
| 5     | MBConv6, k5x5 | 14×14      | 112      | 3      | 1      |
| 6     | MBConv6, k5x5 | 14×14      | 192      | 4      | 2      |
| 7     | MBConv6, k3x3 | 7×7        | 320      | 1      | 1      |

### 对比结果

✅ 配置匹配（expand_ratio, channels, layers, kernel_size 都正确）

## 可能原因分析

### 假设 1: SE Block 实现缺陷

我的 SE Block 实现：

```python
class SEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        reduced_channels = max(1, channels // reduction)
        self.excitation = nn.Sequential(
            nn.Linear(channels, reduced_channels, bias=False),
            Swish(),
            nn.Linear(reduced_channels, channels, bias=False),
            nn.Sigmoid()
        )
```

**潜在问题**:

- 使用 `bias=False` 可能与官方实现不同
- `reduction=4` 是固定的，但官方可能根据 `se_ratio` 动态计算

### 假设 2: MBConvBlock 实现不完整

需要检查的点：

1. **Expansion phase**: 当 `expand_ratio != 1` 时，是否正确实现了扩展？
2. **Depthwise Conv**: groups 参数是否正确设置？
3. **Projection phase**: 1x1 Conv 是否正确实现？
4. **Bias 设置**: 所有 Conv 层都设置了 `bias=False`，是否正确？

### 假设 3: 官方实现使用了不同的 SE 设计

lukemelas/EfficientNet-PyTorch 实现特点：

- 使用 `Conv2d` 而不是 `Linear` 实现 SE block 的FC层
- SE ratio 计算方式可能不同

## 下一步行动

1. **下载并分析官方实现**
   - 对比 `lukemelas/EfficientNet-PyTorch` 的 `MBConvBlock` 实现
   - 检查 SE block 的详细实现

2. **逐层对比参数量**
   - 创建脚本对比每个 stage 的参数量
   - 找出哪个部分的参数缺失

3. **修复或重构**
   - 根据发现的问题修正实现
   - 重新验证参数量

## 参考

### EfficientNet-B0 参数分布（期望值）

根据论文和官方实现，5.3M 参数大致分布：

- Stem (Conv 3→32): ~864 params
- MBConv blocks: ~4.0M params (主体)
- Head (Conv 320→1280): ~409,600 params
- Classifier (FC 1280→100): ~128,000 params

### 实际测量值

- stem: 928 params (✓ 接近期望)
- blocks: 3,585,120 params (✗ 太少！应该 ~4.0M)
- head: 412,160 params (✓ 接近期望)
- classifier: 128,100 params (✓ 接近期望)

**结论: 问题出在 blocks 部分，少了约 40 万参数！**

## 紧急行动

立即深入分析 MBConvBlock 实现，找出 blocks 少了 40 万参数的原因。
