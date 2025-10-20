# 🚨 PyramidNet 超高 Loss 问题诊断

**症状**: Val Loss = 449,512,185 (vs 正常 4-6)  
**诊断**: 实现有严重 bug！

---

## 🔍 可能的问题

### 1. Zero-padding 实现错误 🚨

**当前实现**:

```python
shortcut = F.pad(
    shortcut,
    (0, 0, 0, 0, 0, pad_channels),  # 可能有问题！
    mode="constant",
    value=0,
)
```

**F.pad 格式** (从最后维度开始):

```
Input: [N, C, H, W]
Pad: (W_left, W_right, H_top, H_bottom, C_left, C_right, N_left, N_right)

(0, 0, 0, 0, 0, pad_channels) 的含义:
- W: 不 pad
- H: 不 pad  
- C: 左边 pad 0, 右边 pad pad_channels
- N: (只有 6 个参数，不涉及)
```

**问题**: C 维度 padding 应该在"右边"（channel 末尾），我写的是对的...

### 2. 初始 BN 问题 🚨

**当前实现**:

```python
# __init__
self.conv1 = nn.Conv2d(3, start_channels, ...)
self.bn1 = nn.BatchNorm2d(start_channels)

# forward
x = self.conv1(x)
x = self.bn1(x)  # 这可能有问题！
x = self.layer1(x)
```

**问题**:

- PyramidNet 用 pre-activation
- BasicBlock 内部已有 BN+ReLU
- 初始 conv 后不应该有 BN

### 3. 渐进通道计算可能有误

让我重
