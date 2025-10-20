# 🚨 NaN Loss 问题诊断

**观察**: Epoch 28 开始 Train Loss = nan, Val Loss = nan  
**症状**: 训练崩溃，梯度爆炸

---

## 🔍 可能原因

### 1. Loss 幅度过大 (4 个分类器)

**问题**:

```python
# 当前实现
for logits in all_logits:  # 4 个分类器
    total_loss += (1 - alpha) * ce_loss

# Loss 幅度 = 正常的 4x!
```

**后果**: 梯度爆炸 → NaN

### 2. KL Divergence 数值不稳定

**问题**: log(0) = -inf

```python
F.kl_div(log_softmax(...), softmax(...))
# 如果 teacher 的某个类概率接近 0
# log(0) → -inf → NaN
```

### 3. Temperature scaling

**T=4.0 可能太大**: softmax 过于平滑 → 接近 uniform → 梯度消失/爆炸

---

## 🔧 修复方案

### 1. Loss 归一化 (除以分类器数量)

### 2. KL Divergence 使用 reduction='sum' 然后手动归一化  

### 3. 降低 temperature (4.0 → 3.0)

### 4. 降低 alpha (0.9 → 0.7)
