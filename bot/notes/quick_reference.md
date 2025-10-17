# 快速参考指南

**最后更新**: 2025-10-17

---

## 🚀 常用训练命令

### ResNet50 (当前最佳配置)

```bash
python main.py \
    --model resnet50 \
    --no_pretrained \
    --dropout 0.45 \
    --lr 0.0008 \
    --weight_decay 1.2e-3 \
    --warmup_epochs 20 \
    --num_epochs 350 \
    --early_stopping_patience 35 \
    --batch_size 96 \
    --mixup_alpha 0.2 \
    --use_cutmix \
    --cutmix_alpha 0.6 \
    --aug_strength medium \
    --use_online_aug \
    --seed 42
```

**结果**: Val F1 = 0.77

---

### ResNet34 (当前最佳配置 #2)

```bash
python main.py \
    --model resnet34 \
    --no_pretrained \
    --lr 0.0012 \
    --dropout 0.5 \
    --weight_decay 8e-4 \
    --warmup_epochs 10 \
    --aug_strength light \
    --use_cutmix \
    --cutmix_alpha 0.65 \
    --mixup_alpha 0.25 \
    --num_epochs 500 \
    --early_stopping_patience 50 \
    --seed 42
```

**结果**: Val F1 = 0.77

---

### Wide ResNet-28-10 (Phase 1 目标)

```bash
# 待实现模型后更新
python main.py \
    --model wide_resnet28_10 \
    --no_pretrained \
    --dropout 0.3 \
    --lr 0.001 \
    --weight_decay 5e-4 \
    --warmup_epochs 20 \
    --num_epochs 500 \
    --early_stopping_patience 60 \
    --batch_size 128 \
    --mixup_alpha 0.2 \
    --use_cutmix \
    --cutmix_alpha 0.8 \
    --aug_strength medium \
    --use_online_aug \
    --seed 42
```

**预期**: Val F1 ≥ 0.80

---

## 📋 参数速查表

### 模型选择

```bash
--model resnet18          # 11M 参数
--model resnet34          # 21M 参数
--model resnet50          # 23.5M 参数
--model wide_resnet28_10  # 36M 参数 (待实现)
--model convnext_tiny     # 28M 参数 (待实现)
```

### 预训练

```bash
--use_pretrained          # 启用 ImageNet 预训练 (禁止使用!)
--no_pretrained          # 从头训练 (必须使用)
```

### 数据增强

```bash
--aug_strength light      # 轻度增强
--aug_strength medium     # 中度增强 (推荐)
--aug_strength strong     # 重度增强 (可能过度)

--use_online_aug         # 在线增强 (动态, 推荐)
--no_online_aug          # 离线增强 (预生成)

--mixup_alpha 0.2        # Mixup 强度 (0=禁用)
--use_cutmix             # 启用 CutMix
--cutmix_alpha 0.6       # CutMix 强度
```

### 优化器与调度器

```bash
--optimizer adam         # Adam
--optimizer adamw        # AdamW (推荐)
--optimizer sgd          # SGD with momentum

--scheduler plateau      # ReduceLROnPlateau
--scheduler cosine       # CosineAnnealing (推荐)
--scheduler step         # StepLR
--scheduler onecycle     # OneCycleLR
```

### 正则化

```bash
--dropout 0.5            # Dropout 率
--weight_decay 1e-3      # L2 正则化
--label_smoothing 0.1    # 标签平滑 (与 Mixup 互斥)
--max_grad_norm 1.0      # 梯度裁剪
```

### 训练技巧

```bash
--use_amp                # 混合精度训练 (推荐)
--no_amp                 # 禁用 AMP

--use_ema                # 指数移动平均 (推荐)
--no_ema                 # 禁用 EMA
--ema_decay 0.9999       # EMA 衰减率

--use_compile            # torch.compile() 加速
--no_compile             # 禁用编译
```

### 训练参数

```bash
--batch_size 128         # Batch size
--lr 0.001              # 学习率
--num_epochs 300        # 总轮数
--warmup_epochs 10      # 预热轮数
--early_stopping_patience 30  # 早停耐心
```

### 硬件与随机种子

```bash
--device cuda           # 使用 GPU
--device cpu            # 使用 CPU
--num_workers 4         # 数据加载线程数
--seed 42               # 随机种子
```

---

## 🧪 调试命令

### 快速测试 (小规模)

```bash
python main.py \
    --model resnet18 \
    --no_pretrained \
    --batch_size 64 \
    --num_epochs 10 \
    --early_stopping_patience 5 \
    --num_workers 2
```

### 单 GPU 内存测试

```bash
python main.py \
    --model wide_resnet28_10 \
    --batch_size 256 \
    --num_epochs 1 \
    --num_workers 0
```

### 过拟合测试 (验证模型能力)

```bash
# 在小数据集上快速过拟合
python main.py \
    --model resnet34 \
    --batch_size 32 \
    --num_epochs 50 \
    --no_online_aug \
    --mixup_alpha 0 \
    --weight_decay 0
```

---

## 📊 实验对比模板

### Ablation Study: 数据增强

| 配置 | Mixup | CutMix | Aug | Val F1 | 提升 |
|-----|-------|--------|-----|--------|------|
| Baseline | ❌ | ❌ | light | 0.75 | - |
| + Mixup | ✅ | ❌ | light | 0.76 | +0.01 |
| + CutMix | ❌ | ✅ | light | 0.76 | +0.01 |
| + Both | ✅ | ✅ | light | 0.77 | +0.02 |
| + Stronger Aug | ✅ | ✅ | medium | 0.78 | +0.03 |

### Ablation Study: 模型架构

| 模型 | 参数量 | 训练时间/epoch | Val F1 | 备注 |
|-----|--------|---------------|--------|------|
| ResNet-18 | 11M | 20s | 0.75 | 基线 |
| ResNet-34 | 21M | 35s | 0.77 | 当前最佳 |
| ResNet-50 | 23M | 40s | 0.77 | 当前最佳 |
| WRN-28-10 | 36M | 60s | TBD | Phase 1 目标 |

---

## 🔍 日志分析

### 关键指标

```python
# 查看训练日志
grep "Val F1" cifar_pipeline.log | tail -20

# 查看最佳 epoch
grep "Validation improved" cifar_pipeline.log

# 查看早停触发
grep "Early stopping triggered" cifar_pipeline.log
```

### 绘制曲线 (简单版)

```python
import re
import matplotlib.pyplot as plt

with open('cifar_pipeline.log') as f:
    lines = f.readlines()

epochs, val_f1s = [], []
for line in lines:
    if 'Val F1:' in line:
        epoch = int(re.search(r'Epoch (\d+)/', line).group(1))
        f1 = float(re.search(r'Val F1: ([\d.]+)', line).group(1))
        epochs.append(epoch)
        val_f1s.append(f1)

plt.plot(epochs, val_f1s)
plt.xlabel('Epoch')
plt.ylabel('Val F1')
plt.title('Training Progress')
plt.savefig('training_curve.png')
```

---

## 🎯 超参数搜索空间

### Grid Search (Phase 1)

#### Learning Rate

```bash
for lr in 0.0005 0.0008 0.001 0.0012 0.0015; do
    python main.py --lr $lr --seed 42 > logs/lr_${lr}.log 2>&1 &
done
```

#### Weight Decay

```bash
for wd in 5e-4 8e-4 1e-3 1.5e-3 2e-3; do
    python main.py --weight_decay $wd --seed 42 > logs/wd_${wd}.log 2>&1 &
done
```

#### Batch Size + LR (Linear Scaling)

```bash
# batch_size=64  → lr=0.0005
# batch_size=128 → lr=0.001
# batch_size=256 → lr=0.002
for bs in 64 128 256; do
    lr=$(echo "scale=4; 0.001 * $bs / 128" | bc)
    python main.py --batch_size $bs --lr $lr --seed 42 > logs/bs_${bs}.log 2>&1 &
done
```

### Random Search (Phase 2)

```python
import random
import subprocess

# 搜索空间
lr_range = (0.0003, 0.003)
wd_range = (1e-4, 5e-3)
dropout_range = (0.3, 0.6)

for i in range(20):  # 20 次随机试验
    lr = random.uniform(*lr_range)
    wd = random.uniform(*wd_range)
    dropout = random.uniform(*dropout_range)
    
    cmd = f"""python main.py \
        --lr {lr:.6f} \
        --weight_decay {wd:.6f} \
        --dropout {dropout:.3f} \
        --seed {42 + i} \
        > logs/random_{i}.log 2>&1"""
    
    subprocess.Popen(cmd, shell=True)
```

---

## 💾 模型管理

### 保存/加载检查点

```python
# 保存
checkpoint = {
    'epoch': epoch,
    'state_dict': model.state_dict(),
    'optimizer': optimizer.state_dict(),
    'best_f1': best_f1,
}
torch.save(checkpoint, 'checkpoint.pth')

# 加载
checkpoint = torch.load('checkpoint.pth')
model.load_state_dict(checkpoint['state_dict'])
optimizer.load_state_dict(checkpoint['optimizer'])
start_epoch = checkpoint['epoch']
best_f1 = checkpoint['best_f1']
```

### 模型集成

```python
# 加载多个模型
models = []
for path in ['model1.pth', 'model2.pth', 'model3.pth']:
    model = create_model(...)
    model.load_state_dict(torch.load(path))
    model.eval()
    models.append(model)

# Soft voting
def ensemble_predict(models, x):
    probs = []
    with torch.no_grad():
        for model in models:
            logits = model(x)
            probs.append(F.softmax(logits, dim=1))
    
    avg_probs = torch.stack(probs).mean(dim=0)
    return avg_probs.argmax(dim=1)
```

---

## 🐛 故障排查清单

### 训练无法启动

- [ ] 检查 CUDA 是否可用: `torch.cuda.is_available()`
- [ ] 检查数据路径是否正确
- [ ] 检查依赖是否安装: `pip list | grep torch`

### 训练中断

- [ ] 检查显存占用: `nvidia-smi`
- [ ] 检查磁盘空间: `df -h`
- [ ] 查看错误日志: `tail -50 cifar_pipeline.log`

### 性能不达预期

- [ ] 对比训练曲线: 过拟合? 欠拟合?
- [ ] 检查数据增强: 是否过强/过弱?
- [ ] 检查学习率: 是否过大/过小?
- [ ] 检查正则化: dropout, weight_decay

### OOM (Out of Memory)

- [ ] 减小 batch_size
- [ ] 减小模型 (resnet50 → resnet34)
- [ ] 减少 num_workers
- [ ] 启用梯度检查点 (gradient checkpointing)

---

## 📚 常用 Git 命令

```bash
# 保存当前工作
git add bot/ scripts/ main.py
git commit -m "Phase 1: Implemented Wide ResNet-28-10, F1=0.80"
git push

# 创建实验分支
git checkout -b experiment/wrn-28-10

# 回退到之前的版本
git log --oneline  # 查看历史
git checkout <commit-hash> -- scripts/model_architectures.py

# 查看改动
git diff scripts/train_utils.py
```

---

**快速链接**:

- [完整优化路线图](../plans/optimization_roadmap.md)
- [技术笔记](./technical_notes.md)
- [实验追踪表](../experiments/experiment_tracker.md)
