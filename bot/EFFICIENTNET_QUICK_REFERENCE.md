# EfficientNet B0-B4 快速参考

## 模型对比速查表

| 模型 | 参数量 | 显存 (bs=128) | GPU利用率 | 预期 F1 | 训练时间 | 推荐场景 |
|------|--------|--------------|-----------|---------|---------|---------|
| B0 | 4.1M | 4-6GB | 40-60% | 0.83-0.85 | 4-5h | 快速验证 |
| **B1** | **6.7M** | **6-8GB** | **60-70%** | **0.85-0.87** | **6-7h** | **首选** |
| B2 | 8.0M | 8-10GB | 70-80% | 0.86-0.88 | 7-9h | 高精度 |
| B3 | 10.8M | 10-12GB | 80-90% | 0.87-0.89 | 8-10h | 冲刺 |
| B4 | 17.8M | 12-15GB | 85-95% | 0.88-0.90 | 10-12h | 极限 |

## 一键训练脚本

### 推荐配置 (B1)

```powershell
.\temp\train_efficientnet_b1.ps1
```

### 其他选项

```powershell
# B2 (更高精度)
.\temp\train_efficientnet_b2.ps1

# B0 (大batch)
.\temp\train_efficientnet_b0_large_batch.ps1

# B1 快速测试 (50 epochs)
.\temp\test_efficientnet_b1_50epochs.ps1
```

## 超参数对照表

### SGD 配置 (推荐)

| 参数 | B0 | B1 | B2 | B3 | 说明 |
|------|----|----|----|----|------|
| `--lr` | 0.1 | 0.1 | 0.1 | 0.1 | 标准CIFAR lr |
| `--weight_decay` | 5e-4 | 5e-4 | 5e-4 | 5e-4 | SGD经典值 |
| `--warmup_epochs` | 10 | 15 | 15 | 20 | 更大模型需更长warmup |
| `--batch_size` | 256 | 192 | 160 | 128 | 根据显存调整 |
| `--dropout` | 0.2 | 0.2 | 0.3 | 0.3 | 更大模型需更强正则化 |
| `--label_smoothing` | 0.1 | 0.1 | 0.1 | 0.1 | 防止过拟合 |

### AdamW 配置 (备选)

| 参数 | 所有模型 | 说明 |
|------|---------|------|
| `--lr` | 0.001 | AdamW 标准lr |
| `--weight_decay` | 1e-3 | AdamW 推荐值 |
| `--warmup_epochs` | 10 | 比SGD短 |
| `--batch_size` | 96-192 | 根据模型大小 |

## 常见问题

### Q1: 显存不足 (OOM)

**解决方案**:

1. 降低 batch_size (192 → 128 → 96)
2. 使用更小的模型 (B2 → B1 → B0)
3. 检查是否启用了 `--use_amp` (默认开启)

### Q2: GPU 占用率低

**这是正常的！**

- EfficientNet 本就是高效模型
- B0 占用率 40-60% 是正常的
- 想提高占用率 → 使用 B1/B2/B3

### Q3: 训练太慢

**优化建议**:

1. 确保 `--use_amp` 开启 (混合精度)
2. 确保 `--use_compile` 开启 (torch.compile)
3. 增大 batch_size (如果显存允许)
4. 减少 `--num_epochs` 进行测试

### Q4: Loss 震荡或不收敛

**检查清单**:

1. 降低学习率 (0.1 → 0.08 或 0.05)
2. 延长 warmup (15 → 20 epochs)
3. 启用 label_smoothing=0.1
4. 检查数据增强是否过强

### Q5: F1 score 低于预期

**诊断步骤**:

1. 查看 50 epoch 时的 F1:
   - < 0.60 → 超参数有问题
   - 0.60-0.70 → 可以继续，但可能需要调整
   - > 0.70 → 正常，继续训练
2. 检查 training loss 是否正常下降
3. 查看 validation loss 是否过拟合
4. 尝试调整数据增强强度

## 监控指标

### 关键时间点

| Epoch | 检查项 | 正常范围 | 异常 |
|-------|--------|---------|------|
| 10 | Loss是否下降 | < 3.5 | > 4.0 |
| 50 | Val F1 | 0.65-0.72 | < 0.60 |
| 100 | Val F1 | 0.72-0.78 | < 0.68 |
| 200 | Val F1 | 0.78-0.82 | < 0.74 |
| 600 | Final F1 | 0.83-0.87 | < 0.80 |

### 显存占用参考

| 配置 | 预期显存 | 警告阈值 |
|------|---------|---------|
| B0 bs=256 | 6-8GB | > 10GB |
| B1 bs=192 | 6-8GB | > 10GB |
| B2 bs=160 | 8-10GB | > 12GB |
| B3 bs=128 | 10-12GB | > 14GB |

## 代码示例

### Python API

```python
from scripts.model_architectures import create_model

# 创建 EfficientNet-B1
model = create_model(
    num_classes=100,
    device='cuda',
    model_type='efficientnet_b1',
    input_size=64,
    dropout_rate=0.2,
    drop_path_rate=0.2
)

# 检查参数量
params = sum(p.numel() for p in model.parameters())
print(f"Parameters: {params/1e6:.2f}M")  # 应该是 ~6.6M
```

### 自定义训练命令

```powershell
python main.py `
    --model efficientnet_b1 `
    --input_size 64 `
    --batch_size 192 `
    --num_epochs 600 `
    --optimizer sgd `
    --lr 0.1 `
    --weight_decay 5e-4 `
    --warmup_epochs 15 `
    --scheduler cosine `
    --label_smoothing 0.1 `
    --aug_strength randaugment `
    --randaugment_n 2 `
    --randaugment_m 9 `
    --use_cutmix `
    --cutmix_alpha 1.0 `
    --use_amp `
    --use_ema `
    --max_grad_norm 1.0 `
    --seed 42
```

## 性能优化技巧

### 1. 充分利用 GPU

**问题**: GPU 占用率 < 50%

**方案**:

- 使用更大的模型 (B0 → B1 → B2)
- 增大 batch_size
- 减少 data loader 的 num_workers (避免CPU瓶颈)

### 2. 加速训练

**启用所有加速技术**:

- `--use_amp` (混合精度, 1.5-2x 加速)
- `--use_compile` (torch.compile, 1.1-1.2x 加速)
- 增大 batch_size (更少的 iterations)

### 3. 提升精度

**组合技巧**:

- 使用更大的模型 (B1 → B2 → B3)
- 启用 label_smoothing=0.1
- 延长训练时间 (600 → 800 epochs)
- 数据增强: RandAugment + CutMix

### 4. 防止过拟合

**正则化组合**:

- Dropout (0.2-0.3)
- DropPath (0.2)
- Weight Decay (5e-4)
- Label Smoothing (0.1)
- EMA (默认开启)

## 快速决策树

```
需要 F1 ≥ 0.85？
├─ 是 → 时间充裕？
│  ├─ 是 (8h+) → 使用 B2 或 B3
│  └─ 否 (6h) → 使用 B1 (推荐)
└─ 否 (测试/快速验证) → 使用 B0 (4-5h)

显存有限？
├─ < 8GB → B0 (bs=256) 或 B1 (bs=128)
├─ 8-12GB → B1 (bs=192) 或 B2 (bs=160)
└─ > 12GB → B2/B3/B4 (更大 batch)

GPU 占用率低？
├─ < 40% → 使用 B1/B2
├─ 40-60% → 正常 (EfficientNet 特性)
└─ > 60% → 已经很好

训练不收敛？
├─ Loss 震荡 → 降低 lr 或延长 warmup
├─ Loss 不下降 → 检查数据增强
└─ Val loss 上升 → 过拟合，增强正则化
```

## 推荐最终配置

**保守方案** (成功率 90%+):

```powershell
python main.py --model efficientnet_b1 --input_size 64 --batch_size 192 `
    --optimizer sgd --lr 0.08 --weight_decay 5e-4 --warmup_epochs 15 `
    --label_smoothing 0.1 --num_epochs 600
```

**激进方案** (冲击 0.87+):

```powershell
python main.py --model efficientnet_b2 --input_size 64 --batch_size 160 `
    --optimizer sgd --lr 0.1 --weight_decay 5e-4 --warmup_epochs 15 `
    --label_smoothing 0.1 --dropout 0.3 --num_epochs 600
```

**快速测试** (1-2小时):

```powershell
python main.py --model efficientnet_b1 --input_size 64 --batch_size 192 `
    --num_epochs 50 --optimizer sgd --lr 0.1 --weight_decay 5e-4
```

## 联系与支持

如有问题，请参考:

- 完整文档: `bot/EFFICIENTNET_B0_B4_COMPLETE.md`
- 参数量分析: `bot/analysis/efficientnet_parameter_truth.md`
- SGD配置分析: `bot/analysis/efficientnet_sgd_config_analysis.md`
