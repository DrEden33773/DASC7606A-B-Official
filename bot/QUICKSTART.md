# Quick Start Guide - Wide ResNet-28-10

**最后更新**: 2025-10-17  
**当前阶段**: Phase 1 - Wide ResNet 实现完成

---

## 🚀 立即开始

### 1. 确认环境

```powershell
# 激活虚拟环境
.venv\Scripts\activate

# 验证 PyTorch 安装
python -c "import torch; print(f'PyTorch {torch.__version__} - CUDA available: {torch.cuda.is_available()}')"
```

### 2. 快速测试 Wide ResNet

```powershell
# 测试 Wide ResNet-28-10 实现
python -c "from scripts.model_architectures import create_model; import torch; model = create_model(100, 'cpu', 'wide_resnet28_10'); print('Model created successfully!'); print(f'Parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M')"
```

**预期输出**:

```
Model created successfully!
Parameters: 36.54M
```

### 3. 运行基线实验 (Exp #100)

```powershell
# 完整训练 (预计 3-4 小时)
python main.py `
    --model wide_resnet28_10 `
    --dropout 0.3 `
    --lr 0.001 `
    --weight_decay 5e-4 `
    --warmup_epochs 10 `
    --num_epochs 500 `
    --early_stopping_patience 50 `
    --batch_size 128 `
    --mixup_alpha 0.25 `
    --use_cutmix `
    --cutmix_alpha 0.65 `
    --aug_strength medium `
    --use_online_aug `
    --seed 42
```

### 4. 快速测试 (5 epochs)

```powershell
# 快速验证（约 5 分钟）
python main.py `
    --model wide_resnet28_10 `
    --dropout 0.3 `
    --lr 0.001 `
    --weight_decay 5e-4 `
    --num_epochs 5 `
    --batch_size 64 `
    --seed 42
```

---

## 📊 可用模型

| 模型 | 参数量 | 推荐用途 | Dropout |
|-----|--------|---------|---------|
| `wide_resnet28_10` | 36.5M | **Phase 1 默认** | 0.3 |
| `wide_resnet40_10` | 55.8M | Phase 3 大模型 | 0.3 |
| `wide_resnet28_12` | 52.8M | Phase 3 备选 | 0.3 |
| `resnet34` | 21M | 历史基准 | 0.5 |
| `resnet50` | 23.5M | 历史基准 | 0.5 |
| `resnet18` | 11M | 轻量级 | 0.5 |

---

## 🔧 常用命令

### 查看帮助

```powershell
python main.py --help
```

### 更改模型

```powershell
# 使用 Wide ResNet-40-10 (更深)
python main.py --model wide_resnet40_10 --batch_size 96

# 使用 ResNet-34 (历史基准)
python main.py --model resnet34 --dropout 0.5
```

### 调整超参数

```powershell
# 更激进的正则化
python main.py --model wide_resnet28_10 --dropout 0.4 --weight_decay 1e-3

# 更大的 batch size (需要更大学习率)
python main.py --model wide_resnet28_10 --batch_size 256 --lr 0.002

# 更长的训练
python main.py --model wide_resnet28_10 --num_epochs 800 --early_stopping_patience 80
```

### 不同的数据增强

```powershell
# 强增强
python main.py --model wide_resnet28_10 --aug_strength strong

# 更激进的 CutMix
python main.py --model wide_resnet28_10 --cutmix_alpha 1.0 --mixup_alpha 0.4
```

---

## 📁 项目结构

```
项目根目录/
├── main.py                          # 主训练脚本
├── scripts/
│   ├── model_architectures.py       # 模型定义 (含 Wide ResNet)
│   ├── data_augmentation.py         # 数据增强
│   ├── train_utils.py               # 训练工具
│   └── evaluation_metrics.py        # 评估指标
├── bot/
│   ├── README.md                    # Bot 目录索引
│   ├── QUICKSTART.md                # 本文件
│   ├── PHASE1_PROGRESS.md           # Phase 1 进度
│   ├── implementations/
│   │   └── wide_resnet.py           # Wide ResNet 实现
│   ├── experiments/
│   │   ├── experiment_tracker.md    # 实验追踪表
│   │   └── phase1/
│   │       └── exp_100_*.md         # 实验记录
│   ├── plans/
│   │   └── optimization_roadmap.md  # 完整路线图
│   └── notes/
│       ├── project_standards.md     # 开发规范
│       ├── technical_notes.md       # 技术笔记
│       └── quick_reference.md       # 命令速查
├── data/                            # 数据目录 (自动生成)
├── results/                         # 结果目录 (自动生成)
└── .venv/                           # 虚拟环境
```

---

## 🐛 故障排查

### 问题 1: ModuleNotFoundError: No module named 'torch'

**解决方案**:

```powershell
# 激活虚拟环境
.venv\Scripts\activate

# 或重新安装依赖
uv pip install -e .
```

### 问题 2: CUDA Out of Memory

**解决方案**:

```powershell
# 减小 batch size
python main.py --model wide_resnet28_10 --batch_size 64

# 或使用梯度累积（待实现）
```

### 问题 3: 训练速度慢

**解决方案**:

```powershell
# 确保启用 AMP 和 compile
python main.py --model wide_resnet28_10 --use_amp --use_compile

# 增加 num_workers（Windows 建议 4）
python main.py --model wide_resnet28_10 --num_workers 4
```

### 问题 4: 验证 F1 不提升

**分析步骤**:

1. 查看训练日志: `cifar_pipeline.log`
2. 检查是否过拟合（train loss ↓, val loss ↑）
3. 尝试更强的正则化（增大 dropout, weight_decay）
4. 检查数据增强是否过强

---

## 📈 监控训练

### 实时查看日志

```powershell
# PowerShell
Get-Content -Path "cifar_pipeline.log" -Wait -Tail 50

# 或使用编辑器打开
code cifar_pipeline.log
```

### 关键指标

```
Epoch XX/500:
  Train Loss: X.XXXX, Train Acc: XX.XX%, LR: 0.XXXXXX
  Val Loss: X.XXXX, Val Acc: XX.XX%, Val F1: 0.XXXX
  ↳ Validation improved (...)  # 表示保存了新的最佳模型
```

### 早停信号

```
Early stopping triggered after XXX epochs!
```

表示训练提前结束（验证指标 50 轮未提升）

---

## 🎯 实验记录

每次实验后请记录：

1. **实验编号** - 如 Exp #100
2. **配置** - 完整的超参数
3. **结果** - Val F1, Test F1
4. **分析** - 成功/失败原因
5. **下一步** - 改进方向

**模板**: `bot/experiments/experiment_template.md`  
**追踪**: `bot/experiments/experiment_tracker.md`

---

## 🔗 相关文档

- [完整路线图](plans/optimization_roadmap.md) - 三阶段计划
- [项目规范](notes/project_standards.md) - 开发标准
- [技术笔记](notes/technical_notes.md) - 实现细节
- [快速参考](notes/quick_reference.md) - 命令速查
- [Phase 1 进度](PHASE1_PROGRESS.md) - 当前进度

---

## ✅ 检查清单

开始训练前:

- [ ] 虚拟环境已激活
- [ ] CUDA 可用 (torch.cuda.is_available() = True)
- [ ] 数据已下载 (data/raw/train, data/raw/val)
- [ ] 磁盘空间充足 (>5GB)
- [ ] 了解预计训练时间 (3-4 小时)

训练完成后:

- [ ] 记录 Val F1 到实验追踪表
- [ ] 分析困难类别 (哪些类 F1 最低)
- [ ] 填写实验记录 (成功/失败原因)
- [ ] 决定下一步实验方向
- [ ] 更新 TODO list

---

**祝训练成功！** 🚀

有问题请查看 `bot/notes/technical_notes.md` 的故障排查部分。
