# 项目开发规范

**最后更新**: 2025-10-17  
**适用分支**: Phase 1 优化分支

---

## 📋 代码规范

### 1. 类型注解 (Type Annotations)

**强制要求**: 所有函数必须有完整的类型注解

```python
# ✅ 正确示例
def create_model(
    num_classes: int,
    device: str,
    dropout_rate: float = 0.3,
) -> nn.Module:
    """Create and return a model instance."""
    ...

# ❌ 错误示例
def create_model(num_classes, device, dropout_rate=0.3):
    ...
```

**类型检查模式**: `standard` (见 `.vscode/settings.json`)

```json
{
    "cursorpyright.analysis.typeCheckingMode": "standard"
}
```

---

### 2. 类型错误修复优先级

当遇到类型错误时，按以下优先级修复：

#### ✅ 优先级 1: 使用正确的类型注解

```python
# 明确类型
def forward(self, x: torch.Tensor) -> torch.Tensor:
    ...

# 使用 Union/Optional
from typing import Optional, Union
def __init__(self, dropout: Optional[float] = None) -> None:
    ...
```

#### ✅ 优先级 2: 类型窄化 (Type Narrowing)

```python
# 使用 isinstance 检查
if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
    scheduler.step(val_loss)

# 使用 assert 确保类型
assert steps_per_epoch is not None, "steps_per_epoch required"
```

#### ✅ 优先级 3: 使用 cast()

```python
from typing import cast

# 当你确定类型时
compiled_model = torch.compile(model)
model = cast(nn.Module, compiled_model)
```

#### ❌ 最后选择: type: ignore

```python
# 仅在前三种方法都不适用时使用
model.maxpool = nn.Identity()  # type: ignore[assignment]

# 必须添加说明注释
# type: ignore[assignment]  # nn.Identity is compatible with maxpool
```

---

### 3. 文档字符串 (Docstrings)

**强制要求**: 所有公共函数/类必须有 docstring

```python
def wide_resnet28_10(
    num_classes: int = 100,
    dropout_rate: float = 0.3
) -> nn.Module:
    """
    Construct Wide ResNet-28-10 model for CIFAR datasets.
    
    Wide ResNet uses wider layers (more channels) instead of deeper 
    architecture, which is more effective for CIFAR-sized images.
    
    Args:
        num_classes: Number of output classes (100 for CIFAR-100)
        dropout_rate: Dropout rate for regularization (default: 0.3)
        
    Returns:
        Wide ResNet-28-10 model instance
        
    References:
        Zagoruyko & Komodakis. "Wide Residual Networks" (BMVC 2016)
        https://arxiv.org/abs/1605.07146
        
    Example:
        >>> model = wide_resnet28_10(num_classes=100, dropout_rate=0.3)
        >>> print(sum(p.numel() for p in model.parameters()))  # ~36.5M params
    """
    return WideResNet(depth=28, widen_factor=10, num_classes=num_classes, 
                      dropout_rate=dropout_rate)
```

**Docstring 风格**: Google Style

---

### 4. 注释规范

#### 行内注释

```python
# ✅ 解释"为什么"，而不是"做什么"
channels = [16, 160, 320, 640]  # 16*k*widen_factor (k=1,10,20,40)

# ❌ 重复代码内容
x = x + 1  # Add 1 to x
```

#### 块注释

```python
# ✅ 用于复杂逻辑说明
# Wide ResNet uses dropout between BN-ReLU-Conv layers, which differs
# from standard ResNet. This placement was found to be more effective
# for regularization without hurting gradient flow.
out = self.dropout(F.relu(self.bn1(out)))
```

---

### 5. 代码清理规则

#### 删除标准

**必须删除**:

1. ❌ 未使用的导入
2. ❌ 未使用的函数/类
3. ❌ 注释掉的代码（超过1周未使用）
4. ❌ 调试用的 print 语句
5. ❌ 过时的超参数选项

**保留**:

1. ✅ 已实现但当前未启用的功能（如 Focal Loss）
2. ✅ 向后兼容的接口
3. ✅ 文档化的实验性功能

#### 示例

```python
# ❌ 删除：SimpleCNN 已被 ResNet 替代，无复用价值
class SimpleCNN(nn.Module):
    ...

# ✅ 保留：Focal Loss 已实现，可通过参数启用
class FocalLoss(nn.Module):
    ...

# ❌ 删除：调试代码
# print(f"DEBUG: x.shape = {x.shape}")

# ✅ 保留：有意义的日志
logger.info(f"Model created with {param_count/1e6:.1f}M parameters")
```

---

## 🏗️ 架构规范

### 1. 模型实现规范

#### 文件组织

```
scripts/model_architectures.py
├── Protocol/Interface 定义
├── 基础 Block 实现 (BasicBlock, Bottleneck, etc.)
├── 完整模型类 (ResNetCIFAR, WideResNet, etc.)
├── 工厂函数 (resnet18_cifar, wide_resnet28_10, etc.)
└── 统一创建接口 (create_model)
```

#### 模型类结构

```python
class WideResNet(nn.Module):
    """Wide ResNet architecture for CIFAR."""
    
    def __init__(
        self,
        depth: int,
        widen_factor: int,
        num_classes: int = 100,
        dropout_rate: float = 0.3,
    ) -> None:
        """Initialize Wide ResNet.
        
        Args:
            depth: Network depth (e.g., 28, 40). Must satisfy (depth-4)%6==0
            widen_factor: Channel width multiplier (e.g., 10, 12)
            num_classes: Number of output classes
            dropout_rate: Dropout rate in residual blocks
        """
        super().__init__()
        # 1. 验证参数
        # 2. 初始化层
        # 3. 初始化权重
        
    def _make_layer(...) -> nn.Sequential:
        """Create a layer with multiple blocks."""
        ...
        
    def _initialize_weights(self) -> None:
        """Initialize model weights using He initialization."""
        ...
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor [batch_size, 3, 32, 32]
            
        Returns:
            Output logits [batch_size, num_classes]
        """
        ...
```

---

### 2. 超参数规范

#### main.py 参数组织

```python
parser = argparse.ArgumentParser()

# 1. Dataset & Paths
parser.add_argument("--dataset", ...)
parser.add_argument("--data_dir", ...)

# 2. Model Architecture
parser.add_argument("--model", ...)
parser.add_argument("--dropout", ...)

# 3. Data Augmentation
parser.add_argument("--aug_strength", ...)
parser.add_argument("--mixup_alpha", ...)

# 4. Training Parameters
parser.add_argument("--batch_size", ...)
parser.add_argument("--lr", ...)

# 5. Training Techniques
parser.add_argument("--use_amp", ...)
parser.add_argument("--use_ema", ...)
```

#### 删除过时参数

```python
# ❌ 删除：预训练相关（已禁止使用）
# parser.add_argument("--use_pretrained", ...)
# parser.add_argument("--freeze_backbone", ...)

# ❌ 删除：未使用的 loss 选项
# parser.add_argument("--use_class_weights", ...)  # 效果不佳

# ✅ 保留：已实现且有用的功能
parser.add_argument("--use_cutmix", ...)  # 证明有效
```

---

## 🧪 实验规范

### 1. 实现新模型的步骤

1. **研究最佳实践** (搜索论文/代码)

   ```bash
   # 搜索关键词
   "Wide ResNet CIFAR-100 best practices"
   "WRN-28-10 hyperparameters CIFAR"
   ```

2. **创建实现文件**

   ```bash
   bot/implementations/wide_resnet.py
   ```

3. **集成到主项目**
   - 添加到 `scripts/model_architectures.py`
   - 添加到 `create_model()` 函数
   - 添加命令行参数选项

4. **测试实现**

   ```python
   # 单元测试
   python -c "from scripts.model_architectures import create_model; \
              model = create_model(100, 'cpu', 'wide_resnet28_10'); \
              print(sum(p.numel() for p in model.parameters()))"
   ```

5. **运行基线实验**

   ```bash
   python main.py --model wide_resnet28_10 --num_epochs 5 --batch_size 32
   ```

6. **记录结果**

   ```bash
   bot/experiments/phase1/exp_100_wide_resnet_baseline.md
   ```

---

### 2. 实验记录规范

每个实验必须记录：

```markdown
## Experiment #100 - Wide ResNet-28-10 Baseline

**日期**: 2025-10-17
**目标**: 验证 Wide ResNet-28-10 实现正确性
**预期**: Val F1 ≥ 0.77 (当前 ResNet50 水平)

### 配置
- Model: wide_resnet28_10
- Dropout: 0.3
- LR: 0.001
- Weight Decay: 5e-4
- Batch Size: 128
- Epochs: 500

### 结果
- Val F1: 0.XX
- Training Time: X.X hours
- 参数量: 36.5M

### 分析
- [成功/失败原因]
- [与 ResNet50 对比]
- [下一步计划]
```

---

## 🔧 开发工作流

### 1. 分支管理

```bash
# 当前分支
resnet-best-practice  # Phase 1 开发分支

# 命名规范（未来）
phase1-wide-resnet    # 功能分支
phase2-convnext       # 功能分支
experiment/wrn-sam    # 实验分支
```

### 2. 提交规范

```bash
# 格式: [类型] 简短描述

# 类型标签
feat:     # 新功能
fix:      # Bug 修复
refactor: # 代码重构
docs:     # 文档更新
exp:      # 实验相关

# 示例
git commit -m "feat: Implement Wide ResNet-28-10 for CIFAR-100"
git commit -m "exp: #100 Wide ResNet baseline, F1=0.XX"
git commit -m "refactor: Remove unused SimpleCNN and pretrained code"
git commit -m "fix: Correct Wide ResNet dropout placement"
```

---

## 🐛 Linting & 类型检查

### 1. 运行检查

```bash
# 类型检查（Pyright）
# 自动运行，查看 VS Code 问题面板

# 代码格式检查
# （如果配置了 black/ruff）
black scripts/ --check
ruff check scripts/
```

### 2. 修复优先级

1. **类型错误** (Type Errors) - 最高优先级
2. **未使用的导入/变量** - 高优先级
3. **代码风格** - 中优先级
4. **文档缺失** - 低优先级（但必须补充）

---

## 🌐 跨平台注意事项

### Windows 特定

```python
# ✅ 使用 os.path.join 而非字符串拼接
data_dir = os.path.join(base_dir, "raw", "train")

# ❌ 硬编码路径分隔符
data_dir = base_dir + "/raw/train"  # Windows 用 \

# ✅ 使用 pathlib
from pathlib import Path
data_dir = Path(base_dir) / "raw" / "train"
```

### Shell 命令

```bash
# Windows 环境下运行 Linux 命令
wsl bash -c "cd /mnt/e/path/to/project && ./script.sh"

# 或使用 PowerShell 等效命令
# 避免直接调用 bash 脚本
```

---

## 📦 依赖管理

### pyproject.toml

```toml
[project]
name = "dasc7606-cifar"
version = "1.0.0"
requires-python = ">=3.13"
dependencies = [
    "torch>=2.0.0",
    "torchvision>=0.15.0",
    "albumentations>=1.3.0",
    "scikit-learn>=1.3.0",
    # 仅添加必要依赖
]
```

**规则**:

- ✅ 添加新依赖前先评估必要性
- ✅ 固定主版本号（避免破坏性更新）
- ❌ 不添加仅用于实验的库

---

## 📚 参考资源优先级

### 搜索优先级

1. **Papers with Code** - SOTA 结果和代码链接
2. **官方实现** - 作者提供的代码
3. **timm 库** - 高质量 PyTorch 模型实现
4. **PyTorch Forums** - 实现细节讨论
5. **GitHub Issues** - 常见问题解答

### 可信度评估

```
✅ 高可信度:
- 官方论文 + 官方代码
- timm 库实现
- PyTorch 官方教程

⚠️ 中可信度:
- Star > 1000 的 GitHub 项目
- 知名博客（如 Distill.pub）

❌ 低可信度:
- 个人博客（未验证）
- Stack Overflow 未接受答案
- 过时的教程（> 3 年）
```

---

## ✅ 检查清单

### 实现新功能前

- [ ] 阅读相关论文和官方实现
- [ ] 搜索 CIFAR-100 最佳实践
- [ ] 设计 API 接口（类型注解）
- [ ] 准备测试用例

### 实现完成后

- [ ] 所有函数都有类型注解
- [ ] 所有公共接口都有 docstring
- [ ] 运行类型检查（无错误）
- [ ] 删除未使用的代码和导入
- [ ] 测试基本功能（前向传播）
- [ ] 记录参数量和 FLOPs
- [ ] 更新 `bot/experiments/experiment_tracker.md`

### 提交前

- [ ] 检查 linting errors
- [ ] 检查类型错误
- [ ] 运行快速测试（5 epochs）
- [ ] 提交信息清晰明确
- [ ] 相关文档已更新

---

**维护**: 随着项目发展持续更新此文档  
**强制执行**: 所有 Phase 1+ 的代码必须遵守此规范
