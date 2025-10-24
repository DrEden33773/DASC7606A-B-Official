# generate_class_weights 函数重构完成 ✅

**Date**: 2025-10-25  
**File**: `scripts/train_utils.py`  
**Lines Reduced**: ~300 lines → ~85 lines (节省 **70%** 代码量)

---

## 🎯 重构目标

消除 `generate_class_weights` 函数中的大量重复代码，提高可维护性。

---

## 📊 重构前后对比

### 重构前 ❌

```python
elif strategy == "long_board":
    extreme_high = {...}  # 6 classes
    high_score = {...}    # 13 classes
    mid_high = {...}      # 47 classes
    ... 
    # 150 lines of code

elif strategy == "long_board_v2":
    extreme_high = {...}  # 6 classes (重复!)
    high_score = {...}    # 13 classes (重复!)
    mid_high = {...}      # 47 classes (重复!)
    ...
    # 150 lines of code (重复!)

elif strategy == "long_board_v2.5":
    extreme_high = {...}  # 6 classes (重复!)
    high_score = {...}    # 13 classes (重复!)
    mid_high = {...}      # 47 classes (重复!)
    ...
    # 150 lines of code (重复!)
```

**问题**:

- ❌ 类组定义重复3次（~300行重复代码）
- ❌ 维护困难：修改类组需要同时修改3个地方
- ❌ 易出错：可能导致不同策略的类组不一致

---

### 重构后 ✅

```python
elif strategy in ["long_board", "long_board_v2", "long_board_v2.5"]:
    # 定义类组（只需定义一次）
    class_groups = {
        "extreme_high": {...},  # 6 classes
        "high_score": {...},    # 13 classes
        "mid_high": {...},      # 47 classes
        "medium": {...},        # 18 classes
        "detail_sensitive": {...},  # 8 classes
        "other_low": {...}      # 9 classes
    }
    
    # 定义权重映射（每个策略只需定义6个数字）
    weight_mappings = {
        "long_board": {
            "extreme_high": 0.75,
            "high_score": 0.9,
            "mid_high": 1.1,
            "medium": 1.6,
            "detail_sensitive": 1.0,
            "other_low": 1.4
        },
        "long_board_v2": {
            "extreme_high": 0.3,
            "high_score": 0.5,
            "mid_high": 2.0,
            "medium": 1.0,
            "detail_sensitive": 0.2,
            "other_low": 1.3
        },
        "long_board_v2.5": {
            "extreme_high": 0.6,
            "high_score": 0.7,
            "mid_high": 1.4,
            "medium": 1.5,
            "detail_sensitive": 0.5,
            "other_low": 1.3
        }
    }
    
    # 应用权重
    weight_map = weight_mappings[strategy]
    for idx, class_name in enumerate(cifar100_classes):
        for group_name, group_classes in class_groups.items():
            if class_name in group_classes:
                weights[idx] = weight_map[group_name]
                break
```

**优势**:

- ✅ **类组定义只有1次**（消除重复）
- ✅ **权重映射一目了然**（每个策略只需6个数字）
- ✅ **易于维护**：修改类组只需改1个地方
- ✅ **易于扩展**：添加新策略只需添加新的权重映射
- ✅ **不易出错**：所有策略共享相同的类组定义

---

## 📈 代码量统计

| 指标 | 重构前 | 重构后 | 节省 |
|------|--------|--------|------|
| **总行数** | ~300 lines | ~85 lines | **-215 lines (-70%)** |
| **类组定义重复次数** | 3次 | 1次 | **-2次** |
| **每个策略代码量** | ~150 lines | ~6 lines (权重映射) | **-144 lines (-96%)** |

---

## 🔧 技术细节

### 数据结构设计

**class_groups** (字典):

```python
{
    "extreme_high": set of 6 classes,
    "high_score": set of 13 classes,
    "mid_high": set of 47 classes,
    "medium": set of 18 classes,
    "detail_sensitive": set of 8 classes,
    "other_low": set of 9 classes
}
```

**weight_mappings** (嵌套字典):

```python
{
    "long_board": {group_name: weight_value},
    "long_board_v2": {group_name: weight_value},
    "long_board_v2.5": {group_name: weight_value}
}
```

---

### 算法优化

**查找逻辑**:

```python
# 重构前: 多个if-elif检查
if class_name in extreme_high:
    weights[idx] = 0.75
elif class_name in high_score:
    weights[idx] = 0.9
...

# 重构后: 循环遍历字典
for group_name, group_classes in class_groups.items():
    if class_name in group_classes:
        weights[idx] = weight_map[group_name]
        break
```

**复杂度**: O(6) per class (最多检查6个组) → 性能影响可忽略

---

## ✅ 质量保证

- [x] **Linter检查通过**: 无错误
- [x] **逻辑等价性**: 重构前后生成的权重完全相同
- [x] **类型安全**: 保留所有类型注解
- [x] **文档完整**: Docstring保持完整

---

## 🎉 总结

**核心改进**:

1. ✅ 消除 ~215行重复代码（节省70%）
2. ✅ 提高可维护性（类组定义集中化）
3. ✅ 增强可扩展性（新策略只需添加权重映射）
4. ✅ 降低出错风险（单一数据源）

**开发体验提升**:

- 📝 添加新策略：仅需6行代码（权重映射）
- 🔧 修改类组：只需修改1个地方
- 🐛 调试更简单：权重配置一目了然

---

**重构完成，代码更优雅！** ✨
