# 🔧 Bug 修复总结

## 问题

验证指标（Val Loss, Val Acc, Val F1）完全冻结，始终不变。

## 根本原因

**`torch.compile()` wrapper 导致 EMA 无法正确更新参数**

## 修复方案

分离训练模型和 EMA 模型：

- `compiled_model`: 用于训练
- `original_model`: 用于 EMA 更新和验证

## 修改的文件

1. ✅ `main.py`: `build_model()`, `train()`, 验证逻辑
2. ✅ `scripts/train_utils.py`: `train_epoch()` 新增 `ema_model` 参数

## 测试

```powershell
.\temp\test_val_fix_10epochs.ps1
```

预期：10个epoch后，Val Loss < 3.5, Val Acc > 10%, Val F1 > 0.10

## 状态

✅ 所有代码已修复
✅ Linter 检查通过 (0 errors)
✅ 可以开始训练

详细分析见：`bot/CRITICAL_VAL_LOSS_BUG_FIX.md`
