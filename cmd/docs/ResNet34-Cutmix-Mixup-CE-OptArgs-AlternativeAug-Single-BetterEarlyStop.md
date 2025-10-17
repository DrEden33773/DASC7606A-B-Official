# 使用 ResNet34 模型, 启用 Cutmix 和 Mixup (6:4), 采用 CE Loss, 可切换离线/在线增强 (等级为 Light), 更好的早停策略

可达成 f1-score: 0.77

- powershell

  ```powershell
  python main.py `
  --model resnet34 `
  --lr 0.0012 `
  --dropout 0.5 `
  --weight_decay 8e-4 `
  --warmup_epochs 10 `
  --aug_strength light `
  --use_cutmix `
  --cutmix_alpha 0.65 `
  --mixup_alpha 0.25 `
  --num_epochs 500 `
  --early_stopping_patience 50
  ```

- bash / zsh

  ```bash
  python main.py \
  --model resnet34 \
  --lr 0.0012 \
  --dropout 0.5 \
  --weight_decay 8e-4 \
  --warmup_epochs 10 \
  --aug_strength light \
  --use_cutmix \
  --cutmix_alpha 0.65 \
  --mixup_alpha 0.25 \
  --num_epochs 500 \
  --early_stopping_patience 50
  ```
