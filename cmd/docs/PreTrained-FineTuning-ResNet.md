# 使用 torchvision.resnet 预训练模型进行微调

## resnet50 从头开始训练 (不启用预训练)

达成分数: 0.77

- powershell

    ```powershell
    python main.py `
        --model resnet50 `
        --no_pretrained `
        --dropout 0.45 `
        --lr 0.0008 `
        --weight_decay 1.2e-3 `
        --warmup_epochs 20 `
        --num_epochs 350 `
        --early_stopping_patience 35 `
        --batch_size 96 `
        --mixup_alpha 0.2 `
        --use_cutmix `
        --cutmix_alpha 0.6 `
        --aug_strength medium `
        --use_online_aug
    ```

- bash

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
        --use_online_aug
    ```

## 最优方案 (resnet50)

达成分数: 0.81

- powershell

    ```powershell
    python main.py `
        --model resnet50 `
        --use_pretrained `
        --dropout 0.45 `
        --lr 0.0008 `
        --weight_decay 1.2e-3 `
        --warmup_epochs 20 `
        --num_epochs 350 `
        --early_stopping_patience 35 `
        --batch_size 96 `
        --mixup_alpha 0.2 `
        --use_cutmix `
        --cutmix_alpha 0.6 `
        --aug_strength medium `
        --use_online_aug
    ```

- bash

    ```bash
    python main.py \
        --model resnet50 \
        --use_pretrained \
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
        --use_online_aug
    ```

## 方案一 (resnet34)

达成分数: 0.80

- powershell

    ```powershell
    python main.py `
        --dataset cifar100 `
        --model resnet34 `
        --use_pretrained `
        --use_online_aug `
        --aug_strength medium `
        --use_cutmix `
        --cutmix_alpha 0.65 `
        --mixup_alpha 0.25 `
        --num_epochs 300 `
        --batch_size 128 `
        --dropout 0.5 `
        --lr 0.001 `
        --weight_decay 1e-3 `
        --warmup_epochs 10 `
        --early_stopping_patience 30 `
        --num_workers 4 `
        --seed 42
    ```

- bash

    ```bash
    python main.py \
        --dataset cifar100 \
        --model resnet34 \
        --use_pretrained \
        --use_online_aug \
        --aug_strength medium \
        --use_cutmix \
        --cutmix_alpha 0.65 \
        --mixup_alpha 0.25 \
        --num_epochs 300 \
        --batch_size 128 \
        --dropout 0.5 \
        --lr 0.001 \
        --weight_decay 1e-3 \
        --warmup_epochs 10 \
        --early_stopping_patience 30 \
        --num_workers 4 \
        --seed 42
    ```

## 方案二 (resnet34)

达成分数: 0.80

- powershell

    ```powershell
    python main.py `
        --dataset cifar100 `
        --model resnet34 `
        --use_pretrained `
        --use_online_aug `
        --aug_strength medium `
        --use_cutmix `
        --cutmix_alpha 0.65 `
        --mixup_alpha 0.25 `
        --num_epochs 400 `
        --batch_size 128 `
        --dropout 0.5 `
        --lr 0.0008 `
        --weight_decay 1e-3 `
        --warmup_epochs 20 `
        --early_stopping_patience 50 `
        --num_workers 4 `
        --seed 42
    ```

- bash

    ```bash
    python main.py \
        --dataset cifar100 \
        --model resnet34 \
        --use_pretrained \
        --use_online_aug \
        --aug_strength medium \
        --use_cutmix \
        --cutmix_alpha 0.65 \
        --mixup_alpha 0.25 \
        --num_epochs 400 \
        --batch_size 128 \
        --dropout 0.5 \
        --lr 0.0008 \
        --weight_decay 1e-3 \
        --warmup_epochs 20 \
        --early_stopping_patience 50 \
        --num_workers 4 \
        --seed 42
    ```
