import torch

OPTIMIZERS = {
    "Adam": torch.optim.Adam,
    "SGD": torch.optim.SGD,
    "RMSprop": torch.optim.RMSprop,
    "AdamW": torch.optim.AdamW,
}

LR_SCHEDULERS = {
    "PolynomialLR": torch.optim.lr_scheduler.PolynomialLR,
    "CosineAnnealingLR": torch.optim.lr_scheduler.CosineAnnealingLR,
    "CosineAnnealingWarmRestarts": torch.optim.lr_scheduler.CosineAnnealingWarmRestarts,
}

def build_optimizer(model, cfg: dict):
    cfg = cfg.copy()         # 避免修改原始字典
    name = cfg.pop("name")
    if name not in OPTIMIZERS:
        raise ValueError(f"Unknown optimizer {name}, available: {list(OPTIMIZERS)}")
    return OPTIMIZERS[name](model.parameters(), **cfg)

def build_scheduler(optimizer, cfg: dict):
    cfg = cfg.copy()
    name = cfg.pop("name")
    if name not in LR_SCHEDULERS:
        raise ValueError(f"Unknown scheduler {name}, available: {list(LR_SCHEDULERS)}")
    return LR_SCHEDULERS[name](optimizer, **cfg)


# ===== 使用示例 =====
if __name__ == "__main__":
    model = torch.nn.Linear(10, 2)

    optimizer_cfg = (
        "AdamW",
        {
            "lr": 5e-4,
            "weight_decay": 1e-4,
            "eps": 1e-8,
            "amsgrad": False,
            "betas": (0.9, 0.999),
        },
    )

    scheduler_cfg = (
        "CosineAnnealingLR",
        {
            "T_max": 100,
            "eta_min": 1e-6,
        },
    )

    optimizer = build_optimizer(model, optimizer_cfg)
    scheduler = build_scheduler(optimizer, scheduler_cfg)

    print(optimizer)
    print(scheduler)
