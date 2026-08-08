#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader

from tcstf.data.synthetic import TrajectoryDataset
from tcstf.factory import build_tcstf
from tcstf.tasks import BatteryTaskFamily, InventoryTaskFamily
from tcstf.training import TCSTFTrainer, TrainerConfig
from tcstf.utils.device import resolve_device
from tcstf.utils.io import load_yaml, save_json
from tcstf.utils.seed import seed_everything


def load_npz(path: str) -> TrajectoryDataset:
    d = np.load(path, allow_pickle=False)
    return TrajectoryDataset(d["x"], d["y"])


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Train TCS-TF on pre-split application windows. Split files must be created without leakage."
    )
    ap.add_argument("--application", choices=["inventory", "battery"], required=True)
    ap.add_argument("--train", required=True, help="Training NPZ with x/y")
    ap.add_argument("--val", required=True, help="Validation NPZ with x/y")
    ap.add_argument("--config", default="configs/engineering_defaults.yaml")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--inventory-i0-index", type=int, default=None)
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    seed = int(cfg.get("seed", 7))
    seed_everything(seed)
    train_ds, val_ds = load_npz(args.train), load_npz(args.val)
    h, c = train_ds.y.shape[1:]
    if args.application == "inventory":
        if c != 1:
            raise ValueError("Inventory Y must have one demand channel")
        task = InventoryTaskFamily(h, initial_inventory_index=args.inventory_i0_index)
    else:
        if c < 3:
            raise ValueError("Battery Y must have net-demand, buy-price and sell-price channels")
        task = BatteryTaskFamily(h)

    model = build_tcstf(cfg, task=task, x_dim=train_ds.x.shape[1], horizon=h, channels=c)
    tcfg = TrainerConfig(**{k: v for k, v in cfg["training"].items() if k in TrainerConfig.__dataclass_fields__})
    device = resolve_device(cfg.get("device", "auto"))
    bsz = int(cfg["training"].get("batch_size", 128))
    train_loader = DataLoader(train_ds, batch_size=bsz, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=bsz, shuffle=False)
    trainer = TCSTFTrainer(model, task, tcfg, device, outdir=args.outdir)
    trainer.fit(train_loader, val_loader)
    save_json(trainer.audit_factorization(val_loader), Path(args.outdir) / "validation_audit.json")
    print(Path(args.outdir).resolve())


if __name__ == "__main__":
    main()
