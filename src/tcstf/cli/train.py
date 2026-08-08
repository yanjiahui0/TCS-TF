from __future__ import annotations

import argparse
from pathlib import Path

from torch.utils.data import DataLoader

from tcstf.data import TrajectoryDataset, append_nuisance_channels, generate_s1_aliasing, generate_s2_tails
from tcstf.factory import build_tcstf
from tcstf.tasks import PathRiskTaskFamily, PrecautionTaskFamily
from tcstf.training import TCSTFTrainer, TrainerConfig
from tcstf.utils.device import resolve_device
from tcstf.utils.io import load_yaml, save_json
from tcstf.utils.manifest import environment_manifest, write_manifest
from tcstf.utils.seed import seed_everything


def _make_data(cfg: dict, split: str, seed: int):
    n = int(cfg["data"][split])
    suite = cfg["data"].get("suite", "s2").lower()
    h = int(cfg["data"].get("horizon", 24))
    x_dim = int(cfg["data"].get("x_dim", 4))
    if suite == "s1":
        x, y, _ = generate_s1_aliasing(n, horizon=h, seed=seed)
        task = PrecautionTaskFamily()
    else:
        x, y = generate_s2_tails(n, horizon=h, x_dim=x_dim, seed=seed)
        if suite == "s3":
            y = append_nuisance_channels(
                x,
                y,
                cfg["data"].get("nuisance_channels", 16),
                seed=seed + 99,
            )
        task = PathRiskTaskFamily()
    return TrajectoryDataset(x, y), task


def main() -> None:
    ap = argparse.ArgumentParser(description="Train a configurable TCS-TF synthetic experiment")
    ap.add_argument("--config", default="configs/synthetic_tiny.yaml")
    ap.add_argument("--outdir", default="runs/synthetic")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    seed = int(cfg.get("seed", 7))
    seed_everything(seed)
    train_ds, task = _make_data(cfg, "train", seed)
    val_ds, _ = _make_data(cfg, "val", seed + 1)
    bsz = int(cfg["training"].get("batch_size", 128))
    train_loader = DataLoader(train_ds, batch_size=bsz, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=bsz, shuffle=False)

    x_dim = train_ds.x.shape[1]
    horizon, channels = train_ds.y.shape[1], train_ds.y.shape[2]
    model = build_tcstf(cfg, task=task, x_dim=x_dim, horizon=horizon, channels=channels)
    tcfg = TrainerConfig(
        **{k: v for k, v in cfg["training"].items() if k in TrainerConfig.__dataclass_fields__}
    )
    device = resolve_device(cfg.get("device", "auto"))
    trainer = TCSTFTrainer(model, task, tcfg, device, outdir=args.outdir)
    trainer.fit(train_loader, val_loader)
    audit = trainer.audit_factorization(val_loader)
    save_json(audit, Path(args.outdir) / "validation_audit.json")
    write_manifest(
        Path(args.outdir) / "environment.json",
        environment_manifest({"config": str(args.config), "seed": seed}),
    )
    print("Final validation audit:", audit)
    print("Run directory:", Path(args.outdir).resolve())


if __name__ == "__main__":
    main()
