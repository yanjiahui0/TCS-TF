import torch
from torch.utils.data import DataLoader

from tcstf.data.synthetic import TrajectoryDataset, generate_s2_tails
from tcstf.factory import build_tcstf
from tcstf.tasks.path_risk import PathRiskTaskFamily
from tcstf.training import TCSTFTrainer, TrainerConfig


def test_tiny_training_smoke(tmp_path):
    x, y = generate_s2_tails(64, horizon=8, x_dim=3, seed=1)
    ds = TrajectoryDataset(x, y)
    loader = DataLoader(ds, batch_size=16, shuffle=False)
    task = PathRiskTaskFamily()
    cfg = {
        "model": {
            "known_stats": ["mean", "max"],
            "learned_bank_width": 5,
            "deployed_dim": 4,
            "representation_radius": 4.0,
            "encoder_hidden": [16],
            "decoder_hidden": [16],
            "history_hidden": [16],
            "mixture_components": 2,
            "low_rank": 1,
        },
        "training": {"gate_temperature_start": 1.0},
    }
    model = build_tcstf(cfg, task=task, x_dim=3, horizon=8, channels=1)
    tcfg = TrainerConfig(
        pretrain_epochs=1,
        generator_epochs=1,
        joint_epochs=1,
        samples_per_context=3,
        witnesses_per_context=3,
        lambda_lip=0.0,
    )
    trainer = TCSTFTrainer(model, task, tcfg, torch.device("cpu"), outdir=tmp_path)
    hist = trainer.fit(loader, loader)
    assert len(hist) == 3
    assert (tmp_path / "model.pt").exists()
