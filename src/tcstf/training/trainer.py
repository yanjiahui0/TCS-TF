from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from tcstf.losses import dipm_loss, energy_score, factorization_loss, lipschitz_penalty
from tcstf.models import TCSTF
from tcstf.tasks.base import TaskFamily
from tcstf.utils.io import save_json


@dataclass
class TrainerConfig:
    lr_quotient: float = 1e-3
    lr_generator: float = 1e-3
    weight_decay: float = 1e-5
    pretrain_epochs: int = 10
    generator_epochs: int = 10
    joint_epochs: int = 20
    samples_per_context: int = 8
    witnesses_per_context: int = 8
    tau_factor: float = 0.02
    tau_dipm: float = 0.02
    gate_temperature_start: float = 2.0
    gate_temperature_end: float = 0.35
    lambda_fac: float = 1.0
    lambda_prob: float = 1.0
    lambda_dec: float = 1.0
    lambda_gate: float = 1e-3
    lambda_lip: float = 1e-3
    lip_target: float = 2.0
    grad_clip: float = 5.0


@contextmanager
def _frozen(*modules: torch.nn.Module):
    old: list[list[bool]] = []
    for module in modules:
        flags = [p.requires_grad for p in module.parameters()]
        old.append(flags)
        for p in module.parameters():
            p.requires_grad_(False)
    try:
        yield
    finally:
        for module, flags in zip(modules, old):
            for p, flag in zip(module.parameters(), flags):
                p.requires_grad_(flag)


class TCSTFTrainer:
    """Staged trainer following Section 5.6 of the manuscript.

    The code separates quotient and generator updates in the joint stage to reduce
    encoder-generator collusion. Exact disagreement-maximizing witness refreshes
    are application-specific; the generic trainer samples a fresh feasible witness
    bank from the declared TaskFamily on every batch.
    """

    def __init__(
        self,
        model: TCSTF,
        task: TaskFamily,
        config: TrainerConfig,
        device: torch.device,
        outdir: str | Path | None = None,
    ):
        self.model = model.to(device)
        self.task = task
        self.cfg = config
        self.device = device
        self.outdir = Path(outdir) if outdir is not None else None
        if self.outdir:
            self.outdir.mkdir(parents=True, exist_ok=True)
        self.quotient_optimizer = torch.optim.AdamW(
            list(self.model.encoder.parameters()) + list(self.model.decoder.parameters()),
            lr=config.lr_quotient,
            weight_decay=config.weight_decay,
        )
        self.generator_optimizer = torch.optim.AdamW(
            self.model.generator.parameters(),
            lr=config.lr_generator,
            weight_decay=config.weight_decay,
        )
        self.history: list[dict[str, float | int | str]] = []

    def _move(self, batch):
        x, y = batch
        return x.to(self.device), y.to(self.device)

    def _factorization_terms(self, x: torch.Tensor, y: torch.Tensor):
        b = x.shape[0]
        eta = self.task.sample_params(b, self.device)
        actions = self.task.sample_actions(eta, x, self.cfg.witnesses_per_context)
        eta_f, x_f, y_f, a_f = self.task.flatten_witnesses(eta, x, y, actions)
        z = self.model.encoder(x, y)
        w = actions.shape[1]
        z_f = z[:, None, :].expand(b, w, -1).reshape(b * w, -1)
        a0_f = self.task.reference_action(eta_f, x_f)
        pred = self.model.decoder(eta_f, a_f, x_f, z_f, a0_f)
        target = self.task.relative_loss(eta_f, a_f, x_f, y_f).detach()
        return eta, actions, z, pred, target

    def _decoder_witness_values(
        self,
        eta: torch.Tensor,
        x: torch.Tensor,
        actions: torch.Tensor,
        z: torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate g for observed Z [B,r] -> [B,W]."""
        b, w, a_dim = actions.shape
        eta_f = eta[:, None, :].expand(b, w, -1).reshape(b * w, -1)
        x_f = x[:, None, :].expand(b, w, -1).reshape(b * w, -1)
        a_f = actions.reshape(b * w, a_dim)
        z_f = z[:, None, :].expand(b, w, -1).reshape(b * w, -1)
        a0 = self.task.reference_action(eta_f, x_f)
        return self.model.decoder(eta_f, a_f, x_f, z_f, a0).view(b, w)

    def _decoder_generated_values(
        self,
        eta: torch.Tensor,
        x: torch.Tensor,
        actions: torch.Tensor,
        z_samples: torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate g for generated Z [B,M,r] -> [B,M,W]."""
        b, m, r = z_samples.shape
        w, a_dim = actions.shape[1], actions.shape[2]
        eta_f = eta[:, None, None, :].expand(b, m, w, -1).reshape(b * m * w, -1)
        x_f = x[:, None, None, :].expand(b, m, w, -1).reshape(b * m * w, -1)
        a_f = actions[:, None, :, :].expand(b, m, w, a_dim).reshape(b * m * w, a_dim)
        z_f = z_samples[:, :, None, :].expand(b, m, w, r).reshape(b * m * w, r)
        a0 = self.task.reference_action(eta_f, x_f)
        return self.model.decoder(eta_f, a_f, x_f, z_f, a0).view(b, m, w)

    def _anneal_gate(self, global_epoch: int, total_epochs: int) -> float:
        if total_epochs <= 1:
            t = 1.0
        else:
            t = global_epoch / (total_epochs - 1)
        start, end = self.cfg.gate_temperature_start, self.cfg.gate_temperature_end
        temperature = start * (end / start) ** t
        self.model.encoder.set_gate_temperature(temperature)
        return float(temperature)

    def fit(self, train_loader: DataLoader, val_loader: DataLoader | None = None) -> list[dict]:
        total = self.cfg.pretrain_epochs + self.cfg.generator_epochs + self.cfg.joint_epochs
        epoch_index = 0

        for epoch in range(self.cfg.pretrain_epochs):
            temp = self._anneal_gate(epoch_index, total)
            metrics = self._run_quotient_epoch(train_loader, include_lip=False)
            self._record("quotient_pretrain", epoch, temp, metrics, val_loader)
            epoch_index += 1

        for epoch in range(self.cfg.generator_epochs):
            temp = self._anneal_gate(epoch_index, total)
            metrics = self._run_generator_epoch(train_loader, include_dipm=False)
            self._record("generator_fit", epoch, temp, metrics, val_loader)
            epoch_index += 1

        for epoch in range(self.cfg.joint_epochs):
            temp = self._anneal_gate(epoch_index, total)
            qmetrics = self._run_quotient_epoch(train_loader, include_lip=True)
            gmetrics = self._run_generator_epoch(train_loader, include_dipm=True)
            metrics = {**qmetrics, **gmetrics}
            self._record("joint_alternating", epoch, temp, metrics, val_loader)
            epoch_index += 1

        if self.outdir:
            torch.save(
                {
                    "model": self.model.state_dict(),
                    "trainer_config": asdict(self.cfg),
                    "selected_indices": self.model.encoder.selected_indices().detach().cpu(),
                },
                self.outdir / "model.pt",
            )
            save_json(self.history, self.outdir / "history.json")
        return self.history

    def _run_quotient_epoch(self, loader: Iterable, include_lip: bool) -> dict[str, float]:
        self.model.train()
        totals = {"fac": 0.0, "gate": 0.0, "lip": 0.0, "q_total": 0.0}
        n = 0
        with _frozen(self.model.generator):
            for batch in tqdm(loader, leave=False, desc="quotient"):
                x, y = self._move(batch)
                self.quotient_optimizer.zero_grad(set_to_none=True)
                eta, actions, z, pred, target = self._factorization_terms(x, y)
                fac = factorization_loss(pred, target, self.cfg.tau_factor)
                gate = self.model.encoder.gate_l1()
                lip = torch.zeros((), device=self.device)
                if include_lip and actions.shape[1] > 1:
                    # Do not use the first witness because task implementations reserve it for a0.
                    chosen = actions[:, 1, :]
                    a0 = self.task.reference_action(eta, x)
                    values = self.model.decoder(eta, chosen, x, z, a0)
                    lip, _ = lipschitz_penalty(values, z, self.cfg.lip_target)
                loss = self.cfg.lambda_fac * fac + self.cfg.lambda_gate * gate + self.cfg.lambda_lip * lip
                loss.backward()
                clip_grad_norm_(
                    list(self.model.encoder.parameters()) + list(self.model.decoder.parameters()),
                    self.cfg.grad_clip,
                )
                self.quotient_optimizer.step()
                bs = x.shape[0]
                n += bs
                totals["fac"] += float(fac.detach()) * bs
                totals["gate"] += float(gate.detach()) * bs
                totals["lip"] += float(lip.detach()) * bs
                totals["q_total"] += float(loss.detach()) * bs
        return {k: v / max(n, 1) for k, v in totals.items()}

    def _run_generator_epoch(self, loader: Iterable, include_dipm: bool) -> dict[str, float]:
        self.model.train()
        totals = {"es": 0.0, "dipm": 0.0, "g_total": 0.0}
        n = 0
        with _frozen(self.model.encoder, self.model.decoder):
            for batch in tqdm(loader, leave=False, desc="generator"):
                x, y = self._move(batch)
                self.generator_optimizer.zero_grad(set_to_none=True)
                with torch.no_grad():
                    z_obs = self.model.encoder(x, y)
                z_samples = self.model.generator.sample(
                    x, self.cfg.samples_per_context, training_relaxation=True
                )
                es = energy_score(z_samples, z_obs)
                dipm = torch.zeros((), device=self.device)
                if include_dipm:
                    eta = self.task.sample_params(x.shape[0], self.device)
                    actions = self.task.sample_actions(eta, x, self.cfg.witnesses_per_context)
                    with torch.no_grad():
                        observed = self._decoder_witness_values(eta, x, actions, z_obs)
                    generated = self._decoder_generated_values(eta, x, actions, z_samples)
                    dipm = dipm_loss(observed, generated, self.cfg.tau_dipm)
                loss = self.cfg.lambda_prob * es + self.cfg.lambda_dec * dipm
                loss.backward()
                clip_grad_norm_(self.model.generator.parameters(), self.cfg.grad_clip)
                self.generator_optimizer.step()
                bs = x.shape[0]
                n += bs
                totals["es"] += float(es.detach()) * bs
                totals["dipm"] += float(dipm.detach()) * bs
                totals["g_total"] += float(loss.detach()) * bs
        return {k: v / max(n, 1) for k, v in totals.items()}

    @torch.no_grad()
    def audit_factorization(self, loader: Iterable, max_batches: int | None = None) -> dict[str, float]:
        self.model.eval()
        residuals = []
        for i, batch in enumerate(loader):
            if max_batches is not None and i >= max_batches:
                break
            x, y = self._move(batch)
            _, _, _, pred, target = self._factorization_terms(x, y)
            residuals.append(torch.abs(pred - target).detach().cpu())
        r = torch.cat(residuals) if residuals else torch.tensor([float("nan")])
        return {
            "residual_mean": float(r.mean()),
            "residual_p95": float(torch.quantile(r, 0.95)),
            "residual_max": float(r.max()),
            "active_dimension": float(self.model.encoder.effective_dimension().detach().cpu()),
        }

    def _record(
        self,
        stage: str,
        epoch: int,
        gate_temperature: float,
        metrics: dict[str, float],
        val_loader: DataLoader | None,
    ) -> None:
        row: dict[str, float | int | str] = {
            "stage": stage,
            "epoch": epoch,
            "gate_temperature": gate_temperature,
            **metrics,
        }
        if val_loader is not None:
            audit = self.audit_factorization(val_loader, max_batches=8)
            row.update({f"val_{k}": v for k, v in audit.items()})
        self.history.append(row)
        if self.outdir:
            save_json(self.history, self.outdir / "history.json")
