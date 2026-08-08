import torch

from tcstf.tasks.battery import BatteryTaskFamily, battery_path_loss
from tcstf.tasks.inventory import InventoryTaskFamily, inventory_path_loss
from tcstf.tasks.path_risk import PathRiskTaskFamily


def test_path_risk_batch_loss():
    task = PathRiskTaskFamily()
    b, h = 10, 24
    x = torch.randn(b, 4)
    y = torch.randn(b, h, 1)
    eta = task.sample_params(b, x.device)
    actions = task.sample_actions(eta, x, 3)
    out = task.loss(eta, actions[:, 1], x, y)
    assert out.shape == (b,)
    assert torch.isfinite(out).all()


def test_inventory_loss_shapes():
    task = InventoryTaskFamily(horizon=8)
    b = 5
    eta = task.sample_params(b, torch.device("cpu"))
    x = torch.randn(b, 3)
    y = torch.rand(b, 8, 1) * 6
    actions = task.sample_actions(eta, x, 4)
    loss = task.loss(eta, actions[:, 1], x, y)
    assert loss.shape == (b,)
    assert torch.isfinite(loss).all()
    direct = inventory_path_loss(eta, actions[:, 1], y[..., 0], initial_inventory=torch.zeros(b))
    assert torch.allclose(loss, direct)


def test_battery_loss_shapes():
    task = BatteryTaskFamily(horizon=6)
    b = 4
    eta = task.sample_params(b, torch.device("cpu"))
    x = torch.randn(b, 5)
    net = torch.rand(b, 6, 1) * 5
    buy = 0.1 + torch.rand(b, 6, 1) * 0.2
    sell = 0.5 * buy
    y = torch.cat([net, buy, sell], dim=-1)
    actions = task.sample_actions(eta, x, 3)
    loss = task.loss(eta, actions[:, 1], x, y)
    assert loss.shape == (b,)
    assert torch.isfinite(loss).all()
    assert torch.allclose(loss, battery_path_loss(eta, actions[:, 1], y))
