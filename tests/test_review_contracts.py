"""Checkpoint-free boundary and backend dispatch regression checks."""
import pytest
import torch
import fused_ssim as ssim


def test_non_cuda_channels_last_is_made_contiguous(monkeypatch):
    received = []
    def forward(c1, c2, x, y, train):
        received.append(x)
        return (torch.ones_like(x),) + tuple(torch.zeros_like(x) for _ in range(3))
    monkeypatch.setattr(ssim, 'fusedssim', forward, raising=False)
    x = torch.rand(1, 3, 13, 17).contiguous(memory_format=torch.channels_last)
    assert not x.is_contiguous()
    assert ssim.fused_ssim(x, x).item() == 1
    assert received[0].is_contiguous()
    torch.testing.assert_close(received[0], x)


@pytest.mark.parametrize('height,width', [(8, 8), (8, 12), (12, 8), (10, 12), (11, 11)])
@pytest.mark.parametrize('combined', [False, True])
def test_empty_valid_region_matches_empty_map_mean(monkeypatch, height, width, combined):
    def forward(c1, c2, x, y, train, pad, with_l1):
        region = torch.ones_like(x)[:, :, pad:-pad, pad:-pad]
        sums = torch.stack((region.sum(), (x - y).abs().sum())) if with_l1 else region.sum().reshape(1)
        return (sums,) + tuple(torch.zeros_like(x) for _ in range(3))
    monkeypatch.setattr(ssim, 'fusedssim_sum', forward, raising=False)
    x, y = torch.ones(1, 3, height, width), torch.zeros(1, 3, height, width)
    if combined:
        l1, actual = ssim.FusedL1SSIMMean.apply(.0001, .0009, x, y, 5)
        assert l1.item() == 1
    else:
        actual = ssim.FusedSSIMMean.apply(.0001, .0009, x, y, 5)
    expected = torch.ones_like(x)[:, :, 5:-5, 5:-5].mean()
    torch.testing.assert_close(actual, expected, equal_nan=True)


def test_combined_rejects_cpu_before_native_call(monkeypatch):
    def native(*args):
        pytest.fail('CPU input reached the CUDA extension')
    monkeypatch.setattr(ssim, 'fusedssim_sum', native, raising=False)
    x = torch.zeros(1, 3, 12, 13)
    with pytest.raises(ValueError, match='CUDA'):
        ssim.fused_l1_ssim(x, x)


def test_combined_rejects_trainable_target_before_native_call(monkeypatch):
    # Exercise Python validation without allocating a GPU tensor.
    monkeypatch.setattr(torch.Tensor, 'is_cuda', property(lambda self: True))
    def native(*args):
        pytest.fail('Trainable target reached the CUDA extension')
    monkeypatch.setattr(ssim, 'fusedssim_sum', native, raising=False)
    x = torch.zeros(1, 3, 12, 13)
    target = torch.zeros_like(x, requires_grad=True)
    with pytest.raises(ValueError, match='fixed target'):
        ssim.fused_l1_ssim(x, target)


@pytest.mark.parametrize('kind,message', [('dtype', 'float32'), ('shape', '4D'), ('rank', '4D')])
def test_combined_rejects_invalid_contract_before_native_call(monkeypatch, kind, message):
    monkeypatch.setattr(torch.Tensor, 'is_cuda', property(lambda self: True))
    def native(*args):
        pytest.fail('Invalid inputs reached the CUDA extension')
    monkeypatch.setattr(ssim, 'fusedssim_sum', native, raising=False)
    x = torch.zeros(1, 3, 12, 13)
    y = torch.zeros_like(x)
    if kind == 'dtype':
        y = y.half()
    elif kind == 'shape':
        y = y[:, :, :-1]
    else:
        x, y = x[0], y[0]
    with pytest.raises(ValueError, match=message):
        ssim.fused_l1_ssim(x, y)


def test_deterministic_mode_uses_map_and_separate_l1(monkeypatch):
    monkeypatch.setattr(torch.Tensor, 'is_cuda', property(lambda self: True))
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch, 'are_deterministic_algorithms_enabled', lambda: True)
    def sum_forward(*args):
        pytest.fail('Deterministic mode reached the atomic sum kernel')
    def map_forward(c1, c2, x, y, train):
        return (torch.ones_like(x),) + tuple(torch.zeros_like(x) for _ in range(3))
    monkeypatch.setattr(ssim, 'fusedssim_sum', sum_forward, raising=False)
    monkeypatch.setattr(ssim, 'fusedssim', map_forward, raising=False)
    x = torch.ones(1, 3, 12, 13, requires_grad=True)
    target = torch.zeros_like(x)
    assert ssim.fused_ssim(x, target, padding='valid').item() == 1
    l1, similarity = ssim.fused_l1_ssim(x, target)
    assert l1.item() == 1 and similarity.item() == 1
