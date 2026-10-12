"""CUDA mean/map and combined-loss regression checks; no benchmark timing."""
import pytest
import torch
import torch.nn.functional as F
import fused_ssim as ssim

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA extension required')


def inputs(channels_last, shape=(2, 3, 19, 27)):
    torch.manual_seed(29)
    x = torch.rand(shape, device='cuda')
    y = torch.rand_like(x)
    if channels_last:
        x = x.contiguous(memory_format=torch.channels_last)
        y = y.contiguous(memory_format=torch.channels_last)
    return x.requires_grad_(), y


def map_mean(x, y, padding):
    return ssim.FusedSSIMMap.apply(.0001, .0009, x.contiguous(), y, padding, True, 2).mean()


@pytest.mark.parametrize('padding', ['same', 'valid'])
@pytest.mark.parametrize('channels_last', [False, True])
@pytest.mark.parametrize('nonlinear', [False, True])
def test_mean_and_combined_loss_match_map_path(padding, channels_last, nonlinear):
    reference, target = inputs(channels_last)
    mean_input = reference.detach().clone().requires_grad_()
    combined_input = reference.detach().clone().requires_grad_()
    ref_ssim = map_mean(reference, target, padding)
    actual_ssim = ssim.fused_ssim(mean_input, target, padding=padding)
    combined_l1, combined_ssim = ssim.fused_l1_ssim(combined_input, target, padding=padding)
    ref_l1 = F.l1_loss(reference, target)
    torch.testing.assert_close(actual_ssim, ref_ssim, atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(combined_ssim, ref_ssim, atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(combined_l1, ref_l1, atol=2e-6, rtol=2e-5)
    def loss(l1, similarity):
        value = .8 * l1 + .2 * (1 - similarity)
        return value.square() if nonlinear else value
    loss(ref_l1, ref_ssim).backward()
    loss(F.l1_loss(mean_input, target), actual_ssim).backward()
    loss(combined_l1, combined_ssim).backward()
    # Nonlinear outer losses use the slightly different reduced scalar as their
    # upstream gradient, so unconditional bitwise gradient equality is invalid.
    torch.testing.assert_close(mean_input.grad, reference.grad, atol=1e-7, rtol=2e-5)
    torch.testing.assert_close(combined_input.grad, reference.grad, atol=1e-7, rtol=2e-5)


@pytest.mark.parametrize('shape', [(1, 3, 8, 8), (1, 3, 8, 12), (1, 3, 12, 8), (1, 3, 10, 12)])
def test_empty_valid_region_value_and_gradient(shape):
    reference, target = inputs(True, shape)
    mean_input = reference.detach().clone().requires_grad_()
    combined_input = reference.detach().clone().requires_grad_()
    expected = map_mean(reference, target, 'valid')
    actual = ssim.fused_ssim(mean_input, target, padding='valid')
    _, combined = ssim.fused_l1_ssim(combined_input, target, padding='valid')
    assert torch.isnan(expected) and torch.isnan(actual) and torch.isnan(combined)
    expected.backward()
    actual.backward()
    combined.backward()
    assert torch.count_nonzero(reference.grad) == 0
    torch.testing.assert_close(mean_input.grad, reference.grad, rtol=0, atol=0)
    torch.testing.assert_close(combined_input.grad, reference.grad, rtol=0, atol=0)


def test_combined_rejects_trainable_cuda_target():
    x, target = inputs(False)
    target.requires_grad_()
    with pytest.raises(ValueError, match='fixed target'):
        ssim.fused_l1_ssim(x, target)


def test_combined_rejects_cpu_target_before_kernel_launch():
    x, target = inputs(False)
    with pytest.raises(ValueError, match='same CUDA/HIP device'):
        ssim.fused_l1_ssim(x, target.cpu())


@pytest.mark.parametrize('deterministic', [False, True])
def test_forward_backward_use_current_cuda_stream(deterministic):
    previous = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.use_deterministic_algorithms(deterministic)
        base, target = inputs(False, (1, 3, 37, 45))
        reference = base.detach().clone().requires_grad_()
        expected = ssim.fused_ssim(reference, target, padding='valid')
        expected.backward()
        torch.cuda.synchronize()
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            # An incorrect default-stream launch bypasses this delay and reads
            # inputs or the mean accumulator before this stream initializes them.
            torch.cuda._sleep(30_000_000)
            image = base.detach().clone().requires_grad_()
            stream_target = target.clone()
            actual = ssim.fused_ssim(image, stream_target, padding='valid')
            actual.backward()
        stream.synchronize()
        torch.cuda.synchronize()
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
        torch.testing.assert_close(image.grad, reference.grad, atol=1e-7, rtol=2e-5)
    finally:
        torch.use_deterministic_algorithms(previous, warn_only=warn_only)
