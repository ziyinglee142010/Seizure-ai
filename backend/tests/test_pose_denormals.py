"""The pose network's denormal weights are zeroed at load (26-31x faster on CPU)."""

import unittest

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

from backend.app.video_detection import runtime


@unittest.skipIf(torch is None, "PyTorch is not installed")
class DenormalWeightTests(unittest.TestCase):
    def test_only_denormal_values_are_zeroed(self) -> None:
        layer = torch.nn.Linear(4, 1)
        with torch.no_grad():
            layer.weight.copy_(torch.tensor([[1e-40, -3e-39, 0.5, 1.2e-38]]))
            layer.bias.fill_(1e-41)
        changed = runtime._zero_denormal_weights(torch, layer)
        self.assertEqual(changed, 3)
        self.assertEqual(layer.weight.tolist(), [[0.0, 0.0, 0.5, layer.weight[0, 3].item()]])
        self.assertGreater(layer.weight[0, 3].item(), 1e-38)     # smallest normal value is kept
        self.assertEqual(layer.bias.item(), 0.0)

    def test_outputs_are_unchanged(self) -> None:
        torch.manual_seed(0)
        layer = torch.nn.Conv2d(3, 8, 3)
        with torch.no_grad():
            layer.weight[0, 0, 0, 0] = 5e-40
        x = torch.randn(1, 3, 16, 16)
        before = layer(x)
        runtime._zero_denormal_weights(torch, layer)
        self.assertTrue(torch.equal(before, layer(x)))


if __name__ == "__main__":
    unittest.main()
