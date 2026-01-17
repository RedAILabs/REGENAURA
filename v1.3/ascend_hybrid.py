# File: Red/regen/ascend_hybrid.py

import math

class AscendHybrid:
    """
    Continuous architecture scaling controller.
    Adjusts hidden size (width) and depth (layer count) smoothly.
    """

    def __init__(self, base_config):
        """
        base_config must contain:
        - hidden_size
        - num_layers
        """
        # keep full config for reuse
        self.base_config = dict(base_config)

        self.base_hidden = self.base_config["hidden_size"]
        self.base_layers = self.base_config["num_layers"]

        # Default gains (no scaling yet)
        self.width_gain = 0.0
        self.depth_gain = 0.0

    @staticmethod
    def _round_tensor_core(x):
        """
        Round to nearest multiple that fits GPU Tensor Core blocks.
        64 is safe for T5, GPT, LLaMA, DeepSeek.
        """
        return max(64, int(round(x / 64) * 64))

    def set_scaling(self, width_gain: float, depth_gain: float):
        """
        width_gain and depth_gain both in range [0.0 → 1.0].
        """
        self.width_gain = max(0.0, min(width_gain, 1.0))
        self.depth_gain = max(0.0, min(depth_gain, 1.0))

    def compute_new_architecture(self):
        """
        Returns a dictionary containing the new scaled architecture.
        """
        new_hidden = self.base_hidden * (1.0 + self.width_gain)
        new_layers = self.base_layers * (1.0 + self.depth_gain)

        new_hidden = self._round_tensor_core(new_hidden)
        new_layers = int(max(1, round(new_layers)))

        return {
            "hidden_size": new_hidden,
            "num_layers": new_layers
        }

    def summary(self):
        scaled = self.compute_new_architecture()
        print("\n=== AURASCEND Architecture Scaling ===")
        print(f"Base Hidden Size : {self.base_hidden}")
        print(f"Base Layers      : {self.base_layers}")
        print(f"Width Gain       : {self.width_gain:.2f}")
        print(f"Depth Gain       : {self.depth_gain:.2f}")
        print(f"> New Hidden Size: {scaled['hidden_size']}")
        print(f"> New Layer Count: {scaled['num_layers']}")
        print("=====================================\n")
