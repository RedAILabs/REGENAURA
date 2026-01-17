# File: Red/regen/ascend_manual.py

from ascend_hybrid import AscendHybrid

class ManualScaler:
    """
    Direct, human-controlled scaling of width and depth.
    Safe for early-stage Red.
    """

    def __init__(self, base_config):
        self.engine = AscendHybrid(base_config)

    def set(self, width_gain: float, depth_gain: float):
        self.engine.set_scaling(width_gain, depth_gain)
        return self.engine.compute_new_architecture()

    def preview(self, width_gain: float, depth_gain: float):
        """
        Preview scaling without mutating the internal engine.
        """
        tmp = AscendHybrid(self.engine.base_config)
        tmp.set_scaling(width_gain, depth_gain)
        return tmp.compute_new_architecture()
