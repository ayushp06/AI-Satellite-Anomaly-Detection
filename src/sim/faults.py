import numpy as np
from dataclasses import dataclass

@dataclass
class FaultWindow:
    start: float
    end: float
    kind: str
    params: dict

    def active(self, t: float) -> bool:
        return self.start <= t < self.end

class FaultController:
    """
    Applies faults to measured telemetry while keeping the truth state untouched.
    """
    def __init__(self, windows=None, seed: int = 0):
        self.windows = windows or []
        self.rng = np.random.default_rng(seed)
        self._frozen_w = None  # remember last value for freeze faults

    def apply_w(self, t: float, w_true: np.ndarray):
        """
        Returns: (w_measured, fault_flag, fault_kind)
        """
        w = np.array(w_true, dtype=float)
        fault_flag = 0
        fault_kind = "none"

        # apply the first active fault window (simple, deterministic)
        for win in self.windows:
            if win.active(t):
                fault_flag = 1
                fault_kind = win.kind

                if win.kind == "gyro_bias":
                    bias = np.array(win.params.get("bias", [0, 0, 0]), dtype=float)
                    w = w + bias

                elif win.kind == "noise_burst":
                    sigma = float(win.params.get("sigma", 0.01))
                    w = w + self.rng.normal(0.0, sigma, size=3)

                elif win.kind == "freeze":
                    if self._frozen_w is None:
                        self._frozen_w = w.copy()
                    w = self._frozen_w

                elif win.kind == "dropout":
                    # represent dropout as NaNs (later you'll decide how to handle)
                    w[:] = np.nan

                break  # only one fault at a time

        # reset freeze memory when not in freeze
        if fault_kind != "freeze":
            self._frozen_w = None

        return w, fault_flag, fault_kind

