import numpy as np
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class FaultWindow:
    start: float
    end: float
    kind: str
    params: dict

    def active(self, t: float) -> bool:
        return self.start <= t < self.end

    def to_dict(self):
        """Serialize to dictionary for metadata logging."""
        return {
            "start": self.start,
            "end": self.end,
            "kind": self.kind,
            "params": self.params
        }

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


def sample_fault_windows(run_duration: float, rng: np.random.Generator) -> List[FaultWindow]:
    """
    Randomly sample 0 or 1 fault window for a simulation run.
    
    Parameters:
    -----------
    run_duration : float
        Total duration of the simulation in seconds
    rng : np.random.Generator
        NumPy random generator for reproducibility
        
    Returns:
    --------
    List[FaultWindow]
        Empty list (no fault) or list with single FaultWindow
        
    Fault sampling rules:
    - 30% chance of no fault
    - 70% chance of exactly one fault
    - Fault types: gyro_bias, noise_burst, freeze (equal probability)
    - Fault timing: start between 10% and 70% of run duration
    - Fault duration: 5-15 seconds
    - Fault severity: randomly sampled within realistic bounds
    """
    # Decide if fault occurs (70% probability)
    if rng.random() < 0.3:
        return []  # No fault this run
    
    # Select fault type uniformly
    fault_types = ["gyro_bias", "noise_burst", "freeze"]
    fault_kind = rng.choice(fault_types)
    
    # Sample fault timing
    # Start: between 10% and 70% of run duration (leave time for normal data)
    earliest_start = 0.1 * run_duration
    latest_start = 0.7 * run_duration
    fault_start = rng.uniform(earliest_start, latest_start)
    
    # Duration: 5 to 15 seconds
    fault_duration = rng.uniform(5.0, 15.0)
    fault_end = min(fault_start + fault_duration, run_duration)
    
    # Sample fault-specific parameters
    params = {}
    
    if fault_kind == "gyro_bias":
        # Bias magnitude: 0.01 to 0.05 rad/s on random axis
        bias_magnitude = rng.uniform(0.01, 0.05)
        # Randomly choose dominant axis (x, y, or z)
        axis = rng.integers(0, 3)
        bias = np.zeros(3)
        bias[axis] = bias_magnitude * rng.choice([-1, 1])  # random sign
        params["bias"] = bias.tolist()
        
    elif fault_kind == "noise_burst":
        # Noise standard deviation: 0.01 to 0.03 rad/s
        sigma = rng.uniform(0.01, 0.03)
        params["sigma"] = float(sigma)
        
    elif fault_kind == "freeze":
        # Freeze has no additional parameters
        params = {}
    
    window = FaultWindow(
        start=fault_start,
        end=fault_end,
        kind=fault_kind,
        params=params
    )
    
    return [window]

