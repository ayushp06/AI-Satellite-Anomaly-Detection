"""
Main simulation entrypoint.

Responsibilities:
1. Initialize attitude state
2. Step dynamics at fixed dt
3. Build telemetry records with randomized faults
4. Log telemetry to Parquet
5. Gracefully shut down after fixed duration
"""

import time 
import numpy as np
from sim.attitude import attitudeStep
from sim.telemetry import teleBuild
from sim.logger import teleLogger
from sim.faults import FaultController, sample_fault_windows


def main():
    dt = 0.1
    t = 0.0
    t_end = 60.0  # 1 minute simulation
    
    # Simulation seed for reproducibility
    sim_seed = 42
    rng = np.random.default_rng(sim_seed)

    # Initial state
    q = np.array([1.0, 0.0, 0.0, 0.0])
    w = np.array([0.05, 0.0, 0.0])

    # Satellite properties
    I = np.diag([0.02, 0.02, 0.01])
    torque = np.zeros(3)

    # Initialize telemetry logger
    logger = teleLogger(batch_size=50)

    print(f"Running attitude sim at 10 Hz for {t_end}s (seed={sim_seed})")
    
    # Sample random fault windows (0 or 1 fault)
    fault_windows = sample_fault_windows(t_end, rng)
    
    if fault_windows:
        fw = fault_windows[0]
        print(f"Injecting {fw.kind} fault from {fw.start:.1f}s to {fw.end:.1f}s")
        print(f"Fault params: {fw.params}")
    else:
        print("No faults injected this run")
    
    # Initialize fault controller
    fc = FaultController(windows=fault_windows, seed=sim_seed)
    
    print("\nPress Ctrl+C to stop.\n")
    
    try:
        while t <= t_end:
            # Step simulation -> truth state
            q, w_true = attitudeStep(q, w, t_b=torque, I=I, dt=dt)

            # Apply faults -> measured telemetry
            w_meas, fault_flag, fault_kind = fc.apply_w(t, w_true)
            
            # Build telemetry using measured w and fault label
            telemetry = teleBuild(t, q, w_meas, faultFlag=fault_flag)

            # Log telemetry automatically
            logger.log(
                telemetry["t"],
                telemetry["q"],
                telemetry["w"],
                telemetry["fault"]
            )

            # Optional: print to console
            print(telemetry)

            # Increment time and sleep to maintain 10 Hz
            t += dt
            time.sleep(dt)

    except KeyboardInterrupt:
        print("\nSimulation interrupted by user.")
        
    finally: 
        print("\nStopping telemetry logging...")
        logger.stop()
        print("Telemetry logging stopped.")


if __name__ == "__main__":
    main()

