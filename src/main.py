"""
Main simulation entrypoint.

Responsibilities:
1. Initialize attitude state
2. Step dynamics at fixed dt
3. Build telemetry records
4. Log telemetry to Parquet
5. Gracefully shut down after fixed duration

Later:
- Enable real-time anomaly detection
- Enable UI hooks
"""

import time 
import numpy as np
from sim.attitude import attitudeStep
from sim.telemetry import teleBuild
from sim.logger import teleLogger
from sim.faults import FaultController, FaultWindow


def main():
    dt = 0.1
    t = 0.0
    t_end = 60.0 #this will make it run for 1 min

    q = np.array([1.0, 0.0, 0.0, 0.0])
    w = np.array([0.05, 0.0, 0.0])

    I = np.diag([0.02, 0.02, 0.01])
    torque = np.zeros(3)

    # Initialize telemetry logger
    logger = teleLogger(batch_size=50)

    print(f"Running attitude sim at 10 Hz for {t_end}. Ctrl+C to stop.")

    
    faults = [
        FaultWindow(start=10.0, end=20.0, kind="gyro_bias", params={"bias": [0.03, 0.0, 0.0]}),
        FaultWindow(start=30.0, end=35.0, kind="noise_burst", params={"sigma": 0.02}),
        FaultWindow(start=45.0, end=50.0, kind="freeze", params={}),
    ]
    fc = FaultController(windows=faults, seed=42)
    
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
        # Gracefully stop the logger when user presses Ctrl+C
        print("\nSimulation interrupted by user.")
        
    finally: 
        print("\nStopping telemetry logging...")
        logger.stop()
        print("Telemetry logging stopped.")


if __name__ == "__main__":
    main()

