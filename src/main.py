import time 
import numpy as np
from sim.attitude import attitudeStep
from sim.telemetry import teleBuild
from sim.logger import teleLogger


def main():
    dt = 0.1
    t = 0.0

    q = np.array([1.0, 0.0, 0.0, 0.0])
    w = np.array([0.05, 0.0, 0.0])

    I = np.diag([0.02, 0.02, 0.01])
    torque = np.zeros(3)

    # Initialize telemetry logger
    logger = teleLogger(batch_size=50)

    print("Running attitude sim at 10 Hz. Ctrl+C to stop.")

    try:
        while True:
            # Step simulation
            q, w = attitudeStep(q, w, t_b=torque, I=I, dt=dt)
            
            # Build telemetry
            telemetry = teleBuild(t, q, w)

            # Log telemetry automatically
            logger.log(*telemetry.values())

            # Optional: print to console
            print(telemetry)

            # Increment time and sleep to maintain 10 Hz
            t += dt
            time.sleep(dt)

    except KeyboardInterrupt:
        # Gracefully stop the logger when user presses Ctrl+C
        print("\nStopping telemetry logging...")
        logger.stop()
        print("Telemetry logging stopped.")


if __name__ == "__main__":
    main()

