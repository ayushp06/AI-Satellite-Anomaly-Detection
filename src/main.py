import time 
import numpy as np
from sim.attitude import attitudeStep
from sim.telemetry import teleBuild

#Runs 10x per second with no torque (for now) and simple diagonal inertia
def main():
    dt = 0.1
    t = 0.0
    
    q = np.array([1.0, 0.0, 0.0, 0.0])
    w = np.array([0.05, 0.0, 0.0])

    I = np.diag([0.02, 0.02, 0.01])

    torque = np.zeros(3)

    print("Running attitude sim at 10 Hz. Ctrl+C to stop.")
    while True:
        q, w = attitudeStep(q, w, t_b=torque, I=I, dt=dt)
        telemetry = teleBuild(t, q, w)
        print(telemetry)
        t += dt
        time.sleep(dt)

if __name__ == "__main__":
    main()