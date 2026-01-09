Added fault injection and plot visuals for different fault injections.

Examples:
Gyro Bias: W drifts
Noise Bursts: W spikes
Sensor Freeze: W stuck

Plots used to visualize these faults. 
Run plot_faults.py after running main.py

How the plots work:
Red line drawn as a step function, when a fault is active (1) or when its not (0).