import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_parquet("telemetry.parquet")

# Plot angular rates
plt.figure()
plt.plot(df["t"], df["w0"], label="w0")
plt.plot(df["t"], df["w1"], label="w1")
plt.plot(df["t"], df["w2"], label="w2")

# Overlay fault flag (scaled so you can see it)
plt.plot(df["t"], df["fault"] * 0.1, label="fault (scaled)")

plt.xlabel("Time [s]")
plt.ylabel("Angular rate [rad/s]")
plt.legend()
plt.title("Telemetry with Fault Injection")
plt.show()
