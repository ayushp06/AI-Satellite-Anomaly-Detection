"""
Dataset Generator v1 - Randomized Fault Injection

Generates a reproducible dataset of satellite telemetry with random faults
suitable for training unsupervised anomaly detection models.

Usage:
    python scripts/generate_dataset_v1.py --num-runs 100 --output-dir data/dataset_v1

Output structure:
    data/dataset_v1/
    |-- telemetry_run_0000.parquet
    |-- metadata_run_0000.json
    |-- telemetry_run_0001.parquet
    |-- metadata_run_0001.json
    `-- ...
"""


import sys
from pathlib import Path

# Add src directory to Python path
script_dir = Path(__file__).parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root / "src"))

import argparse
import json
import time
import numpy as np

from sim.attitude import attitudeStep
from sim.telemetry import teleBuild
from sim.logger import teleLogger
from sim.faults import FaultController, sample_fault_windows


def run_single_simulation(
    run_id: int,
    seed: int,
    output_dir: Path,
    run_duration: float = 60.0,
    dt: float = 0.1,
    verbose: bool = False
):
    """
    Run a single simulation with randomized fault injection.
    
    Parameters:
    -----------
    run_id : int
        Unique identifier for this run
    seed : int
        Random seed for reproducibility
    output_dir : Path
        Directory to save telemetry and metadata
    run_duration : float
        Total simulation time in seconds
    dt : float
        Time step in seconds (10 Hz = 0.1s)
    verbose : bool
        Print progress messages
    """
    # Create RNG for this run
    rng = np.random.default_rng(seed)
    
    # Sample fault windows (0 or 1 fault)
    fault_windows = sample_fault_windows(run_duration, rng)
    
    # Initialize simulation state
    t = 0.0
    q = np.array([1.0, 0.0, 0.0, 0.0])
    w = np.array([0.05, 0.0, 0.0])  # small initial rotation
    I = np.diag([0.02, 0.02, 0.01])  # moment of inertia
    torque = np.zeros(3)
    
    # Initialize fault controller
    fc = FaultController(windows=fault_windows, seed=seed)
    
    # Create unique output filename
    run_id_str = f"{run_id:04d}"
    telemetry_file = output_dir / f"telemetry_run_{run_id_str}.parquet"
    
    # Initialize logger (will write to specified file)
    logger = teleLogger(batch_size=50, output_file=str(telemetry_file))
    
    if verbose:
        fault_desc = "no fault"
        if fault_windows:
            fw = fault_windows[0]
            fault_desc = f"{fw.kind} from {fw.start:.1f}s to {fw.end:.1f}s"
        print(f"Run {run_id_str}: seed={seed}, {fault_desc}")
    
    # Run simulation loop
    try:
        while t <= run_duration:
            # Step physics (truth state)
            q, w_true = attitudeStep(q, w, t_b=torque, I=I, dt=dt)
            
            # Apply faults to measurements
            w_meas, fault_flag, fault_kind = fc.apply_w(t, w_true)
            
            # Build telemetry record
            telemetry = teleBuild(t, q, w_meas, faultFlag=fault_flag)
            
            # Log to Parquet
            logger.log(
                telemetry["t"],
                telemetry["q"],
                telemetry["w"],
                telemetry["fault"]
            )
            
            t += dt
            
    finally:
        # Ensure logger flushes to disk
        logger.stop()
    
    # Save metadata
    metadata = {
        "run_id": run_id,
        "seed": seed,
        "run_duration": run_duration,
        "dt": dt,
        "initial_quaternion": q.tolist(),
        "initial_angular_velocity": w.tolist(),
        "fault_injected": len(fault_windows) > 0,
        "fault_windows": [fw.to_dict() for fw in fault_windows]
    }
    
    metadata_file = output_dir / f"metadata_run_{run_id_str}.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return metadata


def generate_dataset(
    num_runs: int,
    output_dir: str,
    base_seed: int = 42,
    run_duration: float = 60.0,
    dt: float = 0.1
):
    """
    Generate complete dataset with N randomized simulation runs.
    
    Parameters:
    -----------
    num_runs : int
        Number of simulation runs to generate
    output_dir : str
        Output directory path
    base_seed : int
        Base random seed (each run gets base_seed + run_id)
    run_duration : float
        Duration of each run in seconds
    dt : float
        Simulation time step
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating dataset_v1 with {num_runs} runs...")
    print(f"Output directory: {output_path.absolute()}")
    print(f"Base seed: {base_seed}")
    print(f"Run duration: {run_duration}s at {1/dt:.0f} Hz\n")
    
    start_time = time.time()
    
    # Track statistics
    fault_counts = {"none": 0, "gyro_bias": 0, "noise_burst": 0, "freeze": 0}
    
    # Generate runs
    for run_id in range(num_runs):
        seed = base_seed + run_id
        
        metadata = run_single_simulation(
            run_id=run_id,
            seed=seed,
            output_dir=output_path,
            run_duration=run_duration,
            dt=dt,
            verbose=True
        )
        
        # Update statistics
        if metadata["fault_injected"]:
            fault_kind = metadata["fault_windows"][0]["kind"]
            fault_counts[fault_kind] += 1
        else:
            fault_counts["none"] += 1
    
    elapsed = time.time() - start_time
    
    # Save dataset summary
    summary = {
        "dataset_version": "v1",
        "num_runs": num_runs,
        "base_seed": base_seed,
        "run_duration": run_duration,
        "dt": dt,
        "samples_per_run": int(run_duration / dt) + 1,
        "fault_distribution": fault_counts,
        "generation_time_seconds": elapsed,
        "output_directory": str(output_path.absolute())
    }
    
    summary_file = output_path / "dataset_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Print summary
    print("\n" + "="*60)
    print("Dataset generation complete!")
    print("="*60)
    print(f"Total runs: {num_runs}")
    print(f"Generation time: {elapsed:.1f}s ({elapsed/num_runs:.2f}s per run)")
    print(f"\nFault distribution:")
    for fault_type, count in fault_counts.items():
        percentage = 100 * count / num_runs
        print(f"  {fault_type:15s}: {count:4d} runs ({percentage:5.1f}%)")
    print(f"\nDataset saved to: {output_path.absolute()}")
    print(f"Summary: {summary_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate randomized satellite telemetry dataset for anomaly detection"
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=100,
        help="Number of simulation runs to generate (default: 100)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/dataset_v1",
        help="Output directory for dataset (default: data/dataset_v1)"
    )
    parser.add_argument(
        "--base-seed",
        type=int,
        default=42,
        help="Base random seed (default: 42)"
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=60.0,
        help="Duration of each run in seconds (default: 60.0)"
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=0.1,
        help="Simulation time step in seconds (default: 0.1)"
    )
    
    args = parser.parse_args()
    
    generate_dataset(
        num_runs=args.num_runs,
        output_dir=args.output_dir,
        base_seed=args.base_seed,
        run_duration=args.duration,
        dt=args.dt
    )


if __name__ == "__main__":
    main()