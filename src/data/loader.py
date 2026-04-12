"""
Synthetic CMAPSS-style dataset generator.
Mimics NASA's C-MAPSS turbofan engine degradation dataset:
  - 21 sensors (temperature, pressure, speed, flow metrics)
  - 3 operational settings
  - Time-series degradation with injected anomalies
  - Text maintenance logs (multimodal component)
"""

import numpy as np
import pandas as pd
from pathlib import Path

SENSOR_NAMES = [
    "T2",     # Fan inlet temperature (K)
    "T24",    # LPC outlet temperature (K)
    "T30",    # HPC outlet temperature (K)
    "T50",    # LPT outlet temperature (K)
    "P2",     # Fan inlet pressure (psi)
    "P15",    # Bypass-duct pressure (psi)
    "P30",    # HPC outlet pressure (psi)
    "Nf",     # Physical fan speed (rpm)
    "Nc",     # Physical core speed (rpm)
    "epr",    # Engine pressure ratio
    "Ps30",   # HPC static pressure (psi)
    "phi",    # Fuel flow / Ps30 ratio
    "NRf",    # Corrected fan speed
    "NRc",    # Corrected core speed
    "BPR",    # Bypass ratio
    "farB",   # Burner fuel-air ratio
    "htBleed",# Bleed enthalpy
    "Nf_dmd", # Demanded fan speed
    "PCNfR",  # Fan speed at inlet
    "W31",    # HPT coolant bleed
    "W32",    # LPT coolant bleed
]

# Baseline healthy values (approximate real CMAPSS ranges)
SENSOR_BASELINES = {
    "T2":      518.67, "T24":    642.15, "T30":    1583.04,
    "T50":    1400.60, "P2":      14.62, "P15":      21.61,
    "P30":     549.19, "Nf":     2388.01,"Nc":     9046.19,
    "epr":       1.30, "Ps30":    47.20, "phi":     521.66,
    "NRf":    2388.01, "NRc":    8138.62,"BPR":       8.44,
    "farB":      0.03, "htBleed": 392.0, "Nf_dmd": 2388.01,
    "PCNfR":   100.00, "W31":     38.94, "W32":      23.39,
}

SENSOR_NOISE = {s: v * 0.005 for s, v in SENSOR_BASELINES.items()}  # 0.5% noise

# Sensors that degrade (increase) over lifetime
DEGRADING_UP   = ["T24", "T30", "T50", "P30", "phi", "W31", "W32"]
# Sensors that degrade (decrease) over lifetime
DEGRADING_DOWN = ["Ps30", "NRf", "NRc", "BPR", "epr"]

MAINTENANCE_TEMPLATES = [
    "Cycle {cycle}: Routine inspection completed. All systems nominal.",
    "Cycle {cycle}: Fan blade tip clearance measured. Within tolerance.",
    "Cycle {cycle}: HPT rotor inspection. Minor oxidation detected.",
    "Cycle {cycle}: LPT blade inspection. No cracks found.",
    "Cycle {cycle}: Borescope inspection of HPC stages 1-3. Light coating degradation.",
    "Cycle {cycle}: Oil analysis completed. Elevated metal particles detected.",
    "Cycle {cycle}: Vibration signature elevated on fan module. Monitoring.",
    "Cycle {cycle}: Fuel nozzle inspection. One nozzle showing partial blockage.",
    "Cycle {cycle}: Turbine blade temperature spread increasing. Combustor check needed.",
    "Cycle {cycle}: HPT tip clearance out of spec. Maintenance advisory issued.",
]


def generate_synthetic_cmapss(
    n_engines: int = 50,
    min_cycles: int = 100,
    max_cycles: int = 250,
    anomaly_fraction: float = 0.15,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate synthetic CMAPSS-like sensor data + text maintenance logs.

    Returns:
        sensor_df: columns = [unit, cycle, op1, op2, op3, s1..s21]
        logs_df:   columns = [unit, cycle, log_text]
    """
    rng = np.random.default_rng(seed)
    sensor_rows = []
    log_rows = []

    for unit in range(1, n_engines + 1):
        n_cycles = rng.integers(min_cycles, max_cycles)
        degradation = np.linspace(0, 1, n_cycles)  # 0 = healthy, 1 = failure

        for cycle_idx, deg in enumerate(degradation):
            cycle = cycle_idx + 1
            row = {"unit": unit, "cycle": cycle}

            # Operational settings (3 clusters)
            op_mode = rng.choice([0, 1, 2])
            row["op1"] = [0.0, 0.42, 1.0][op_mode] + rng.normal(0, 0.002)
            row["op2"] = [0.0, 0.21, 0.84][op_mode] + rng.normal(0, 0.002)
            row["op3"] = [100.0, 60.0, 80.0][op_mode]

            # Inject anomaly near end of life
            is_anomalous = deg > (1 - anomaly_fraction)
            anomaly_boost = rng.uniform(0.05, 0.15) if is_anomalous else 0.0

            for s in SENSOR_NAMES:
                base = SENSOR_BASELINES[s]
                noise = rng.normal(0, SENSOR_NOISE[s])
                if s in DEGRADING_UP:
                    drift = base * deg * 0.08 + base * anomaly_boost
                elif s in DEGRADING_DOWN:
                    drift = -base * deg * 0.06 - base * anomaly_boost * 0.5
                else:
                    drift = 0.0
                row[s] = round(base + drift + noise, 4)

            sensor_rows.append(row)

            # Add maintenance log ~every 20 cycles
            if cycle % 20 == 0 or is_anomalous:
                tmpl = rng.choice(MAINTENANCE_TEMPLATES)
                log_rows.append({
                    "unit": unit,
                    "cycle": cycle,
                    "log_text": tmpl.format(cycle=cycle),
                })

    sensor_df = pd.DataFrame(sensor_rows)
    logs_df = pd.DataFrame(log_rows)
    return sensor_df, logs_df


def load_engine_data(
    sensor_df: pd.DataFrame,
    logs_df: pd.DataFrame,
    unit: int,
    cycle_range: tuple[int, int] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return sensor + log data for a single engine, optionally sliced by cycle range."""
    s = sensor_df[sensor_df["unit"] == unit].copy()
    l = logs_df[logs_df["unit"] == unit].copy()

    if cycle_range:
        s = s[(s["cycle"] >= cycle_range[0]) & (s["cycle"] <= cycle_range[1])]
        l = l[(l["cycle"] >= cycle_range[0]) & (l["cycle"] <= cycle_range[1])]

    return s.reset_index(drop=True), l.reset_index(drop=True)


def compute_rolling_stats(sensor_df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    """Add rolling mean, std, and z-score columns for each sensor."""
    df = sensor_df.copy()
    for s in SENSOR_NAMES:
        if s not in df.columns:
            continue
        roll = df[s].rolling(window=window, min_periods=3)
        df[f"{s}_mean"] = roll.mean()
        df[f"{s}_std"]  = roll.std().fillna(1e-6)
        df[f"{s}_zscore"] = (df[s] - df[f"{s}_mean"]) / df[f"{s}_std"].clip(lower=1e-6)
    return df


def save_to_csv(sensor_df: pd.DataFrame, logs_df: pd.DataFrame, out_dir: str = "data") -> None:
    Path(out_dir).mkdir(exist_ok=True)
    sensor_df.to_csv(f"{out_dir}/sensor_data.csv", index=False)
    logs_df.to_csv(f"{out_dir}/maintenance_logs.csv", index=False)
    print(f"Saved {len(sensor_df)} sensor rows and {len(logs_df)} log entries to {out_dir}/")


if __name__ == "__main__":
    sensor_df, logs_df = generate_synthetic_cmapss()
    save_to_csv(sensor_df, logs_df)
    print(sensor_df.head())
    print(logs_df.head())
