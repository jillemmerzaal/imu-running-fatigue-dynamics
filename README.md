# IMU Running Fatigue Dynamics

A Python pipeline for analyzing running gait dynamics from inertial measurement unit (IMU) data, with a focus on fatigue-related changes in movement quality and variability.

## Overview

This pipeline processes raw IMU accelerometer data (`.mat` files) collected from a lumbar sensor during pre- and post-fatigue running trials. It computes a suite of linear and nonlinear metrics that characterize gait symmetry, smoothness, complexity, and dynamic stability, then exports results to Excel for statistical analysis.

## Pipeline

```
Raw .mat files
    └── extract2zoo     — convert .mat → structured .json (zoo format)
    └── TiltCorrection  — correct sensor tilt/orientation
    └── Analysis        — compute gait metrics (stored as events in .json)
    └── FFT             — frequency-domain decomposition
    └── Results export  — aggregate metrics into results.xlsx
```

## Metrics

| Metric | Description |
|---|---|
| Step Symmetry & Stride Regularity | Autocorrelation-based symmetry index |
| RMS | Root mean square acceleration (variability proxy) |
| Sample Entropy | Signal complexity |
| Sample Entropy with Delay | Complexity with time-delay embedding |
| Log Dimensionless Jerk (LDLJ) | Movement smoothness |
| Lyapunov Exponent (LyE) | Local dynamic stability (short and long horizon) |
| FFT | Power spectrum of lumbar accelerations |

## Data Format

Input files are `.mat` files named `{subjectID}_{sex}.mat`. The pipeline splits each file into pre- and post-fatigue conditions, producing files named `{subjectID}_{sex}_{Pre|Post}.json`.

The lumbar sensor provides three acceleration axes: vertical (`avert`), mediolateral (`amedlat`), and anteroposterior (`aantpost`), plus a resultant vector (`aresvec`).

## Usage

1. Place raw `.mat` files in a `data40/` folder at the project root. Sample data of 1 female and one male are provided.
2. Run the pipeline:

```bash
python main.py
```

Outputs are written to subfolders within `data40/`:
- `mat2zoo/` — extracted JSON files
- `tiltcorrected/` — tilt-corrected JSON files
- `nld/` — files with computed metric events
- `stats/results.xlsx` — summary table with one sheet per metric

## Installation

```bash
pip install -r requirements.txt
```

Requires Python 3.8+.

## Project Structure

```
├── main.py              # Main pipeline script
├── requirements.txt
└── src/
    ├── engine.py        # File discovery utility
    ├── extract2zoo.py   # .mat → .json conversion
    ├── TiltCorrection.py
    ├── symmetry.py      # Step symmetry & stride regularity
    ├── RMS.py
    ├── sample_entropy.py
    ├── sample_entropy_delay.py
    ├── ldlj.py          # Log dimensionless jerk
    ├── LyE.py           # Lyapunov exponent
    ├── grab.py          # .mat data loader
    ├── zsave.py         # JSON save utility
    ├── add_channel.py   # Add channels to zoo data structure
    ├── fldOrganize.py   # File organization helper
    ├── fileparts.py     # Path parsing utility
    └── setZoosystem.py  # Initialize zoo data structure
```
