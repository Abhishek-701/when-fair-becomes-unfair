"""Global configuration constants for the CS 516 EO Fairness project."""

import os

RANDOM_SEED = 42
DATA_RAW_PATH = "data/compas_raw.csv"
DATA_CLEAN_PATH = "results/compas_clean.csv"
RESULTS_DIR = "results/"
FIGURES_DIR = "results/figures/"
TEST_SIZE = 0.20
K_NEIGHBORS = 5
BOOTSTRAP_N = 10_000
EPSILON_GRID = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]

# Feature list used for modeling (no race)
FEATURES_NR = [
    'age',
    'sex',
    'juv_fel_count',
    'juv_misd_count',
    'juv_other_count',
    'priors_count',
    'charge_degree',
]

# Colorblind-accessible palette
COLOR_WHITE = '#0077BB'
COLOR_BLACK = '#EE7733'
COLOR_NEUTRAL = '#BBBBBB'

# Ensure results directories exist when config is imported
for _dir in [RESULTS_DIR, FIGURES_DIR, "results/robustness"]:
    os.makedirs(_dir, exist_ok=True)
