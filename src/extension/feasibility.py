"""
Extension Module 2: Feasibility analysis across the (L, epsilon) grid.

The key scientific output is the feasibility frontier: for each EO tolerance
epsilon, the minimum Lipschitz constant L at which the joint QP becomes
feasible. This quantifies the fundamental trade-off between individual
fairness (L) and group fairness (epsilon).
"""

import logging
import os
import sys

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from config import RANDOM_SEED, RESULTS_DIR  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

EXT_RESULTS_DIR = os.path.join(RESULTS_DIR, "extension") + os.sep
EXT_FIGURES_DIR = os.path.join(RESULTS_DIR, "extension", "figures") + os.sep

_FEASIBLE_STATUSES = {"optimal", "optimal_inaccurate"}


def compute_feasibility_frontier(grid_df):
    """
    Identify the minimum feasible L for each epsilon from the grid results.

    For each epsilon value, finds the smallest L at which the joint QP is
    feasible. This is the feasibility frontier in (L, epsilon) space: it
    shows how much individual fairness (lower L = stricter) must be relaxed
    to achieve group fairness at each EO tolerance level.

    Parameters
    ----------
    grid_df : pd.DataFrame
        Output of solve_joint_grid, with columns:
        L, epsilon, status, inconsistency_rate, fpr_gap.

    Returns
    -------
    pd.DataFrame
        Columns: epsilon, L_min_feasible, inconsistency_rate_at_Lmin,
        fpr_gap_at_Lmin. Infeasible epsilons have None values.
        Saved to results/extension/feasibility_frontier.csv.
    """
    os.makedirs(EXT_RESULTS_DIR, exist_ok=True)

    feasible = grid_df[grid_df["status"].isin(_FEASIBLE_STATUSES)].copy()

    rows = []
    for eps in sorted(grid_df["epsilon"].unique()):
        sub = feasible[feasible["epsilon"] == eps]
        if sub.empty:
            rows.append(
                {
                    "epsilon": eps,
                    "L_min_feasible": None,
                    "inconsistency_rate_at_Lmin": None,
                    "fpr_gap_at_Lmin": None,
                }
            )
        else:
            idx_min = sub["L"].idxmin()
            r = sub.loc[idx_min]
            rows.append(
                {
                    "epsilon": float(eps),
                    "L_min_feasible": float(r["L"]),
                    "inconsistency_rate_at_Lmin": r["inconsistency_rate"],
                    "fpr_gap_at_Lmin": r["fpr_gap"],
                }
            )

    frontier_df = pd.DataFrame(rows)
    out_path = EXT_RESULTS_DIR + "feasibility_frontier.csv"
    frontier_df.to_csv(out_path, index=False)
    logger.info("Feasibility frontier saved to %s", out_path)
    return frontier_df


def plot_feasibility_heatmap(grid_df, save_path):
    """
    Heatmap of feasibility over the (L, epsilon) grid.

    Cells are colored green (feasible) or red (infeasible). Contour lines
    overlay the inconsistency rate on the feasible region.

    Parameters
    ----------
    grid_df : pd.DataFrame
        Output of solve_joint_grid.
    save_path : str
        Destination file path for the PNG figure (300 DPI).
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    epsilons = sorted(grid_df["epsilon"].unique())
    Ls = sorted(grid_df["L"].unique())
    ne, nL = len(epsilons), len(Ls)

    eps_idx = {e: i for i, e in enumerate(epsilons)}
    L_idx = {L: i for i, L in enumerate(Ls)}

    feasible_mat = np.zeros((nL, ne))
    ir_mat = np.full((nL, ne), np.nan)

    for _, row in grid_df.iterrows():
        i = L_idx[row["L"]]
        j = eps_idx[row["epsilon"]]
        is_ok = row["status"] in _FEASIBLE_STATUSES
        feasible_mat[i, j] = 1.0 if is_ok else 0.0
        if is_ok and row["inconsistency_rate"] is not None:
            ir_mat[i, j] = float(row["inconsistency_rate"])

    fig, ax = plt.subplots(figsize=(9, 6))

    cmap = plt.cm.RdYlGn
    ax.imshow(
        feasible_mat,
        aspect="auto",
        origin="lower",
        cmap=cmap,
        vmin=0,
        vmax=1,
        extent=[-0.5, ne - 0.5, -0.5, nL - 0.5],
    )

    # Contour lines for inconsistency rate on feasible region
    valid = ~np.isnan(ir_mat)
    if valid.any():
        X_mesh, Y_mesh = np.meshgrid(np.arange(ne), np.arange(nL))
        ir_filled = np.where(valid, ir_mat, np.nanmean(ir_mat[valid]))
        try:
            cs = ax.contour(
                X_mesh, Y_mesh, ir_filled,
                levels=5, colors="navy", linewidths=0.8, alpha=0.7,
            )
            ax.clabel(cs, fmt="%.2f", fontsize=8)
        except Exception:
            pass  # contour may fail if all values identical

    ax.set_xticks(range(ne))
    ax.set_xticklabels([f"{e:.2f}" for e in epsilons], fontsize=9)
    ax.set_yticks(range(nL))
    ax.set_yticklabels([f"{L:.2f}" for L in Ls], fontsize=9)
    ax.set_xlabel("Epsilon  (EO tolerance)", fontsize=12)
    ax.set_ylabel("L  (Lipschitz constant)", fontsize=12)
    ax.set_title(
        "Feasibility Heatmap: Joint EO + Lipschitz QP\n"
        "Green = feasible | Red = infeasible | Contours = inconsistency rate",
        fontsize=11,
    )

    red_patch = mpatches.Patch(color="red", label="Infeasible")
    green_patch = mpatches.Patch(color="green", label="Feasible")
    ax.legend(handles=[red_patch, green_patch], loc="upper left", fontsize=9)

    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Feasibility heatmap saved to %s", save_path)


def plot_three_way_frontier(grid_df, save_path):
    """
    2D scatter/Pareto plot of the three-way trade-off.

    x-axis : Group fairness (FPR gap, lower is better)
    y-axis : Individual fairness (Lipschitz L, lower = stricter constraint)
    color  : Accuracy

    Parameters
    ----------
    grid_df : pd.DataFrame
        Output of solve_joint_grid.
    save_path : str
        Destination file path for the PNG figure (300 DPI).
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    feasible = grid_df[
        grid_df["status"].isin(_FEASIBLE_STATUSES)
    ].dropna(subset=["fpr_gap", "accuracy"]).copy()

    if feasible.empty:
        logger.warning("No feasible cells to plot in three-way frontier.")
        return

    fig, ax = plt.subplots(figsize=(8, 6))

    sc = ax.scatter(
        feasible["fpr_gap"],
        feasible["L"],
        c=feasible["accuracy"],
        cmap="viridis",
        s=90,
        edgecolors="black",
        linewidths=0.5,
        alpha=0.85,
    )

    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("Accuracy", fontsize=11)

    for _, row in feasible.iterrows():
        ax.annotate(
            f"ε={row['epsilon']:.2f}",
            (row["fpr_gap"], row["L"]),
            textcoords="offset points",
            xytext=(4, 3),
            fontsize=6,
            alpha=0.7,
        )

    ax.set_xlabel("FPR Gap |Black − White|  (Group Fairness ↓)", fontsize=12)
    ax.set_ylabel(
        "Lipschitz Constant L  (Individual Fairness: lower = stricter ↓)", fontsize=11
    )
    ax.set_title(
        "Three-Way Trade-off: Group Fairness vs Individual Fairness vs Accuracy",
        fontsize=11,
    )

    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Three-way frontier saved to %s", save_path)
