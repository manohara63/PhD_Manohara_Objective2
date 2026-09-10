# ============================================================
# PHYSICALLY CONSISTENT DF–SLOPE–RAINFALL ANALYSIS
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.interpolate import PchipInterpolator
from mpl_toolkits.mplot3d import Axes3D

# ============================================================
# USER PATHS (EDIT ONLY THESE)
# ============================================================
input_file = r"D:\PhD work\Objective 2\FE modelling\quantile threshold curves\Slope_Rainfall_Df_Quantiles_is0.2.xlsx"
save_dir   = r"D:\PhD work\Objective 2\FE modelling\quantile threshold curves\plots_is0.2"

os.makedirs(save_dir, exist_ok=True)

# ============================================================
# GLOBAL PLOT SETTINGS
# ============================================================
plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 12,
    "axes.labelsize": 14,
    "axes.titlesize": 16
})

# ============================================================
# 1. LOAD DATA
# ============================================================
df = pd.read_excel(input_file)
df = df.apply(pd.to_numeric, errors="coerce").dropna()

# ============================================================
# 2. GRID DEFINITION
# ============================================================
slopes = np.sort(df["Slope"].unique())
rain   = np.sort(df["Rainfall"].unique())

slope_f = np.linspace(slopes.min(), slopes.max(), 120)
rain_f  = np.linspace(rain.min(), rain.max(), 150)

S_f, R_f = np.meshgrid(slope_f, rain_f, indexing="ij")

# ============================================================
# 3. PHYSICALLY CONSTRAINED (MONOTONIC) INTERPOLATION
# ============================================================
def monotone_surface(df, value_col):
    """
    Monotone in rainfall, linear across slope
    Guarantees no closed contours or artefacts
    """
    Z = np.zeros((len(slope_f), len(rain_f)))

    # Rainfall-wise monotone interpolation
    for i, s in enumerate(slope_f):
        d = df[df["Slope"] == s].sort_values("Rainfall")

        if len(d) < 3:
            Z[i, :] = np.nan
            continue

        f = PchipInterpolator(d["Rainfall"], d[value_col], extrapolate=True)
        Z[i, :] = f(rain_f)

    # Linear interpolation across slope
    for j in range(len(rain_f)):
        col = Z[:, j]
        mask = ~np.isnan(col)
        Z[:, j] = np.interp(slope_f, slope_f[mask], col[mask])

    return Z

Zm  = monotone_surface(df, "Df_mean")
Zsd = monotone_surface(df, "Df_std")
Zl  = Zm - Zsd
Zh  = Zm + Zsd
Zmd = monotone_surface(df, "Df_median")
Z10 = monotone_surface(df, "Df_10th_percentile")
Z90 = monotone_surface(df, "Df_90th_percentile")

# ============================================================
# 4. PLOTTING FUNCTIONS
# ============================================================
def save_3d(Z, title, fname, cmap="viridis"):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(S_f, R_f, Z, cmap=cmap, linewidth=0)
    ax.set_xlabel("Slope (°)")
    ax.set_ylabel("Rainfall (mm)")
    ax.set_zlabel("Days to Failure")
    ax.set_title(title)
    ax.view_init(25, 135)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, fname), dpi=600)
    plt.close()

def save_contour(
    Z,
    title,
    fname,
    start_level=1,
    step=5,
    cmap="viridis"
):
    """
    Explicit, controlled contour levels
    """
    Zmin = np.nanmin(Z)
    Zmax = np.nanmax(Z)

    first = max(start_level, int(np.ceil(Zmin)))
    levels = np.arange(first, Zmax + step, step) #for all other initial suction

    plt.figure(figsize=(7, 6))
    cs = plt.contour(
        S_f, R_f, Z,
        levels=levels,
        cmap=cmap,
        linewidths=0.9
    )
    plt.clabel(cs, fmt="%d", fontsize=9)

    plt.xlabel("Slope (°)")
    plt.ylabel("Rainfall (mm)")
    plt.title(title)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, fname), dpi=600)
    plt.close()

# ============================================================
# 5. 3D SURFACES (INDIVIDUAL)
# ============================================================
save_3d(Zm,  "Mean Days to Failure",            "3D_Mean2.png")
save_3d(Zl,  "Mean − 1 SD Days to Failure",     "3D_Mean_minus_SD2.png")
save_3d(Zh,  "Mean + 1 SD Days to Failure",     "3D_Mean_plus_SD2.png")
save_3d(Zmd, "Median Days to Failure",          "3D_Median2.png", "plasma")
save_3d(Z10, "10th Percentile Days to Failure", "3D_10th2.png",   "plasma")
save_3d(Z90, "90th Percentile Days to Failure", "3D_90th2.png",   "plasma")

# ============================================================
# 6. COMBINED 3D SURFACES
# ============================================================
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")
ax.plot_surface(S_f, R_f, Zm, cmap="viridis", alpha=0.9)
ax.plot_surface(S_f, R_f, Zl, color="orange", alpha=0.25)
ax.plot_surface(S_f, R_f, Zh, color="orange", alpha=0.25)
ax.set_xlabel("Slope (°)")
ax.set_ylabel("Rainfall (mm)")
ax.set_zlabel("Days to Failure")
ax.set_title("Mean Days to Failure with ±1 SD")
ax.view_init(25, 135)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "3D_Mean_with_SD2.png"), dpi=600)
plt.close()

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")
ax.plot_surface(S_f, R_f, Zmd, cmap="plasma", alpha=0.9)
ax.plot_surface(S_f, R_f, Z10, color="grey", alpha=0.3)
ax.plot_surface(S_f, R_f, Z90, color="grey", alpha=0.3)
ax.set_xlabel("Slope (°)")
ax.set_ylabel("Rainfall (mm)")
ax.set_zlabel("Days to Failure")
ax.set_title("Median Days to Failure with 10–90 Percentiles")
ax.view_init(25, 135)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "3D_Median_10_90_2.png"), dpi=600)
plt.close()

# ============================================================
# 7. CONTOUR PLOTS (CONTROLLED LEVELS)
# ============================================================
save_contour(Zm,  "Mean Days to Failure",            "Contour_Mean2.png")
save_contour(Zl,  "Mean − 1 SD Days to Failure",     "Contour_Mean_minus_SD2.png")
save_contour(Zh,  "Mean + 1 SD Days to Failure",     "Contour_Mean_plus_SD2.png")
save_contour(Zmd, "Median Days to Failure",          "Contour_Median2.png")
save_contour(Z10, "10th Percentile Days to Failure", "Contour_10th2.png")
save_contour(Z90, "90th Percentile Days to Failure", "Contour_90th2.png")

# ============================================================
# 8. RAINFALL–DF CURVES (MEAN ± SD)
# ============================================================
plt.figure(figsize=(8, 6))
rain_dense = np.linspace(rain.min(), rain.max(), 300)

for s in slopes:
    d = df[df["Slope"] == s].sort_values("Rainfall")
    f_m = PchipInterpolator(d["Rainfall"], d["Df_mean"])
    f_s = PchipInterpolator(d["Rainfall"], d["Df_std"])

    m = f_m(rain_dense)
    sdev = f_s(rain_dense)

    plt.plot(rain_dense, m, label=f"{s}°")
    plt.fill_between(rain_dense, m - sdev, m + sdev, alpha=0.25)

plt.xlabel("Rainfall (mm)")
plt.ylabel("Days to Failure")
plt.title("Rainfall–Days to Failure (Mean ± SD)")
plt.legend(title="Slope")
plt.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "Rainfall_vs_Df_Mean_SD2.png"), dpi=600)
plt.close()

print("ALL PLOTS GENERATED SUCCESSFULLY")
