# -*- coding: utf-8 -*-
"""
Created on Mon Feb 16 11:44:11 2026

@author: Dr. Arindam Dey
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy.interpolate import LinearNDInterpolator

# ============================================================
# USER INPUT
# ============================================================
input_file = r"D:\PhD work\Objective 2\FE modelling\quantile threshold curves\IS_Sl_Rf_Df input.xlsx"
sheet_name = "Exclude 3"

# Grid resolution (increase carefully — memory heavy)
grid_res = 80   # 40^3 = 64,000 grid points (safe)
# 60^3 = 216,000 (moderate)
# 80^3 = 512,000 (heavy)

# ============================================================
# 1. LOAD DATA
# ============================================================
df = pd.read_excel(input_file, sheet_name=sheet_name)

df = df[["Slope", "Rainfall", "Initial_Suction", "Df_90th_percentile"]].apply(
    pd.to_numeric, errors="coerce"
).dropna()

# Remove non-positive (log requirement)
df = df[
    (df["Slope"] > 0) &
    (df["Rainfall"] > 0) &
    (df["Initial_Suction"] > 0) &
    (df["Df_90th_percentile"] > 0)
]

# ============================================================
# 2. BUILD 3D INTERPOLATED SURFACE
# ============================================================
interp = LinearNDInterpolator(
    list(zip(df["Slope"], df["Rainfall"], df["Initial_Suction"])),
    df["Df_90th_percentile"]
)

# ============================================================
# 3. CREATE REGULAR 3D GRID
# ============================================================
Sg = np.linspace(df["Slope"].min(), df["Slope"].max(), grid_res)
Rg = np.linspace(df["Rainfall"].min(), df["Rainfall"].max(), grid_res)
ISg = np.linspace(df["Initial_Suction"].min(), df["Initial_Suction"].max(), grid_res)

Sg, Rg, ISg = np.meshgrid(Sg, Rg, ISg)

# Interpolated Df values
Df_grid = interp(Sg, Rg, ISg)

# ============================================================
# 4. REMOVE NaNs
# ============================================================
mask = ~np.isnan(Df_grid)

S_fit = Sg[mask]
R_fit = Rg[mask]
IS_fit = ISg[mask]
Df_fit = Df_grid[mask]

# ============================================================
# 5. LOG–LOG POWER LAW FIT
#     Df = A · S^α · R^β · IS^γ
# ============================================================
X = np.column_stack([
    np.log(S_fit),
    np.log(R_fit),
    np.log(IS_fit)
])

y = np.log(Df_fit)

model = LinearRegression()
model.fit(X, y)

lnA = model.intercept_
alpha, beta, gamma = model.coef_

A = np.exp(lnA)
R2 = model.score(X, y)

# ============================================================
# 6. RESULTS
# ============================================================
print("\n=== POWER-LAW FIT TO MONOTONIC 3D SURFACE ===\n")
print("Df = A · S^α · R^β · IS^γ\n")
print(f"A     = {A:.6f}")
print(f"α     = {alpha:.4f}  (slope sensitivity)")
print(f"β     = {beta:.4f}  (rainfall sensitivity)")
print(f"γ     = {gamma:.4f}  (initial suction sensitivity)")
print(f"R²    = {R2:.4f}")

print("\nFinal Equation:\n")
print(
    f"Df = {A:.6f} · S^{alpha:.4f} · R^{beta:.4f} · IS^{gamma:.4f}"
)
