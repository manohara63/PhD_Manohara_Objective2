import numpy as np
import pandas as pd
from scipy.stats import qmc

# Define factor levels per subgroup (example shown for all combined)

# Factor levels (numerical or discrete categories)
unsat_params = np.array([3.45, 5, 8.17])
initial_suction = np.array([0.2, 3, 6, 10, 20, 50, 200])
rainfall = np.array([5, 10, 20, 40, 80, 160])

# Subgroup constraints summarized:
subgroups = {
    'NS_small': {
        'SlopeType': ['NS'],
        'SlopeAngle': [28, 32],
        'SoilThickness': [1.5, 3.5, 11]
    },
    'NS_large': {
        'SlopeType': ['NS'],
        'SlopeAngle': [36, 40, 44],
        'SoilThickness': [1.5]
    },
    'FCS': {
        'SlopeType': ['FCS'],
        'SlopeAngle': [24, 28, 32, 36],
        'SoilThickness': [1.5]
    },
    'PCS': {
        'SlopeType': ['PCS'],
        'SlopeAngle': [24, 28],
        'SoilThickness': [3.5, 11]
    }
}

# Number of samples desired per subgroup (small initial sparse sampling)
samples_per_subgroup = 25

# Helper function to map LHS samples to discrete factor levels
def map_to_levels(lhs_samples, levels):
    # lhs_samples assumed between 0 and 1
    idx = (lhs_samples * len(levels)).astype(int)
    idx = np.clip(idx, 0, len(levels)-1)
    return np.array(levels)[idx]

# Generate samples for each subgroup
all_samples = []

for subgroup, factors in subgroups.items():
    # Define dimension count: SlopeAngle, SoilThickness, UnsatParameterA, Initial Suc, Rainfall
    dim = 5
    sampler = qmc.LatinHypercube(d=dim, seed=42)
    sample = sampler.random(n=samples_per_subgroup)
    
    # Map each LHS column to corresponding factor levels
    slope_angle = map_to_levels(sample[:, 0], factors['SlopeAngle'])
    soil_thickness = map_to_levels(sample[:, 1], factors['SoilThickness'])
    unsat_param = map_to_levels(sample[:, 2], unsat_params)
    initial_suction_ = map_to_levels(sample[:, 3], initial_suction)
    rainfall_ = map_to_levels(sample[:, 4], rainfall)
    
    # Construct DataFrame for subgroup samples
    df_sub = pd.DataFrame({
        'Slope type': factors['SlopeType']*samples_per_subgroup,
        'Slope (°)': slope_angle,
        'Soil thickness (m)': soil_thickness,
        'Unsat parameter (a)': unsat_param,
        'Initial suction (kPa)': initial_suction_,
        'Rainfall (mm)': rainfall_
    })
    all_samples.append(df_sub)

# Combine all subgroup samples
initial_sparse_samples = pd.concat(all_samples, ignore_index=True)

# Export to Excel
initial_sparse_samples.to_excel('D:\PhD work\Objective 2\FE modelling\Design of experiments\initial_sparse_lhs_samples_exclude 24.xlsx', index=False)

print(f"Generated {len(initial_sparse_samples)} initial LHS samples.")
print(initial_sparse_samples.head())
