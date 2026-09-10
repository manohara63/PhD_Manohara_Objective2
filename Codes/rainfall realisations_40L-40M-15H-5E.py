import numpy as np
import pandas as pd
import os

def generate_distribution(shape_type, n_hours=24):
    if shape_type == 'uniform':
        arr = np.ones(n_hours) / n_hours
    elif shape_type == 'increasing':
        arr = np.linspace(0.1, 1, n_hours)
    elif shape_type == 'bell':
        arr = np.exp(-((np.arange(n_hours) - (n_hours / 2)) ** 2) / (2 * (n_hours / 8) ** 2))
    elif shape_type == 'exponential':
        arr = np.exp(np.linspace(0, 2, n_hours))
    elif shape_type == 'bimodal':
        arr = (
            np.exp(-((np.arange(n_hours) - n_hours / 4) ** 2) / (2 * (n_hours / 10) ** 2))
            + np.exp(-((np.arange(n_hours) - 3 * n_hours / 4) ** 2) / (2 * (n_hours / 10) ** 2))
        )
    else:
        arr = np.zeros(n_hours)
    return arr

def drizzle_pattern(n_hours=24):
    # Light, low-intensity almost uniform drizzle with small noise
    base = np.ones(n_hours) * 0.8
    noise = np.random.rand(n_hours) * 0.4
    arr = base + noise
    arr = arr / arr.sum()
    return arr

def moderate_showers_pattern(n_hours=24):
    # Several (2-4) moderate rainfall clusters randomly distributed
    arr = np.zeros(n_hours)
    num_showers = np.random.randint(2, 5)
    for _ in range(num_showers):
        center = np.random.randint(0, n_hours)
        width = np.random.randint(1, 4)
        spread = np.exp(-0.5 * ((np.arange(n_hours) - center) / width)**2)
        intensity = np.random.uniform(0.5, 1.5)
        arr += intensity * spread
    arr = arr + 0.1 * np.random.rand(n_hours)  # baseline noise
    arr = arr.clip(min=0)
    arr = arr / arr.sum()
    return arr

def intense_showers_pattern(n_hours=24):
    # 1-2 intense rainfall periods, sharp peaks, rest dry
    arr = np.zeros(n_hours)
    num_peaks = np.random.choice([1, 2])
    for _ in range(num_peaks):
        center = np.random.randint(0, n_hours)
        width = np.random.randint(1, 2)
        spread = np.exp(-0.5 * ((np.arange(n_hours) - center) / width) ** 2)
        intensity = np.random.uniform(1.5, 3.0)
        arr += intensity * spread
    arr = arr.clip(min=0)
    arr = arr / arr.sum()
    return arr

def sporadic_extreme_pattern(n_hours=24):
    # Rare intense event: all rain in 1 or 2 hours
    arr = np.zeros(n_hours)
    spike_hours = np.random.choice(n_hours, np.random.choice([1, 2]), replace=False)
    spike_values = np.random.rand(len(spike_hours))
    spike_values = spike_values / spike_values.sum()
    arr[spike_hours] = spike_values
    residual = 1 - arr.sum()
    arr += residual * np.random.rand(n_hours) * 0.1  # tiny drizzle elsewhere
    arr = arr.clip(min=0)
    arr = arr / arr.sum()
    return arr

def generate_daily_realistic_pattern():
    # Choose pattern probabilistically
    pattern_types = ['drizzle', 'moderate', 'intense', 'sporadic']
    probabilities = [0.25, 0.60, 0.12, 0.03]
    choice = np.random.choice(pattern_types, p=probabilities)
    if choice == 'drizzle':
        return drizzle_pattern()
    elif choice == 'moderate':
        return moderate_showers_pattern()
    elif choice == 'intense':
        return intense_showers_pattern()
    else:  # sporadic
        return sporadic_extreme_pattern()

def generate_daily_pattern(realisation_index, day_index, n_hours=24, fixed_shapes=None):
    if fixed_shapes is not None and realisation_index < len(fixed_shapes):
        shape_type = fixed_shapes[realisation_index]
        arr = generate_distribution(shape_type, n_hours)
    else:
        arr = generate_daily_realistic_pattern()
    return arr

def generate_realisation(day_total_mm=5, n_days=90, n_hours=24, realisation_index=0, fixed_shapes=None):
    data = []
    for day in range(n_days):
        distribution = generate_daily_pattern(realisation_index, day, n_hours, fixed_shapes)
        hourly_rainfall = distribution / distribution.sum() * day_total_mm
        data.append(hourly_rainfall)
    data = np.array(data)
    return data

def save_realisation_to_csv(data, realisation_index, output_folder):
    columns = [f'Hour_{i + 1}' for i in range(data.shape[1])]
    df = pd.DataFrame(data, columns=columns)
    df.insert(0, 'Day', np.arange(1, data.shape[0] + 1))
    os.makedirs(output_folder, exist_ok=True)
    filename = os.path.join(output_folder, f'rainfall_realisations_{realisation_index + 1}.csv')
    df.to_csv(filename, index=False)
    print(f'Saved realisation {realisation_index + 1} to {filename}')

def main():
    day_total_mm = 1
    n_days = 90
    n_hours = 24
    n_realisations = 20
    
    # Specify your desired directory path here (absolute or relative)
    output_folder = r'D:\PhD work\Objective 2\FE modelling\Geostudio\Model\Slope stability for random rainfall realisations\rfr_1mm'
    
    np.random.seed(42)

    fixed_shapes = ['uniform', 'increasing', 'bell', 'exponential', 'bimodal']

    for i in range(n_realisations):
        if i < len(fixed_shapes):
            data = generate_realisation(day_total_mm, n_days, n_hours, i, fixed_shapes)
        else:
            data = generate_realisation(day_total_mm, n_days, n_hours, i, fixed_shapes=None)
        save_realisation_to_csv(data, i, output_folder)

if __name__ == "__main__":
    main()
