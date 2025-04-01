"""
Unnormalize a data given its statistcs

This is needed as all training data are normalized to the range [-1, 1] before training 
"""
import numpy as np 

def normalize_mag(unnormalized_data : np.ndarray, stats : dict, eps : float = 1e-8):
    """
    unnormalized_data (np.ndarray) : A 3xWxH image where the 3 rows are the RGB magnitude 
    
    stats (dict) : Statistics of the respective array, it should at least contain
                   the max and min of the data array BEFORE normalization 

                   The naming convetions for min and max for mag and phase are:
                   mag_R_max, mag_G_max, mag_B_max
                   mag_R_min, mag_G_MIN, mag_B_MIN
                   phase_R_max, phase_G_max, phase_B_max
                   phase_R_min, phase_G_MIN, phase_B_MIN
                   

    eps (float) : A small constant added during normalization to prevent overflow 

    return : normalized data in the range [-1, 1]
    """
    assert unnormalized_data.shape[0] == 3, f"The first dimension needs to be 3"


    # Formula: normalized_data = 2 * ((data - min) / (max - min + 1e-8)) - 1    
    # -> data =  (normalized_data + 1) / 2 * (max - min + 1e-8) + min
    print(stats)
    maxs = np.array([stats["mag_R_max"], stats["mag_G_max"], stats["mag_B_max"]])
    mins = np.array([stats["mag_R_min"], stats["mag_G_min"], stats["mag_B_min"]])
    print("maxs", maxs)
    print("mins", mins)
    normalized_data = np.empty_like(unnormalized_data)
    for i in range(3):
        normalized_data[i] = 2 * (unnormalized_data[i] - mins[i])  / (maxs[i] - mins[i] + eps) - 1

    return normalized_data

def normalize_mag_and_phase(unnormalized_data : np.ndarray, stats : dict, eps : float = 1e-8):
    """
    unnormalized_data (np.ndarray) : A 6xWxH image where the first 3 rows are the 
                                    RGB magnitude and the next 3 rows are the RGB phase 
            
    stats (dict) : Statistics of the respective array, it should at least contain
                   the max and min of the data array BEFORE normalization 

                   The naming convetions for min and max for mag and phase are:
                   mag_R_max, mag_G_max, mag_B_max
                   mag_R_min, mag_G_MIN, mag_B_MIN
                   phase_R_max, phase_G_max, phase_B_max
                   phase_R_min, phase_G_MIN, phase_B_MIN
                   

    eps (float) : A small constant added during normalization to prevent overflow 

    return : normalized data in the range [-1, 1]
    """

    assert unnormalized_data.shape[0] == 6, f"The first dimension needs to be 6"

    # Formula: normalized_data = 2 * ((data - min) / (max - min + 1e-8)) - 1    
    # -> data =  (normalized_data + 1) / 2 * (max - min + 1e-8) + min
    maxs = np.array([stats["mag_R_max"], stats["mag_G_max"], stats["mag_B_max"], 
                     stats["phase_R_max"], stats["phase_G_max"], stats["phase_B_max"]])
    mins = np.array([stats["mag_R_min"], stats["mag_G_min"], stats["mag_B_min"], 
                     stats["phase_R_min"], stats["phase_G_min"], stats["phase_B_min"]])

    normalized_data = np.empty_like(unnormalized_data)

    for i in range(6):
        normalized_data[i] = 2 * (unnormalized_data[i] - mins[i])  / (maxs[i] - mins[i] + eps) - 1

    return normalized_data



def unnormalize_mag(normalized_data : np.ndarray, stats : dict, eps : float = 1e-8):
    """
    normalized_data (np.ndarray) : A 3xWxH image where the 3 rows are the RGB magnitude 
                                   These 3 rows should always be in the range [-1, 1]
            
    stats (dict) : Statistics of the respective array, it should at least contain
                   the max and min of the data array BEFORE normalization 

                   The naming convetions for min and max for mag and phase are:
                   mag_R_max, mag_G_max, mag_B_max
                   mag_R_min, mag_G_MIN, mag_B_MIN
                   phase_R_max, phase_G_max, phase_B_max
                   phase_R_min, phase_G_MIN, phase_B_MIN
                   

    eps (float) : A small constant added during normalization to prevent overflow 

    return : unnormalized data in the original range (tips: it should be around [-18, 18] for log data)
    """

    assert normalized_data.shape[0] == 3, f"The first dimension needs to be 3"

    # Formula: normalized_data = 2 * ((data - min) / (max - min + 1e-8)) - 1    
    # -> data =  (normalized_data + 1) / 2 * (max - min + 1e-8) + min
    maxs = np.array([stats["mag_R_max"], stats["mag_G_max"], stats["mag_B_max"]])
    mins = np.array([stats["mag_R_min"], stats["mag_G_min"], stats["mag_B_min"]])
    print("maxs", maxs)
    print("mins", mins)
    unnormalized_data = np.empty_like(normalized_data)

    for i in range(3):
        print("b4", np.any(np.isnan(unnormalized_data[i])))
        unnormalized_data[i] = ((normalized_data[i] + 1) / 2) * (maxs[i] - mins[i] + eps) + mins[i]
        nan_indices = np.where(np.isnan(unnormalized_data[i]))
        print(nan_indices)
        print("After", np.any(np.isnan(unnormalized_data[i])))
    return unnormalized_data

def unnormalize_mag_and_phase(normalized_data : np.ndarray, stats : dict, permute: bool = False, eps : float = 1e-8):
    """
    normalized_data (np.ndarray) : A 6xWxH image where the first 3 rows are the 
                                   RGB magnitude and the next 3 rows are the RGB phase 
                                   These 6 rows should always be in the range [-1, 1]
            
    stats (dict) : Statistics of the respective array, it should at least contain
                   the max and min of the data array BEFORE normalization 

                   The naming convetions for min and max for mag and phase are:
                   mag_R_max, mag_G_max, mag_B_max
                   mag_R_min, mag_G_MIN, mag_B_MIN
                   phase_R_max, phase_G_max, phase_B_max
                   phase_R_min, phase_G_MIN, phase_B_MIN
                   

    eps (float) : A small constant added during normalization to prevent overflow 

    return : unnormalized data
    """
    if permute:
        assert normalized_data.shape[2] == 6, f"The first dimension needs to be 6"
        normalized_data = np.transpose(normalized_data, (2, 0, 1))
    else:
        assert normalized_data.shape[0] == 6, f"The first dimension needs to be 6"


    # Formula: normalized_data = 2 * ((data - min) / (max - min + 1e-8)) - 1    
    # -> data =  (normalized_data + 1) / 2 * (max - min + 1e-8) + min
    maxs = np.array([stats["mag_R_max"], stats["mag_G_max"], stats["mag_B_max"], 
                     stats["phase_R_max"], stats["phase_G_max"], stats["phase_B_max"]])
    mins = np.array([stats["mag_R_min"], stats["mag_G_min"], stats["mag_B_min"], 
                     stats["phase_R_min"], stats["phase_G_min"], stats["phase_B_min"]])

    unnormalized_data = np.empty_like(normalized_data)

    for i in range(6):
        unnormalized_data[i] = ((normalized_data[i] + 1) / 2) * (maxs[i] - mins[i] + eps) + mins[i]

    if permute:
        unnormalized_data = np.transpose(unnormalized_data, (1, 2, 0))
    return unnormalized_data


def signed_log_scale(x, epsilon=1e-6):
    # Preserve sign but compress magnitude
    return np.sign(x) * np.log1p(np.abs(x) + epsilon)

def signed_log_inverse(x_scaled, epsilon=1e-6):
    return np.sign(x_scaled) * (np.exp(np.abs(x_scaled)) - 1 - epsilon)

# useful for ignoring outliers
def compute_robust_stats(data, lower=1, upper=99):
    lower_val = np.percentile(data, lower)
    upper_val = np.percentile(data, upper)
    return lower_val, upper_val

def robust_normalize(channel_data, lower_percentile=1, upper_percentile=99, new_min=-1, new_max=1):
    lower_val = np.percentile(channel_data, lower_percentile)
    upper_val = np.percentile(channel_data, upper_percentile)
    # Clip the values to remove extreme outliers
    channel_clipped = np.clip(channel_data, lower_val, upper_val)
    # Scale to the new range [new_min, new_max]
    normalized = (channel_clipped - lower_val) / (upper_val - lower_val + 1e-8)
    normalized = normalized * (new_max - new_min) + new_min
    return normalized
