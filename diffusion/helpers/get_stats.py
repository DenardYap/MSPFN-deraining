import csv
def get_stats(stats_csv_file):
    column_names = [
        "mag_R_max", "mag_G_max", "mag_B_max",
        "phase_R_max", "phase_G_max", "phase_B_max",
        "mag_R_min", "mag_G_min", "mag_B_min",
        "phase_R_min", "phase_G_min", "phase_B_min",
        "mag_R_mean", "mag_G_mean", "mag_B_mean",
        "phase_R_mean", "phase_G_mean", "phase_B_mean",
        "mag_R_std", "mag_G_std", "mag_B_std",
        "phase_R_std", "phase_G_std", "phase_B_std"
    ]

    with open(stats_csv_file, mode='r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header row
        row = next(reader)  # Read the only data row
        stats = {key: float(value) for key, value in zip(column_names, row)}

    return stats

def get_stats_YCrCb(stats_csv_file):
    column_names = [
        "mag_Y_max", "mag_Cr_max", "mag_Cb_max",
        "phase_Y_max", "phase_Cr_max", "phase_Cb_max",
        "mag_Y_min", "mag_Cr_min", "mag_Cb_min",
        "phase_Y_min", "phase_Cr_min", "phase_Cb_min",
        "mag_Y_mean", "mag_Cr_mean", "mag_Cb_mean",
        "phase_Y_mean", "phase_Cr_mean", "phase_Cb_mean",
        "mag_Y_std", "mag_Cr_std", "mag_Cb_std",
        "phase_Y_std", "phase_Cr_std", "phase_Cb_std"
    ]

    with open(stats_csv_file, mode='r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header row
        row = next(reader)  # Read the only data row
        stats = {key: float(value) for key, value in zip(column_names, row)}

    return stats
