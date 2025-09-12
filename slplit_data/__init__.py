import splitfolders


# Chia dữ liệu: 70% train, 15% val, 15% test
splitfolders.ratio("dataset", output="dataset_split", seed=42, ratio=(0.7, 0.15, 0.15))
