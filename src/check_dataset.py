import os
import pandas as pd

CSV_PATH = "dataset/train.csv"
IMAGE_DIR = "dataset/gaussian_filtered_images/gaussian_filtered_images"

df = pd.read_csv(CSV_PATH)

print("First 5 rows:")
print(df.head())

print("\nCSV columns:")
print(df.columns)

print("\nLabel distribution:")
print(df["diagnosis"].value_counts().sort_index())

print("\nFolders inside image directory:")
print(os.listdir(IMAGE_DIR))