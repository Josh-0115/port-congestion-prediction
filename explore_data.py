
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/synthetic_ports.csv")

print("Dataset shape:", df.shape)
print("\nColumn names:")
print(df.columns.tolist())

print("\nData types:")
print(df.dtypes)

print("\nMissing values:")
print(df.isnull().sum())

print("\nDescriptive statistics:")
print(df.describe())

print("\nLabel distribution:")
print(df["congestion_label"].value_counts())

print("\nCongestion rate:")
print(f"{df['congestion_label'].mean():.1%}")

# Visualize the two target classes
df["congestion_label"].value_counts().sort_index().plot(
    kind="bar"
)
plt.title("Distribution of Congestion Labels")
plt.xlabel("Congestion label (0 = no significant delay, 1 = delay)")
plt.ylabel("Number of records")
plt.tight_layout()
plt.savefig("data/label_distribution.png")
plt.show()
