import pandas as pd

df = pd.read_csv('../data/train.csv')

print("Store range:", df['store'].min(), "to", df['store'].max())
print("Item range:", df['item'].min(), "to", df['item'].max())