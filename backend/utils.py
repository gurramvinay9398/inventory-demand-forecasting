import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv('../data/sales.csv')

print(df.head())
print(df.info())
print(df.describe())


df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date')

plt.plot(df['Date'], df['Total'])
plt.title("Sales Over Time")
plt.show()