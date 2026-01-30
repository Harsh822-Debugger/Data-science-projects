import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Open the file
data = pd.read_csv("Unemployment in India.csv")

# Step 2: Clean column names (remove extra spaces)
data.columns = data.columns.str.strip()

# Step 3: Look at column names
print(data.columns)

# Step 4: Make Date readable
data['Date'] = pd.to_datetime(data['Date'])

# Step 5: Average unemployment by date
average = data.groupby('Date')['Estimated Unemployment Rate (%)'].mean()

# Step 6: Draw unemployment graph
plt.plot(average)
plt.title("Unemployment Rate in India")
plt.xlabel("Year")
plt.ylabel("Unemployment (%)")
plt.show()

# Step 7: Covid year (2020)
covid = data[data['Date'].dt.year == 2020]

sns.lineplot(x='Date', y='Estimated Unemployment Rate (%)', data=covid)
plt.title("Unemployment During Covid-19")
plt.show()
