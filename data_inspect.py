import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load CSV file into a DataFrame
df = pd.read_csv('./normal/normal/12.288.csv', header=None)

column_names = [
    'tachometer_signal',  # Column 1
    'underhang_bearing_axial', 'underhang_bearing_radial', 'underhang_bearing_tangential',  # Columns 2 to 4
    'overhang_bearing_axial', 'overhang_bearing_radial', 'overhang_bearing_tangential',  # Columns 5 to 7
    'microphone'  # Column 8
]

df.columns = column_names

# Display the first few rows of the DataFrame
print("First few rows of the DataFrame:")
print(df.head())

# Perform some data manipulation (e.g., calculate mean)
mean_values = df.mean()

# Display mean values
print("\nMean values:")
print(mean_values)

# Plotting the data
# For example, let's plot the tachometer signal
plt.figure(figsize=(10, 6))
plt.plot(df.index, df['tachometer_signal'], marker='o', linestyle='-')
plt.title('Plot of Tachometer Signal')
plt.xlabel('Index')
plt.ylabel('Tachometer Signal')
plt.grid(True)
plt.show()