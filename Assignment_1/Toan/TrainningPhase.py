import pandas as pd
import matplotlib.pyplot as plt
from Compute_trainning_error import compute_training_error

'''   Plot data from train.sDAT.csv and train.sNC.csv   '''

# Read the 2D grid points CSV file
df_sDAT = pd.read_csv('train.sDAT.csv', header=None, names=['X1_sDAT', 'X2_sDAT'])
df_sNC = pd.read_csv('train.sNC.csv', header=None, names=['X1_sNC', 'X2_sNC'])

fig, axes = plt.subplots(2, 2, figsize=(10, 10))  # 2 rows, 2 columns
ax1, ax2, ax3, ax4 = axes.flat  # Flatten the 2D array into 4 axes

# Plot sDAT data
ax1.scatter(df_sDAT['X1_sDAT'], df_sDAT['X2_sDAT'], marker='x', alpha=0.6, s=50, color='red', label='sDAT')
ax1.set_xlabel('X1')
ax1.set_ylabel('X2')
ax1.set_xlim(0.5, 2.5)
ax1.set_ylim(0.5, 2.5)
ax1.set_title('sDAT Data')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot sNC data
ax2.scatter(df_sNC['X1_sNC'], df_sNC['X2_sNC'], marker='o', alpha=0.6, s=50, color='blue', label='sNC')
ax2.set_xlabel('X1')
ax2.set_ylabel('X2')
ax2.set_xlim(0.5, 2.5)
ax2.set_ylim(0.5, 2.5)
ax2.set_title('sNC Data')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot sDAT data


ax3.scatter(df_sDAT['X1_sDAT'], df_sDAT['X2_sDAT'], marker='x', alpha=0.9, s=1, color='red', label='sDAT')
ax3.scatter(df_sNC['X1_sNC'], df_sNC['X2_sNC'], marker='o', alpha=0.9, s=1, color='blue', label='sNC')
ax3.set_xlabel('X1')
ax3.set_ylabel('X2')
ax3.set_xlim(0.5, 2.5)
ax3.set_ylim(0.5, 2.5)
ax3.set_title('sDAT Data and sNC Data')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Plot Training error result
k = [1, 3, 5, 10, 20, 30, 50, 100, 150, 200]
training_errors = []
for model in k:
    error, _ = compute_training_error(df_sDAT, df_sNC, model)
    training_errors.append(error)

ax4.plot(k, training_errors, marker='o', linestyle='-', color='green')
ax4.set_xlabel('k')
ax4.set_ylabel('Training Error')
ax4.set_title('Training Error vs k')
ax4.grid(True, alpha=0.3)


plt.tight_layout()
plt.show()
print("Training Phase plot displayed.")
print(training_errors)