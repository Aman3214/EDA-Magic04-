import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

dataset_path = "magic04.data"
sns.set(style="whitegrid")
column_names = ["fLength", "fWidth", "fSize", "fConc", "fConc1", "fAsym", "fM3Long", "fM3Trans","fAlpha", "fDist", "class"]
df_data = pd.read_csv(dataset_path, header = None,names = column_names)

# checking the enteries of the dataset
 
# displaying the first 5 rows
print("\nFirst 5 rows of the dataset")
print(df_data.head())

#checking data details
print("\nData details")
print(df_data.info())

# checking for missing values
print("\nMissing values")
print(df_data.isnull().sum())

#data description
print("\nData description")
print(df_data.describe())

# distribution of target values
'''observation is that more of the data is in the gamma class'''
print("\nDistibution of target values")
print(df_data['class'].value_counts())
plt.figure(figsize=(6, 4))
sns.countplot(x='class', data=df_data, palette='viridis')
plt.title('Distribution of Target Classes (g=gamma, h=hadron)')
plt.xlabel('Class')
plt.ylabel('Count')
plt.show()

# Univariate analysis
print("\nUnivariate Analysis (Feature Distributions)")


# Plot density plots for potentially skewed features
'''most of the features are positively skewed but some like fAsym and fDist are symmetric, fM3Trans is bimodal and symmetric'''
features = df_data.columns[:-1]
plt.figure(figsize=(15, 10))
for i, col in enumerate(features):
    plt.subplot(4, 3, i + 1)
    sns.kdeplot(df_data[col], fill=True)
    plt.title(f'Density Plot of {col}')
    plt.xlabel('')
    plt.ylabel('')
plt.suptitle('Density Plots of Numerical Features', y=1.02)
plt.tight_layout(rect=[0, 0, 1, 1])
plt.show()

# Plot box plots to check for outliers and spread
'''except fConc and fAlpha other features have a lot of outliers'''
plt.figure(figsize=(15, 10))
for i, col in enumerate(features):
    plt.subplot(4, 3, i + 1)
    sns.boxplot(y=df_data[col])
    plt.title(f'Box Plot of {col}')
    plt.ylabel('')
plt.suptitle('Box Plots of Numerical Features', y=1.02)
plt.tight_layout(rect=[0, 0, 1, 1])
plt.show()

print("\nBivariate Analysis (Feature vs. Feature)")

# Calculate the correlation matrix
correlation_matrix = df_data[features].corr()

# Plot the heatmap
'''shows some highly correlated features'''
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Correlation Matrix of Numerical Features')
plt.show()

# Example: Scatter plot for two potentially interesting features
plt.figure(figsize=(8, 6))
sns.scatterplot(x='fLength', y='fWidth', data=df_data, hue='class', alpha=0.6, palette='viridis')
plt.title('fLength vs fWidth (colored by class)')
plt.show()

# plotting box plots for feature distributions in each class
'''here we can see whaich class has more outliers'''
plt.figure(figsize=(18,12))
for i, col in enumerate(features):
    plt.subplot(4, 3, i + 1)
    sns.boxplot(x='class', y=df_data[col], data=df_data)
    plt.title(f'Box Plot of {col} by Class')
    plt.xlabel('class g=gamma, h=hadron')
    plt.ylabel(col)
plt.suptitle('Box Plots of Numerical Features by Class', y=1.02)
plt.tight_layout(rect=[0, 0, 1, 1])
plt.show()

print("\nBivariate Analysis (Feature vs. Target)")

# Box plots comparing feature distributions for each class
plt.figure(figsize=(18, 12))
for i, col in enumerate(features):
    plt.subplot(4, 3, i + 1)
    sns.boxplot(x='class', y=col, data=df_data, palette='viridis')
    plt.title(f'{col} distribution by Class')
    plt.xlabel('Class (g=gamma, h=hadron)')
    plt.ylabel(col)
plt.suptitle('Feature Distributions Grouped by Target Class (Box Plots)', y=1.02)
plt.tight_layout(rect=[0, 0, 1, 1])
plt.show()

# Density plots comparing feature distributions for each class
plt.figure(figsize=(18, 12))
for i, col in enumerate(features):
    plt.subplot(4, 3, i + 1)
    sns.kdeplot(data=df_data, x=col, hue='class', fill=True, common_norm=False, palette='viridis')
    plt.title(f'{col} density by Class')
    plt.xlabel(col)
    plt.ylabel('Density')
plt.suptitle('Feature Distributions Grouped by Target Class (Density Plots)', y=1.02)
plt.tight_layout(rect=[0, 0, 1, 1])
plt.show()

# pair plots of all features
plt.figure(figsize=(18, 12))
sns.pairplot(df_data, hue='class', palette='viridis',diag_kind='kde',corner=True)
plt.title('Pair Plot of All Features')
plt.show()