#8- boxplots

'''feature_columns = [col for col in df.columns if col != 'Outcome']

num_features = len(feature_columns)
cols = 4  # number of subplots per row
rows = (num_features + cols - 1) // cols  # ceiling division

fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
axes = axes.flatten()  # Flatten in case of multiple rows

for i, col in enumerate(feature_columns):
    sns.boxplot(data=df, y=col, ax=axes[i])
    axes[i].set_title(f'Box plot of {col}')

# Hide any unused subplots
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()'''


# 2 features with label scatter-plot

'''


# All features except the target
features = [col for col in df.columns if col != 'Outcome']

combs = list(combinations(features, 2))  # 28 combinations

# Grid layout (e.g., 4 rows × 7 cols = 28 plots)
rows, cols = 4, 7
fig, axes = plt.subplots(rows, cols, figsize=(24, 14))
axes = axes.flatten()

# Create each subplot
for i, (f1, f2) in enumerate(combs):
    sns.scatterplot(
        data=df, x=f1, y=f2, hue='Outcome',
        palette='Set1', ax=axes[i], legend=False, s=15
    )
    axes[i].set_title(f'{f1} vs {f2}', fontsize=9)
    axes[i].tick_params(labelsize=7)

# Remove unused axes if any
for j in range(i+1, len(axes)):
    fig.delaxes(axes[j])

# Add a common legend outside
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, title='Outcome', loc='upper center', ncol=2)

plt.suptitle('Scatter Plots of Feature Pairs by Outcome', fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave space for the legend
plt.show()'''



'''plt.figure(figsize=(18, 4))

for i, col in enumerate(cols_to_impute):
    plt.subplot(1, 3, i+1)
    sns.histplot(df[col], bins=30, kde=True)
    plt.title(f'Histogram of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')

plt.tight_layout()
plt.show()

plt.figure(figsize=(18, 4))

for i, col in enumerate(cols_to_impute):
    plt.subplot(1, 3, i+1)
    sns.boxplot(y=df[col])
    plt.title(f'Boxplot of {col}')
    plt.ylabel(col)

plt.tight_layout()
plt.show()'''


# Plots after all outlier and 0 handling

'''
plt.figure(figsize=(18, 4))

for i, col in enumerate(cols_to_impute_2):
    plt.subplot(1, 4, i+1)
    sns.histplot(df[col], bins=30, kde=True)
    plt.title(f'Histogram of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')

plt.tight_layout()
plt.show()

plt.figure(figsize=(18, 4))

for i, col in enumerate(cols_to_impute_2):
    plt.subplot(1, 4, i+1)
    sns.boxplot(y=df[col])
    plt.title(f'Boxplot of {col}')
    plt.ylabel(col)

plt.tight_layout()
plt.show()


feature_columns = [col for col in df.columns if col != 'Outcome']

num_features = len(feature_columns)
cols = 4  # number of subplots per row
rows = (num_features + cols - 1) // cols  # ceiling division

fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
axes = axes.flatten()  # Flatten in case of multiple rows

for i, col in enumerate(feature_columns):
    sns.boxplot(data=df, y=col, ax=axes[i])
    axes[i].set_title(f'Box plot of {col}')

# Hide any unused subplots
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()
'''

#plot of feature vs outcome

'''

features = [col for col in df.columns if col != "Outcome"]

# Set plot style
sns.set(style='whitegrid')

# Create subplots
n_cols = 4  # You can change based on how many you want per row
n_rows = -(-len(features) // n_cols)  # Ceiling division

fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
axes = axes.flatten()  # To index like a 1D array

for i, feature in enumerate(features):
    ax = axes[i]
    sns.scatterplot(data=df, x=feature, y="Outcome", hue="Outcome", ax=ax, palette='Set1', alpha=0.6)
    ax.set_title(f'{feature} vs Outcome')
    ax.set_ylabel('Outcome')
    ax.set_xlabel(feature)

# Hide any unused subplots
for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

plt.tight_layout()
plt.show()
'''

#histograms
'''

feature_columns = [col for col in df.columns if col != 'Outcome']

num_features = len(feature_columns)
cols = 4  # number of subplots per row
rows = (num_features + cols - 1) // cols  # ceiling division

fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
axes = axes.flatten()  # Flatten in case of multiple rows

for i, col in enumerate(feature_columns):
    sns.histplot(data=df, x=col, bins=30, kde=False, ax=axes[i])
    axes[i].set_title(f'Histogram of {col}')

# Hide any unused subplots
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()
'''
