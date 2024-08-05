
import matplotlib.pyplot as plt
import seaborn as sns

def plot_histograms(df, num_cols):
    df[num_cols].hist(figsize=(14, 14))
    plt.show()

def plot_correlation_heatmap(df, num_cols):
    plt.figure(figsize=(15, 8))
    sns.heatmap(df[num_cols].corr(), annot=True, fmt='0.2f', cmap='YlGnBu')
    plt.show()