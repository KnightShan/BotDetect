import seaborn as sns
import matplotlib.pyplot as plt

def visualize_target_distribution(df):
    sns.countplot(x='Bot Label', data=df)
    plt.title("Bot vs Human Count")
    plt.show()

def plot_distributions(df):
    for i, col in enumerate(df.drop(columns='Bot Label').columns):
        plt.figure(i)
        sns.histplot(df[col], kde=True)
        plt.title(f'Distribution of {col}')
        plt.show()

def plot_correlation_heatmap(df):
    plt.figure(figsize=(8, 6))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
    plt.title("Feature Correlation Heatmap")
    plt.show()
