"""
EDA and Visualization Pipeline - Quick Data Understanding
Fast templates for exploratory data analysis and visualization
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# ============================================================================
# QUICK EDA FUNCTIONS
# ============================================================================

def quick_eda_summary(df):
    """Get comprehensive data summary"""
    print("="*50)
    print("DATASET OVERVIEW")
    print("="*50)
    print(f"Shape: {df.shape}")
    print(f"\nData Types:\n{df.dtypes.value_counts()}")
    print(f"\nMemory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    print(f"\nMissing Values:\n{df.isnull().sum()[df.isnull().sum() > 0]}")
    print(f"\nDuplicate Rows: {df.duplicated().sum()}")
    print("\n" + "="*50)
    print("STATISTICAL SUMMARY")
    print("="*50)
    print(df.describe())
    
    # Categorical columns
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    if len(cat_cols) > 0:
        print("\n" + "="*50)
        print("CATEGORICAL COLUMNS")
        print("="*50)
        for col in cat_cols:
            print(f"\n{col}: {df[col].nunique()} unique values")
            print(df[col].value_counts().head())

def plot_missing_data(df, figsize=(12, 6)):
    """Visualize missing data patterns"""
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Missing data heatmap
    sns.heatmap(df.isnull(), cbar=False, yticklabels=False, cmap='viridis', ax=axes[0])
    axes[0].set_title('Missing Data Pattern')
    
    # Missing data percentage
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    missing_pct = (missing / len(df)) * 100
    missing_pct.plot(kind='barh', ax=axes[1], color='coral')
    axes[1].set_title('Missing Data Percentage')
    axes[1].set_xlabel('Percentage')
    
    plt.tight_layout()
    return fig

def plot_correlation_matrix(df, method='pearson', figsize=(10, 8)):
    """Plot correlation matrix for numerical features"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    corr = df[numeric_cols].corr(method=method)
    
    plt.figure(figsize=figsize)
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                square=True, linewidths=1, cbar_kws={"shrink": 0.8})
    plt.title(f'Correlation Matrix ({method})')
    plt.tight_layout()
    return plt.gcf()

def plot_distributions(df, figsize=(15, 10), bins=30):
    """Plot distributions of all numerical columns"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    n_cols = len(numeric_cols)
    n_rows = (n_cols + 2) // 3
    
    fig, axes = plt.subplots(n_rows, 3, figsize=figsize)
    axes = axes.flatten()
    
    for idx, col in enumerate(numeric_cols):
        axes[idx].hist(df[col].dropna(), bins=bins, edgecolor='black', alpha=0.7)
        axes[idx].set_title(f'{col}\nSkew: {df[col].skew():.2f}')
        axes[idx].set_ylabel('Frequency')
    
    # Remove empty subplots
    for idx in range(n_cols, len(axes)):
        fig.delaxes(axes[idx])
    
    plt.tight_layout()
    return fig

def plot_categorical_distributions(df, max_categories=10, figsize=(12, 8)):
    """Plot distributions of categorical columns"""
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    n_cols = len(cat_cols)
    
    if n_cols == 0:
        print("No categorical columns found")
        return None
    
    n_rows = (n_cols + 1) // 2
    fig, axes = plt.subplots(n_rows, 2, figsize=figsize)
    axes = axes.flatten() if n_cols > 1 else [axes]
    
    for idx, col in enumerate(cat_cols):
        value_counts = df[col].value_counts().head(max_categories)
        value_counts.plot(kind='barh', ax=axes[idx], color='skyblue')
        axes[idx].set_title(f'{col} (Top {max_categories})')
        axes[idx].set_xlabel('Count')
    
    # Remove empty subplots
    for idx in range(n_cols, len(axes)):
        fig.delaxes(axes[idx])
    
    plt.tight_layout()
    return fig

def plot_outliers_boxplot(df, figsize=(15, 6)):
    """Detect outliers using boxplots"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    fig, ax = plt.subplots(figsize=figsize)
    df[numeric_cols].plot(kind='box', ax=ax, vert=False)
    ax.set_title('Outlier Detection - Boxplots')
    plt.tight_layout()
    return fig

def detect_outliers_iqr(df, columns=None):
    """Detect outliers using IQR method"""
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    outlier_info = {}
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        outlier_info[col] = {
            'count': len(outliers),
            'percentage': (len(outliers) / len(df)) * 100,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound
        }
    
    return pd.DataFrame(outlier_info).T

def plot_target_analysis(df, target_col, figsize=(15, 5)):
    """Analyze target variable"""
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Distribution
    df[target_col].hist(bins=30, ax=axes[0], edgecolor='black')
    axes[0].set_title(f'{target_col} Distribution')
    axes[0].set_ylabel('Frequency')
    
    # Box plot
    df.boxplot(column=target_col, ax=axes[1])
    axes[1].set_title(f'{target_col} Boxplot')
    
    # Q-Q plot
    stats.probplot(df[target_col].dropna(), dist="norm", plot=axes[2])
    axes[2].set_title('Q-Q Plot (Normality Check)')
    
    plt.tight_layout()
    return fig

# ============================================================================
# COMPLETE EDA PIPELINE
# ============================================================================

def full_eda_pipeline(df, target_col=None, save_plots=False, output_dir='eda_output'):
    """Run complete EDA pipeline"""
    import os
    
    if save_plots and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 1. Summary statistics
    print("Running Quick EDA Summary...")
    quick_eda_summary(df)
    
    # 2. Missing data
    print("\nGenerating Missing Data Visualization...")
    fig = plot_missing_data(df)
    if save_plots:
        fig.savefig(f'{output_dir}/missing_data.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # 3. Correlation matrix
    print("\nGenerating Correlation Matrix...")
    fig = plot_correlation_matrix(df)
    if save_plots:
        fig.savefig(f'{output_dir}/correlation_matrix.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # 4. Distributions
    print("\nGenerating Distribution Plots...")
    fig = plot_distributions(df)
    if save_plots:
        fig.savefig(f'{output_dir}/distributions.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # 5. Categorical distributions
    print("\nGenerating Categorical Distributions...")
    fig = plot_categorical_distributions(df)
    if fig and save_plots:
        fig.savefig(f'{output_dir}/categorical_dist.png', dpi=150, bbox_inches='tight')
    if fig:
        plt.show()
    
    # 6. Outlier detection
    print("\nDetecting Outliers...")
    fig = plot_outliers_boxplot(df)
    if save_plots:
        fig.savefig(f'{output_dir}/outliers.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\nOutlier Statistics:")
    print(detect_outliers_iqr(df))
    
    # 7. Target analysis
    if target_col and target_col in df.columns:
        print(f"\nAnalyzing Target Variable: {target_col}...")
        fig = plot_target_analysis(df, target_col)
        if save_plots:
            fig.savefig(f'{output_dir}/target_analysis.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    print("\n" + "="*50)
    print("EDA PIPELINE COMPLETED")
    print("="*50)

# ============================================================================
# QUICK USAGE EXAMPLES
# ============================================================================

if __name__ == "__main__":
    # Example: Load data and run EDA
    # df = pd.read_csv('your_data.csv')
    # full_eda_pipeline(df, target_col='target', save_plots=True)
    
    # Or use individual functions:
    # quick_eda_summary(df)
    # plot_correlation_matrix(df)
    # plot_distributions(df)
    pass
