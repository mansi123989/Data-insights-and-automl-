# visualization_utils.py
# Visualization utilities

import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix


def plot_target_distribution(data, target_col):
    """Plot target distribution"""
    fig, ax = plt.subplots(figsize=(8, 4))
    if data[target_col].dtype in ['object', 'category'] or data[target_col].nunique() <= 10:
        data[target_col].value_counts().plot(kind='bar', ax=ax)
        ax.set_title(f'Distribution of {target_col}')
        ax.set_xlabel(target_col)
        ax.set_ylabel('Count')
        plt.xticks(rotation=45)
    else:
        data[target_col].hist(bins=30, ax=ax)
        ax.set_title(f'Distribution of {target_col}')
        ax.set_xlabel(target_col)
        ax.set_ylabel('Frequency')
    plt.tight_layout()
    return fig


def plot_model_performance(results, problem_type, best_model):
    """Plot model performance comparison"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    if problem_type == 'regression':
        # MSE comparison
        mse_values = [results[name]['MSE'] for name in results.keys()]
        bars = axes[0].barh(list(results.keys()), mse_values, color='skyblue')
        axes[0].set_title('MSE Comparison (Lower is better)', fontsize=12, fontweight='bold')
        axes[0].set_xlabel('Mean Squared Error')
        # Add value labels
        for bar, val in zip(bars, mse_values):
            axes[0].text(val + max(mse_values)*0.01, bar.get_y() + bar.get_height()/2, 
                        f'{val:.2f}', va='center')
        
        # R2 comparison
        r2_values = [results[name]['R2'] for name in results.keys()]
        bars = axes[1].barh(list(results.keys()), r2_values, color='lightgreen')
        axes[1].set_title('R² Score Comparison (Higher is better)', fontsize=12, fontweight='bold')
        axes[1].set_xlabel('R² Score')
        # Add value labels
        for bar, val in zip(bars, r2_values):
            axes[1].text(val + 0.01, bar.get_y() + bar.get_height()/2, 
                        f'{val:.3f}', va='center')
    else:
        # Accuracy comparison
        acc_values = [results[name]['Accuracy'] for name in results.keys()]
        colors = ['#ff6b6b' if name == best_model else '#4ecdc4' for name in results.keys()]
        bars = axes[0].barh(list(results.keys()), acc_values, color=colors)
        axes[0].set_title('Accuracy Comparison (Higher is better)', fontsize=12, fontweight='bold')
        axes[0].set_xlabel('Accuracy')
        # Add value labels
        for bar, val in zip(bars, acc_values):
            axes[0].text(val + 0.01, bar.get_y() + bar.get_height()/2, 
                        f'{val:.3f}', va='center')
        
        # Confusion matrix for best model
        y_pred = results[best_model]['predictions']
        y_test = st.session_state.automl.y_test
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', ax=axes[1], cmap='Blues', 
                   xticklabels=st.session_state.automl.target_classes, 
                   yticklabels=st.session_state.automl.target_classes)
        axes[1].set_title(f'Confusion Matrix - {best_model}', fontsize=12, fontweight='bold')
        axes[1].set_xlabel('Predicted')
        axes[1].set_ylabel('Actual')
    
    plt.tight_layout()
    return fig


def plot_feature_importance(feature_columns, importances):
    """Plot feature importance"""
    feature_importance_df = pd.DataFrame({
        'Feature': feature_columns,
        'Importance': importances
    }).sort_values('Importance', ascending=False)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=feature_importance_df.head(10), x='Importance', y='Feature', 
               palette='viridis')
    ax.set_title('Top 10 Feature Importances', fontsize=14, fontweight='bold')
    ax.set_xlabel('Importance Score')
    plt.tight_layout()
    return fig, feature_importance_df


def plot_boxplots(data, numeric_cols):
    """Plot boxplots for numeric columns"""
    fig, axes = plt.subplots(1, min(3, len(numeric_cols)), figsize=(12, 4))
    if len(numeric_cols) == 1:
        axes = [axes]
    for idx, col in enumerate(numeric_cols[:3]):
        sns.boxplot(data=data[col], ax=axes[idx])
        axes[idx].set_title(f'{col}')
        axes[idx].set_xlabel('')
    plt.tight_layout()
    return fig


# Export functions
__all__ = [
    'plot_target_distribution',
    'plot_model_performance',
    'plot_feature_importance',
    'plot_boxplots'
]