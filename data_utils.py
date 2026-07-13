# data_utils.py
# Data processing utilities

import pandas as pd
import numpy as np
import tempfile
from ydata_profiling import ProfileReport
import streamlit as st


def generate_profiling_report(df):
    """Generate profiling report using ydata-profiling"""
    try:
        profile = ProfileReport(df, title="Data Profile", explorative=True, 
                            minimal=True, progress_bar=False)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp:
            profile.to_file(tmp.name)
            with open(tmp.name, "r", encoding="utf-8") as f:
                html = f.read()
        return html
    except Exception as e:
        st.error(f"Error generating report: {str(e)}")
        return None


def get_detailed_data_summary(df):
    """Generate comprehensive data summary for AI context"""
    summary = {
        "shape": df.shape,
        "columns": df.columns.tolist(),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "missing_values": df.isnull().sum().to_dict(),
        "numeric_stats": df.describe().to_dict() if len(df.select_dtypes(include=[np.number]).columns) > 0 else {},
        "categorical_info": {}
    }
    
    # Get categorical column info
    for col in df.select_dtypes(include=['object', 'category']).columns:
        value_counts = df[col].value_counts().head(10).to_dict()
        summary["categorical_info"][col] = {
            "unique_values": df[col].nunique(),
            "top_values": value_counts,
            "missing": df[col].isnull().sum()
        }
    
    # Get correlations for numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 1:
        correlations = df[numeric_cols].corr().to_dict()
        summary["correlations"] = correlations
    
    # Get sample data
    summary["sample_data"] = df.head(10).to_dict()
    
    return summary


def highlight_best(row):
    """Highlight the best performing model in the results table"""
    if row.name == "MSE (↓)":
        best = row.min()
    else:
        best = row.max()
    
    return [
        "background-color: lightgreen" if x == best else ""
        for x in row
    ]


# Make sure functions are exported
__all__ = ['generate_profiling_report', 'get_detailed_data_summary', 'highlight_best']