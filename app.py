# app.py
# Main Streamlit Application - Entry Point

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import all modules
from config import CUSTOM_CSS
from automl_core import AutoML
from data_utils import generate_profiling_report, highlight_best
from visualization_utils import (
    plot_target_distribution, 
    plot_model_performance, 
    plot_feature_importance,
    plot_boxplots
)
from ai_assistant import ask_groq_question
from prediction_utils import display_prediction_result

# Page configuration
st.set_page_config(page_title="AutoML Data Science Platform", layout="wide")

# Apply custom CSS
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'target' not in st.session_state:
    st.session_state.target = None
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'best_model' not in st.session_state:
    st.session_state.best_model = None
if 'scaler' not in st.session_state:
    st.session_state.scaler = None
if 'feature_columns' not in st.session_state:
    st.session_state.feature_columns = None
if 'automl' not in st.session_state:
    st.session_state.automl = None
if 'categorical_mappings' not in st.session_state:
    st.session_state.categorical_mappings = {}
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# ... rest of the code continues


def main():
    st.markdown('<div class="main-header">🚀 AutoML Data Science Platform</div>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 📊 Data Upload")
        uploaded_file = st.file_uploader("Upload CSV or Excel file", type=['csv', 'xlsx', 'xls'])
        
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith('.csv'):
                    data = pd.read_csv(uploaded_file)
                else:
                    data = pd.read_excel(uploaded_file)
                
                st.session_state.data = data
                st.success(f"✅ Data loaded! {data.shape[0]} rows, {data.shape[1]} columns")
                
                # Display basic info
                st.markdown("### 📋 Data Info")
                st.write(f"**Rows:** {data.shape[0]}")
                st.write(f"**Columns:** {data.shape[1]}")
                st.write(f"**Missing Values:** {data.isnull().sum().sum()}")
                
                # Display data types
                st.markdown("### 📝 Data Types")
                dtype_counts = data.dtypes.value_counts()
                for dtype, count in dtype_counts.items():
                    st.write(f"- {dtype}: {count}")
                
                
                
                
                
                
            except Exception as e:
                st.error(f"Error loading file: {str(e)}")
    
    # Main content
    if st.session_state.data is not None:
        data = st.session_state.data
        
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Data Profiling", "🔧 Data Preprocessing", "🎯 Model Training", "📈 Results", "🔮 Predict"])
        
        with tab1:
            st.markdown("### 📊 Data Profiling Report")
            st.markdown("*Comprehensive analysis including statistics, correlations, missing values, and more*")
            
            if st.button("🔄 Generate Profiling Report", use_container_width=True):
                with st.spinner("Generating comprehensive data profiling report..."):
                    html = generate_profiling_report(data)
                    if html:
                        components.html(html, height=1200, scrolling=True)
                        st.success("✅ Report generated successfully!")
            
            # Show quick stats without generating full report
            with st.expander("📈 Quick Statistics", expanded=False):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Rows", data.shape[0])
                with col2:
                    st.metric("Total Columns", data.shape[1])
                with col3:
                    st.metric("Memory Usage", f"{data.memory_usage().sum() / 1024**2:.2f} MB")
                
                st.write("### Numeric Columns Statistics")
                numeric_cols = data.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    st.dataframe(data[numeric_cols].describe())
                
                st.write("### Categorical Columns Statistics")
                categorical_cols = data.select_dtypes(include=['object', 'category']).columns
                if len(categorical_cols) > 0:
                    for col in categorical_cols[:3]:  # Show first 3
                        st.write(f"**{col}**")
                        st.write(data[col].value_counts().head())
        
        with tab2:
            st.markdown("### 🔧 Data Preprocessing")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Missing Values")
                missing_data = data.isnull().sum()
                missing_data = missing_data[missing_data > 0]
                if len(missing_data) > 0:
                    st.warning(f"⚠️ Found {len(missing_data)} columns with missing values")
                    st.dataframe(pd.DataFrame({
                        'Column': missing_data.index,
                        'Missing Count': missing_data.values,
                        'Missing %': (missing_data.values / len(data) * 100).round(2)
                    }))
                    
                    if st.button("Remove Missing Values", use_container_width=True):
                        data = data.dropna()
                        st.session_state.data = data
                        st.success("✅ Missing values removed!")
                        st.rerun()
                else:
                    st.success("✅ No missing values found!")
            
            with col2:
                st.markdown("#### Outlier Detection")
                numeric_cols = data.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    # Show outliers count
                    outliers_count = {}
                    for col in numeric_cols[:5]:  # Check first 5 numeric columns
                        Q1 = data[col].quantile(0.25)
                        Q3 = data[col].quantile(0.75)
                        IQR = Q3 - Q1
                        lower_bound = Q1 - 1.5 * IQR
                        upper_bound = Q3 + 1.5 * IQR
                        outliers = ((data[col] < lower_bound) | (data[col] > upper_bound)).sum()
                        outliers_count[col] = outliers
                    
                    if any(outliers_count.values()):
                        st.warning("⚠️ Outliers detected in numeric columns")
                        st.dataframe(pd.DataFrame({
                            'Column': list(outliers_count.keys()),
                            'Outliers Count': list(outliers_count.values())
                        }))
                    else:
                        st.success("✅ No significant outliers detected!")
                    
                    # Plot boxplots
                    if len(numeric_cols) > 0:
                        st.markdown("#### Boxplot Visualization")
                        fig = plot_boxplots(data, numeric_cols)
                        st.pyplot(fig)
                    
                    if st.button("Remove Outliers", use_container_width=True):
                        for col in numeric_cols:
                            Q1 = data[col].quantile(0.25)
                            Q3 = data[col].quantile(0.75)
                            IQR = Q3 - Q1
                            lower_bound = Q1 - 1.5 * IQR
                            upper_bound = Q3 + 1.5 * IQR
                            data[col] = data[col].clip(lower_bound, upper_bound)
                        st.session_state.data = data
                        st.success("✅ Outliers handled!")
                        st.rerun()
        
        with tab3:
            st.markdown("### 🎯 Model Training")
            
            if st.session_state.data is not None:
                # Select target column
                target_col = st.selectbox("Select Target Column to Predict", data.columns.tolist())
                st.session_state.target = target_col
                
                # Show target distribution
                st.markdown("#### Target Distribution")
                fig = plot_target_distribution(data, target_col)
                st.pyplot(fig)
                
                # Feature selection options
                st.markdown("#### Feature Selection")
                num_features = st.slider("Number of top features to select", 3, min(20, len(data.columns)-1), min(10, len(data.columns)-1))
                
                if st.button("🚀 Train Models", use_container_width=True):
                    with st.spinner("Training models..."):
                        try:
                            # Initialize AutoML
                            automl = AutoML(data, target_col)
                            
                            # Detect problem type
                            problem_type = automl.detect_problem_type()
                            st.info(f"📌 Problem Type: **{problem_type.upper()}**")
                            
                            # Preprocess
                            X, y = automl.preprocess_data()
                            st.success("✅ Data preprocessing complete!")
                            
                            # Feature selection
                            st.write(f"📊 Selecting top {num_features} features based on correlation...")
                            X = automl.feature_selection(k=num_features)
                            st.success(f"✅ Selected {len(automl.feature_columns)} features")
                            
                            # Show selected features
                            with st.expander("📋 Selected Features"):
                                st.write("**Numeric Features:**")
                                numeric_selected = [f for f in automl.feature_columns if f in automl.numeric_columns]
                                if numeric_selected:
                                    st.write(numeric_selected)
                                else:
                                    st.write("None")
                                
                                st.write("**Categorical Features:**")
                                categorical_selected = [f for f in automl.feature_columns if f in automl.categorical_columns]
                                if categorical_selected:
                                    st.write(categorical_selected)
                                    # Show category mappings
                                    for col in categorical_selected:
                                        if col in automl.original_categories:
                                            st.write(f"  - {col}: {automl.original_categories[col]}")
                                else:
                                    st.write("None")
                            
                            # Train models
                            results, best_model = automl.train_models()
                            
                            if results:
                                # Store in session
                                st.session_state.automl = automl
                                st.session_state.results = results
                                st.session_state.best_model = best_model
                                st.session_state.problem_type = problem_type
                                st.session_state.model_trained = True
                                
                                st.success(f"✅ Models trained successfully! 🏆 Best model: **{best_model}**")
                            else:
                                st.error("❌ No models trained successfully. Please check your data.")
                                
                        except Exception as e:
                            st.error(f"Error during training: {str(e)}")
                            st.info("💡 Tip: Make sure your target column is appropriate and data is clean.")
        
        with tab4:
            st.markdown("### 📈 Model Performance")
            
            if st.session_state.model_trained and st.session_state.automl is not None:
                results = st.session_state.results
                problem_type = st.session_state.problem_type
                best_model = st.session_state.best_model
                automl = st.session_state.automl
                
                # Results table
                st.markdown("#### Model Comparison")
                results_df = pd.DataFrame()
                for name, metrics in results.items():
                    if problem_type == 'regression':
                        results_df[name] = [metrics['MSE'], metrics['R2']]
                    else:
                        results_df[name] = [metrics['Accuracy']]
                
                if problem_type == 'regression':
                    results_df.index = ['MSE (↓)', 'R² Score (↑)']
                else:
                    results_df.index = ['Accuracy (↑)']
                
                # Apply highlight function
                styled_df = results_df.style.apply(highlight_best, axis=1)
                st.dataframe(styled_df)
                
                # Plot results
                st.markdown("#### Performance Visualization")
                fig = plot_model_performance(results, problem_type, best_model)
                st.pyplot(fig)
                
                # Feature Importance
                st.markdown("### 🔍 Feature Importance Analysis")
                if hasattr(automl.results[best_model]['model'], 'feature_importances_'):
                    importances = automl.results[best_model]['model'].feature_importances_
                    fig, feature_importance_df = plot_feature_importance(automl.feature_columns, importances)
                    st.pyplot(fig)
                    st.dataframe(feature_importance_df)
                else:
                    st.info("ℹ️ Feature importance not available for the best model.")
                
                # Model details
                with st.expander("📝 Model Details"):
                    st.write(f"**Best Model:** {best_model}")
                    st.write(f"**Problem Type:** {problem_type.upper()}")
                    st.write(f"**Number of Features:** {len(automl.feature_columns)}")
                    st.write(f"**Categorical Columns:** {automl.categorical_columns}")
                    st.write(f"**Numeric Columns:** {automl.numeric_columns}")
                    if problem_type == 'regression':
                        st.write(f"**Best MSE:** {results[best_model]['MSE']:.4f}")
                        st.write(f"**Best R² Score:** {results[best_model]['R2']:.4f}")
                    else:
                        st.write(f"**Best Accuracy:** {results[best_model]['Accuracy']:.4f}")
                        st.write(f"**Target Classes:** {automl.target_classes}")
            else:
                st.info("ℹ️ Please train models first in the 'Model Training' tab.")
        
        with tab5:
            st.markdown("### 🔮 Make Predictions")
            
            if st.session_state.model_trained and st.session_state.automl is not None:
                automl = st.session_state.automl
                
                st.success(f"✅ Using best model: **{st.session_state.best_model}**")
                st.info(f"📌 Problem Type: **{st.session_state.problem_type.upper()}**")
                
                if st.session_state.problem_type == 'classification':
                    st.info(f"🎯 Target Classes: {', '.join(automl.target_classes)}")
                
                st.markdown("#### Enter values for prediction")
                st.markdown("*Enter values for all features to get a prediction*")
                
                # Create input fields with appropriate types
                input_values = {}
                cols = st.columns(3)
                
                for idx, feature in enumerate(automl.feature_columns):
                    with cols[idx % 3]:
                        # Check if this is a categorical feature
                        if feature in automl.categorical_columns:
                            # Get the original categories
                            categories = automl.original_categories.get(feature, [])
                            if categories:
                                # Create dropdown for categorical input
                                input_values[feature] = st.selectbox(
                                    f"{feature} (Categorical)",
                                    options=categories,
                                    key=f"input_{feature}"
                                )
                            else:
                                input_values[feature] = st.text_input(
                                    f"{feature} (Categorical)",
                                    value="",
                                    key=f"input_{feature}"
                                )
                        else:
                            # Numeric input with default from training data
                            default_val = 0.0
                            if feature in automl.numeric_columns:
                                # Get median from training data for better default
                                if feature in automl.X.columns:
                                    default_val = automl.X[feature].median()
                            
                            input_values[feature] = st.number_input(
                                f"{feature} (Numeric)", 
                                value=float(default_val),
                                format="%.2f",
                                key=f"input_{feature}"
                            )
                
                if st.button("🔮 Predict", use_container_width=True):
                    try:
                        # Store input for later use in probability display
                        st.session_state._prediction_input = input_values
                        
                        # Show what's being input
                        with st.expander("📋 Input Values", expanded=False):
                            st.write(pd.DataFrame([input_values]))
                        
                        # Get prediction with label
                        prediction_value, label, confidence = automl.predict_with_label(input_values)
                        
                        if prediction_value is not None:
                            # Display the prediction
                            display_prediction_result(
                                prediction_value, 
                                label, 
                                confidence, 
                                st.session_state.problem_type,
                                automl,
                                st.session_state.best_model
                            )
                        else:
                            st.error("Failed to make prediction. Please check your input values.")
                            
                    except Exception as e:
                        st.error(f"Error making prediction: {str(e)}")
                        st.info("💡 Make sure all input values are valid.")
            else:
                st.info("ℹ️ Please train models first in the 'Model Training' tab.")
    
    else:
        # Welcome message when no data
        st.markdown("""
        ### 👋 Welcome to AutoML Data Science Platform!
        
        **Get started by uploading a dataset in the sidebar.**
        
        #### What you can do:
        - 📊 **Upload** CSV or Excel files
        - 📈 **Profile** your data automatically
        - 🔧 **Clean** data with automatic preprocessing
        - 🎯 **Train** multiple ML models automatically
        - 📊 **Compare** model performance
        - 🔮 **Make** predictions with the best model
        - 🤖 **Query** your data using AI (Groq API)
        
        #### Supported features:
        - ✅ Automatic handling of missing values
        - ✅ Outlier detection and removal
        - ✅ Feature selection based on correlation
        - ✅ Automatic encoding of categorical variables
        - ✅ Multiple ML models for classification & regression
        - ✅ Model performance comparison
        - ✅ Feature importance analysis
        - ✅ Categorical input support in predictions
        - ✅ AI-powered data insights
        - ✅ Prediction with class labels and confidence
        """)
        
        # Example data option
        st.markdown("---")
        st.markdown("### 📂 Try with Sample Data")
        
        sample_options = {
            "Iris (Classification)": "iris",
            "California Housing (Regression)": "housing"
        }
        
        selected_sample = st.selectbox("Choose a sample dataset", list(sample_options.keys()))
        
        if st.button("Load Sample Dataset", use_container_width=True):
            try:
                if sample_options[selected_sample] == "iris":
                    from sklearn.datasets import load_iris
                    iris = load_iris()
                    df = pd.DataFrame(iris.data, columns=iris.feature_names)
                    df['target'] = iris.target
                    df['target'] = df['target'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})
                    st.session_state.data = df
                    st.success("✅ Iris dataset loaded! (Classification)")
                else:
                    from sklearn.datasets import fetch_california_housing
                    housing = fetch_california_housing()
                    df = pd.DataFrame(housing.data, columns=housing.feature_names)
                    df['target'] = housing.target
                    st.session_state.data = df
                    st.success("✅ California Housing dataset loaded! (Regression)")
                st.rerun()
            except Exception as e:
                st.error(f"Error loading sample: {str(e)}")


if __name__ == "__main__":
    main()