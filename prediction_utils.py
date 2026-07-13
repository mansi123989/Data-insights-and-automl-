# prediction_utils.py
# Prediction display utilities

import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def display_prediction_result(prediction_value, label, confidence, problem_type, automl, best_model):
    """Display prediction results in a nice format"""
    
    st.markdown("### 🎯 Prediction Result")
    
    # Create columns for layout
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        if problem_type == 'classification' and label is not None:
            # Display classification result
            st.markdown(f"""
            <div class="prediction-card">
                <div class="prediction-label">Predicted Class</div>
                <div class="prediction-value">{label}</div>
                <div style="margin-top: 1rem; font-size: 1rem; color: #aaa;">
                    Encoded Value: {prediction_value:.2f}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Show confidence if available
            if confidence is not None:
                st.markdown(f"""
                <div style="text-align: center; margin-top: 1rem;">
                    <div style="background: #1e1e3f; padding: 1rem; border-radius: 10px;">
                        <span style="color: #4fc3f7;">Confidence: </span>
                        <span style="color: #00ff88; font-weight: bold; font-size: 1.2rem;">{confidence:.1f}%</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            # Show all possible classes with their probabilities if available
            if hasattr(automl.results[best_model]['model'], 'predict_proba'):
                try:
                    # Get probabilities for all classes
                    input_data = st.session_state._prediction_input
                    if input_data is not None:
                        input_scaled = automl.prepare_prediction_input(input_data)
                        proba = automl.results[best_model]['model'].predict_proba(input_scaled)
                        
                        st.markdown("#### Class Probabilities")
                        proba_df = pd.DataFrame({
                            'Class': automl.target_classes,
                            'Probability': proba[0] * 100
                        }).sort_values('Probability', ascending=False)
                        
                        # Create a bar chart
                        fig, ax = plt.subplots(figsize=(8, 4))
                        bars = ax.barh(proba_df['Class'], proba_df['Probability'], color='skyblue')
                        ax.set_xlabel('Probability (%)')
                        ax.set_title('Prediction Probabilities by Class')
                        
                        # Add value labels
                        for bar, val in zip(bars, proba_df['Probability']):
                            ax.text(val + 1, bar.get_y() + bar.get_height()/2, 
                                   f'{val:.1f}%', va='center')
                        
                        st.pyplot(fig)
                except Exception as e:
                    st.warning(f"Could not show probabilities: {str(e)}")
                    
        else:
            # Display regression result
            st.markdown(f"""
            <div class="prediction-card">
                <div class="prediction-label">Predicted Value</div>
                <div class="prediction-value">{prediction_value:.4f}</div>
            </div>
            """, unsafe_allow_html=True)
    
    # Add feature importance for this prediction (if available)
    with st.expander("🔍 Feature Contributions", expanded=False):
        if hasattr(automl.results[best_model]['model'], 'feature_importances_'):
            importances = automl.results[best_model]['model'].feature_importances_
            feature_importance_df = pd.DataFrame({
                'Feature': automl.feature_columns,
                'Importance': importances
            }).sort_values('Importance', ascending=False)
            
            st.dataframe(feature_importance_df.head(10))


# Export function
__all__ = ['display_prediction_result']