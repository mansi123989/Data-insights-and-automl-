# automl_core.py
# Core AutoML class implementation

import pandas as pd
import numpy as np
import streamlit as st  # ADD THIS IMPORT
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, confusion_matrix
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
import warnings
warnings.filterwarnings('ignore')


class AutoML:
    def __init__(self, data, target_column):
        self.data = data
        self.target_column = target_column
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.problem_type = None
        self.models = {}
        self.results = {}
        self.best_model = None
        self.scaler = StandardScaler()
        self.feature_columns = None
        self.label_encoders = {}
        self.categorical_columns = []
        self.numeric_columns = []
        self.original_categories = {}
        self.target_encoder = None
        self.target_classes = None
        
  
    def detect_problem_type(self):
        target = self.data[self.target_column]

        if target.dtype == object or str(target.dtype) == "category":
            self.problem_type = "classification"

        elif target.nunique() <= 20:
            self.problem_type = "classification"

        else:
            self.problem_type = "regression"

        return self.problem_type
    
    def preprocess_data(self):
        """Handle missing values and outliers with proper categorical handling"""
        # Separate features and target
        self.X = self.data.drop(columns=[self.target_column])
        self.y = self.data[self.target_column]
        
        # Encode target if it's categorical
        if self.problem_type == 'classification':
            self.target_encoder = LabelEncoder()
            self.y = self.target_encoder.fit_transform(self.y)
            self.target_classes = self.target_encoder.classes_
        
        # Identify column types
        self.numeric_columns = self.X.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_columns = self.X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Handle missing values
        if len(self.numeric_columns) > 0:
            num_imputer = SimpleImputer(strategy='median')
            self.X[self.numeric_columns] = num_imputer.fit_transform(self.X[self.numeric_columns])
        
        if len(self.categorical_columns) > 0:
            cat_imputer = SimpleImputer(strategy='most_frequent')
            self.X[self.categorical_columns] = cat_imputer.fit_transform(self.X[self.categorical_columns])
        
        # Handle outliers using IQR method (only for numeric)
        for col in self.numeric_columns:
            Q1 = self.X[col].quantile(0.25)
            Q3 = self.X[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            self.X[col] = self.X[col].clip(lower_bound, upper_bound)
        
        # Encode categorical variables and store mappings
        if len(self.categorical_columns) > 0:
            for col in self.categorical_columns:
                le = LabelEncoder()
                self.X[col] = le.fit_transform(self.X[col].astype(str))
                self.label_encoders[col] = le
                # Store original categories for prediction
                self.original_categories[col] = le.classes_.tolist()
        
        # Combine all features
        self.feature_columns = self.X.columns.tolist()
        
        return self.X, self.y
    
    def feature_selection(self, k=10):
        """Select top k features based on correlation with target"""
        k = min(k, len(self.feature_columns))
        if k == 0:
            return self.X
            
        if self.problem_type == 'regression':
            selector = SelectKBest(score_func=f_regression, k=k)
        else:
            selector = SelectKBest(score_func=f_classif, k=k)
        
        X_selected = selector.fit_transform(self.X, self.y)
        selected_indices = selector.get_support(indices=True)
        self.feature_columns = [self.feature_columns[i] for i in selected_indices]
        self.X = pd.DataFrame(X_selected, columns=self.feature_columns)
        
        return self.X
    
    def train_models(self):
        """Train multiple models based on problem type"""
        # Split data
        stratify_param = self.y if self.problem_type == 'classification' else None
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42, stratify=stratify_param
        )
        
        # Scale features
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)
        
        if self.problem_type == 'regression':
            models = {
                'Linear Regression': LinearRegression(),
                'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
                'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
                'Decision Tree': DecisionTreeRegressor(random_state=42)
            }
        else:
            models = {
                'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000, multi_class='auto'),
                'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
                'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
                'Decision Tree': DecisionTreeClassifier(random_state=42)
            }
        
        self.models = models
        self.results = {}
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for idx, (name, model) in enumerate(models.items()):
            status_text.text(f"Training {name}...")
            try:
                model.fit(self.X_train, self.y_train)
                y_pred = model.predict(self.X_test)
                
                if self.problem_type == 'regression':
                    mse = mean_squared_error(self.y_test, y_pred)
                    r2 = r2_score(self.y_test, y_pred)
                    self.results[name] = {'MSE': mse, 'R2': r2, 'model': model, 'predictions': y_pred}
                else:
                    accuracy = accuracy_score(self.y_test, y_pred)
                    self.results[name] = {'Accuracy': accuracy, 'model': model, 'predictions': y_pred}
            except Exception as e:
                st.warning(f"⚠️ Model {name} failed: {str(e)}")
            
            progress_bar.progress((idx + 1) / len(models))
        
        status_text.empty()
        progress_bar.empty()
        
        # Find best model
        if self.results:
            if self.problem_type == 'regression':
                self.best_model = min(self.results.keys(), key=lambda x: self.results[x]['MSE'])
            else:
                self.best_model = max(self.results.keys(), key=lambda x: self.results[x]['Accuracy'])
        
        return self.results, self.best_model
    
    def prepare_prediction_input(self, input_data):
        """Prepare input data for prediction with proper categorical encoding"""
        # Create a DataFrame with the input data
        input_df = pd.DataFrame([input_data])
        
        # Ensure we only keep the features used in training
        input_df = input_df[self.feature_columns]
        
        # Encode categorical columns if they exist
        for col in self.categorical_columns:
            if col in input_df.columns:
                if col in self.label_encoders:
                    try:
                        # Transform using the fitted encoder
                        input_df[col] = self.label_encoders[col].transform(input_df[col].astype(str))
                    except ValueError as e:
                        # Handle unseen categories
                        st.warning(f"⚠️ Unseen category in {col}. Using most frequent category.")
                        # Use the first category as default
                        default_value = self.label_encoders[col].transform([self.label_encoders[col].classes_[0]])[0]
                        input_df[col] = default_value
        
        # Scale the input
        input_scaled = self.scaler.transform(input_df)
        
        return input_scaled
    
    def predict_with_label(self, input_data):
        """Make prediction and return both value and label if applicable"""
        if self.best_model is None or self.best_model not in self.results:
            return None, None
        
        # Prepare input
        input_scaled = self.prepare_prediction_input(input_data)
        
        # Make prediction
        prediction = self.results[self.best_model]['model'].predict(input_scaled)
        prediction_value = prediction[0] if isinstance(prediction, np.ndarray) else prediction
        
        # Get label for classification
        label = None
        if self.problem_type == 'classification' and self.target_encoder is not None:
            try:
                # Round prediction for classification
                pred_class = int(round(prediction_value))
                # Ensure it's within bounds
                if 0 <= pred_class < len(self.target_classes):
                    label = self.target_encoder.inverse_transform([pred_class])[0]
            except:
                pass
        
        # Get probability/confidence for classification
        confidence = None
        if self.problem_type == 'classification' and hasattr(self.results[self.best_model]['model'], 'predict_proba'):
            try:
                proba = self.results[self.best_model]['model'].predict_proba(input_scaled)
                confidence = np.max(proba) * 100
            except:
                pass
        
        return prediction_value, label, confidence