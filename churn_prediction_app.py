import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import sklearn
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.linear_model import LogisticRegression
from statsmodels.tools.eval_measures import rmse
from sklearn.model_selection import train_test_split
from PIL import Image
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, roc_auc_score
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import warnings
import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title='Customer Churn Prediction',
    page_icon='📡',
    layout='wide',
    initial_sidebar_state='expanded'
)

# Custom CSS for better aesthetics
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
    }
    .nav-container {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        border-left: 4px solid #667eea;
    }
    .upload-container {
        background: #e3f2fd;
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        border: 2px dashed #2196f3;
    }
    .metric-container {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
    .stSelectbox > div > div > select {
        background-color: white;
        border: 2px solid #e1e5e9;
        border-radius: 5px;
    }
    .team-member {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem;
        text-align: center;
        border-left: 4px solid #667eea;
    }
</style>
""", unsafe_allow_html=True)

# Header Section
st.markdown("""
<div class="main-header">
    <h1>Customer Churn Prediction System</h1>
    <p>Advanced Analytics for Customer Retention</p>
</div>
""", unsafe_allow_html=True)

# Top Navigation and File Upload
col1, col2 = st.columns([3, 1])

with col1:
    st.markdown("""
    <div class="nav-container">
        <h3>Navigation</h3>
        <p>Select a page to explore different aspects of the churn prediction system</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="upload-container">
        <h4>Data Upload</h4>
    </div>
    """, unsafe_allow_html=True)
    
    upload_file = st.file_uploader("Upload your CSV file", type=["csv"], help="Upload a CSV file containing customer data")

# Initialize session state for data persistence across pages
if 'df1' not in st.session_state:
    st.session_state.df1 = None
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None
if 'models' not in st.session_state:
    st.session_state.models = {}
if 'model_metrics' not in st.session_state:
    st.session_state.model_metrics = {}

# Check if a file is uploaded
if upload_file is not None:
    try:
        st.session_state.df1 = pd.read_csv(upload_file)
        df1 = st.session_state.df1
        st.success("File uploaded successfully.")
    except Exception as e:
        st.error(f"Error reading the file: {e}")

# Page Selection
pages = {
    'Home & Data Overview': 'page1',
    'Data Preprocessing': 'page2', 
    'Model Training': 'page3',
    'Model Evaluation': 'page4',
    'Prediction Interface': 'page5',
    'Insights & Conclusions': 'page6',
    'Batch Prediction': 'page7'
}

selected_page = st.selectbox("Choose Page", list(pages.keys()), 
                           help="Navigate through different sections of the application")

# Helper functions
def preprocess_data(df1):
    processed_data = df1.copy()
    
    # Handle TotalCharges column (convert to numeric and handle missing values)
    processed_data['TotalCharges'] = pd.to_numeric(processed_data['TotalCharges'], errors='coerce')
    processed_data['TotalCharges'].fillna(processed_data['TotalCharges'].median(), inplace=True)
    categorical_cols = df1.select_dtypes(include=['object']).columns
    mode_imputer = SimpleImputer(strategy='most_frequent')
    df1[categorical_cols] = mode_imputer.fit_transform(df1[categorical_cols])
    
    # Create label encoders for categorical variables
    label_encoders = {}
    categorical_columns = ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
                          'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                          'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
                          'PaperlessBilling', 'PaymentMethod', 'Churn']
    
    for col in categorical_columns:
        if col in processed_data.columns:
            le = LabelEncoder()
            processed_data[col] = le.fit_transform(processed_data[col])
            label_encoders[col] = le
    
    return processed_data, label_encoders

def page1():
    st.markdown("## Team Members")
    
    # Team members in a nice grid layout
    team_cols = st.columns(3)
    team_members = [
        "Ruth Mensah - 22253087",
        "Emmanuel Oduro Dwamena - 11410636", 
        "Zoe Akua Ohene-Ampofo - 22252412",
        "Sandra Animwaa Bamfo - 22256394",
        "Joshua Kwaku Mensah - 22257672"
    ]
    
    for i, member in enumerate(team_members):
        with team_cols[i % 3]:
            st.markdown(f"""
            <div class="team-member">
                <strong>{member.split(' - ')[0]}</strong><br>
                <small>{member.split(' - ')[1]}</small>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("## Preview of Uploaded Data")
    
    if 'df1' not in st.session_state or st.session_state.df1 is None:
        st.warning("Please upload a CSV file first.")
        return

    df1 = st.session_state.df1

    # Display basic dataset information
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Customers", len(df1))
    with col2:
        st.metric("Features", len(df1.columns))
    with col3:
        churn_count = df1['Churn'].value_counts()['Yes'] if 'Yes' in df1['Churn'].values else df1['Churn'].sum()
        st.metric("Churned Customers", churn_count)
    with col4:
        churn_rate = (churn_count / len(df1)) * 100
        st.metric("Churn Rate", f"{churn_rate:.1f}%")

    if st.checkbox('Preview Dataset'):
        st.dataframe(df1, use_container_width=True)
    
    if st.checkbox("Summary Statistics"):
        if 'tenure' in df1.columns and 'MonthlyCharges' in df1.columns:
            num_columns = df1[['tenure','MonthlyCharges']]
            st.dataframe(num_columns.describe(), use_container_width=True)

    if st.checkbox("Exploratory Data Analysis"):
        col1, col2, col3 = st.columns(3)

        with col1:
            # Churn distribution
            churn_counts = df1['Churn'].value_counts()
            fig_churn = px.pie(values=churn_counts.values, 
                              names=churn_counts.index,
                              title="Customer Churn Distribution",
                              color_discrete_sequence=['#90EE90', '#FF6B6B'])
            st.plotly_chart(fig_churn, use_container_width=True)
        
        with col2:
            # Internet Service distribution by churn
            if 'InternetService' in df1.columns:
                fig_tenure1 = px.bar(df1, x='InternetService', color='Churn', 
                                    title="Customer InternetService Distribution by Churn")
                st.plotly_chart(fig_tenure1, use_container_width=True)   

        # Additional visualizations
        col3, col4 = st.columns(2)

        with col3:
            # Tenure distribution by churn
            if 'tenure' in df1.columns:
                fig_tenure = px.histogram(df1, x='tenure', color='Churn', 
                                        title="Customer Tenure Distribution by Churn",
                                        barmode='overlay', opacity=0.7)
                st.plotly_chart(fig_tenure, use_container_width=True) 

        with col4:
            # Monthly charges by churn
            if 'MonthlyCharges' in df1.columns:
                fig_charges = px.box(df1, x='Churn', y='MonthlyCharges',
                                   title="Monthly Charges by Churn Status",
                                   color='Churn')
                st.plotly_chart(fig_charges, use_container_width=True) 

        # Contract type analysis
        if 'Contract' in df1.columns:
            contract_churn = df1.groupby(['Contract', 'Churn']).size().reset_index(name='Count')
            fig = px.bar(contract_churn, x='Contract', y='Count', color='Churn',
                     title="Churn by Contract Type", barmode='group',
                     color_discrete_sequence=['#ff9999', '#66b3ff'])
            st.plotly_chart(fig, use_container_width=True)

def page2():
    st.subheader("Data Preprocessing")

    if st.checkbox('Check for null values'):
        if 'df1' in st.session_state and st.session_state['df1'] is not None:
            df1 = st.session_state['df1'].copy()

            # Replace common placeholders with NaN
            null_placeholders = ["", " ", "NA", "N/A", "null", "Null", "NaN", "-", "--"]
            df1.replace(to_replace=null_placeholders, value=np.nan, inplace=True)

            # Save cleaned data back to session
            st.session_state['df1'] = df1

            # Compute missing values
            missing_count = df1.isna().sum()
            missing_percent = (missing_count / len(df1)) * 100
            missing_df = pd.DataFrame({
                "Missing Values": missing_count,
                "Percent Missing": missing_percent.round(2)
            })
            missing_df = missing_df[missing_df["Missing Values"] > 0]

            if not missing_df.empty:
                st.warning(f"Found {missing_df.shape[0]} columns with missing values.")
                st.dataframe(missing_df)
            else:
                st.success("No missing values found!")

        else:
            st.error("No dataset loaded. Please load the dataset first.")

    # Data types analysis
    if st.checkbox('Data Types Overview'):
        if 'df1' in st.session_state and st.session_state['df1'] is not None:
            df1= st.session_state['df1']
            st.markdown("### Data Types Overview")
            data_types_df = pd.DataFrame({
                'Column': df1.columns,
                'Data Type': df1.dtypes,
                'Unique Values': [df1[col].nunique() for col in df1.columns],
                'Example Values': [str(df1[col].unique()[:3])[1:-1] for col in df1.columns]
            })
            st.dataframe(data_types_df, use_container_width=True)
        else:
            st.error("No dataset loaded. Please load the dataset first.")

    # Preprocessing steps
    if st.checkbox('Preprocess Data'):
        if 'df1' in st.session_state and st.session_state['df1'] is not None:
            df1= st.session_state['df1']

            if st.button("Start Preprocessing", type="primary"):
                with st.spinner("Processing data..."):
                    processed_data, label_encoders = preprocess_data(df1)
                    st.session_state.processed_data = processed_data
                    st.session_state.label_encoders = label_encoders

                st.success("Data preprocessing completed!")

                # Show preprocessing summary
                st.markdown("### Preprocessing Summary:")
                st.write("1. Converted TotalCharges to numeric format")
                st.write("2. Handled missing values(if any) using median imputation")
                st.write("3. Applied Label Encoding to categorical variables")

            # Display processed data if available
            if st.session_state.processed_data is not None:
                st.markdown("### Processed Data Preview")

                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("**Original Data:**")
                    st.dataframe(df1.head(), use_container_width=True)

                with col2:
                    st.markdown("**Processed Data:**")
                    st.dataframe(st.session_state.processed_data.head(), use_container_width=True)

        else:
            st.error("No dataset loaded. Please load the dataset first.")

    if st.checkbox('Check Heat Map'):       
        # Correlation heatmap of processed data
        st.markdown("### Feature Correlation Analysis of Processed Data")

        if 'processed_data' not in st.session_state or st.session_state.processed_data is None:
            st.warning("Please preprocess data first.")
        else:
            correlation_matrix_num = st.session_state.processed_data.select_dtypes(include='number')
            correlation_matrix = correlation_matrix_num.corr()

            fig = px.imshow(
                correlation_matrix, 
                text_auto=True, 
                aspect="auto",
                title="Feature Correlation Matrix",
                color_continuous_scale='RdBu_r'
            )
            fig.update_layout(width=100, height=700)
            st.plotly_chart(fig, use_container_width=True)

def page3():
    st.subheader("Model Training")
    if st.session_state.processed_data is not None:
        df2 = st.session_state.processed_data

        # Feature selection
        st.markdown("### Feature Selection")
        
        # Separate features and target
        x_predict = df2.drop(['customerID', 'Churn'], axis=1, errors='ignore')
        y_output = df2['Churn']
        
        st.info(f"Training with {x_predict.shape[1]} features and {x_predict.shape[0]} samples")
        
        # Train-test split
        test_size = st.slider("Test Size (proportion)", 0.1, 0.5, 0.2, 0.05)
        random_state = st.number_input("Random State", value=40, min_value=0)
        
        x_train, x_test, y_train, y_test = train_test_split(
            x_predict, y_output, test_size=test_size, random_state=random_state, stratify=y_output
        )
        
        st.success(f"Data split: {len(x_train)} training samples, {len(x_test)} testing samples")
        
        # Model configuration
        st.markdown("### Model Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Random Forest Parameters:**")
            n_estimators = st.slider("Number of Trees", 10, 200, 100, 10)
            max_depth = st.slider("Max Depth", 3, 20, 10)
            rf_random_state = st.number_input("RF Random State", value=40, min_value=0)
        
        with col2:
            st.markdown("**SVM Parameters:**")
            svm_kernel = st.selectbox("Kernel", ['rbf', 'linear', 'poly'])
            svm_C = st.slider("Regularization (C)", 0.1, 10.0, 1.0, 0.1)
            svm_random_state = st.number_input("SVM Random State", value=40, min_value=0)
        
        # Train models
        if st.button("Train Models", type="primary"):
            with st.spinner("Training models... This may take sometime."):
                
                # Step 1: Impute missing values
                imputer = SimpleImputer(strategy='most_frequent')
                x_train_imputed = imputer.fit_transform(x_train)
                x_test_imputed = imputer.transform(x_test)

                # Step 2: Scale features for SVM
                scaler = StandardScaler()
                x_train_scaled = scaler.fit_transform(x_train_imputed)
                x_test_scaled = scaler.transform(x_test_imputed)

                # Step 3: Train Random Forest
                rf_model = RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    random_state=rf_random_state
                )
                rf_model.fit(x_train_imputed, y_train)

                # Step 4: Train SVM
                svm_model = SVC(
                    kernel=svm_kernel,
                    C=svm_C,
                    random_state=svm_random_state,
                    probability=True
                )
                svm_model.fit(x_train_scaled, y_train)

                # Step 5: Store models and preprocessing objects
                st.session_state.models = {
                    'Random Forest': rf_model,
                    'SVM': svm_model
                }
                st.session_state.imputer = imputer
                st.session_state.scaler = scaler
                st.session_state.X_train = x_train
                st.session_state.X_test = x_test
                st.session_state.y_train = y_train
                st.session_state.y_test = y_test
                st.session_state.X_train_scaled = x_train_scaled
                st.session_state.X_test_scaled = x_test_scaled
                st.session_state.feature_names = x_predict.columns.tolist()

                st.success("Models trained successfully!")

            # Display model information
            st.markdown("### Model Summary")

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("""
                **Random Forest:**
                - Ensemble method using multiple decision trees  
                - Handles feature interactions well  
                - Provides feature importance scores  
                - Less prone to overfitting
                """)
            with col2:
                st.markdown("""
                **Support Vector Machine:**
                - Finds optimal decision boundary  
                - Works well with high-dimensional data  
                - Uses kernel trick for non-linear patterns  
                - Requires feature scaling
                """)

        # Feature importance (if Random Forest is trained)
        if 'Random Forest' in st.session_state.models:
            st.markdown("### Feature Importance (Random Forest)")
            rf_model = st.session_state.models['Random Forest']
            feature_importance = pd.DataFrame({
                'Feature': st.session_state.feature_names,
                'Importance': rf_model.feature_importances_
            }).sort_values('Importance', ascending=False)

            fig_importance = px.bar(
                feature_importance,
                x='Importance', y='Feature',
                orientation='h',
                title="Order of Importance of Features",
            )
            fig_importance.update_layout(yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig_importance, use_container_width=True)

        if 'SVM' in st.session_state.models:
            svm_model = st.session_state.models['SVM']

            # Ensure this is only done for linear kernel
            if svm_model.kernel == 'linear':
                st.markdown("### Feature Importance (SVM - Linear Kernel)")

                coef = svm_model.coef_[0]  # For binary classification
                feature_importance_svm = pd.DataFrame({
                    'Feature': st.session_state.feature_names,
                    'Importance': np.abs(coef)  # Absolute importance
                }).sort_values('Importance', ascending=True)

                fig_svm_importance = px.bar(
                    feature_importance_svm,
                    x='Importance', y='Feature',
                    orientation='h',
                    title="Order of Importance of Features"
                )
                fig_svm_importance.update_layout(
                    yaxis=dict(categoryorder='total ascending'),
                    xaxis_title='Absolute Coefficient Value',
                    yaxis_title='Feature',
                    title_x=0.5
                )
                st.plotly_chart(fig_svm_importance, use_container_width=True)
    
    else:
        st.error("No processed data available. Please complete the data preprocessing step first.")

def page4():
    st.subheader("Model Evaluation")
    if st.session_state.models:
        
        # Calculate predictions and metrics for both models
        results = {}
        
        for model_name, model in st.session_state.models.items():
            if model_name == 'Random Forest':
                X_test_input = st.session_state.imputer.transform(st.session_state.X_test)
            else:  # SVM
                X_test_input = st.session_state.X_test_scaled
            
            y_pred = model.predict(X_test_input)
            y_pred_proba = model.predict_proba(X_test_input)[:, 1]
            
            # Calculate metrics
            metrics = {
                'Accuracy': accuracy_score(st.session_state.y_test, y_pred),
                'Precision': precision_score(st.session_state.y_test, y_pred),
                'Recall': recall_score(st.session_state.y_test, y_pred),
                'F1-Score': f1_score(st.session_state.y_test, y_pred),
                'ROC-AUC': roc_auc_score(st.session_state.y_test, y_pred_proba)
            }
            
            results[model_name] = {
                'predictions': y_pred,
                'probabilities': y_pred_proba,
                'metrics': metrics
            }
        
        # Store results for prediction page
        st.session_state.model_metrics = results
        
        # Display metrics comparison
        st.markdown("### Model Performance Comparison")
        
        metrics_df = pd.DataFrame({
            model_name: result['metrics'] 
            for model_name, result in results.items()
        }).T
        
        st.dataframe(metrics_df.round(4), use_container_width=True)
        
        # Visual metrics comparison
        fig_metrics = px.bar(metrics_df.reset_index(), 
                           x='index', y=['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC'],
                           title="Model Performance Metrics Comparison",
                           barmode='group')
        fig_metrics.update_layout(xaxis_title="Models", yaxis_title="Score")
        st.plotly_chart(fig_metrics, use_container_width=True)
        
        # Confusion Matrices
        st.markdown("### Confusion Matrices")
        
        col1, col2 = st.columns(2)
        
        for i, (model_name, result) in enumerate(results.items()):
            cm = confusion_matrix(st.session_state.y_test, result['predictions'])
            
            fig_cm = px.imshow(cm, text_auto=True, aspect="auto",
                              title=f"Confusion Matrix - {model_name}",
                              labels=dict(x="Predicted", y="Actual"),
                              x=['No Churn', 'Churn'],
                              y=['No Churn', 'Churn'])
            
            if i == 0:
                col1.plotly_chart(fig_cm, use_container_width=True)
            else:
                col2.plotly_chart(fig_cm, use_container_width=True)
        
        # ROC Curves
        st.markdown("### ROC Curves Comparison")
        
        fig_roc = go.Figure()
        
        for model_name, result in results.items():
            fpr, tpr, _ = roc_curve(st.session_state.y_test, result['probabilities'])
            auc_score = auc(fpr, tpr)
            
            fig_roc.add_trace(go.Scatter(
                x=fpr, y=tpr,
                mode='lines',
                name=f'{model_name} (AUC = {auc_score:.3f})',
                line=dict(width=3)
            ))
        
        # Add diagonal line
        fig_roc.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            name='Random Classifier',
            line=dict(dash='dash', color='gray')
        ))
        
        fig_roc.update_layout(
            title='ROC Curves Comparison',
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            width=800, height=500
        )
        
        st.plotly_chart(fig_roc, use_container_width=True)
        
        # Best model recommendation
        st.markdown("### Model Recommendation")
        
        # Calculate overall score (weighted average of metrics)
        overall_scores = {}
        for model_name, result in results.items():
            metrics = result['metrics']
            # Weight: Accuracy(0.2) + Precision(0.2) + Recall(0.2) + F1(0.2) + ROC-AUC(0.2)
            overall_score = (metrics['Accuracy'] * 0.2 + 
                           metrics['Precision'] * 0.2 + 
                           metrics['Recall'] * 0.2 + 
                           metrics['F1-Score'] * 0.2 + 
                           metrics['ROC-AUC'] * 0.2)
            overall_scores[model_name] = overall_score
        
        best_model = max(overall_scores, key=overall_scores.get)
        
        st.success(f"**Recommended Model: {best_model}**")
        st.info(f"Overall Score: {overall_scores[best_model]:.4f}")
        
        # Detailed classification reports
        st.markdown("### Detailed Classification Reports")
        
        for model_name, result in results.items():
            with st.expander(f"{model_name} Classification Report"):
                report = classification_report(
                    st.session_state.y_test, 
                    result['predictions'], 
                    output_dict=True
                )
                report_df = pd.DataFrame(report).transpose()
                st.dataframe(report_df.round(4), use_container_width=True)
    
    else:
        st.error("No trained models available. Please complete the model training step first.")

def page5():
    st.subheader("Prediction Interface")

    if st.session_state.models:
        st.markdown("### Enter Customer Information")
        
        # Create input form with organized layout
        with st.form("prediction_form"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**Demographics**")
                gender = st.selectbox("Gender", ["Female", "Male"])
                senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
                partner = st.selectbox("Has Partner", ["No", "Yes"])
                dependents = st.selectbox("Has Dependents", ["No", "Yes"])
                tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=12)
            
            with col2:
                st.markdown("**Services**")
                phone_service = st.selectbox("Phone Service", ["No", "Yes"])
                multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
                internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
                online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
                online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
            
            with col3:
                st.markdown("**Additional Services**")
                device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
                tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
                streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
                streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Billing Information**")
                contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
                paperless_billing = st.selectbox("Paperless Billing", ["No", "Yes"])
                payment_method = st.selectbox("Payment Method", 
                                            ["Electronic check", "Mailed check", 
                                             "Bank transfer (automatic)", "Credit card (automatic)"])
            
            with col2:
                st.markdown("**Charges**")
                monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=200.0, value=50.0)
                total_charges = st.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0, value=500.0)
            
            # Model selection
            selected_model = st.selectbox("Choose Prediction Model", list(st.session_state.models.keys()))
            
            predict_button = st.form_submit_button("Predict Churn", type="primary")
        
        if predict_button:
            # Prepare input data
            input_data = {
                'gender': 1 if gender == "Male" else 0,
                'SeniorCitizen': 1 if senior_citizen == "Yes" else 0,
                'Partner': 1 if partner == "Yes" else 0,
                'Dependents': 1 if dependents == "Yes" else 0,
                'tenure': tenure,
                'PhoneService': 1 if phone_service == "Yes" else 0,
                'MultipleLines': 0 if multiple_lines == "No" else (1 if multiple_lines == "Yes" else 2),
                'InternetService': 0 if internet_service == "DSL" else (1 if internet_service == "Fiber optic" else 2),
                'OnlineSecurity': 0 if online_security == "No" else (1 if online_security == "Yes" else 2),
                'OnlineBackup': 0 if online_backup == "No" else (1 if online_backup == "Yes" else 2),
                'DeviceProtection': 0 if device_protection == "No" else (1 if device_protection == "Yes" else 2),
                'TechSupport': 0 if tech_support == "No" else (1 if tech_support == "Yes" else 2),
                'StreamingTV': 0 if streaming_tv == "No" else (1 if streaming_tv == "Yes" else 2),
                'StreamingMovies': 0 if streaming_movies == "No" else (1 if streaming_movies == "Yes" else 2),
                'Contract': 0 if contract == "Month-to-month" else (1 if contract == "One year" else 2),
                'PaperlessBilling': 1 if paperless_billing == "Yes" else 0,
                'PaymentMethod': {"Electronic check": 0, "Mailed check": 1, 
                                "Bank transfer (automatic)": 2, "Credit card (automatic)": 3}[payment_method],
                'MonthlyCharges': monthly_charges,
                'TotalCharges': total_charges
            }
            
            # Convert to DataFrame
            input_df = pd.DataFrame([input_data])
            
            # Make prediction
            model = st.session_state.models[selected_model]
            
            if selected_model == "SVM":
                input_processed = st.session_state.imputer.transform(input_df)
                input_scaled = st.session_state.scaler.transform(input_processed)
                prediction = model.predict(input_scaled)[0]
                probability = model.predict_proba(input_scaled)[0]
            else:  # Random Forest
                input_processed = st.session_state.imputer.transform(input_df)
                prediction = model.predict(input_processed)[0]
                probability = model.predict_proba(input_processed)[0]
            
            # Display prediction results
            st.markdown("### Prediction Results")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if prediction == 1:
                    st.error("**HIGH RISK: Customer likely to CHURN**")
                    risk_level = "HIGH"
                    risk_color = "#ff4444"
                else:
                    st.success("**LOW RISK: Customer likely to STAY**")
                    risk_level = "LOW"
                    risk_color = "#44ff44"
            
            with col2:
                churn_prob = probability[1] * 100
                st.metric("Churn Probability", f"{churn_prob:.2f}%")
            
            # Risk assessment and recommendations
            st.markdown("### Risk Assessment & Recommendations")
            
            if churn_prob > 70:
                st.markdown("""
                **HIGH RISK**
                - Contact customer within 24 hours
                - Offer personalized retention package
                - Investigate service issues
                - Consider contract upgrade incentives
                """)
            elif churn_prob > 40:
                st.markdown("""
                **MODERATE RISK**
                - Send satisfaction survey
                - Offer service upgrades
                - Provide loyalty rewards
                - Monitor usage patterns
                """)
            else:
                st.markdown("""
                **LOW RISK**
                - Continue excellent service
                - Opportunity for upselling
                - Regular satisfaction check
                - Consider referral programs
                """)
            
            # Feature impact analysis (for Random Forest)
            if selected_model == "Random Forest" and 'feature_names' in st.session_state:
                st.markdown("### Key Factors Influencing This Prediction")
                
                # Get feature importance for this specific prediction
                rf_model = st.session_state.models['Random Forest']
                feature_importance = pd.DataFrame({
                    'Feature': st.session_state.feature_names,
                    'Importance': rf_model.feature_importances_,
                    'Customer_Value': [input_data.get(feature, 0) for feature in st.session_state.feature_names]
                }).sort_values('Importance', ascending=False).head(8)
                
                fig_factors = px.bar(feature_importance, 
                                   x='Importance', y='Feature',
                                   orientation='h',
                                   title="Most Influential Factors for This Customer",
                                   color='Importance',
                                   color_continuous_scale='plasma')
                fig_factors.update_layout(yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig_factors, use_container_width=True)
    
    else:
        st.error("No trained models available. Please complete the model training step first.")

def page6():
    st.subheader("Insights & Conclusions")
    if st.session_state.df1 is not None:
        data = st.session_state.df1

        # Calculate financial impact
        total_customers1 = len(data)
        churned_customers1 = len(data[data['Churn'] == 'Yes'])
        monthly_revenue_lost1 = data[data['Churn'] == 'Yes']['MonthlyCharges'].sum()
        annual_revenue_lost1 = monthly_revenue_lost1 * 12
        churn_rate1 = (data['Churn'] == 'Yes').mean() * 100
        
        # Final summary
        st.markdown("### Executive Summary")
        
        st.markdown(f"""
        **Project Outcome Summary**
        - Analyzed {len(data):,} customer records with {churn_rate1:.1f}% churn rate
        - Identified ${annual_revenue_lost1:,.0f} in annual revenue at risk
        - Achieved high accuracy in churn prediction with actionable insights
        - Enabled proactive customer retention with potential 5-15% churn reduction
        """)

        # Business insights from data analysis
        st.markdown("### Key Business Insights")
        
        # Calculate key statistics
        churn_rate = (data['Churn'] == 'Yes').mean() * 100
        avg_tenure_churn = data[data['Churn'] == 'Yes']['tenure'].mean()
        avg_tenure_stay = data[data['Churn'] == 'No']['tenure'].mean()
        avg_monthly_churn = data[data['Churn'] == 'Yes']['MonthlyCharges'].mean()
        avg_monthly_stay = data[data['Churn'] == 'No']['MonthlyCharges'].mean()
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Overall Churn Rate", f"{churn_rate:.1f}%")
            st.metric("Avg Tenure (Churned)", f"{avg_tenure_churn:.1f} months")
        
        with col2:
            st.metric("Avg Monthly Charges (Churned)", f"${avg_monthly_churn:.2f}")
            st.metric("Avg Tenure (Retained)", f"{avg_tenure_stay:.1f} months")
        
        with col3:
            st.metric("Avg Monthly Charges (Retained)", f"${avg_monthly_stay:.2f}")
            revenue_at_risk = len(data[data['Churn'] == 'Yes']) * avg_monthly_churn * 12
            st.metric("Annual Revenue at Risk", f"${revenue_at_risk:,.2f}")
        
        # Detailed insights
        st.markdown("### Detailed Analysis")
        
        insights_tabs = st.tabs(["Churn Drivers", "Business Impact", "Recommendations", "Model Performance"])
        
        with insights_tabs[0]:
            st.markdown("### Primary Churn Drivers Identified:")
            
            # Contract type analysis
            contract_churn = data.groupby('Contract')['Churn'].apply(lambda x: (x == 'Yes').mean() * 100)
            
            # Payment method analysis
            payment_churn = data.groupby('PaymentMethod')['Churn'].apply(lambda x: (x == 'Yes').mean() * 100)
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig_contract = px.bar(x=contract_churn.index, y=contract_churn.values,
                                    title="Churn Rate by Contract Type",
                                    labels={'x': 'Contract Type', 'y': 'Churn Rate (%)'},
                                    color=contract_churn.values,
                                    color_continuous_scale='Reds')
                st.plotly_chart(fig_contract, use_container_width=True)
            
            with col2:
                fig_payment = px.bar(x=payment_churn.index, y=payment_churn.values,
                                   title="Churn Rate by Payment Method",
                                   labels={'x': 'Payment Method', 'y': 'Churn Rate (%)'},
                                   color=payment_churn.values,
                                   color_continuous_scale='Reds')
                fig_payment.update_xaxes(tickangle=45)
                st.plotly_chart(fig_payment, use_container_width=True)
            
            st.markdown("""
            **Key Findings:**
            - Month-to-month contracts show highest churn risk
            - Electronic check payments correlate with higher churn
            - Fiber optic customers have mixed retention patterns
            - Senior citizens show different churn behaviors
            - New customers (low tenure) are most vulnerable
            """)
        
        with insights_tabs[1]:
            st.markdown("### Business Impact Analysis:")
            
            # Calculate financial impact
            total_customers = len(data)
            churned_customers = len(data[data['Churn'] == 'Yes'])
            monthly_revenue_lost = data[data['Churn'] == 'Yes']['MonthlyCharges'].sum()
            annual_revenue_lost = monthly_revenue_lost * 12
            
            # Customer acquisition cost (estimated)
            estimated_cac = 200  # Average customer acquisition cost
            total_acquisition_cost = churned_customers * estimated_cac
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"""
                **Financial Impact:**
                - Customers Lost: {churned_customers:,}
                - Monthly Revenue Lost: ${monthly_revenue_lost:,.2f}
                - Annual Revenue Lost: ${annual_revenue_lost:,.2f}
                - Replacement Cost: ${total_acquisition_cost:,.2f}
                - **Total Annual Impact: ${annual_revenue_lost + total_acquisition_cost:,.2f}**
                """)
            
            with col2:
                # Revenue impact visualization
                impact_data = pd.DataFrame({
                    'Category': ['Revenue Lost', 'Acquisition Cost'],
                    'Amount': [annual_revenue_lost, total_acquisition_cost]
                })
                
                fig_impact = px.pie(impact_data, values='Amount', names='Category',
                                  title="Annual Financial Impact Breakdown",
                                  color_discrete_sequence=['#ff6b6b', '#feca57'])
                st.plotly_chart(fig_impact, use_container_width=True)
        
        with insights_tabs[2]:
            st.markdown("### Strategic Recommendations:")
            
            st.markdown("""
            **Immediate Actions (0-30 days):**
            1. **Target High-Risk Segments**: Focus on month-to-month contract customers
            2. **Payment Method Strategy**: Incentivize automatic payment methods
            3. **Proactive Outreach**: Contact customers with tenure < 6 months
            4. **Retention Offers**: Design personalized packages for at-risk customers
            
            **Medium-term Strategies (1-6 months):**
            1. **Predictive Analytics**: Implement real-time churn scoring
            2. **Customer Success Program**: Dedicated support for new customers
            3. **Pricing Optimization**: Review pricing structure for fiber services
            4. **Digital Experience**: Improve online service management tools
            
            **Long-term Initiatives (6+ months):**
            1. **Segmentation Strategy**: Develop customer lifecycle programs
            2. **Continuous Monitoring**: Regular model updates and retraining
            3. **Value Demonstration**: Clearer communication of service benefits
            4. **Loyalty Programs**: Reward long-term customers
            """)
            
            # ROI calculation for retention efforts
            st.markdown("### ROI Potential:")
            
            retention_scenarios = pd.DataFrame({
                'Scenario': ['5% Reduction', '10% Reduction', '15% Reduction'],
                'Customers_Saved': [churned_customers * 0.05, churned_customers * 0.10, churned_customers * 0.15],
                'Annual_Savings': [annual_revenue_lost * 0.05, annual_revenue_lost * 0.10, annual_revenue_lost * 0.15]
            })
            
            fig_roi = px.bar(retention_scenarios, x='Scenario', y='Annual_Savings',
                           title="Potential Annual Savings from Churn Reduction",
                           color='Annual_Savings',
                           color_continuous_scale='Blues')
            fig_roi.update_layout(yaxis_title="Annual Savings ($)")
            st.plotly_chart(fig_roi, use_container_width=True)
        
        with insights_tabs[3]:
            if st.session_state.model_metrics:
                st.markdown("### Model Performance Summary:")
                
                # Best performing model
                best_model = None
                best_f1 = 0
                
                for model_name, results in st.session_state.model_metrics.items():
                    if results['metrics']['F1-Score'] > best_f1:
                        best_f1 = results['metrics']['F1-Score']
                        best_model = model_name
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.success(f"**Best Performing Model: {best_model}**")
                    st.markdown(f"**F1-Score: {best_f1:.4f}**")
                    
                    st.markdown("""
                    **Model Strengths:**
                    - High accuracy in identifying churn patterns
                    - Good balance between precision and recall
                    - Robust feature importance insights
                    - Reliable probability estimates
                    """)
                
                with col2:
                    st.markdown("""
                    **Implementation Benefits:**
                    - **Early Warning System**: Identify at-risk customers before they churn
                    - **Cost Reduction**: Lower customer acquisition costs
                    - **Revenue Protection**: Maintain recurring revenue streams
                    - **Personalization**: Tailored retention strategies
                    """)
                
                # Model comparison summary
                if len(st.session_state.model_metrics) > 1:
                    st.markdown("### Model Comparison Insights:")
                    
                    comparison_df = pd.DataFrame({
                        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC'],
                        'Random Forest': [
                            st.session_state.model_metrics['Random Forest']['metrics']['Accuracy'],
                            st.session_state.model_metrics['Random Forest']['metrics']['Precision'],
                            st.session_state.model_metrics['Random Forest']['metrics']['Recall'],
                            st.session_state.model_metrics['Random Forest']['metrics']['F1-Score'],
                            st.session_state.model_metrics['Random Forest']['metrics']['ROC-AUC']
                        ],
                        'SVM': [
                            st.session_state.model_metrics['SVM']['metrics']['Accuracy'],
                            st.session_state.model_metrics['SVM']['metrics']['Precision'],
                            st.session_state.model_metrics['SVM']['metrics']['Recall'],
                            st.session_state.model_metrics['SVM']['metrics']['F1-Score'],
                            st.session_state.model_metrics['SVM']['metrics']['ROC-AUC']
                        ]
                    })
                    
                    st.dataframe(comparison_df.round(4), use_container_width=True)
            
            else:
                st.info("Model performance metrics will appear here after training models.")
    
    else:
        st.error("No data available for analysis. Please load your dataset first.")

def page7():
    st.subheader("Batch Prediction")

    # Check if models are available from main app
    if not st.session_state.models:
        st.error("No trained models available. Please complete the model training step first.")
        st.info("Go to 'Model Training' page and train models before using batch prediction.")
        return

    st.markdown("### Upload CSV File for Batch Predictions")
    st.info("Upload a CSV file with customer data to get churn predictions for multiple customers at once.")
    
    # File uploader for batch prediction
    batch_file = st.file_uploader("Choose CSV file for batch prediction", type="csv", key="batch_upload")
    
    if batch_file is not None:
        try:
            # Read the uploaded file
            batch_df = pd.read_csv(batch_file)
            
            st.success("File uploaded successfully!")
            st.markdown("### Data Preview")
            st.dataframe(batch_df.head(), use_container_width=True)
            
            # Show data info
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Records", len(batch_df))
            with col2:
                st.metric("Features", len(batch_df.columns))
            with col3:
                missing_data = batch_df.isnull().sum().sum()
                st.metric("Missing Values", missing_data)
            
            # Model selection for batch prediction
            selected_model = st.selectbox("Choose Model for Batch Prediction", 
                                        list(st.session_state.models.keys()),
                                        key="batch_model_select")
            
            if st.button("Run Batch Prediction", type="primary"):
                with st.spinner("Processing batch predictions..."):
                    try:
                        # Prepare the data
                        batch_processed = batch_df.copy()
                        
                        # Handle customer ID if present
                        if 'customerID' in batch_processed.columns:
                            customer_ids = batch_processed['customerID']
                            batch_processed = batch_processed.drop('customerID', axis=1)
                        else:
                            customer_ids = [f"Customer_{i+1}" for i in range(len(batch_processed))]
                        
                        # Remove target column if present
                        if 'Churn' in batch_processed.columns:
                            batch_processed = batch_processed.drop('Churn', axis=1)
                        
                        # Handle TotalCharges conversion
                        if 'TotalCharges' in batch_processed.columns:
                            batch_processed['TotalCharges'] = pd.to_numeric(batch_processed['TotalCharges'], errors='coerce')
                            batch_processed['TotalCharges'].fillna(batch_processed['TotalCharges'].median(), inplace=True)
                        
                        # Encode categorical variables using the same mapping as training
                        if 'label_encoders' in st.session_state:
                            for col, le in st.session_state.label_encoders.items():
                                if col in batch_processed.columns and col != 'Churn':
                                    # Handle unseen categories
                                    unique_values = batch_processed[col].unique()
                                    for val in unique_values:
                                        if val not in le.classes_:
                                            # Add new class to encoder
                                            le.classes_ = np.append(le.classes_, val)
                                    batch_processed[col] = le.transform(batch_processed[col])
                        else:
                            # If no label encoders, do basic encoding
                            categorical_cols = batch_processed.select_dtypes(include=['object']).columns
                            for col in categorical_cols:
                                batch_processed[col] = LabelEncoder().fit_transform(batch_processed[col])
                        
                        # Ensure all required features are present
                        if 'feature_names' in st.session_state:
                            required_features = st.session_state.feature_names
                            for feature in required_features:
                                if feature not in batch_processed.columns:
                                    batch_processed[feature] = 0  # Default value for missing features
                            
                            # Select only required features in correct order
                            batch_processed = batch_processed[required_features]
                        
                        # Get the trained model and preprocessors
                        model = st.session_state.models[selected_model]
                        
                        # Apply preprocessing
                        if selected_model == "SVM":
                            batch_imputed = st.session_state.imputer.transform(batch_processed)
                            batch_scaled = st.session_state.scaler.transform(batch_imputed)
                            predictions = model.predict(batch_scaled)
                            probabilities = model.predict_proba(batch_scaled)[:, 1]
                        else:  # Random Forest
                            batch_imputed = st.session_state.imputer.transform(batch_processed)
                            predictions = model.predict(batch_imputed)
                            probabilities = model.predict_proba(batch_imputed)[:, 1]
                        
                        # Create results DataFrame
                        results_df = pd.DataFrame({
                            'CustomerID': customer_ids,
                            'Churn_Prediction': ['Yes' if pred == 1 else 'No' for pred in predictions],
                            'Churn_Probability': probabilities.round(4),
                            'Risk_Level': ['High' if prob > 0.7 else 'Medium' if prob > 0.4 else 'Low' for prob in probabilities]
                        })
                        
                        st.success("Batch predictions completed successfully!")
                        
                        # Display results summary
                        st.markdown("### Prediction Summary")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            total_customers = len(results_df)
                            st.metric("Total Customers", total_customers)
                        
                        with col2:
                            churned_predictions = len(results_df[results_df['Churn_Prediction'] == 'Yes'])
                            st.metric("Predicted Churners", churned_predictions)
                        
                        with col3:
                            churn_rate_pred = (churned_predictions / total_customers) * 100
                            st.metric("Predicted Churn Rate", f"{churn_rate_pred:.1f}%")
                        
                        with col4:
                            high_risk = len(results_df[results_df['Risk_Level'] == 'High'])
                            st.metric("High Risk Customers", high_risk)
                        
                        # Display detailed results
                        st.markdown("### Detailed Results")
                        st.dataframe(results_df, use_container_width=True)
                        
                        # Risk distribution chart
                        st.markdown("### Risk Distribution")
                        risk_counts = results_df['Risk_Level'].value_counts()
                        fig_risk = px.pie(values=risk_counts.values, 
                                        names=risk_counts.index,
                                        title="Customer Risk Distribution",
                                        color_discrete_sequence=['#ff6b6b', '#feca57', '#90EE90'])
                        st.plotly_chart(fig_risk, use_container_width=True)
                        
                        # Probability distribution
                        fig_prob = px.histogram(results_df, x='Churn_Probability', 
                                              title="Churn Probability Distribution",
                                              nbins=20)
                        st.plotly_chart(fig_prob, use_container_width=True)
                        
                        # Download button for results
                        csv_data = results_df.to_csv(index=False)
                        st.download_button(
                            label="Download Results as CSV",
                            data=csv_data,
                            file_name=f"batch_churn_predictions_{selected_model.lower().replace(' ', '_')}.csv",
                            mime="text/csv"
                        )
                        
                        # Action recommendations
                        st.markdown("### Recommended Actions")
                        
                        high_risk_customers = results_df[results_df['Risk_Level'] == 'High']
                        medium_risk_customers = results_df[results_df['Risk_Level'] == 'Medium']
                        
                        if len(high_risk_customers) > 0:
                            st.error(f"**High Priority**: {len(high_risk_customers)} customers need immediate attention")
                            with st.expander("View High Risk Customers"):
                                st.dataframe(high_risk_customers, use_container_width=True)
                        
                        if len(medium_risk_customers) > 0:
                            st.warning(f"**Medium Priority**: {len(medium_risk_customers)} customers need proactive engagement")
                            with st.expander("View Medium Risk Customers"):
                                st.dataframe(medium_risk_customers, use_container_width=True)
                        
                    except Exception as e:
                        st.error(f"Error in batch prediction: {str(e)}")
                        st.info("Please ensure your data has the same format as the training data.")
        
        except Exception as e:
            st.error(f"Error reading the file: {str(e)}")
    
    else:
        st.info("Please upload a CSV file to start batch prediction.")
        
        # Show expected format
        st.markdown("### Expected Data Format")
        st.markdown("""
        Your CSV file should contain the following columns:
        - customerID (optional)
        - gender, SeniorCitizen, Partner, Dependents
        - tenure, PhoneService, MultipleLines, InternetService
        - OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport
        - StreamingTV, StreamingMovies, Contract, PaperlessBilling
        - PaymentMethod, MonthlyCharges, TotalCharges
        """)

# Page routing
if selected_page == 'Home & Data Overview':
    page1()
elif selected_page == 'Data Preprocessing':
    page2()
elif selected_page == 'Model Training':
    page3()
elif selected_page == 'Model Evaluation':
    page4()
elif selected_page == 'Prediction Interface':
    page5()
elif selected_page == 'Insights & Conclusions':
    page6()
elif selected_page == 'Batch Prediction':
    page7()