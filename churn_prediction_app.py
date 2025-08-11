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
    page_title='Customer Churn Prediction Dashboard',
    page_icon='📊',
    layout='wide',
    initial_sidebar_state='collapsed'
)

# Professional CSS styling with dark theme
st.markdown("""
<style>
    /* Import professional font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Dark theme base */
    .main {
        padding: 0rem 1rem;
        font-family: 'Inter', sans-serif;
        background-color: #1a1a1a;
        color: #ffffff;
    }
    
    /* Dark theme for streamlit elements */
    .stApp {
        background-color: #1a1a1a;
        color: #ffffff;
    }
    
    /* Header styling - dark theme */
    .main-header {
        background: linear-gradient(135deg, #2d3748 0%, #4a5568 100%);
        padding: 2.5rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        border: 1px solid #4a5568;
    }
    
    /* Navigation styling - dark theme */
    .nav-container {
        background: linear-gradient(90deg, #2d3748 0%, #4a5568 100%);
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.3);
        border: 1px solid #4a5568;
    }
    
    /* Enhanced navigation buttons with different colors */
    .stButton > button {
        font-weight: 500;
        font-family: 'Inter', sans-serif;
        transition: all 0.3s ease;
        border: none;
        border-radius: 8px;
        padding: 0.7rem 1.5rem;
        margin: 0.2rem;
        font-size: 0.9rem;
    }
    
    /* Different colors for navigation buttons */
    .stButton > button:nth-child(1) {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
    }
    
    .stButton > button:nth-child(2) {
        background: linear-gradient(135deg, #f093fb, #f5576c);
        color: white;
    }
    
    .stButton > button:nth-child(3) {
        background: linear-gradient(135deg, #4facfe, #00f2fe);
        color: white;
    }
    
    .stButton > button:nth-child(4) {
        background: linear-gradient(135deg, #43e97b, #38f9d7);
        color: white;
    }
    
    .stButton > button:nth-child(5) {
        background: linear-gradient(135deg, #fa709a, #fee140);
        color: white;
    }
    
    .stButton > button:nth-child(6) {
        background: linear-gradient(135deg, #a8edea, #fed6e3);
        color: #2d3748;
    }
    
    .stButton > button:nth-child(7) {
        background: linear-gradient(135deg, #ffecd2, #fcb69f);
        color: #2d3748;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.4);
    }
    
    /* Enhanced tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
        background: #2d3748;
        border-radius: 8px;
        padding: 0.5rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: linear-gradient(135deg, #4a5568, #2d3748);
        color: white;
        border-radius: 6px;
        padding: 0.8rem 1.5rem;
        font-weight: 500;
        border: 1px solid #4a5568;
        transition: all 0.3s ease;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: linear-gradient(135deg, #667eea, #764ba2);
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea, #764ba2) !important;
        color: white !important;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    
    /* Dark theme metric cards */
    .metric-card {
        background: linear-gradient(135deg, #2d3748 0%, #4a5568 100%);
        padding: 1.8rem;
        border-radius: 12px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
        border: 1px solid #4a5568;
        margin: 1rem 0;
        transition: transform 0.2s ease;
        color: white;
    }
    
    .metric-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.4);
        border: 1px solid #667eea;
    }
    
    /* Form styling - dark theme */
    .stSelectbox > div > div, .stNumberInput > div > div > input {
        background-color: #2d3748 !important;
        border: 1px solid #4a5568 !important;
        border-radius: 6px;
        font-family: 'Inter', sans-serif;
        color: white !important;
    }
    
    /* File uploader styling - dark theme */
    .uploadedFile {
        border: 2px dashed #667eea;
        border-radius: 8px;
        padding: 2rem;
        text-align: center;
        background: #2d3748;
        color: white;
    }
    
    /* Alert styling - dark theme */
    .stSuccess {
        background: linear-gradient(90deg, #48bb78, #38a169);
        border-radius: 6px;
        color: white;
        border: 1px solid #48bb78;
    }
    
    .stError {
        background: linear-gradient(90deg, #f56565, #e53e3e);
        border-radius: 6px;
        color: white;
        border: 1px solid #f56565;
    }
    
    .stWarning {
        background: linear-gradient(90deg, #ed8936, #dd6b20);
        border-radius: 6px;
        color: white;
        border: 1px solid #ed8936;
    }
    
    .stInfo {
        background: linear-gradient(90deg, #4299e1, #3182ce);
        border-radius: 6px;
        color: white;
        border: 1px solid #4299e1;
    }
    
    /* Team member card - dark theme */
    .team-card {
        background: linear-gradient(135deg, #2d3748 0%, #4a5568 100%);
        color: white;
        padding: 2.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        text-align: center;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        border: 1px solid #4a5568;
    }
    
    /* Dark theme insight boxes */
    .insight-box {
        background: linear-gradient(135deg, #2d3748 0%, #4a5568 100%);
        padding: 1.8rem;
        border-radius: 8px;
        margin: 1rem 0;
        border-left: 4px solid #667eea;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.3);
        color: white;
        border: 1px solid #4a5568;
    }
    
    /* Status boxes - dark theme */
    .status-high {
        background: linear-gradient(135deg, #f56565, #e53e3e);
        color: white;
        padding: 2rem;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 6px 25px rgba(245, 101, 101, 0.4);
        border: 1px solid #f56565;
    }
    
    .status-low {
        background: linear-gradient(135deg, #48bb78, #38a169);
        color: white;
        padding: 2rem;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 6px 25px rgba(72, 187, 120, 0.4);
        border: 1px solid #48bb78;
    }
    
    /* Dark theme table styling */
    .dataframe {
        background-color: #2d3748 !important;
        color: white !important;
        border: 1px solid #4a5568 !important;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.3);
        border-radius: 8px;
    }
    
    /* Expander styling - dark theme */
    .streamlit-expanderHeader {
        background: linear-gradient(90deg, #2d3748, #4a5568) !important;
        color: white !important;
        border: 1px solid #4a5568 !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        padding: 1rem !important;
        transition: all 0.3s ease !important;
    }
    
    .streamlit-expanderHeader:hover {
        background: linear-gradient(90deg, #667eea, #764ba2) !important;
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    .streamlit-expanderContent {
        background: #1a202c !important;
        border: 1px solid #4a5568 !important;
        border-top: none !important;
        border-radius: 0 0 8px 8px !important;
        color: white !important;
    }
    
    /* Text styling for dark theme */
    .stMarkdown, .stText, p, div, span {
        color: #ffffff !important;
    }
    
    h1, h2, h3, h4, h5, h6 {
        color: #ffffff !important;
    }
    
    /* Hide streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# Professional Header
st.markdown("""
<div class="main-header">
    <h1>Customer Churn Prediction Dashboard</h1>
    <h3>Advanced Analytics for Customer Retention</h3>
</div>
""", unsafe_allow_html=True)

# Initialize session state
if 'df1' not in st.session_state:
    st.session_state.df1 = None
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None
if 'models' not in st.session_state:
    st.session_state.models = {}
if 'model_metrics' not in st.session_state:
    st.session_state.model_metrics = {}

# Professional navigation and file upload
col1, col2, col3 = st.columns([2, 1, 1])

with col1:
    st.markdown("""
    <div class="nav-container">
        <h4 style="color: white; margin: 0;">Data Upload Center</h4>
    </div>
    """, unsafe_allow_html=True)

with col2:
    upload_file = st.file_uploader("Upload CSV file", type=["csv"], help="Upload your customer data CSV file")

with col3:
    if upload_file is not None:
        try:
            st.session_state.df1 = pd.read_csv(upload_file)
            st.success("File uploaded successfully")
        except Exception as e:
            st.error(f"Error: {e}")

# Navigation
st.markdown("""
<div class="nav-container">
    <h4 style="color: white; margin-bottom: 1rem;">Navigation Dashboard</h4>
</div>
""", unsafe_allow_html=True)

# Professional navigation
nav_cols = st.columns(7)
page_names = [
    'Home & Overview', 'Data Preprocessing', 'Model Training', 
    'Model Evaluation', 'Prediction Interface', 'Business Insights', 'Batch Processing'
]

selected_page_index = 0
for i, (col, page_name) in enumerate(zip(nav_cols, page_names)):
    with col:
        if st.button(page_name, key=f"nav_{i}"):
            st.session_state.selected_page = i
            selected_page_index = i

if 'selected_page' not in st.session_state:
    st.session_state.selected_page = 0

selected_page_index = st.session_state.selected_page

def preprocess_data(df1):
    processed_data = df1.copy()
    
    # Handle TotalCharges column
    processed_data['TotalCharges'] = pd.to_numeric(processed_data['TotalCharges'], errors='coerce')
    processed_data['TotalCharges'].fillna(processed_data['TotalCharges'].median(), inplace=True)
    categorical_cols = df1.select_dtypes(include=['object']).columns
    mode_imputer = SimpleImputer(strategy='most_frequent')
    df1[categorical_cols] = mode_imputer.fit_transform(df1[categorical_cols])
    
    # Create label encoders
    label_encoders = {}
    categorical_columns = ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
                          'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                          'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
                          'PaperlessBilling', 'PaymentMethod', 'Churn']
    
    for col in categorical_columns:
        le = LabelEncoder()
        processed_data[col] = le.fit_transform(processed_data[col])
        label_encoders[col] = le
    
    return processed_data, label_encoders

def page1():
    st.markdown("""
    <div class="team-card">
        <h2>Group 7 Team Members</h2>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 1rem; margin-top: 1rem;">
            <div>Ruth Mensah - 22253087</div>
            <div>Emmanuel Oduro Dwamena - 11410636</div>
            <div>Zoe Akua Ohene-Ampofo - 22252412</div>
            <div>Sandra Animwaa Bamfo - 22256394</div>
            <div>Joshua Kwaku Mensah - 22257672</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("## Dataset Overview")
    
    if 'df1' not in st.session_state or st.session_state.df1 is None:
        st.markdown("""
        <div class="insight-box">
            <h4>No Data Loaded</h4>
            <p>Please upload a CSV file using the upload center above to get started with the analysis.</p>
        </div>
        """, unsafe_allow_html=True)
        return

    df1 = st.session_state.df1

    # Professional metrics display
    col1, col2, col3, col4 = st.columns(4)
    
    churn_count = df1['Churn'].value_counts()['Yes'] if 'Yes' in df1['Churn'].values else df1['Churn'].sum()
    churn_rate = (churn_count / len(df1)) * 100
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="color: #3498db;">Total Customers</h3>
            <h2 style="color: #2c3e50;">{len(df1):,}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="color: #27ae60;">Features</h3>
            <h2 style="color: #2c3e50;">{len(df1.columns)}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="color: #e74c3c;">Churned Customers</h3>
            <h2 style="color: #2c3e50;">{churn_count:,}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="color: #9b59b6;">Churn Rate</h3>
            <h2 style="color: #2c3e50;">{churn_rate:.1f}%</h2>
        </div>
        """, unsafe_allow_html=True)

    # Professional expandable sections
    with st.expander("Dataset Preview", expanded=False):
        st.dataframe(df1, use_container_width=True, height=400)
    
    with st.expander("Summary Statistics", expanded=False):
        num_columns = df1[['tenure','MonthlyCharges']]
        st.dataframe(num_columns.describe(), use_container_width=True)

    with st.expander("Exploratory Data Analysis", expanded=True):
        col1, col2 = st.columns(2)

        with col1:
            # Professional churn distribution
            churn_counts = df1['Churn'].value_counts()
            fig_churn = px.pie(values=churn_counts.values, 
                              names=churn_counts.index,
                              title="Customer Churn Distribution",
                              color_discrete_sequence=['#3498db', '#e74c3c'])
            fig_churn.update_layout(title_font_size=16, font_size=12)
            st.plotly_chart(fig_churn, use_container_width=True)
        
        with col2:
            # Internet service analysis
            fig_internet = px.bar(df1, x='InternetService', color='Churn', 
                                title="Internet Service Distribution by Churn",
                                color_discrete_sequence=['#27ae60', '#e74c3c'])
            fig_internet.update_layout(title_font_size=16)
            st.plotly_chart(fig_internet, use_container_width=True)

        col3, col4 = st.columns(2)

        with col3:
            # Tenure distribution
            fig_tenure = px.histogram(df1, x='tenure', color='Churn', 
                                    title="Customer Tenure Distribution by Churn",
                                    barmode='overlay', opacity=0.7,
                                    color_discrete_sequence=['#27ae60', '#e74c3c'])
            fig_tenure.update_layout(title_font_size=16)
            st.plotly_chart(fig_tenure, use_container_width=True) 

        with col4:
            # Monthly charges
            fig_charges = px.box(df1, x='Churn', y='MonthlyCharges',
                               title="Monthly Charges by Churn Status",
                               color='Churn',
                               color_discrete_sequence=['#27ae60', '#e74c3c'])
            fig_charges.update_layout(title_font_size=16)
            st.plotly_chart(fig_charges, use_container_width=True)

        # Contract analysis
        contract_churn = df1.groupby(['Contract', 'Churn']).size().reset_index(name='Count')
        fig_contract = px.bar(contract_churn, x='Contract', y='Count', color='Churn',
                 title="Churn Analysis by Contract Type", barmode='group',
                 color_discrete_sequence=['#27ae60', '#e74c3c'])
        fig_contract.update_layout(title_font_size=18, font_size=12)
        st.plotly_chart(fig_contract, use_container_width=True)

def page2():
    st.markdown("## Data Preprocessing")

    if 'df1' not in st.session_state or st.session_state.df1 is None:
        st.markdown("""
        <div class="insight-box">
            <h4>No Data Available</h4>
            <p>Please upload your dataset first from the Home page.</p>
        </div>
        """, unsafe_allow_html=True)
        return

    df1 = st.session_state.df1

    # Professional tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Data Quality Check", "Data Types Analysis", "Process Data", "Correlation Analysis"])

    with tab1:
        st.markdown("### Missing Values Analysis")
        
        df_clean = df1.copy()
        null_placeholders = ["", " ", "NA", "N/A", "null", "Null", "NaN", "-", "--"]
        df_clean.replace(to_replace=null_placeholders, value=np.nan, inplace=True)
        st.session_state['df1'] = df_clean

        missing_count = df_clean.isna().sum()
        missing_percent = (missing_count / len(df_clean)) * 100
        missing_df = pd.DataFrame({
            "Missing Values": missing_count,
            "Percent Missing": missing_percent.round(2)
        })
        missing_df = missing_df[missing_df["Missing Values"] > 0]

        if not missing_df.empty:
            st.warning(f"Found {missing_df.shape[0]} columns with missing values.")
            st.dataframe(missing_df, use_container_width=True)
        else:
            st.success("No missing values found!")

    with tab2:
        st.markdown("### Data Types Overview")
        data_types_df = pd.DataFrame({
            'Column': df1.columns,
            'Data Type': df1.dtypes,
            'Unique Values': [df1[col].nunique() for col in df1.columns],
            'Example Values': [str(df1[col].unique()[:3])[1:-1] for col in df1.columns]
        })
        st.dataframe(data_types_df, use_container_width=True)

    with tab3:
        st.markdown("### Data Preprocessing")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            if st.button("Start Processing", type="primary"):
                with st.spinner("Processing data..."):
                    processed_data, label_encoders = preprocess_data(df1)
                    st.session_state.processed_data = processed_data
                    st.session_state.label_encoders = label_encoders
                st.success("Processing completed successfully!")
        
        with col2:
            if st.session_state.processed_data is not None:
                st.info("Processing Summary: TotalCharges converted, missing values handled, categorical encoding applied")

        if st.session_state.processed_data is not None:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Original Data:**")
                st.dataframe(df1.head(), use_container_width=True)
            with col2:
                st.markdown("**Processed Data:**")
                st.dataframe(st.session_state.processed_data.head(), use_container_width=True)

    with tab4:
        st.markdown("### Feature Correlation Analysis")
        
        if st.session_state.processed_data is not None:
            correlation_matrix_num = st.session_state.processed_data.select_dtypes(include='number')
            correlation_matrix = correlation_matrix_num.corr()

            fig = px.imshow(
                correlation_matrix, 
                text_auto=True, 
                aspect="auto",
                title="Feature Correlation Heatmap",
                color_continuous_scale='RdBu_r'
            )
            fig.update_layout(height=700, title_font_size=18)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Please process the data first to view correlations.")

def page3():
    st.markdown("## Model Training")
    
    if st.session_state.processed_data is None:
        st.markdown("""
        <div class="insight-box">
            <h4>Data Not Processed</h4>
            <p>Please complete data preprocessing first.</p>
        </div>
        """, unsafe_allow_html=True)
        return

    df2 = st.session_state.processed_data

    # Professional model configuration
    st.markdown("### Model Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h4 style="color: #27ae60;">Random Forest Parameters</h4>
        </div>
        """, unsafe_allow_html=True)
        
        n_estimators = st.slider("Number of Trees", 10, 200, 100, 10)
        max_depth = st.slider("Max Depth", 3, 20, 10)
        
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h4 style="color: #e74c3c;">SVM Parameters</h4>
        </div>
        """, unsafe_allow_html=True)
        
        svm_kernel = st.selectbox("Kernel", ['rbf', 'linear', 'poly'])
        svm_C = st.slider("Regularization (C)", 0.1, 10.0, 1.0, 0.1)

    # Train-test split configuration
    col1, col2 = st.columns(2)
    with col1:
        test_size = st.slider("Test Size", 0.1, 0.5, 0.2, 0.05)
    with col2:
        random_state = st.number_input("Random State", value=40, min_value=0)

    # Feature information
    x_predict = df2.drop(['customerID', 'Churn'], axis=1)
    y_output = df2['Churn']
    
    st.info(f"Training with {x_predict.shape[1]} features and {x_predict.shape[0]} samples")
    
    x_train, x_test, y_train, y_test = train_test_split(
        x_predict, y_output, test_size=test_size, random_state=random_state, stratify=y_output
    )

    # Professional training button
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("Train Models", type="primary"):
            with st.spinner("Training models... Please wait..."):
                
                # Preprocessing
                imputer = SimpleImputer(strategy='most_frequent')
                x_train_imputed = imputer.fit_transform(x_train)
                x_test_imputed = imputer.transform(x_test)

                scaler = StandardScaler()
                x_train_scaled = scaler.fit_transform(x_train_imputed)
                x_test_scaled = scaler.transform(x_test_imputed)

                # Train models
                rf_model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)
                rf_model.fit(x_train_imputed, y_train)

                svm_model = SVC(kernel=svm_kernel, C=svm_C, random_state=random_state, probability=True)
                svm_model.fit(x_train_scaled, y_train)

                # Store results
                st.session_state.models = {'Random Forest': rf_model, 'SVM': svm_model}
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

    # Model information
    if st.session_state.models:
        st.markdown("### Model Summary")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            <div class="insight-box">
                <h4>Random Forest</h4>
                <ul>
                    <li>Ensemble method using multiple decision trees</li>
                    <li>Handles feature interactions well</li>
                    <li>Provides feature importance scores</li>
                    <li>Less prone to overfitting</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
        with col2:
            st.markdown("""
            <div class="insight-box">
                <h4>Support Vector Machine</h4>
                <ul>
                    <li>Finds optimal decision boundary</li>
                    <li>Works well with high-dimensional data</li>
                    <li>Uses kernel trick for non-linear patterns</li>
                    <li>Requires feature scaling</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

        # Feature importance
        if 'Random Forest' in st.session_state.models:
            st.markdown("### Feature Importance Analysis")
            rf_model = st.session_state.models['Random Forest']
            feature_importance = pd.DataFrame({
                'Feature': st.session_state.feature_names,
                'Importance': rf_model.feature_importances_
            }).sort_values('Importance', ascending=False)

            fig_importance = px.bar(
                feature_importance.head(10),
                x='Importance', y='Feature',
                orientation='h',
                title="Top 10 Most Important Features",
                color='Importance',
                color_continuous_scale='viridis'
            )
            fig_importance.update_layout(yaxis={'categoryorder': 'total ascending'}, title_font_size=16)
            st.plotly_chart(fig_importance, use_container_width=True)

def page4():
    st.markdown("## Model Evaluation")
    
    if not st.session_state.models:
        st.markdown("""
        <div class="insight-box">
            <h4>No Models Available</h4>
            <p>Please train models first in the Model Training section.</p>
        </div>
        """, unsafe_allow_html=True)
        return
        
    # Calculate metrics
    results = {}
    
    for model_name, model in st.session_state.models.items():
        X_test_input = st.session_state.X_test if model_name == 'Random Forest' else st.session_state.X_test_scaled
        
        y_pred = model.predict(X_test_input)
        y_pred_proba = model.predict_proba(X_test_input)[:, 1]
        
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
    
    st.session_state.model_metrics = results
    
    # Professional metrics display
    st.markdown("### Performance Comparison")
    
    metrics_df = pd.DataFrame({
        model_name: result['metrics'] 
        for model_name, result in results.items()
    }).T
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.dataframe(metrics_df.round(4), use_container_width=True)
    
    with col2:
        # Best model recommendation
        overall_scores = {}
        for model_name, result in results.items():
            metrics = result['metrics']
            overall_score = sum(metrics.values()) / len(metrics)
            overall_scores[model_name] = overall_score
        
        best_model = max(overall_scores, key=overall_scores.get)
        
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="color: #27ae60;">Best Performing Model</h3>
            <h2 style="color: #2c3e50;">{best_model}</h2>
            <p>Overall Score: {overall_scores[best_model]:.4f}</p>
        </div>
        """, unsafe_allow_html=True)

    # Visual comparison
    fig_metrics = px.bar(metrics_df.reset_index(), 
                       x='index', y=['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC'],
                       title="Model Performance Metrics Comparison",
                       barmode='group',
                       color_discrete_sequence=['#3498db', '#e74c3c', '#27ae60', '#f39c12', '#9b59b6'])
    fig_metrics.update_layout(xaxis_title="Models", yaxis_title="Score", title_font_size=18)
    st.plotly_chart(fig_metrics, use_container_width=True)
    
    # Professional tabs for detailed analysis
    tab1, tab2, tab3 = st.tabs(["Confusion Matrix", "ROC Curves", "Classification Reports"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        for i, (model_name, result) in enumerate(results.items()):
            cm = confusion_matrix(st.session_state.y_test, result['predictions'])
            
            fig_cm = px.imshow(cm, text_auto=True, aspect="auto",
                              title=f"{model_name} Confusion Matrix",
                              labels=dict(x="Predicted", y="Actual"),
                              x=['No Churn', 'Churn'],
                              y=['No Churn', 'Churn'],
                              color_continuous_scale='Blues')
            fig_cm.update_layout(title_font_size=16)
            
            if i == 0:
                col1.plotly_chart(fig_cm, use_container_width=True)
            else:
                col2.plotly_chart(fig_cm, use_container_width=True)
    
    with tab2:
        fig_roc = go.Figure()
        
        colors = ['#3498db', '#e74c3c']
        for i, (model_name, result) in enumerate(results.items()):
            fpr, tpr, _ = roc_curve(st.session_state.y_test, result['probabilities'])
            auc_score = auc(fpr, tpr)
            
            fig_roc.add_trace(go.Scatter(
                x=fpr, y=tpr,
                mode='lines',
                name=f'{model_name} (AUC = {auc_score:.3f})',
                line=dict(width=3, color=colors[i])
            ))
        
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
            title_font_size=18,
            height=500
        )
        
        st.plotly_chart(fig_roc, use_container_width=True)
    
    with tab3:
        for model_name, result in results.items():
            with st.expander(f"{model_name} Classification Report"):
                report = classification_report(
                    st.session_state.y_test, 
                    result['predictions'], 
                    output_dict=True
                )
                report_df = pd.DataFrame(report).transpose()
                st.dataframe(report_df.round(4), use_container_width=True)

def page5():
    st.markdown("## Prediction Interface")

    if not st.session_state.models:
        st.markdown("""
        <div class="insight-box">
            <h4>Models Not Available</h4>
            <p>Please train models first in the Model Training section.</p>
        </div>
        """, unsafe_allow_html=True)
        return

    st.markdown("""
    <div class="team-card">
        <h3>Customer Information Input</h3>
        <p>Enter customer details below to predict churn probability</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Professional input form
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
        
        predict_button = st.form_submit_button("Predict Churn Risk", type="primary")
    
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
            input_scaled = st.session_state.scaler.transform(input_df)
            prediction = model.predict(input_scaled)[0]
            probability = model.predict_proba(input_scaled)[0]
        else:  # Random Forest
            prediction = model.predict(input_df)[0]
            probability = model.predict_proba(input_df)[0]
        
        # Professional prediction results
        st.markdown("### Prediction Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            churn_prob = probability[1] * 100
            if prediction == 1:
                st.markdown(f"""
                <div class="status-high">
                    <h2>HIGH RISK</h2>
                    <h3>Customer Likely to Churn</h3>
                    <h1>{churn_prob:.1f}%</h1>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="status-low">
                    <h2>LOW RISK</h2>
                    <h3>Customer Likely to Stay</h3>
                    <h1>{churn_prob:.1f}%</h1>
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            # Professional risk gauge
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = churn_prob,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Churn Probability (%)"},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "#34495e"},
                    'steps': [
                        {'range': [0, 30], 'color': "#27ae60"},
                        {'range': [30, 70], 'color': "#f39c12"},
                        {'range': [70, 100], 'color': "#e74c3c"}],
                    'threshold': {
                        'line': {'color': "#e74c3c", 'width': 4},
                        'thickness': 0.75,
                        'value': 70}}))
            fig_gauge.update_layout(height=300)
            st.plotly_chart(fig_gauge, use_container_width=True)
        
        with col3:
            # Professional recommendations
            if churn_prob > 70:
                st.markdown("""
                <div class="insight-box">
                    <h4>URGENT ACTIONS REQUIRED</h4>
                    <ul>
                        <li>Contact customer within 24 hours</li>
                        <li>Offer personalized retention package</li>
                        <li>Investigate service issues immediately</li>
                        <li>Consider contract upgrade incentives</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            elif churn_prob > 40:
                st.markdown("""
                <div class="insight-box">
                    <h4>PROACTIVE MEASURES RECOMMENDED</h4>
                    <ul>
                        <li>Send customer satisfaction survey</li>
                        <li>Offer service upgrades or add-ons</li>
                        <li>Provide loyalty rewards or discounts</li>
                        <li>Monitor usage patterns closely</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="insight-box">
                    <h4>MAINTENANCE ACTIONS</h4>
                    <ul>
                        <li>Continue providing excellent service</li>
                        <li>Explore upselling opportunities</li>
                        <li>Conduct regular satisfaction checks</li>
                        <li>Consider referral program enrollment</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
        
        # Feature impact analysis
        if selected_model == "Random Forest":
            st.markdown("### Key Factors Influencing This Prediction")
            
            rf_model = st.session_state.models['Random Forest']
            feature_importance = pd.DataFrame({
                'Feature': st.session_state.feature_names,
                'Importance': rf_model.feature_importances_,
                'Customer_Value': [input_data[feature] for feature in st.session_state.feature_names]
            }).sort_values('Importance', ascending=False).head(8)
            
            fig_factors = px.bar(feature_importance, 
                               x='Importance', y='Feature',
                               orientation='h',
                               title="Most Influential Factors for This Customer",
                               color='Importance',
                               color_continuous_scale='viridis')
            fig_factors.update_layout(yaxis={'categoryorder':'total ascending'}, title_font_size=16)
            st.plotly_chart(fig_factors, use_container_width=True)

def page6():
    st.markdown("## Business Insights & Strategic Recommendations")
    
    if st.session_state.df1 is None:
        st.markdown("""
        <div class="insight-box">
            <h4>No Data Available</h4>
            <p>Please upload your dataset to view business insights.</p>
        </div>
        """, unsafe_allow_html=True)
        return

    data = st.session_state.df1

    # Executive summary
    total_customers1 = len(data)
    churned_customers1 = len(data[data['Churn'] == 'Yes'])
    monthly_revenue_lost1 = data[data['Churn'] == 'Yes']['MonthlyCharges'].sum()
    annual_revenue_lost1 = monthly_revenue_lost1 * 12
    churn_rate1 = (data['Churn'] == 'Yes').mean() * 100
    
    st.markdown(f"""
    <div class="team-card">
        <h2>Executive Summary</h2>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; margin-top: 1rem;">
            <div>Analyzed {len(data):,} customer records</div>
            <div>Identified {churn_rate1:.1f}% churn rate</div>
            <div>${annual_revenue_lost1:,.0f} annual revenue at risk</div>
            <div>Achieved high prediction accuracy</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Key metrics
    churn_rate = (data['Churn'] == 'Yes').mean() * 100
    avg_tenure_churn = data[data['Churn'] == 'Yes']['tenure'].mean()
    avg_tenure_stay = data[data['Churn'] == 'No']['tenure'].mean()
    avg_monthly_churn = data[data['Churn'] == 'Yes']['MonthlyCharges'].mean()
    avg_monthly_stay = data[data['Churn'] == 'No']['MonthlyCharges'].mean()
    
    col1, col2, col3, col4 = st.columns(4)
    
    metrics_data = [
        ("Churn Rate", f"{churn_rate:.1f}%", "#e74c3c"),
        ("Avg Tenure (Churned)", f"{avg_tenure_churn:.1f} months", "#f39c12"),
        ("Avg Monthly (Churned)", f"${avg_monthly_churn:.2f}", "#9b59b6"),
        ("Annual Revenue Risk", f"${len(data[data['Churn'] == 'Yes']) * avg_monthly_churn * 12:,.0f}", "#34495e")
    ]
    
    for i, (metric, value, color) in enumerate(metrics_data):
        with [col1, col2, col3, col4][i]:
            st.markdown(f"""
            <div class="metric-card">
                <h4 style="color: {color};">{metric}</h4>
                <h2 style="color: #2c3e50;">{value}</h2>
            </div>
            """, unsafe_allow_html=True)

    # Professional analysis tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Churn Drivers", "Financial Impact", "Strategic Recommendations", "Model Performance"])
    
    with tab1:
        st.markdown("### Primary Churn Drivers Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            contract_churn = data.groupby('Contract')['Churn'].apply(lambda x: (x == 'Yes').mean() * 100)
            fig_contract = px.bar(x=contract_churn.index, y=contract_churn.values,
                                title="Churn Rate by Contract Type",
                                labels={'x': 'Contract Type', 'y': 'Churn Rate (%)'},
                                color=contract_churn.values,
                                color_continuous_scale='Reds')
            fig_contract.update_layout(title_font_size=16)
            st.plotly_chart(fig_contract, use_container_width=True)
        
        with col2:
            payment_churn = data.groupby('PaymentMethod')['Churn'].apply(lambda x: (x == 'Yes').mean() * 100)
            fig_payment = px.bar(x=payment_churn.index, y=payment_churn.values,
                               title="Churn Rate by Payment Method",
                               labels={'x': 'Payment Method', 'y': 'Churn Rate (%)'},
                               color=payment_churn.values,
                               color_continuous_scale='Reds')
            fig_payment.update_xaxes(tickangle=45)
            fig_payment.update_layout(title_font_size=16)
            st.plotly_chart(fig_payment, use_container_width=True)
        
        st.markdown("""
        <div class="insight-box">
            <h4>Key Findings</h4>
            <ul>
                <li>Month-to-month contracts show significantly higher churn risk</li>
                <li>Electronic check payments correlate with increased churn probability</li>
                <li>Fiber optic customers demonstrate varied retention patterns</li>
                <li>Senior citizens exhibit distinct churn behaviors requiring targeted approaches</li>
                <li>New customers (low tenure) represent the highest risk segment</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with tab2:
        st.markdown("### Financial Impact Analysis")
        
        # Financial calculations
        total_customers = len(data)
        churned_customers = len(data[data['Churn'] == 'Yes'])
        monthly_revenue_lost = data[data['Churn'] == 'Yes']['MonthlyCharges'].sum()
        annual_revenue_lost = monthly_revenue_lost * 12
        estimated_cac = 200
        total_acquisition_cost = churned_customers * estimated_cac
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h4>Financial Impact Summary</h4>
                <ul>
                    <li>Customers Lost: {churned_customers:,}</li>
                    <li>Monthly Revenue Lost: ${monthly_revenue_lost:,.2f}</li>
                    <li>Annual Revenue Lost: ${annual_revenue_lost:,.2f}</li>
                    <li>Customer Acquisition Cost: ${total_acquisition_cost:,.2f}</li>
                    <li><strong>Total Annual Impact: ${annual_revenue_lost + total_acquisition_cost:,.2f}</strong></li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            impact_data = pd.DataFrame({
                'Category': ['Revenue Lost', 'Acquisition Cost'],
                'Amount': [annual_revenue_lost, total_acquisition_cost]
            })
            
            fig_impact = px.pie(impact_data, values='Amount', names='Category',
                              title="Annual Financial Impact Breakdown",
                              color_discrete_sequence=['#e74c3c', '#f39c12'])
            fig_impact.update_layout(title_font_size=16)
            st.plotly_chart(fig_impact, use_container_width=True)
        
        # ROI scenarios
        st.markdown("### Return on Investment Potential")
        
        retention_scenarios = pd.DataFrame({
            'Scenario': ['5% Reduction', '10% Reduction', '15% Reduction'],
            'Customers_Saved': [churned_customers * 0.05, churned_customers * 0.10, churned_customers * 0.15],
            'Annual_Savings': [annual_revenue_lost * 0.05, annual_revenue_lost * 0.10, annual_revenue_lost * 0.15]
        })
        
        fig_roi = px.bar(retention_scenarios, x='Scenario', y='Annual_Savings',
                       title="Potential Annual Savings from Churn Reduction",
                       color='Annual_Savings',
                       color_continuous_scale='Greens')
        fig_roi.update_layout(yaxis_title="Annual Savings ($)", title_font_size=16)
        st.plotly_chart(fig_roi, use_container_width=True)
    
    with tab3:
        st.markdown("### Strategic Recommendations")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class="insight-box">
                <h4>Immediate Actions (0-30 days)</h4>
                <ul>
                    <li>Target high-risk customer segments with personalized outreach</li>
                    <li>Implement automatic payment incentives</li>
                    <li>Establish proactive contact protocols for new customers</li>
                    <li>Design data-driven retention offers</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="insight-box">
                <h4>Medium-term Strategies (1-6 months)</h4>
                <ul>
                    <li>Deploy real-time churn scoring system</li>
                    <li>Launch comprehensive customer success programs</li>
                    <li>Optimize pricing structures for high-risk segments</li>
                    <li>Enhance digital customer experience platforms</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="insight-box">
                <h4>Long-term Initiatives (6+ months)</h4>
                <ul>
                    <li>Develop sophisticated customer lifecycle programs</li>
                    <li>Implement continuous model monitoring and updates</li>
                    <li>Create comprehensive value communication strategies</li>
                    <li>Establish advanced loyalty and rewards programs</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
    
    with tab4:
        if st.session_state.model_metrics:
            st.markdown("### Model Performance Analysis")
            
            # Best model identification
            best_model = None
            best_f1 = 0
            
            for model_name, results in st.session_state.model_metrics.items():
                if results['metrics']['F1-Score'] > best_f1:
                    best_f1 = results['metrics']['F1-Score']
                    best_model = model_name
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <h3 style="color: #27ae60;">Best Performing Model</h3>
                    <h2>{best_model}</h2>
                    <p>F1-Score: {best_f1:.4f}</p>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("""
                <div class="insight-box">
                    <h4>Model Capabilities</h4>
                    <ul>
                        <li>High accuracy in identifying churn patterns</li>
                        <li>Balanced precision and recall performance</li>
                        <li>Robust feature importance insights</li>
                        <li>Reliable probability estimation</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown("""
                <div class="insight-box">
                    <h4>Business Implementation Benefits</h4>
                    <ul>
                        <li>Early warning system for at-risk customers</li>
                        <li>Reduced customer acquisition costs</li>
                        <li>Protected recurring revenue streams</li>
                        <li>Enabled personalized retention strategies</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("Model performance metrics will be displayed after completing model training.")

def page7():
    st.markdown("## Batch Prediction Processing")

    if not st.session_state.models:
        st.markdown("""
        <div class="insight-box">
            <h4>Models Not Available</h4>
            <p>Please complete model training before using batch prediction functionality.</p>
        </div>
        """, unsafe_allow_html=True)
        return

    st.markdown("""
    <div class="team-card">
        <h3>Batch Prediction Center</h3>
        <p>Upload a CSV file to generate churn predictions for multiple customers simultaneously</p>
    </div>
    """, unsafe_allow_html=True)

    # Professional file upload
    uploaded_file = st.file_uploader("Upload CSV file for batch prediction", type="csv", 
                                    help="Upload a CSV file containing customer data for bulk predictions")
    
    if uploaded_file:
        user_df = pd.read_csv(uploaded_file)
        
        st.markdown("### Data Preview")
        st.dataframe(user_df.head(), use_container_width=True)
        
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            st.metric("Total Records", len(user_df))
        with col2:
            st.metric("Features", len(user_df.columns))
        with col3:
            selected_model = st.selectbox("Select Model", list(st.session_state.models.keys()))
        
        if st.button("Generate Batch Predictions", type="primary"):
            with st.spinner("Processing predictions..."):
                
                # Prepare data
                if "customerID" in user_df.columns:
                    ids = user_df["customerID"]
                    prediction_df = user_df.drop(columns=["customerID"])
                else:
                    ids = pd.Series([f"Customer-{i+1}" for i in range(len(user_df))])
                    prediction_df = user_df.copy()
                
                try:
                    # Get model and make predictions
                    model = st.session_state.models[selected_model]
                    
                    if selected_model == "SVM":
                        predictions = model.predict(prediction_df)
                        probabilities = model.predict_proba(prediction_df)[:, 1]
                    else:
                        predictions = model.predict(prediction_df)
                        probabilities = model.predict_proba(prediction_df)[:, 1]
                    
                    # Create results
                    results_df = pd.DataFrame({
                        "CustomerID": ids,
                        "Churn_Prediction": ["High Risk" if p == 1 else "Low Risk" for p in predictions],
                        "Churn_Probability": (probabilities * 100).round(1),
                        "Risk_Category": ["High" if p > 0.7 else "Medium" if p > 0.4 else "Low" for p in probabilities]
                    })
                    
                    st.success("Predictions completed successfully!")
                    
                    # Display results
                    st.markdown("### Prediction Results")
                    st.dataframe(results_df, use_container_width=True)
                    
                    # Summary statistics
                    col1, col2, col3, col4 = st.columns(4)
                    
                    high_risk = sum(probabilities > 0.7)
                    medium_risk = sum((probabilities > 0.4) & (probabilities <= 0.7))
                    low_risk = sum(probabilities <= 0.4)
                    avg_prob = probabilities.mean() * 100
                    
                    with col1:
                        st.markdown(f"""
                        <div class="metric-card">
                            <h4 style="color: #e74c3c;">High Risk</h4>
                            <h2>{high_risk}</h2>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown(f"""
                        <div class="metric-card">
                            <h4 style="color: #f39c12;">Medium Risk</h4>
                            <h2>{medium_risk}</h2>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col3:
                        st.markdown(f"""
                        <div class="metric-card">
                            <h4 style="color: #27ae60;">Low Risk</h4>
                            <h2>{low_risk}</h2>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col4:
                        st.markdown(f"""
                        <div class="metric-card">
                            <h4 style="color: #3498db;">Average Probability</h4>
                            <h2>{avg_prob:.1f}%</h2>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Risk distribution visualization
                    risk_dist = pd.DataFrame({
                        'Risk Level': ['Low Risk', 'Medium Risk', 'High Risk'],
                        'Count': [low_risk, medium_risk, high_risk]
                    })
                    
                    fig_risk = px.pie(risk_dist, values='Count', names='Risk Level',
                                    title="Risk Distribution Analysis",
                                    color_discrete_sequence=['#27ae60', '#f39c12', '#e74c3c'])
                    fig_risk.update_layout(title_font_size=18)
                    st.plotly_chart(fig_risk, use_container_width=True)
                    
                    # Download functionality
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        label="Download Results as CSV",
                        data=csv,
                        file_name="churn_predictions.csv",
                        mime="text/csv",
                        type="primary"
                    )
                    
                except Exception as e:
                    st.error(f"Error during prediction: {str(e)}")
                    st.info("Please ensure your CSV file has the same structure as the training data.")

# Page mapping
pages = [page1, page2, page3, page4, page5, page6, page7]

# Display selected page
pages[selected_page_index]()