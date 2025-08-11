import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
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
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

warnings.filterwarnings('ignore')

# ----------------------- Page Config -----------------------
st.set_page_config(
    page_title='Customer Churn Prediction',
    page_icon='📡',
    layout='wide'
)

# ----------------------- Professional Dark Theme -----------------------
def inject_theme():
    st.markdown("""
    <style>
    /* =========================================================
       Professional Dark Theme for Streamlit
       ======================================================= */

    /* -------- Design tokens -------- */
    :root{
      --bg:#0b0f19;            /* app background */
      --surface-1:#0f172a;     /* panels/cards */
      --surface-2:#111827;     /* hover state */
      --text:#e5e7eb;          /* primary text */
      --muted:#9ca3af;         /* secondary text */
      --border:#1f2937;        /* borders */
      --brand:#6366f1;         /* primary */
      --brand-strong:#4f46e5;  /* primary - strong */
      --shadow:0 8px 24px rgba(0,0,0,.45);
      --radius:12px;
    }

    /* Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    /* -------- Base app -------- */
    html, body, .stApp{
      background:var(--bg) !important;
      color:var(--text) !important;
      font-family:'Inter', system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
    }
    .block-container{ padding:0 1rem 2rem 1rem !important; }

    @media (prefers-reduced-motion:no-preference){
      *{ transition:background-color .2s ease, color .2s ease, border-color .2s ease, box-shadow .2s ease, transform .2s ease; }
    }

    /* -------- Header -------- */
    .main-header{
      background:var(--surface-1);
      padding:2rem;
      border-radius:var(--radius);
      margin-bottom:2rem;
      color:var(--text);
      text-align:center;
      border:1px solid var(--border);
      box-shadow:var(--shadow);
    }
    .main-header h1{ margin:0 0 .25rem 0; font-weight:700; }
    .main-header h3{ margin:0; font-weight:500; color:var(--muted); }

    /* -------- Sidebar -------- */
    section[data-testid="stSidebar"]{
      background: linear-gradient(180deg, #0c1324, #0b0f19);
      border-right: 1px solid var(--border);
    }

    /* -------- Buttons -------- */
    .stButton > button{
      color:var(--text);
      background:linear-gradient(180deg, var(--surface-1), var(--surface-2));
      border:1px solid var(--border);
      border-radius:10px;
      padding:.7rem 1.1rem;
      font-weight:600;
      box-shadow:0 4px 14px rgba(0,0,0,.35);
    }
    .stButton > button:hover{
      background:var(--surface-2);
      border-color:#2b3447;
    }
    .stButton > button:focus-visible{
      outline:3px solid rgba(99,102,241,.45);
      outline-offset:1px;
    }

    /* -------- Tabs (page_names highlight) -------- */
    .stTabs [data-baseweb="tab-list"]{
      gap:6px;
      background:var(--surface-1);
      border:1px solid var(--border);
      border-radius:10px;
      padding:.45rem;
    }
    .stTabs [data-baseweb="tab"]{
      position:relative;
      background:transparent;
      color:var(--text);
      border-radius:9px;
      padding:.75rem 1.25rem;
      font-weight:600;
      border:1px solid transparent;
    }
    .stTabs [data-baseweb="tab"]:hover{ background:var(--surface-2); }
    .stTabs [data-baseweb="tab"]:focus-visible{
      outline:3px solid rgba(99,102,241,.45);
      outline-offset:2px;
    }
    .stTabs [aria-selected="true"]{
      background:linear-gradient(180deg, var(--brand), var(--brand-strong)) !important;
      color:#fff !important;
      border-color:transparent !important;
      box-shadow:0 8px 26px rgba(99,102,241,.35);
      transform:translateY(-1px);
    }
    .stTabs [aria-selected="true"]::after{
      content:"";
      position:absolute;
      left:10px; right:10px; bottom:-6px;
      height:3px; border-radius:2px;
      background:var(--brand);
      box-shadow:0 0 0 2px rgba(99,102,241,.15);
    }

    /* -------- Cards -------- */
    .card{
      background:var(--surface-1);
      color:var(--text);
      padding:1.2rem 1.4rem;
      border-radius:var(--radius);
      border:1px solid var(--border);
      box-shadow:var(--shadow);
      margin-bottom:1rem;
    }

    /* -------- Inputs -------- */
    .stSelectbox > div > div,
    .stNumberInput > div > div > input{
      background:var(--surface-1) !important;
      border:1px solid var(--border) !important;
      border-radius:10px !important;
      color:var(--text) !important;
    }

    /* -------- File uploader -------- */
    [data-testid="stFileUploader"] > section{ padding:0 !important; }
    [data-testid="stFileUploaderDropzone"]{
      background:var(--surface-1) !important;
      border:2px dashed var(--brand) !important;
      border-radius:var(--radius) !important;
      color:var(--text) !important;
      box-shadow:var(--shadow);
    }
    [data-testid="stFileUploaderDropzone"]:hover{
      background:var(--surface-2) !important;
      border-color:var(--brand-strong) !important;
    }
    [data-testid="stFileUploader"] label,
    [data-testid="stFileUploader"] small{ color:var(--muted) !important; }

    /* -------- Alerts (custom) -------- */
    .stSuccess{
      background:rgba(16,185,129,.15);
      border:1px solid rgba(16,185,129,.5);
      color:#d1fae5; border-radius:10px; padding:.9rem 1rem;
    }
    .stError{
      background:rgba(239,68,68,.15);
      border:1px solid rgba(239,68,68,.5);
      color:#fee2e2; border-radius:10px; padding:.9rem 1rem;
    }
    .stWarning{
      background:rgba(245,158,11,.15);
      border:1px solid rgba(245,158,11,.5);
      color:#ffedd5; border-radius:10px; padding:.9rem 1rem;
    }
    .stInfo{
      background:rgba(59,130,246,.15);
      border:1px solid rgba(59,130,246,.5);
      color:#dbeafe; border-radius:10px; padding:.9rem 1rem;
    }

    /* -------- Tables -------- */
    .dataframe{
      background:var(--surface-1) !important;
      color:var(--text) !important;
      border:1px solid var(--border) !important;
      box-shadow:var(--shadow);
      border-radius:10px !important;
    }

    /* -------- Expanders -------- */
    .streamlit-expanderHeader{
      background:var(--surface-1) !important;
      color:var(--text) !important;
      border:1px solid var(--border) !important;
      border-radius:10px !important;
      font-weight:600 !important;
      padding:1rem !important;
    }
    .streamlit-expanderHeader:hover{ background:var(--surface-2) !important; }
    .streamlit-expanderContent{
      background:#0c1324 !important;
      border:1px solid var(--border) !important;
      border-top:none !important;
      border-radius:0 0 10px 10px !important;
      color:var(--text) !important;
    }

    /* -------- Typography -------- */
    .stMarkdown, .stText, p, div, span{ color:var(--text) !important; }
    h1, h2, h3, h4, h5, h6{ color:#fff !important; }

    /* -------- Hide Streamlit chrome (optional) -------- */
    #MainMenu{visibility:hidden;}
    footer{visibility:hidden;}
    header{visibility:hidden;}
    </style>
    """, unsafe_allow_html=True)

inject_theme()

# ----------------------- Branding / Header -----------------------
logo_path = "telco_logo.jpg"
col_logo, col_title = st.columns([1,6], vertical_alignment="center")
with col_logo:
    if os.path.exists(logo_path):
        st.image(Image.open(logo_path), width=90)
with col_title:
    st.markdown("""
    <div class="main-header">
      <h1>Customer Churn Prediction Dashboard</h1>
      <h3>Advanced Analytics for Customer Retention</h3>
    </div>
    """, unsafe_allow_html=True)

# ----------------------- Session State -----------------------
if 'df1' not in st.session_state:
    st.session_state.df1 = None
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None
if 'models' not in st.session_state:
    st.session_state.models = {}
if 'model_metrics' not in st.session_state:
    st.session_state.model_metrics = {}

# ----------------------- Sidebar -----------------------
with st.sidebar:
    st.title("📂 Data & Navigation")
    upload_file = st.file_uploader("Upload CSV", type=["csv"])
    st.markdown("---")
    st.markdown("### Pages")

# Process upload
if upload_file is not None:
    try:
        st.session_state.df1 = pd.read_csv(upload_file)
        st.sidebar.success("File uploaded successfully.")
    except Exception as e:
        st.sidebar.error(f"Error reading the file: {e}")

# ----------------------- Helper: Preprocess -----------------------
def preprocess_data(df1: pd.DataFrame):
    processed_data = df1.copy()

    # Clean placeholders -> NaN
    null_placeholders = ["", " ", "NA", "N/A", "null", "Null", "NaN", "-", "--"]
    processed_data.replace(to_replace=null_placeholders, value=np.nan, inplace=True)

    # Convert TotalCharges and impute
    processed_data['TotalCharges'] = pd.to_numeric(processed_data.get('TotalCharges', np.nan), errors='coerce')
    processed_data['TotalCharges'].fillna(processed_data['TotalCharges'].median(), inplace=True)

    # Impute categoricals
    categorical_cols = processed_data.select_dtypes(include=['object']).columns
    if len(categorical_cols) > 0:
        mode_imputer = SimpleImputer(strategy='most_frequent')
        processed_data[categorical_cols] = mode_imputer.fit_transform(processed_data[categorical_cols])

    # Label encode select categorical columns (to stay consistent with existing app logic)
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

# ----------------------- Pages -----------------------
def page1():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 👥 Group 7 Team Members")
    cols = st.columns(5)
    members = [
        "Ruth Mensah - 22253087",
        "Emmanuel Oduro Dwamena - 11410636",
        "Zoe Akua Ohene-Ampofo - 22252412",
        "Sandra Animwaa Bamfo - 22256394",
        "Joshua Kwaku Mensah - 22257672"
    ]
    for c, m in zip(cols, members):
        with c: st.markdown(f"- {m}")
    st.markdown('</div>', unsafe_allow_html=True)

    st.write("### Preview of Uploaded Data")
    if st.session_state.df1 is None:
        st.warning("Please upload a CSV file first.")
        return

    df1 = st.session_state.df1

    # KPIs
    k1, k2, k3, k4 = st.columns(4)
    with k1: st.metric("Total Customers", len(df1))
    with k2: st.metric("Features", len(df1.columns))
    with k3:
        churn_count = (df1['Churn'] == 'Yes').sum() if 'Churn' in df1.columns else 0
        st.metric("Churned Customers", int(churn_count))
    with k4:
        churn_rate = (churn_count / len(df1) * 100) if len(df1) else 0
        st.metric("Churn Rate", f"{churn_rate:.1f}%")

    with st.expander("Dataset Preview", expanded=False):
        st.dataframe(df1, use_container_width=True)

    with st.expander("Summary Statistics", expanded=False):
        if set(['tenure','MonthlyCharges']).issubset(df1.columns):
            num_columns = df1[['tenure','MonthlyCharges']]
            st.dataframe(num_columns.describe(), use_container_width=True)

    # EDA
    st.write("### Exploratory Data Analysis")
    if 'Churn' in df1.columns:
        c1, c2 = st.columns(2)
        with c1:
            churn_counts = df1['Churn'].value_counts()
            fig_churn = px.pie(values=churn_counts.values,
                               names=churn_counts.index,
                               title="Customer Churn Distribution")
            fig_churn.update_layout(template='plotly_dark', legend_title_text='Churn')
            st.plotly_chart(fig_churn, use_container_width=True)
        with c2:
            if 'InternetService' in df1.columns:
                fig_tenure1 = px.bar(df1, x='InternetService', color='Churn',
                                     title="Internet Service Distribution by Churn")
                fig_tenure1.update_layout(template='plotly_dark')
                st.plotly_chart(fig_tenure1, use_container_width=True)

        c3, c4 = st.columns(2)
        with c3:
            if 'tenure' in df1.columns:
                fig_tenure = px.histogram(df1, x='tenure', color='Churn',
                                          title="Tenure Distribution by Churn",
                                          barmode='overlay', opacity=0.7)
                fig_tenure.update_layout(template='plotly_dark')
                st.plotly_chart(fig_tenure, use_container_width=True)
        with c4:
            if set(['Churn','MonthlyCharges']).issubset(df1.columns):
                fig_charges = px.box(df1, x='Churn', y='MonthlyCharges',
                                     title="Monthly Charges by Churn Status", color='Churn')
                fig_charges.update_layout(template='plotly_dark')
                st.plotly_chart(fig_charges, use_container_width=True)

        if set(['Contract','Churn']).issubset(df1.columns):
            contract_churn = df1.groupby(['Contract', 'Churn']).size().reset_index(name='Count')
            fig = px.bar(contract_churn, x='Contract', y='Count', color='Churn',
                         title="Churn by Contract Type", barmode='group')
            fig.update_layout(template='plotly_dark')
            st.plotly_chart(fig, use_container_width=True)

def page2():
    st.subheader("Data Preprocessing")

    if st.checkbox('Check for null values'):
        if st.session_state.df1 is not None:
            df1 = st.session_state.df1.copy()

            # Replace common placeholders with NaN
            null_placeholders = ["", " ", "NA", "N/A", "null", "Null", "NaN", "-", "--"]
            df1.replace(to_replace=null_placeholders, value=np.nan, inplace=True)

            st.session_state['df1'] = df1

            missing_count = df1.isna().sum()
            missing_percent = (missing_count / len(df1)) * 100
            missing_df = pd.DataFrame({
                "Missing Values": missing_count,
                "Percent Missing": missing_percent.round(2)
            })
            missing_df = missing_df[missing_df["Missing Values"] > 0]

            if not missing_df.empty:
                st.warning(f"⚠️ Found {missing_df.shape[0]} columns with missing values.")
                st.dataframe(missing_df, use_container_width=True)
            else:
                st.success("✅ No missing values found!")
        else:
            st.error("No dataset loaded. Please load the dataset first.")

    if st.checkbox('Data Types Overview'):
        if st.session_state.df1 is not None:
            df1= st.session_state.df1
            st.markdown("### Data Types Overview")
            data_types_df = pd.DataFrame({
                'Column': df1.columns,
                'Data Type': df1.dtypes.astype(str),
                'Unique Values': [df1[col].nunique() for col in df1.columns],
                'Example Values': [str(df1[col].unique()[:3])[1:-1] for col in df1.columns]
            })
            st.dataframe(data_types_df, use_container_width=True)
        else:
            st.error("No dataset loaded. Please load the dataset first.")

    if st.checkbox('Preprocess Data'):
        if st.session_state.df1 is not None:
            df1= st.session_state['df1']

            if st.button("Start Preprocessing", type="primary"):
                with st.spinner("Processing data..."):
                    processed_data, label_encoders = preprocess_data(df1)
                    st.session_state.processed_data = processed_data
                    st.session_state.label_encoders = label_encoders

                st.success("Data preprocessing completed!")

                st.markdown("### Preprocessing Summary:")
                st.write("1. Converted TotalCharges to numeric format")
                st.write("2. Handled missing values using median/mode imputation")
                st.write("3. Label-encoded selected categorical variables")

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
        st.markdown("### Feature Correlation Analysis of Processed Data")

        if st.session_state.processed_data is None:
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
            fig.update_layout(template='plotly_dark', height=700)
            st.plotly_chart(fig, use_container_width=True)

def page3():
    st.subheader("Model Training")
    if st.session_state.processed_data is not None:
        df2 = st.session_state.processed_data

        st.markdown("### Feature Selection")
        x_predict = df2.drop([c for c in ['customerID','Churn'] if c in df2.columns], axis=1)
        y_output = df2['Churn']
        st.info(f"Training with {x_predict.shape[1]} features and {x_predict.shape[0]} samples")

        test_size = st.slider("Test Size (proportion)", 0.1, 0.5, 0.2, 0.05)
        random_state = st.number_input("Random State", value=40, min_value=0)

        x_train, x_test, y_train, y_test = train_test_split(
            x_predict, y_output, test_size=test_size, random_state=random_state, stratify=y_output
        )
        st.success(f"Data split: {len(x_train)} training samples, {len(x_test)} testing samples")

        st.markdown("### Model Configuration")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Random Forest Parameters:**")
            n_estimators = st.slider("Number of Trees", 10, 300, 120, 10)
            max_depth = st.slider("Max Depth", 3, 30, 12)
            rf_random_state = st.number_input("RF Random State", value=40, min_value=0)
        with col2:
            st.markdown("**SVM Parameters:**")
            svm_kernel = st.selectbox("Kernel", ['rbf', 'linear', 'poly'])
            svm_C = st.slider("Regularization (C)", 0.1, 10.0, 1.0, 0.1)
            svm_random_state = st.number_input("SVM Random State", value=40, min_value=0)

        if st.button("Train Models", type="primary"):
            with st.spinner("Training models..."):
                imputer = SimpleImputer(strategy='most_frequent')
                x_train_imputed = imputer.fit_transform(x_train)
                x_test_imputed = imputer.transform(x_test)

                scaler = StandardScaler()
                x_train_scaled = scaler.fit_transform(x_train_imputed)
                x_test_scaled = scaler.transform(x_test_imputed)

                rf_model = RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    random_state=rf_random_state
                )
                rf_model.fit(x_train_imputed, y_train)

                svm_model = SVC(
                    kernel=svm_kernel,
                    C=svm_C,
                    random_state=svm_random_state,
                    probability=True
                )
                svm_model.fit(x_train_scaled, y_train)

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

        if 'Random Forest' in st.session_state.models:
            st.markdown("### Feature Importance (Random Forest)")
            rf_model = st.session_state.models['Random Forest']
            feature_importance = pd.DataFrame({
                'Feature': st.session_state.feature_names,
                'Importance': rf_model.feature_importances_
            }).sort_values('Importance', ascending=False)

            fig_importance = px.bar(
                feature_importance, x='Importance', y='Feature',
                orientation='h', title="Order of Importance of Features",
            )
            fig_importance.update_layout(yaxis={'categoryorder': 'total ascending'}, template='plotly_dark')
            st.plotly_chart(fig_importance, use_container_width=True)

        if 'SVM' in st.session_state.models:
            svm_model = st.session_state.models['SVM']
            if svm_model.kernel == 'linear':
                st.markdown("### Feature Importance (SVM - Linear Kernel)")
                coef = svm_model.coef_[0]
                feature_importance_svm = pd.DataFrame({
                    'Feature': st.session_state.feature_names,
                    'Importance': np.abs(coef)
                }).sort_values('Importance', ascending=True)

                fig_svm_importance = px.bar(
                    feature_importance_svm, x='Importance', y='Feature',
                    orientation='h', title="Order of Importance of Features"
                )
                fig_svm_importance.update_layout(
                    yaxis=dict(categoryorder='total ascending'),
                    xaxis_title='Absolute Coefficient Value',
                    yaxis_title='Feature',
                    title_x=0.5,
                    template='plotly_dark'
                )
                st.plotly_chart(fig_svm_importance, use_container_width=True)
    else:
        st.error("No processed data available. Please complete the data preprocessing step first.")

def page4():
    st.subheader("Model Evaluation")
    if st.session_state.models:

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
            results[model_name] = {'predictions': y_pred, 'probabilities': y_pred_proba, 'metrics': metrics}

        st.session_state.model_metrics = results

        st.markdown("### Model Performance Comparison")
        metrics_df = pd.DataFrame({ m: r['metrics'] for m, r in results.items() }).T
        st.dataframe(metrics_df.round(4), use_container_width=True)

        fig_metrics = px.bar(metrics_df.reset_index(),
                             x='index', y=['Accuracy','Precision','Recall','F1-Score','ROC-AUC'],
                             title="Model Performance Metrics Comparison", barmode='group')
        fig_metrics.update_layout(xaxis_title="Models", yaxis_title="Score", template='plotly_dark')
        st.plotly_chart(fig_metrics, use_container_width=True)

        st.markdown("### Confusion Matrices")
        col1, col2 = st.columns(2)
        for i, (model_name, result) in enumerate(results.items()):
            cm = confusion_matrix(st.session_state.y_test, result['predictions'])
            fig_cm = px.imshow(cm, text_auto=True, aspect="auto",
                               title=f"Confusion Matrix - {model_name}",
                               labels=dict(x="Predicted", y="Actual"),
                               x=['No Churn', 'Churn'], y=['No Churn', 'Churn'])
            fig_cm.update_layout(template='plotly_dark')
            (col1 if i==0 else col2).plotly_chart(fig_cm, use_container_width=True)

        st.markdown("### ROC Curves Comparison")
        fig_roc = go.Figure()
        for model_name, result in results.items():
            fpr, tpr, _ = roc_curve(st.session_state.y_test, result['probabilities'])
            auc_score = auc(fpr, tpr)
            fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines',
                                         name=f'{model_name} (AUC = {auc_score:.3f})', line=dict(width=3)))
        fig_roc.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines', name='Random Classifier',
                                     line=dict(dash='dash')))
        fig_roc.update_layout(title='ROC Curves Comparison',
                              xaxis_title='False Positive Rate',
                              yaxis_title='True Positive Rate',
                              template='plotly_dark')
        st.plotly_chart(fig_roc, use_container_width=True)

        st.markdown("### Model Recommendation")
        overall_scores = {}
        for model_name, result in results.items():
            m = result['metrics']
            overall_scores[model_name] = np.mean([m['Accuracy'], m['Precision'], m['Recall'], m['F1-Score'], m['ROC-AUC']])
        best_model = max(overall_scores, key=overall_scores.get)
        st.success(f"**Recommended Model: {best_model}**")
        st.info(f"Overall Score: {overall_scores[best_model]:.4f}")

        st.markdown("### Detailed Classification Reports")
        for model_name, result in results.items():
            with st.expander(f"{model_name} Classification Report"):
                report = classification_report(st.session_state.y_test, result['predictions'], output_dict=True)
                report_df = pd.DataFrame(report).transpose()
                st.dataframe(report_df.round(4), use_container_width=True)
    else:
        st.error("No trained models available. Please complete the model training step first.")

def page5():
    st.subheader("Prediction Interface")

    if st.session_state.models:
        st.markdown("### 👤 Enter Customer Information")

        with st.form("prediction_form"):
            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown("**👥 Demographics**")
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

            col4, col5 = st.columns(2)
            with col4:
                st.markdown("**Billing Information**")
                contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
                paperless_billing = st.selectbox("Paperless Billing", ["No", "Yes"])
                payment_method = st.selectbox("Payment Method",
                                             ["Electronic check", "Mailed check",
                                              "Bank transfer (automatic)", "Credit card (automatic)"])

            with col5:
                st.markdown("**Charges**")
                monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=200.0, value=50.0)
                total_charges = st.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0, value=500.0)

            selected_model = st.selectbox("Choose Prediction Model", list(st.session_state.models.keys()))
            predict_button = st.form_submit_button("Predict Churn", type="primary")

        if predict_button:
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
            input_df = pd.DataFrame([input_data])

            model = st.session_state.models[selected_model]
            if selected_model == "SVM":
                input_scaled = st.session_state.scaler.transform(input_df)
                prediction = model.predict(input_scaled)[0]
                probability = model.predict_proba(input_scaled)[0]
            else:
                prediction = model.predict(input_df)[0]
                probability = model.predict_proba(input_df)[0]

            st.markdown("### Prediction Results")
            c1, c2 = st.columns(2)
            with c1:
                if prediction == 1:
                    st.error("**HIGH RISK: Customer likely to CHURN**")
                else:
                    st.success("**LOW RISK: Customer likely to STAY**")
            with c2:
                churn_prob = probability[1] * 100
                st.metric("Churn Probability", f"{churn_prob:.2f}%")

            st.markdown("### Risk Assessment & Recommendations")
            if churn_prob > 70:
                st.markdown("""**HIGH RISK**  
- Contact customer within 24 hours  
- Offer personalized retention package  
- Investigate service issues  
- Consider contract upgrade incentives""")
            elif churn_prob > 40:
                st.markdown("""**MODERATE RISK**  
- Send satisfaction survey  
- Offer service upgrades  
- Provide loyalty rewards  
- Monitor usage patterns""")
            else:
                st.markdown("""**LOW RISK**  
- Continue excellent service  
- Opportunity for upselling  
- Regular satisfaction check  
- Consider referral programs""")

            if selected_model == "Random Forest":
                st.markdown("### Key Factors Influencing This Prediction")
                rf_model = st.session_state.models['Random Forest']
                feature_importance = pd.DataFrame({
                    'Feature': st.session_state.feature_names,
                    'Importance': rf_model.feature_importances_,
                    'Customer_Value': [input_data[feature] for feature in st.session_state.feature_names]
                }).sort_values('Importance', ascending=False).head(8)

                fig_factors = px.bar(feature_importance,
                                     x='Importance', y='Feature', orientation='h',
                                     title="Most Influential Factors for This Customer", color='Importance',
                                     color_continuous_scale='plasma')
                fig_factors.update_layout(yaxis={'categoryorder':'total ascending'}, template='plotly_dark')
                st.plotly_chart(fig_factors, use_container_width=True)
    else:
        st.error("No trained models available. Please complete the model training step first.")

def page6():
    st.subheader("Insights & Conclusions")
    if st.session_state.df1 is not None:
        data = st.session_state.df1

        total_customers1 = len(data)
        churned_customers1 = (data['Churn'] == 'Yes').sum() if 'Churn' in data.columns else 0
        monthly_revenue_lost1 = data.loc[data['Churn'] == 'Yes', 'MonthlyCharges'].sum() if 'MonthlyCharges' in data.columns and 'Churn' in data.columns else 0.0
        annual_revenue_lost1 = monthly_revenue_lost1 * 12
        churn_rate1 = (data['Churn'] == 'Yes').mean() * 100 if 'Churn' in data.columns else 0

        st.markdown("### Executive Summary")
        st.markdown(f"""
- Analyzed {total_customers1:,} customer records with {churn_rate1:.1f}% churn rate  
- Identified ${annual_revenue_lost1:,.0f} in annual revenue at risk  
- Achieved high accuracy in churn prediction with actionable insights  
- Enabled proactive customer retention with potential 5–15% churn reduction
""")

        st.markdown("### Key Business Insights")
        if set(['Churn','tenure','MonthlyCharges']).issubset(data.columns):
            churn_rate = (data['Churn'] == 'Yes').mean() * 100
            avg_tenure_churn = data.loc[data['Churn'] == 'Yes','tenure'].mean()
            avg_tenure_stay = data.loc[data['Churn'] == 'No','tenure'].mean()
            avg_monthly_churn = data.loc[data['Churn'] == 'Yes','MonthlyCharges'].mean()
            avg_monthly_stay = data.loc[data['Churn'] == 'No','MonthlyCharges'].mean()

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Overall Churn Rate", f"{churn_rate:.1f}%")
                st.metric("Avg Tenure (Churned)", f"{avg_tenure_churn:.1f} months")
            with col2:
                st.metric("Avg Monthly Charges (Churned)", f"${avg_monthly_churn:.2f}")
                st.metric("Avg Tenure (Retained)", f"{avg_tenure_stay:.1f} months")
            with col3:
                st.metric("Avg Monthly Charges (Retained)", f"${avg_monthly_stay:.2f}")
                revenue_at_risk = churned_customers1 * (avg_monthly_churn if not np.isnan(avg_monthly_churn) else 0) * 12
                st.metric("Annual Revenue at Risk", f"${revenue_at_risk:,.2f}")

        st.markdown("### Detailed Analysis")
        insights_tabs = st.tabs(["Churn Drivers", "Business Impact", "Recommendations", "Model Performance"])

        with insights_tabs[0]:
            st.markdown("### Primary Churn Drivers Identified:")
            if set(['Contract','Churn']).issubset(data.columns):
                contract_churn = data.groupby('Contract')['Churn'].apply(lambda x: (x == 'Yes').mean() * 100)
                fig_contract = px.bar(x=contract_churn.index, y=contract_churn.values,
                                      title="Churn Rate by Contract Type",
                                      labels={'x':'Contract Type','y':'Churn Rate (%)'},
                                      color=contract_churn.values, color_continuous_scale='Reds')
                fig_contract.update_layout(template='plotly_dark')
                st.plotly_chart(fig_contract, use_container_width=True)
            if set(['PaymentMethod','Churn']).issubset(data.columns):
                payment_churn = data.groupby('PaymentMethod')['Churn'].apply(lambda x: (x == 'Yes').mean() * 100)
                fig_payment = px.bar(x=payment_churn.index, y=payment_churn.values,
                                     title="Churn Rate by Payment Method",
                                     labels={'x':'Payment Method','y':'Churn Rate (%)'},
                                     color=payment_churn.values, color_continuous_scale='Reds')
                fig_payment.update_xaxes(tickangle=45)
                fig_payment.update_layout(template='plotly_dark')
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
            churned_customers = (data['Churn'] == 'Yes').sum() if 'Churn' in data.columns else 0
            monthly_revenue_lost = data.loc[data['Churn'] == 'Yes','MonthlyCharges'].sum() if set(['Churn','MonthlyCharges']).issubset(data.columns) else 0.0
            annual_revenue_lost = monthly_revenue_lost * 12
            estimated_cac = 200
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
                impact_data = pd.DataFrame({'Category':['Revenue Lost','Acquisition Cost'],
                                            'Amount':[annual_revenue_lost, total_acquisition_cost]})
                fig_impact = px.pie(impact_data, values='Amount', names='Category',
                                    title="Annual Financial Impact Breakdown")
                fig_impact.update_layout(template='plotly_dark')
                st.plotly_chart(fig_impact, use_container_width=True)

        with insights_tabs[2]:
            st.markdown("### Strategic Recommendations:")
            st.markdown("""
**Immediate (0–30 days)**
1. Target month-to-month contract customers
2. Incentivize automatic payment methods
3. Proactive outreach for tenure < 6 months
4. Personalized retention packages

**Medium-term (1–6 months)**
1. Implement real-time churn scoring
2. Customer success program for new customers
3. Review pricing structure for fiber services
4. Enhance digital self-service tools

**Long-term (6+ months)**
1. Lifecycle-based segmentation programs
2. Continuous model monitoring & retraining
3. Communicate value & service benefits clearly
4. Loyalty & referral programs
""")
            churned = (data['Churn'] == 'Yes').sum() if 'Churn' in data.columns else 0
            monthly_lost = data.loc[data['Churn'] == 'Yes','MonthlyCharges'].sum() if set(['Churn','MonthlyCharges']).issubset(data.columns) else 0.0
            annual_lost = monthly_lost * 12
            retention_scenarios = pd.DataFrame({
                'Scenario':['5% Reduction','10% Reduction','15% Reduction'],
                'Annual_Savings':[annual_lost*0.05, annual_lost*0.10, annual_lost*0.15]
            })
            fig_roi = px.bar(retention_scenarios, x='Scenario', y='Annual_Savings',
                             title="Potential Annual Savings from Churn Reduction",
                             color='Annual_Savings', color_continuous_scale='Blues')
            fig_roi.update_layout(yaxis_title="Annual Savings ($)", template='plotly_dark')
            st.plotly_chart(fig_roi, use_container_width=True)

        with insights_tabs[3]:
            if st.session_state.model_metrics:
                st.markdown("### Model Performance Summary:")
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
- Early Warning: Identify at-risk customers early
- Cost Reduction: Lower acquisition costs
- Revenue Protection: Stabilize recurring revenue
- Personalization: Tailored retention offers
""")
                if len(st.session_state.model_metrics) > 1:
                    st.markdown("### Model Comparison:")
                    comparison_df = pd.DataFrame({
                        'Metric': ['Accuracy','Precision','Recall','F1-Score','ROC-AUC'],
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

    if not st.session_state.models:
        st.error("No trained models available. Please complete the model training step first.")
        st.info("Go to 'Model Training' page and train models before using batch prediction.")
        return

    MODEL_PATH = "random_forest_churn_model.pkl"
    PIPELINE_PATH = "preprocessing_pipeline.pkl"

    @st.cache_data
    def load_training_data():
        df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        df.dropna(inplace=True)
        return df

    @st.cache_resource
    def get_model_and_pipeline():
        if os.path.exists(MODEL_PATH) and os.path.exists(PIPELINE_PATH):
            model = joblib.load(MODEL_PATH)
            pipeline = joblib.load(PIPELINE_PATH)
            return model, pipeline

        df = load_training_data()
        X = df.drop(columns=["customerID", "Churn"])
        y = df["Churn"].map({"Yes": 1, "No": 0})

        numeric_features = ["tenure", "MonthlyCharges", "TotalCharges"]
        categorical_features = X.select_dtypes(include="object").columns.tolist()

        preprocessor = ColumnTransformer(transformers=[
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ]), numeric_features),
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore"))
            ]), categorical_features)
        ])

        pipeline = Pipeline(steps=[("preprocessor", preprocessor)])
        X_preprocessed = pipeline.fit_transform(X)

        model = RandomForestClassifier()
        model.fit(X_preprocessed, y)

        joblib.dump(model, MODEL_PATH)
        joblib.dump(pipeline, PIPELINE_PATH)
        return model, pipeline

    model, pipeline = get_model_and_pipeline()

    uploaded_file = st.file_uploader("Upload a CSV file for batch prediction", type="csv")
    if uploaded_file:
        user_df = pd.read_csv(uploaded_file)
        st.write("📄 Uploaded Data Preview:", user_df.head())

        if "customerID" in user_df.columns:
            ids = user_df["customerID"]
            user_df = user_df.drop(columns=["customerID"])
        else:
            ids = pd.Series([f"ID-{i}" for i in range(len(user_df))])

        X_user = pipeline.transform(user_df)

        predictions = model.predict(X_user)
        proba = model.predict_proba(X_user)[:, 1]

        result_df = pd.DataFrame({
            "CustomerID": ids,
            "Churn_Prediction": ["Yes" if p == 1 else "No" for p in predictions],
            "Churn_Probability": proba.round(3)
        })

        st.success("✅ Predictions Completed")
        st.dataframe(result_df, use_container_width=True)

        csv = result_df.to_csv(index=False).encode('utf-8')
        st.download_button("⬇️ Download Results as CSV", data=csv, file_name="churn_predictions.csv", mime="text/csv")

# ----------------------- Page Router -----------------------
pages = {
    'Home & Data Overview': page1,
    'Data Preprocessing': page2,
    'Model Training': page3,
    'Model Evaluation': page4,
    'Prediction Interface': page5,
    'Insights & Conclusions': page6,
    'Batch Prediction': page7
}

select_page = st.sidebar.selectbox("Select page", list(pages.keys()))
pages[select_page]()
