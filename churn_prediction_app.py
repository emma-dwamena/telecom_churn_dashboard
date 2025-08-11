import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from PIL import Image
import warnings
import os

# ML / Pipeline
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, auc, roc_auc_score,
    precision_score, recall_score, f1_score, accuracy_score
)

warnings.filterwarnings('ignore')

# ----------------------- Page configuration -----------------------
st.set_page_config(
    page_title='Customer Churn Prediction',
    page_icon='📡',
    layout='wide',
    initial_sidebar_state='collapsed'
)

# ----------------------- Professional CSS (fixed header + sticky tabs) -----------------------
st.markdown("""
<style>
  /* Provide space for fixed header */
  .reportview-container .main .block-container { padding-top: 96px; }

  /* Fixed header container */
  .fixed-header {
      position: fixed;
      top: 0; left: 0; right: 0;
      z-index: 1000;
      background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
      padding: 14px 0 10px 0;
      box-shadow: 0 6px 16px rgba(0,0,0,0.18);
  }
  .main-title {
      color: white; text-align: center; margin: 0;
      padding: 0 20px 4px 20px; font-size: 2.1rem; font-weight: 700;
      text-shadow: 1px 2px 4px rgba(0,0,0,0.35);
  }
  .main-subtitle {
      color: #e8f4f8; text-align: center; margin: 0;
      font-size: 1.05rem; font-weight: 400;
  }

  /* Content card */
  .content-container {
      background: white; border-radius: 12px; padding: 26px; margin: 16px 0;
      box-shadow: 0 4px 20px rgba(30, 60, 114, 0.08); border: 1px solid #e3f2fd;
  }

  /* Section headers */
  .section-header {
      background: linear-gradient(135deg, #f8fbff 0%, #e3f2fd 100%);
      padding: 16px 18px; border-radius: 10px; margin-bottom: 18px;
      border-left: 5px solid #2a5298; color: #1e3c72;
  }
  .section-header h2, .section-header h3 { color: #1e3c72; margin: 0; font-weight: 650; }

  /* Metric cards */
  .metric-container {
      background: linear-gradient(135deg, #ffffff 0%, #f8fbff 100%);
      padding: 16px; border-radius: 10px; box-shadow: 0 2px 10px rgba(30, 60, 114, 0.08);
      margin: 8px 0; border: 1px solid #e3f2fd; text-align: center;
  }

  /* Team member */
  .team-member {
      background: linear-gradient(135deg, #ffffff 0%, #f0f7ff 100%);
      padding: 10px; border-radius: 8px; margin: 6px; text-align: center;
      border-left: 3px solid #2a5298; box-shadow: 0 1px 4px rgba(30, 60, 114, 0.1);
      color: #1e3c72; font-size: 0.86rem;
  }

  /* Uploader */
  .upload-container {
      background: linear-gradient(135deg, #e3f2fd 0%, #f0f7ff 100%);
      padding: 22px; border-radius: 12px; margin-bottom: 16px;
      border: 2px dashed #2a5298; text-align: center; color: #1e3c72;
  }

  /* Inputs */
  .stSelectbox > div > div > select,
  .stNumberInput > div > div > input,
  .stTextInput > div > div > input {
      background-color: white; border: 2px solid #e3f2fd; border-radius: 8px; color: #1e3c72;
  }
  .stSelectbox > div > div > select:focus,
  .stNumberInput > div > div > input:focus,
  .stTextInput > div > div > input:focus {
      border-color: #2a5298; box-shadow: 0 0 0 3px rgba(42, 82, 152, 0.12);
  }

  /* Buttons */
  .stButton > button {
      background: linear-gradient(135deg, #2a5298 0%, #1e3c72 100%);
      color: white; border: none; border-radius: 8px; padding: 10px 18px; font-weight: 650;
      transition: all 0.25s ease;
  }
  .stButton > button:hover { transform: translateY(-2px); box-shadow: 0 6px 18px rgba(30, 60, 114, 0.28); }

  /* Tabs (sticky) */
  .stTabs [data-baseweb="tab-list"] {
      gap: 8px; background: #ffffff; border-radius: 10px; padding: 6px 6px;
      position: sticky; top: 72px;  /* sticks beneath header */
      z-index: 999; box-shadow: 0 8px 18px rgba(0,0,0,0.06); border: 1px solid #e6eefc;
  }
  .stTabs [data-baseweb="tab"] {
      background: transparent; border: none; color: #1e3c72;
      padding: 8px 16px; border-radius: 8px; font-weight: 600;
  }
  .stTabs [data-baseweb="tab"]:hover { background: #eff4ff; }
  .stTabs [aria-selected="true"] {
      background: #2a5298; color: white; box-shadow: 0 6px 16px rgba(30, 60, 114, 0.24);
  }

  /* Alerts */
  .stSuccess { background: linear-gradient(135deg, #e8f5e8 0%, #f0f8f0 100%); border-left: 4px solid #4caf50; color: #2e7d32; }
  .stError   { background: linear-gradient(135deg, #ffebee 0%, #fff5f5 100%); border-left: 4px solid #f44336; color: #c62828; }
  .stWarning { background: linear-gradient(135deg, #fff8e1 0%, #fffbf0 100%); border-left: 4px solid #ff9800; color: #e65100; }
  .stInfo    { background: linear-gradient(135deg, #e3f2fd 0%, #f0f7ff 100%); border-left: 4px solid #2196f3; color: #1565c0; }

  /* Hide Streamlit chrome */
  #MainMenu{visibility:hidden;} footer{visibility:hidden;} header{visibility:hidden;}
</style>
""", unsafe_allow_html=True)

# ----------------------- Fixed header -----------------------
st.markdown("""
<div class="fixed-header">
  <h1 class="main-title">Customer Churn Prediction System</h1>
  <p class="main-subtitle">Advanced Analytics for Customer Retention</p>
</div>
""", unsafe_allow_html=True)

# ----------------------- Session state -----------------------
if 'df1' not in st.session_state:
    st.session_state.df1 = None
if 'feature_columns' not in st.session_state:
    st.session_state.feature_columns = None     # X columns used for training
if 'y_name' not in st.session_state:
    st.session_state.y_name = 'Churn'
if 'pipelines' not in st.session_state:
    st.session_state.pipelines = {}             # {'Random Forest': pipeline, 'SVM': pipeline}
if 'model_metrics' not in st.session_state:
    st.session_state.model_metrics = {}
if 'splits' not in st.session_state:
    st.session_state.splits = {}                # store raw X_train, X_test, y_train, y_test

# ----------------------- Upload area -----------------------
st.markdown("""
<div class="upload-container">
  <h3 style="color:#1e3c72;margin-bottom:8px;">📁 Upload Your Dataset</h3>
  <p style="color:#2a5298;margin:0;">Upload a CSV with customer data to begin</p>
</div>
""", unsafe_allow_html=True)

upload_file = st.file_uploader("Choose CSV", type=["csv"], key="main_upload",
                               help="Telco data with columns like tenure, MonthlyCharges, Contract, etc.")

if upload_file is not None:
    try:
        st.session_state.df1 = pd.read_csv(upload_file)
        st.success("✅ File uploaded successfully!")
    except Exception as e:
        st.error(f"❌ Error reading the file: {e}")

# ----------------------- Utilities -----------------------
def clean_base(df: pd.DataFrame) -> pd.DataFrame:
    """Minimal cleaning prior to modeling. Keep categoricals as strings; don't encode here."""
    df = df.copy()
    # Replace common null placeholders
    null_marks = ["", " ", "NA", "N/A", "null", "Null", "NaN", "-", "--"]
    df.replace(to_replace=null_marks, value=np.nan, inplace=True)

    # Convert TotalCharges to numeric if present
    if 'TotalCharges' in df.columns:
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)

    return df

def split_xy(df: pd.DataFrame, y_name: str):
    X = df.drop(columns=[c for c in [y_name, 'customerID'] if c in df.columns])
    if df[y_name].dtype == 'O':
        y = df[y_name].map({'Yes':1, 'No':0}).astype(int)
    else:
        y = df[y_name]
    return X, y

def build_preprocessor(X: pd.DataFrame, scale_numeric: bool) -> ColumnTransformer:
    num_cols = X.select_dtypes(include=['number']).columns.tolist()
    cat_cols = X.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    num_pipeline_steps = [('imputer', SimpleImputer(strategy='median'))]
    if scale_numeric:
        num_pipeline_steps.append(('scaler', StandardScaler()))
    preprocessor = ColumnTransformer([
        ('num', Pipeline(num_pipeline_steps), num_cols),
        ('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')),
                          ('onehot', OneHotEncoder(handle_unknown='ignore'))]), cat_cols)
    ])
    return preprocessor

def get_feature_names(preprocessor: ColumnTransformer) -> list:
    try:
        return preprocessor.get_feature_names_out().tolist()
    except Exception:
        # Fallback for older sklearn
        names = []
        for name, trans, cols in preprocessor.transformers_:
            if hasattr(trans, 'get_feature_names_out'):
                names.extend([f"{name}__{n}" for n in trans.get_feature_names_out(cols)])
            else:
                names.extend([f"{name}__{c}" for c in cols])
        return names

# ----------------------- Tabs (pages) -----------------------
tabs = st.tabs([
    "🏠 Home & Overview", "🔧 Data Preprocessing", "🤖 Model Training",
    "📈 Model Evaluation", "🎯 Prediction Interface", "💡 Insights & Conclusions", "📦 Batch Prediction"
])

# ----------------------- Tab 1: Home -----------------------
with tabs[0]:
    st.markdown('<div class="content-container">', unsafe_allow_html=True)
    st.markdown("""
    <div class="section-header">
      <h2>📊 Dataset Overview</h2>
      <p>Comprehensive analysis of your uploaded customer data</p>
    </div>
    """, unsafe_allow_html=True)

    if st.session_state.df1 is None:
        st.warning("⚠️ Please upload a CSV file first using the upload area above.")
    else:
        df1 = st.session_state.df1.copy()
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f'<div class="metric-container"><h3 style="color:#1e3c72;margin:0;">{len(df1):,}</h3><p style="color:#2a5298;margin:6px 0 0 0;">Total Customers</p></div>', unsafe_allow_html=True)
        with col2:
            st.markdown(f'<div class="metric-container"><h3 style="color:#1e3c72;margin:0;">{len(df1.columns)}</h3><p style="color:#2a5298;margin:6px 0 0 0;">Features</p></div>', unsafe_allow_html=True)
        with col3:
            if 'Churn' in df1.columns:
                churn_count = (df1['Churn'] == 'Yes').sum()
            else:
                churn_count = 0
            st.markdown(f'<div class="metric-container"><h3 style="color:#1e3c72;margin:0;">{churn_count:,}</h3><p style="color:#2a5298;margin:6px 0 0 0;">Churned Customers</p></div>', unsafe_allow_html=True)
        with col4:
            churn_rate = (churn_count / len(df1) * 100) if len(df1) else 0
            st.markdown(f'<div class="metric-container"><h3 style="color:#1e3c72;margin:0;">{churn_rate:.1f}%</h3><p style="color:#2a5298;margin:6px 0 0 0;">Churn Rate</p></div>', unsafe_allow_html=True)

        st.checkbox('📋 Preview Dataset', key='preview_ds', value=False)
        if st.session_state.preview_ds:
            st.dataframe(df1, use_container_width=True)

        st.checkbox('📈 Summary Statistics', key='summ_stats', value=False)
        if st.session_state.summ_stats and set(['tenure','MonthlyCharges']).issubset(df1.columns):
            st.dataframe(df1[['tenure','MonthlyCharges']].describe(), use_container_width=True)

        st.checkbox('🔍 Exploratory Data Analysis', key='eda_toggle', value=False)
        if st.session_state.eda_toggle:
            c1, c2 = st.columns(2)
            if 'Churn' in df1.columns:
                with c1:
                    churn_counts = df1['Churn'].value_counts()
                    fig_churn = px.pie(values=churn_counts.values, names=churn_counts.index,
                                       title="Customer Churn Distribution",
                                       color_discrete_sequence=['#4fc3f7', '#2a5298'])
                    st.plotly_chart(fig_churn, use_container_width=True)
            if 'InternetService' in df1.columns:
                with c2:
                    fig_internet = px.bar(df1, x='InternetService', color='Churn',
                                          title="Internet Service Distribution by Churn",
                                          color_discrete_sequence=['#4fc3f7', '#2a5298'])
                    st.plotly_chart(fig_internet, use_container_width=True)
            c3, c4 = st.columns(2)
            if 'tenure' in df1.columns:
                with c3:
                    fig_tenure = px.histogram(df1, x='tenure', color='Churn', barmode='overlay', opacity=0.7,
                                              title="Customer Tenure Distribution by Churn",
                                              color_discrete_sequence=['#4fc3f7', '#2a5298'])
                    st.plotly_chart(fig_tenure, use_container_width=True)
            if 'MonthlyCharges' in df1.columns and 'Churn' in df1.columns:
                with c4:
                    fig_charges = px.box(df1, x='Churn', y='MonthlyCharges',
                                         title="Monthly Charges by Churn Status", color='Churn',
                                         color_discrete_sequence=['#4fc3f7', '#2a5298'])
                    st.plotly_chart(fig_charges, use_container_width=True)
            if set(['Contract','Churn']).issubset(df1.columns):
                contract_churn = df1.groupby(['Contract', 'Churn']).size().reset_index(name='Count')
                fig = px.bar(contract_churn, x='Contract', y='Count', color='Churn',
                             title="Churn by Contract Type", barmode='group',
                             color_discrete_sequence=['#4fc3f7', '#2a5298'])
                st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ----------------------- Tab 2: Data Preprocessing -----------------------
with tabs[1]:
    st.markdown('<div class="content-container">', unsafe_allow_html=True)
    st.markdown('<div class="section-header"><h2>🔧 Data Preprocessing</h2><p>Clean and prepare your data</p></div>', unsafe_allow_html=True)

    if st.session_state.df1 is None:
        st.error("❌ No dataset loaded. Please upload a CSV first.")
    else:
        if st.checkbox("🔍 Check for null values"):
            df1 = clean_base(st.session_state.df1)
            st.session_state.df1 = df1
            missing = df1.isna().sum()
            missing_pct = (missing / len(df1) * 100).round(2)
            miss_df = pd.DataFrame({"Missing Values": missing, "Percent Missing": missing_pct})
            miss_df = miss_df[miss_df["Missing Values"] > 0]
            if len(miss_df):
                st.warning(f"⚠️ Found {len(miss_df)} columns with missing values.")
                st.dataframe(miss_df, use_container_width=True)
            else:
                st.success("✅ No missing values found after cleaning.")

        if st.checkbox("📊 Data Types Overview"):
            df1 = st.session_state.df1
            info_df = pd.DataFrame({
                'Column': df1.columns,
                'Data Type': df1.dtypes.astype(str),
                'Unique Values': [df1[c].nunique() for c in df1.columns],
                'Example Values': [', '.join(map(str, df1[c].dropna().unique()[:3])) for c in df1.columns]
            })
            st.dataframe(info_df, use_container_width=True)

        if st.checkbox("⚙️ Prepare Target & Preview"):
            df1 = clean_base(st.session_state.df1)
            if 'Churn' not in df1.columns:
                st.error("Target column 'Churn' not found.")
            else:
                y_numeric = df1['Churn'].map({'Yes':1, 'No':0})
                df_preview = df1.copy()
                df_preview['Churn_numeric'] = y_numeric
                st.dataframe(df_preview.head(), use_container_width=True)
                st.info("Note: Encoding will be handled with OneHotEncoder in the training pipeline (no label encoding).")
    st.markdown('</div>', unsafe_allow_html=True)

# ----------------------- Tab 3: Model Training -----------------------
with tabs[2]:
    st.markdown('<div class="content-container">', unsafe_allow_html=True)
    st.markdown('<div class="section-header"><h2>🤖 Model Training</h2><p>Pipelines with OneHotEncoder</p></div>', unsafe_allow_html=True)

    if st.session_state.df1 is None or 'Churn' not in st.session_state.df1.columns:
        st.error("❌ Please upload data with a 'Churn' column first.")
    else:
        df2 = clean_base(st.session_state.df1)
        X, y = split_xy(df2, 'Churn')
        st.session_state.feature_columns = X.columns.tolist()

        st.info(f"📊 Training with {X.shape[1]} features and {X.shape[0]} samples")
        test_size = st.slider("📊 Test Size (proportion)", 0.1, 0.5, 0.2, 0.05)
        random_state = st.number_input("🎲 Random State", value=40, min_value=0)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        st.session_state.splits = {"X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test}
        st.success(f"✅ Data split: {len(X_train)} train / {len(X_test)} test")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**🌳 Random Forest Parameters:**")
            n_estimators = st.slider("Number of Trees", 10, 300, 120, 10)
            max_depth = st.slider("Max Depth", 3, 30, 12)
            rf_rs = st.number_input("RF Random State", value=40, min_value=0)
        with col2:
            st.markdown("**🎯 SVM Parameters:**")
            svm_kernel = st.selectbox("Kernel", ['rbf', 'linear', 'poly'])
            svm_C = st.slider("Regularization (C)", 0.1, 10.0, 1.0, 0.1)
            svm_rs = st.number_input("SVM Random State", value=40, min_value=0)

        if st.button("🚀 Train Pipelines", type="primary"):
            with st.spinner("Training models with OneHotEncoder pipelines..."):
                # Preprocessors
                preproc_rf = build_preprocessor(X_train, scale_numeric=False)
                preproc_svm = build_preprocessor(X_train, scale_numeric=True)
                # Pipelines
                rf_pipe = Pipeline([('preprocess', preproc_rf),
                                    ('model', RandomForestClassifier(n_estimators=n_estimators,
                                                                     max_depth=max_depth,
                                                                     random_state=rf_rs))])
                svm_pipe = Pipeline([('preprocess', preproc_svm),
                                     ('model', SVC(kernel=svm_kernel, C=svm_C, random_state=svm_rs, probability=True))])
                # Fit
                rf_pipe.fit(X_train, y_train)
                svm_pipe.fit(X_train, y_train)

                st.session_state.pipelines = {'Random Forest': rf_pipe, 'SVM': svm_pipe}
                st.success("✅ Models trained successfully!")

        # Feature importance for RF
        if 'Random Forest' in st.session_state.pipelines:
            st.markdown("### 📊 Feature Importance (Random Forest)")
            rf_pipe = st.session_state.pipelines['Random Forest']
            rf = rf_pipe.named_steps['model']
            pre = rf_pipe.named_steps['preprocess']
            feat_names = get_feature_names(pre)
            importances = rf.feature_importances_
            # Align sizes (rare edge cases)
            n = min(len(feat_names), len(importances))
            fi_df = pd.DataFrame({'Feature': feat_names[:n], 'Importance': importances[:n]}).sort_values('Importance', ascending=False).head(25)
            fig_imp = px.bar(fi_df, x='Importance', y='Feature', orientation='h',
                             title="Top 25 Feature Importances", color='Importance', color_continuous_scale='Blues')
            fig_imp.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig_imp, use_container_width=True)

        # SVM linear coefficients
        if 'SVM' in st.session_state.pipelines:
            svm_pipe = st.session_state.pipelines['SVM']
            if svm_kernel == 'linear' and hasattr(svm_pipe.named_steps['model'], 'coef_'):
                st.markdown("### 📊 Feature Importance (SVM - Linear)")
                pre = svm_pipe.named_steps['preprocess']
                feat_names = get_feature_names(pre)
                coef = svm_pipe.named_steps['model'].coef_[0]
                n = min(len(feat_names), len(coef))
                svm_df = pd.DataFrame({'Feature': feat_names[:n], 'Importance': np.abs(coef[:n])}).sort_values('Importance', ascending=True).head(25)
                fig_svm = px.bar(svm_df, x='Importance', y='Feature', orientation='h',
                                 title="Top 25 Absolute Coefficients", color='Importance', color_continuous_scale='Blues')
                fig_svm.update_layout(yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig_svm, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ----------------------- Tab 4: Model Evaluation -----------------------
with tabs[3]:
    st.markdown('<div class="content-container">', unsafe_allow_html=True)
    st.markdown('<div class="section-header"><h2>📈 Model Evaluation</h2><p>Compare trained pipelines</p></div>', unsafe_allow_html=True)

    if not st.session_state.pipelines:
        st.error("❌ No trained models available. Please train models in the previous tab.")
    else:
        X_test = st.session_state.splits['X_test']
        y_test = st.session_state.splits['y_test']

        results = {}
        for name, pipe in st.session_state.pipelines.items():
            y_pred = pipe.predict(X_test)
            y_proba = pipe.predict_proba(X_test)[:, 1]
            results[name] = {
                'predictions': y_pred,
                'probabilities': y_proba,
                'metrics': {
                    'Accuracy': accuracy_score(y_test, y_pred),
                    'Precision': precision_score(y_test, y_pred),
                    'Recall': recall_score(y_test, y_pred),
                    'F1-Score': f1_score(y_test, y_pred),
                    'ROC-AUC': roc_auc_score(y_test, y_proba)
                }
            }
        st.session_state.model_metrics = results

        st.markdown("### 📊 Model Performance Comparison")
        metrics_df = pd.DataFrame({m: r['metrics'] for m, r in results.items()}).T
        st.dataframe(metrics_df.round(4), use_container_width=True)

        fig_metrics = px.bar(metrics_df.reset_index(),
                             x='index', y=['Accuracy','Precision','Recall','F1-Score','ROC-AUC'],
                             title="Model Performance Metrics Comparison", barmode='group',
                             color_discrete_sequence=['#1e3c72', '#2a5298', '#4fc3f7', '#81d4fa', '#b3e5fc'])
        fig_metrics.update_layout(xaxis_title="Models", yaxis_title="Score")
        st.plotly_chart(fig_metrics, use_container_width=True)

        st.markdown("### 🔍 Confusion Matrices")
        col1, col2 = st.columns(2)
        for i, (name, r) in enumerate(results.items()):
            cm = confusion_matrix(y_test, r['predictions'])
            fig_cm = px.imshow(cm, text_auto=True, aspect="auto",
                               title=f"Confusion Matrix - {name}", labels=dict(x="Predicted", y="Actual"),
                               x=['No Churn','Churn'], y=['No Churn','Churn'], color_continuous_scale='Blues')
            (col1 if i == 0 else col2).plotly_chart(fig_cm, use_container_width=True)

        st.markdown("### 📈 ROC Curves Comparison")
        fig_roc = go.Figure()
        colors = ['#1e3c72', '#2a5298']
        for i, (name, r) in enumerate(results.items()):
            fpr, tpr, _ = roc_curve(y_test, r['probabilities'])
            fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines',
                                         name=f'{name} (AUC = {auc(fpr, tpr):.3f})',
                                         line=dict(width=3, color=colors[i % len(colors)])))
        fig_roc.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines', name='Random Classifier',
                                     line=dict(dash='dash', color='gray')))
        fig_roc.update_layout(title='ROC Curves Comparison', xaxis_title='False Positive Rate', yaxis_title='True Positive Rate')
        st.plotly_chart(fig_roc, use_container_width=True)

        st.markdown("### 🏆 Model Recommendation")
        overall = {name: np.mean(list(r['metrics'].values())) for name, r in results.items()}
        best = max(overall, key=overall.get)
        st.success(f"**Recommended Model: {best}**")
        st.info(f"Overall Score: {overall[best]:.4f}")

        st.markdown("### 📋 Detailed Classification Reports")
        for name, r in results.items():
            with st.expander(f"{name} Classification Report"):
                report_df = pd.DataFrame(classification_report(y_test, r['predictions'], output_dict=True)).transpose()
                st.dataframe(report_df.round(4), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ----------------------- Tab 5: Prediction Interface -----------------------
with tabs[4]:
    st.markdown('<div class="content-container">', unsafe_allow_html=True)
    st.markdown('<div class="section-header"><h2>🎯 Prediction Interface</h2><p>Use trained pipeline to score one customer</p></div>', unsafe_allow_html=True)

    if not st.session_state.pipelines:
        st.error("❌ No trained models available. Train models first.")
    else:
        st.markdown("### 📝 Enter Customer Information")
        with st.form("prediction_form"):
            col1, col2, col3 = st.columns(3)
            with col1:
                gender = st.selectbox("Gender", ["Female", "Male"])
                senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
                partner = st.selectbox("Has Partner", ["No", "Yes"])
                dependents = st.selectbox("Has Dependents", ["No", "Yes"])
                tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=12)
            with col2:
                phone_service = st.selectbox("Phone Service", ["No", "Yes"])
                multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
                internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
                online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
                online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
            with col3:
                device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
                tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
                streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
                streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
            col4, col5 = st.columns(2)
            with col4:
                contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
                paperless_billing = st.selectbox("Paperless Billing", ["No", "Yes"])
                payment_method = st.selectbox("Payment Method",
                                              ["Electronic check", "Mailed check",
                                               "Bank transfer (automatic)", "Credit card (automatic)"])
            with col5:
                monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=200.0, value=50.0)
                total_charges = st.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0, value=500.0)

            selected_model = st.selectbox("🤖 Choose Prediction Model", list(st.session_state.pipelines.keys()))
            submit = st.form_submit_button("🎯 Predict Churn", type="primary")

        if submit:
            # Build a single-row DataFrame using original column names (not encoded)
            row = {
                'gender': gender,
                'SeniorCitizen': 1 if senior_citizen == "Yes" else 0,
                'Partner': partner,
                'Dependents': dependents,
                'tenure': tenure,
                'PhoneService': phone_service,
                'MultipleLines': multiple_lines,
                'InternetService': internet_service,
                'OnlineSecurity': online_security,
                'OnlineBackup': online_backup,
                'DeviceProtection': device_protection,
                'TechSupport': tech_support,
                'StreamingTV': streaming_tv,
                'StreamingMovies': streaming_movies,
                'Contract': contract,
                'PaperlessBilling': paperless_billing,
                'PaymentMethod': payment_method,
                'MonthlyCharges': monthly_charges,
                'TotalCharges': total_charges
            }
            X_cols = st.session_state.feature_columns or list(row.keys())
            # ensure same cols order; add any missing with NaN
            for c in X_cols:
                row.setdefault(c, np.nan)
            X_input = pd.DataFrame([row])[X_cols]

            pipe = st.session_state.pipelines[selected_model]
            pred = pipe.predict(X_input)[0]
            proba = pipe.predict_proba(X_input)[0][1] * 100

            c1, c2 = st.columns(2)
            with c1:
                if pred == 1:
                    st.error("**⚠️ HIGH RISK: Customer likely to CHURN**")
                else:
                    st.success("**✅ LOW RISK: Customer likely to STAY**")
            with c2:
                st.markdown(f'<div class="metric-container"><h3 style="color:#1e3c72;margin:0;">{proba:.2f}%</h3><p style="color:#2a5298;margin:6px 0 0 0;">Churn Probability</p></div>', unsafe_allow_html=True)

            # Quick recommendations
            st.markdown("### 📋 Risk Assessment & Recommendations")
            if proba > 70:
                st.markdown("**🚨 HIGH RISK**  \n- Contact within 24 hours  \n- Offer retention incentives  \n- Investigate service issues")
            elif proba > 40:
                st.markdown("**⚠️ MODERATE RISK**  \n- Send satisfaction survey  \n- Offer upgrade options  \n- Monitor usage closely")
            else:
                st.markdown("**✅ LOW RISK**  \n- Maintain service quality  \n- Upsell opportunities  \n- Referral programs")

            # Feature importance for RF context (global, already shown in training tab)

    st.markdown('</div>', unsafe_allow_html=True)

# ----------------------- Tab 6: Insights & Conclusions -----------------------
with tabs[5]:
    st.markdown('<div class="content-container">', unsafe_allow_html=True)
    st.markdown('<div class="section-header"><h2>💡 Insights & Conclusions</h2><p>Business insights and ROI</p></div>', unsafe_allow_html=True)

    if st.session_state.df1 is None:
        st.error("❌ No data available for analysis. Please upload your dataset first.")
    else:
        data = st.session_state.df1.copy()
        if 'Churn' in data.columns:
            total_customers = len(data)
            churned_customers = (data['Churn'] == 'Yes').sum()
            monthly_lost = data.loc[data['Churn'] == 'Yes', 'MonthlyCharges'].sum() if 'MonthlyCharges' in data.columns else 0.0
            annual_lost = monthly_lost * 12
            churn_rate = (data['Churn'] == 'Yes').mean() * 100

            st.markdown("### 📊 Executive Summary")
            st.markdown(f"""
- Analyzed **{total_customers:,}** records with **{churn_rate:.1f}%** churn rate  
- Identified **${annual_lost:,.0f}** in annual revenue at risk  
- Predictive models enable proactive retention and targeted incentives
""")

            st.markdown("### 🔍 Key Business Insights")
            if set(['tenure','MonthlyCharges','Churn']).issubset(data.columns):
                avg_tenure_churn = data.loc[data['Churn']=='Yes','tenure'].mean()
                avg_tenure_stay = data.loc[data['Churn']=='No','tenure'].mean()
                avg_monthly_churn = data.loc[data['Churn']=='Yes','MonthlyCharges'].mean()
                avg_monthly_stay = data.loc[data['Churn']=='No','MonthlyCharges'].mean()

                c1,c2,c3 = st.columns(3)
                with c1:
                    st.metric("Avg Tenure (Churned)", f"{avg_tenure_churn:.1f} mo")
                    st.metric("Avg Tenure (Retained)", f"{avg_tenure_stay:.1f} mo")
                with c2:
                    st.metric("Avg Monthly (Churned)", f"${avg_monthly_churn:.2f}")
                    st.metric("Avg Monthly (Retained)", f"${avg_monthly_stay:.2f}")
                with c3:
                    st.metric("Annual Revenue at Risk", f"${(churned_customers * (avg_monthly_churn or 0) * 12):,.0f}")

            st.markdown("### 📈 Detailed Analysis")
            tabs_ins = st.tabs(["Churn Drivers", "Business Impact", "Recommendations", "Model Performance"])
            with tabs_ins[0]:
                if set(['Contract','Churn']).issubset(data.columns):
                    contract_churn = data.groupby('Contract')['Churn'].apply(lambda x: (x=='Yes').mean()*100)
                    fig_contract = px.bar(x=contract_churn.index, y=contract_churn.values,
                                          title="Churn Rate by Contract Type",
                                          labels={'x':'Contract Type','y':'Churn Rate (%)'},
                                          color=contract_churn.values, color_continuous_scale='Reds')
                    st.plotly_chart(fig_contract, use_container_width=True)
                if set(['PaymentMethod','Churn']).issubset(data.columns):
                    payment_churn = data.groupby('PaymentMethod')['Churn'].apply(lambda x: (x=='Yes').mean()*100)
                    fig_payment = px.bar(x=payment_churn.index, y=payment_churn.values,
                                         title="Churn Rate by Payment Method",
                                         labels={'x':'Payment Method','y':'Churn Rate (%)'},
                                         color=payment_churn.values, color_continuous_scale='Reds')
                    fig_payment.update_xaxes(tickangle=30)
                    st.plotly_chart(fig_payment, use_container_width=True)
            with tabs_ins[1]:
                churned_customers = (data['Churn'] == 'Yes').sum()
                monthly_lost = data.loc[data['Churn'] == 'Yes','MonthlyCharges'].sum() if set(['Churn','MonthlyCharges']).issubset(data.columns) else 0.0
                annual_lost = monthly_lost * 12
                est_cac = 200
                total_acq = churned_customers * est_cac
                col1,col2 = st.columns(2)
                with col1:
                    st.markdown(f"""
**Financial Impact**
- Customers Lost: {churned_customers:,}
- Monthly Revenue Lost: ${monthly_lost:,.2f}
- Annual Revenue Lost: ${annual_lost:,.2f}
- Replacement Cost: ${total_acq:,.2f}
- **Total Annual Impact: ${annual_lost + total_acq:,.2f}**
""")
                with col2:
                    impact_df = pd.DataFrame({'Category':['Revenue Lost','Acquisition Cost'],
                                              'Amount':[annual_lost, total_acq]})
                    fig = px.pie(impact_df, values='Amount', names='Category',
                                 title="Annual Financial Impact Breakdown")
                    st.plotly_chart(fig, use_container_width=True)
            with tabs_ins[2]:
                st.markdown("""
**Immediate (0–30 days)**
1. Target month-to-month contracts
2. Incentivize automatic payments
3. Outreach to tenure < 6 months
4. Personalized retention offers

**Medium-term (1–6 months)**
1. Real-time churn scoring
2. Customer success program
3. Pricing review for fiber services
4. Improve self-service tools

**Long-term (6+ months)**
1. Lifecycle-based segmentation
2. Model monitoring & retraining
3. Communicate value clearly
4. Loyalty & referral programs
""")
            with tabs_ins[3]:
                if st.session_state.model_metrics:
                    st.dataframe(pd.DataFrame({k:v['metrics'] for k,v in st.session_state.model_metrics.items()}).T, use_container_width=True)
                else:
                    st.info("Train and evaluate models to see performance here.")
    st.markdown('</div>', unsafe_allow_html=True)

# ----------------------- Tab 7: Batch Prediction -----------------------
with tabs[6]:
    st.markdown('<div class="content-container">', unsafe_allow_html=True)
    st.markdown('<div class="section-header"><h2>📦 Batch Prediction</h2><p>Score multiple customers at once</p></div>', unsafe_allow_html=True)

    if not st.session_state.pipelines:
        st.error("❌ No trained models available. Train models first.")
    else:
        st.info("Upload a CSV with the same feature columns used during training. Any missing columns will be filled with NA.")
        batch_file = st.file_uploader("Choose CSV file for batch prediction", type="csv", key="batch_upload")
        if batch_file is not None:
            try:
                batch_df = pd.read_csv(batch_file)
                st.success("✅ File uploaded")
                st.dataframe(batch_df.head(), use_container_width=True)

                # Ensure required columns exist and order them
                exp_cols = st.session_state.feature_columns
                for c in exp_cols:
                    if c not in batch_df.columns:
                        batch_df[c] = np.nan
                X_batch = batch_df[exp_cols]

                model_name = st.selectbox("🤖 Choose model", list(st.session_state.pipelines.keys()), key="batch_model_select")
                if st.button("🚀 Run Batch Prediction", type="primary"):
                    pipe = st.session_state.pipelines[model_name]
                    preds = pipe.predict(X_batch)
                    probs = pipe.predict_proba(X_batch)[:,1]

                    out = pd.DataFrame({
                        'Churn_Prediction': np.where(preds==1, 'Yes', 'No'),
                        'Churn_Probability': np.round(probs, 4)
                    })
                    if 'customerID' in batch_df.columns:
                        out.insert(0, 'CustomerID', batch_df['customerID'])

                    st.dataframe(out, use_container_width=True)

                    csv = out.to_csv(index=False).encode('utf-8')
                    st.download_button("📥 Download Results CSV", data=csv, file_name=f"batch_churn_predictions_{model_name.replace(' ','_').lower()}.csv", mime="text/csv")

                    # Quick summaries
                    col1,col2,col3,col4 = st.columns(4)
                    with col1: st.markdown(f'<div class="metric-container"><h3 style="color:#1e3c72;margin:0;">{len(out):,}</h3><p style="color:#2a5298;margin:6px 0 0 0;">Total Customers</p></div>', unsafe_allow_html=True)
                    with col2:
                        churned = (out['Churn_Prediction']=='Yes').sum()
                        st.markdown(f'<div class="metric-container"><h3 style="color:#1e3c72;margin:0;">{churned:,}</h3><p style="color:#2a5298;margin:6px 0 0 0;">Predicted Churners</p></div>', unsafe_allow_html=True)
                    with col3:
                        churn_rate = churned/len(out)*100 if len(out) else 0
                        st.markdown(f'<div class="metric-container"><h3 style="color:#1e3c72;margin:0;">{churn_rate:.1f}%</h3><p style="color:#2a5298;margin:6px 0 0 0;">Predicted Churn Rate</p></div>', unsafe_allow_html=True)
                    with col4:
                        high = (out['Churn_Probability']>0.7).sum()
                        st.markdown(f'<div class="metric-container"><h3 style="color:#1e3c72;margin:0;">{high:,}</h3><p style="color:#2a5298;margin:6px 0 0 0;">High Risk Customers</p></div>', unsafe_allow_html=True)
            except Exception as e:
                st.error(f"❌ Error reading the file: {e}")

# Bottom spacing
st.markdown("<br><br>", unsafe_allow_html=True)
