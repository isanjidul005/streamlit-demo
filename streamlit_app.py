import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import warnings

warnings.filterwarnings('ignore')

# =============================
# App Config & CSS
# =============================
st.set_page_config(
    page_title="EduAnalytics Pro+",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.main-header {
    font-size: 3.5rem;
    background: linear-gradient(45deg, #667eea, #764ba2);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-align: center;
    margin-bottom: 1rem;
    font-weight: bold;
}
.sub-header {
    font-size: 2rem;
    color: #2c3e50;
    border-bottom: 4px solid #3498db;
    padding-bottom: 0.5rem;
    margin-top: 2rem;
    margin-bottom: 1.5rem;
    font-weight: 600;
}
.stButton>button {
    background: linear-gradient(45deg, #FF4B2B, #FF416C);
    color: white;
    border: none;
    padding: 1rem 2rem;
    border-radius: 10px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# =============================
# Safe session state init
# =============================
def init_session_state():
    defaults = {
        "df": None,
        "cleaned_df": None,
        "file_uploaded": False,
        "detected_cols": {},
        "column_mapping": {},
        "current_view": "data_upload",
        "selected_students": [],
        "current_class": None
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

# =============================
# App Title
# =============================
st.markdown('<h1 class="main-header">📊 EduAnalytics Pro+</h1>', unsafe_allow_html=True)
st.markdown("### AI-powered student performance analytics with predictive insights.")

# =============================
# Sidebar Navigation
# =============================
with st.sidebar:
    st.markdown("## 🧭 Navigation")
    view_options = {
        "📤 Data Upload & Mapping": "data_upload",
        "📊 Overview Dashboard": "overview",
        "👥 Student Comparison": "comparison",
        "🏫 Class Analytics": "class_analytics",
        "📈 Individual Insights": "individual",
        "🔍 Advanced Analysis": "advanced",
        "📖 Documentation": "documentation"
    }
    selected_view = st.radio(
        "Select Section",
        list(view_options.keys()),
        index=list(view_options.values()).index(st.session_state.current_view)
    )
    st.session_state.current_view = view_options[selected_view]

# =============================
# SECTION: Data Upload & Cleaning
# =============================
def show_data_upload():
    st.markdown('<h2 class="sub-header">📤 Data Upload & Preprocessing</h2>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload CSV/XLSX", type=['csv', 'xlsx', 'xls'])
    
    if uploaded_file:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file, engine='openpyxl')
            st.session_state.df = df
            st.session_state.file_uploaded = True

            # Advanced Column Detection
            detected = {
                'id_cols': [], 'name_cols': [], 'class_cols': [],
                'attendance_cols': [], 'score_cols': [], 'date_cols': [], 'demographic_cols': []
            }
            for col in df.columns:
                cl = col.lower()
                dt = str(df[col].dtype)
                if any(x in cl for x in ['id', 'roll', 'number', 'code', 'studentid', 'reg']):
                    if dt in ['int64', 'float64'] or (df[col].nunique() == len(df)):
                        detected['id_cols'].append(col)
                elif any(x in cl for x in ['name', 'student', 'fullname', 'first', 'last']):
                    if dt == 'object' and df[col].nunique() > 1:
                        detected['name_cols'].append(col)
                elif any(x in cl for x in ['class', 'section', 'grade', 'batch', 'group']):
                    detected['class_cols'].append(col)
                elif any(x in cl for x in ['attendance', 'present', 'absent', 'pct', '%', 'rate']):
                    if dt in ['int64', 'float64']:
                        detected['attendance_cols'].append(col)
                elif any(x in cl for x in ['score', 'mark', 'test', 'exam', 'assessment', 'quiz', 'assignment']):
                    if dt in ['int64', 'float64']:
                        detected['score_cols'].append(col)
                elif any(x in cl for x in ['date', 'time', 'day', 'month', 'year']):
                    detected['date_cols'].append(col)
                elif any(x in cl for x in ['age', 'gender', 'dob', 'background', 'category']):
                    detected['demographic_cols'].append(col)

            # Pick primary columns
            mapping = {
                'id_col': detected['id_cols'][0] if detected['id_cols'] else None,
                'name_col': detected['name_cols'][0] if detected['name_cols'] else None,
                'class_col': detected['class_cols'][0] if detected['class_cols'] else None,
                'attendance_col': detected['attendance_cols'][0] if detected['attendance_cols'] else None,
                'score_cols': detected['score_cols']
            }
            st.session_state.detected_cols = detected
            st.session_state.column_mapping = mapping

            st.success(f"✅ Loaded {df.shape[0]} rows and {df.shape[1]} columns")
            st.dataframe(df.head(10))

            # Cleaning Options
            st.markdown("### 🧹 Data Cleaning")
            cleaning_ops = st.multiselect("Select cleaning operations",
                                          ["Remove duplicates", "Handle missing values", "Remove outliers", "Normalize scores"],
                                          default=["Handle missing values", "Remove outliers"])
            if st.button("🚀 Apply Cleaning"):
                clean_df = df.copy()
                # Remove duplicates
                if "Remove duplicates" in cleaning_ops:
                    clean_df = clean_df.drop_duplicates()
                # Handle missing
                if "Handle missing values" in cleaning_ops:
                    for col in clean_df.select_dtypes(include=[np.number]).columns:
                        clean_df[col] = clean_df[col].fillna(clean_df[col].median())
                # Remove outliers
                if "Remove outliers" in cleaning_ops:
                    for col in mapping['score_cols']:
                        mean = clean_df[col].mean()
                        std = clean_df[col].std()
                        clean_df = clean_df[(clean_df[col] >= mean - 3*std) & (clean_df[col] <= mean + 3*std)]
                # Normalize
                if "Normalize scores" in cleaning_ops:
                    for col in mapping['score_cols']:
                        clean_df[f"{col}_normalized"] = (clean_df[col]-clean_df[col].mean())/clean_df[col].std()
                st.session_state.cleaned_df = clean_df
                st.success("✅ Cleaning applied successfully!")
                st.dataframe(clean_df.head(10))

        except Exception as e:
            st.error(f"Failed to load: {e}")
    else:
        st.info("Upload a file to continue.")

# =============================
# SECTION: Overview Dashboard
# =============================
def show_overview():
    st.markdown('<h2 class="sub-header">📊 Overview Dashboard</h2>', unsafe_allow_html=True)
    if not st.session_state.file_uploaded:
        st.info("Upload and process data first.")
        return
    df = st.session_state.cleaned_df if st.session_state.cleaned_df is not None else st.session_state.df
    mapping = st.session_state.column_mapping

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Students", len(df))
    if mapping['attendance_col']:
        col2.metric("Average Attendance", f"{df[mapping['attendance_col']].mean():.1f}%")
    if mapping['score_cols']:
        col3.metric("Average Score", f"{df[mapping['score_cols']].mean().mean():.1f}")
    if mapping['class_col']:
        col4.metric("Total Classes", df[mapping['class_col']].nunique())

    # Example plots
    if mapping['score_cols']:
        st.plotly_chart(px.bar(df[mapping['score_cols']].mean(), title="Average Score by Test"), use_container_width=True)
    if mapping['attendance_col']:
        st.plotly_chart(px.histogram(df, x=mapping['attendance_col'], title="Attendance Distribution"), use_container_width=True)

# =============================
# SECTION: Comparison (placeholder)
# =============================
def show_comparison():
    st.markdown('<h2 class="sub-header">👥 Student Comparison</h2>', unsafe_allow_html=True)
    st.info("Student comparison feature is coming soon!")

# =============================
# SECTION: Class Analytics (placeholder)
# =============================
def show_class_analytics():
    st.markdown('<h2 class="sub-header">🏫 Class Analytics</h2>', unsafe_allow_html=True)
    st.info("Class analytics feature is coming soon!")

# =============================
# SECTION: Individual Insights (placeholder)
# =============================
def show_individual():
    st.markdown('<h2 class="sub-header">📈 Individual Insights</h2>', unsafe_allow_html=True)
    st.info("Individual student insights coming soon!")

# =============================
# SECTION: Advanced Analysis (placeholder)
# =============================
def show_advanced():
    st.markdown('<h2 class="sub-header">🔍 Advanced Analysis</h2>', unsafe_allow_html=True)
    st.info("Advanced analytics coming soon!")

# =============================
# SECTION: Documentation
# =============================
def show_documentation():
    st.markdown('<h2 class="sub-header">📖 Documentation</h2>', unsafe_allow_html=True)
    st.markdown("""
- **Data Upload**: Upload CSV/XLSX. Columns auto-detected.
- **Overview Dashboard**: Metrics, distributions, performance trends.
- **Student Comparison**: Compare students side by side (radar charts soon).
- **Class Analytics**: Class-wise performance & attendance.
- **Individual Insights**: Track a student's performance over time.
- **Advanced Analysis**: Correlation and predictive insights.
- **Data Cleaning**: Handle missing, remove outliers, normalize scores.
""")

# =============================
# SECTION SWITCHER
# =============================
sections = {
    "data_upload": show_data_upload,
    "overview": show_overview,
    "comparison": show_comparison,
    "class_analytics": show_class_analytics,
    "individual": show_individual,
    "advanced": show_advanced,
    "documentation": show_documentation
}

sections[st.session_state.current_view]()
