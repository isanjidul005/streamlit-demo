import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import warnings

warnings.filterwarnings('ignore')

# =============================
# App Configuration
# =============================
st.set_page_config(
    page_title="Right iTech",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================
# Custom CSS
# =============================
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
# Safe Session State Initialization
# =============================
def init_session_state():
    defaults = {
        "df": None,
        "cleaned_df": None,
        "file_uploaded": False,
        "detected_cols": {},
        "column_mapping": {},
        "current_view": "overview",
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
st.markdown('<h1 class="main-header">📊 Right iTech</h1>', unsafe_allow_html=True)
st.markdown("### AI-Powered Student Performance Analytics Platform")

# =============================
# Sidebar Navigation
# =============================
with st.sidebar:
    st.markdown("## 🧭 Navigation")
    view_options = {
        "📤 Data Upload": "data_upload",
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
# Main Section Switcher
# =============================
def show_data_upload():
    st.markdown('<h2 class="sub-header">📤 Data Upload</h2>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Upload CSV or Excel",
        type=["csv", "xlsx", "xls"]
    )
    if uploaded_file:
        try:
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file, engine="openpyxl")
            st.session_state.df = df
            st.session_state.file_uploaded = True

            # Simple column detection
            detected = {
                "id_col": None,
                "name_col": None,
                "class_col": None,
                "attendance_col": None,
                "score_cols": []
            }
            for col in df.columns:
                cl = col.lower()
                if any(x in cl for x in ["id", "roll", "number"]):
                    detected["id_col"] = col
                elif any(x in cl for x in ["name", "student"]):
                    detected["name_col"] = col
                elif any(x in cl for x in ["class", "section", "grade"]):
                    detected["class_col"] = col
                elif any(x in cl for x in ["attendance", "%"]):
                    detected["attendance_col"] = col
                elif any(x in cl for x in ["score", "mark", "exam", "test"]):
                    detected["score_cols"].append(col)
            st.session_state.detected_cols = detected
            st.session_state.column_mapping = detected

            st.success(f"Loaded {df.shape[0]} rows and {df.shape[1]} columns")
            st.dataframe(df.head(10))
        except Exception as e:
            st.error(f"Failed to load file: {e}")
    else:
        st.info("Please upload a file to proceed.")


def show_overview():
    st.markdown('<h2 class="sub-header">📊 Overview Dashboard</h2>', unsafe_allow_html=True)
    if not st.session_state.file_uploaded:
        st.info("Upload data first.")
        return

    df = st.session_state.df
    col_map = st.session_state.column_mapping

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Students", len(df))
    with col2:
        if col_map["attendance_col"]:
            st.metric("Average Attendance", f"{df[col_map['attendance_col']].mean():.1f}%")
    with col3:
        if col_map["score_cols"]:
            st.metric("Average Score", f"{df[col_map['score_cols']].mean().mean():.1f}")
    with col4:
        if col_map["class_col"]:
            st.metric("Total Classes", df[col_map["class_col"]].nunique())

    # Example plot
    if col_map["score_cols"]:
        fig = px.bar(df[col_map["score_cols"]].mean(), title="Average Score by Test")
        st.plotly_chart(fig, use_container_width=True)


def show_comparison():
    st.markdown('<h2 class="sub-header">👥 Student Comparison</h2>', unsafe_allow_html=True)
    st.info("Comparison section placeholder (to be implemented).")


def show_class_analytics():
    st.markdown('<h2 class="sub-header">🏫 Class Analytics</h2>', unsafe_allow_html=True)
    st.info("Class analytics section placeholder.")


def show_individual():
    st.markdown('<h2 class="sub-header">📈 Individual Insights</h2>', unsafe_allow_html=True)
    st.info("Individual insights section placeholder.")


def show_advanced():
    st.markdown('<h2 class="sub-header">🔍 Advanced Analysis</h2>', unsafe_allow_html=True)
    st.info("Advanced analytics placeholder.")


def show_documentation():
    st.markdown('<h2 class="sub-header">📖 Documentation</h2>', unsafe_allow_html=True)
    st.markdown("""
    **Right iTech Platform Documentation**
    
    - **Data Upload**: Upload CSV/XLSX files. Columns auto-detected.
    - **Overview Dashboard**: High-level statistics & plots.
    - **Student Comparison**: Compare students side by side.
    - **Class Analytics**: Class-wise performance and attendance.
    - **Individual Insights**: Track one student's progress.
    - **Advanced Analysis**: Correlation and trends.
    """)


# =============================
# Show the section based on current_view
# =============================
view_funcs = {
    "data_upload": show_data_upload,
    "overview": show_overview,
    "comparison": show_comparison,
    "class_analytics": show_class_analytics,
    "individual": show_individual,
    "advanced": show_advanced,
    "documentation": show_documentation
}

view_funcs[st.session_state.current_view]()
