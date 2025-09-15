import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from io import StringIO
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
# Custom CSS Styling
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
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 10px 20px rgba(0, 0, 0, 0.2);
        text-align: center;
        margin: 0.5rem;
    }
    .stButton>button {
        background: linear-gradient(45deg, #FF4B2B, #FF416C);
        color: white;
        border: none;
        padding: 1rem 2rem;
        border-radius: 10px;
        font-weight: bold;
        font-size: 1rem;
    }
    .student-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 2rem;
        border-radius: 15px;
        border-left: 6px solid #667eea;
        margin: 1.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# =============================
# Session State Initialization
# =============================
if 'df' not in st.session_state:
    st.session_state.df = None
if 'file_uploaded' not in st.session_state:
    st.session_state.file_uploaded = False
if 'current_view' not in st.session_state:
    st.session_state.current_view = "data_upload"
if 'detected_cols' not in st.session_state:
    st.session_state.detected_cols = {}

# =============================
# App Title
# =============================
st.markdown('<h1 class="main-header">📊 Right iTech</h1>', unsafe_allow_html=True)
st.markdown("### Advanced Educational Data Analysis Platform")

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
    
    selected_view = st.radio("Select Section", list(view_options.keys()))
    st.session_state.current_view = view_options[selected_view]

# =============================
# Helper Function - Detect Columns
# =============================
def detect_columns(df):
    detected = {
        'id_col': None,
        'name_col': None,
        'class_col': None,
        'attendance_col': None,
        'score_cols': []
    }
    
    for col in df.columns:
        col_lower = str(col).lower()
        if any(x in col_lower for x in ['id', 'roll', 'number']):
            detected['id_col'] = col
        elif any(x in col_lower for x in ['name', 'student']):
            detected['name_col'] = col
        elif any(x in col_lower for x in ['class', 'section', 'grade']):
            detected['class_col'] = col
        elif any(x in col_lower for x in ['attendance', 'present', 'absent', '%']):
            detected['attendance_col'] = col
        elif any(x in col_lower for x in ['test', 'exam', 'score', 'mark']):
            detected['score_cols'].append(col)
    
    return detected

# =============================
# Data Upload Section
# =============================
if st.session_state.current_view == "data_upload":
    st.markdown('<h2 class="sub-header">📤 Data Upload</h2>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Upload Student Data File", 
        type=['csv', 'xlsx', 'xls'],
        help="Supported formats: CSV, Excel (XLSX, XLS)"
    )
    
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            st.session_state.df = df
            st.session_state.detected_cols = detect_columns(df)
            st.session_state.file_uploaded = True
            
            st.success(f"✅ Successfully loaded {df.shape[0]} students with {df.shape[1]} columns")
            st.info(f"🔍 Detected columns: {st.session_state.detected_cols}")
            
            st.subheader("📋 Dataset Preview")
            st.dataframe(df.head(), use_container_width=True)
            
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

# =============================
# Block Access if No Data
# =============================
elif not st.session_state.file_uploaded:
    st.warning("📝 Please upload your data first in the 'Data Upload' section.")
    st.stop()

# =============================
# Overview Dashboard
# =============================
elif st.session_state.current_view == "overview":
    st.markdown('<h2 class="sub-header">📊 Comprehensive Overview</h2>', unsafe_allow_html=True)
    df = st.session_state.df
    detected_cols = st.session_state.detected_cols
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Students", len(df))
    with col2:
        if detected_cols['attendance_col']:
            avg_att = df[detected_cols['attendance_col']].mean()
            st.metric("Avg Attendance", f"{avg_att:.1f}%")
    with col3:
        if detected_cols['score_cols']:
            avg_score = df[detected_cols['score_cols']].mean().mean()
            st.metric("Avg Performance", f"{avg_score:.1f}%")
    with col4:
        if detected_cols['class_col']:
            st.metric("Total Classes", df[detected_cols['class_col']].nunique())
    
    # Visualizations
    col1, col2 = st.columns(2)
    with col1:
        if detected_cols['score_cols']:
            score_means = df[detected_cols['score_cols']].mean()
            fig = px.bar(x=score_means.index, y=score_means.values, 
                       title="Average Scores by Test", labels={'x': 'Test', 'y': 'Score'})
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        if detected_cols['attendance_col']:
            fig = px.histogram(df, x=detected_cols['attendance_col'], 
                             title="Attendance Distribution", nbins=20)
            st.plotly_chart(fig, use_container_width=True)

# =============================
# Student Comparison
# =============================
elif st.session_state.current_view == "comparison":
    st.markdown('<h2 class="sub-header">👥 Student Comparison</h2>', unsafe_allow_html=True)
    df = st.session_state.df
    detected_cols = st.session_state.detected_cols
    
    if detected_cols['name_col']:
        student_names = df[detected_cols['name_col']].tolist()
        col1, col2 = st.columns(2)
        with col1:
            student1 = st.selectbox("Select First Student", student_names, key="student1")
        with col2:
            student2 = st.selectbox("Select Second Student", [s for s in student_names if s != student1], key="student2")
        
        if student1 and student2:
            student1_data = df[df[detected_cols['name_col']] == student1].iloc[0]
            student2_data = df[df[detected_cols['name_col']] == student2].iloc[0]
            
            col1, col2 = st.columns(2)
            with col1:
                st.subheader(f"👤 {student1}")
                if detected_cols['class_col']:
                    st.write(f"**Class:** {student1_data[detected_cols['class_col']]}")
                if detected_cols['attendance_col']:
                    st.metric("Attendance", f"{student1_data[detected_cols['attendance_col']]}%")
                if detected_cols['score_cols']:
                    avg_score = student1_data[detected_cols['score_cols']].mean()
                    st.metric("Average Score", f"{avg_score:.1f}%")
            with col2:
                st.subheader(f"👤 {student2}")
                if detected_cols['class_col']:
                    st.write(f"**Class:** {student2_data[detected_cols['class_col']]}")
                if detected_cols['attendance_col']:
                    st.metric("Attendance", f"{student2_data[detected_cols['attendance_col']]}%")
                if detected_cols['score_cols']:
                    avg_score = student2_data[detected_cols['score_cols']].mean()
                    st.metric("Average Score", f"{avg_score:.1f}%")
            
            # Radar chart
            if detected_cols['score_cols']:
                fig = go.Figure()
                fig.add_trace(go.Scatterpolar(r=student1_data[detected_cols['score_cols']].values,
                                              theta=detected_cols['score_cols'],
                                              fill='toself', name=student1))
                fig.add_trace(go.Scatterpolar(r=student2_data[detected_cols['score_cols']].values,
                                              theta=detected_cols['score_cols'],
                                              fill='toself', name=student2))
                fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                                  title="Test Score Comparison")
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No student name column detected in the dataset.")

# =============================
# Class Analytics
# =============================
elif st.session_state.current_view == "class_analytics":
    st.markdown('<h2 class="sub-header">🏫 Class Analytics</h2>', unsafe_allow_html=True)
    df = st.session_state.df
    detected_cols = st.session_state.detected_cols
    
    if detected_cols['class_col']:
        classes = df[detected_cols['class_col']].unique()
        selected_class = st.selectbox("Select Class", classes)
        class_data = df[df[detected_cols['class_col']] == selected_class]
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Students in Class", len(class_data))
        with col2:
            if detected_cols['attendance_col']:
                avg_att = class_data[detected_cols['attendance_col']].mean()
                st.metric("Average Attendance", f"{avg_att:.1f}%")
        with col3:
            if detected_cols['score_cols']:
                avg_score = class_data[detected_cols['score_cols']].mean().mean()
                st.metric("Average Score", f"{avg_score:.1f}%")
        
        if detected_cols['score_cols']:
            class_avg_scores = class_data[detected_cols['score_cols']].mean()
            fig = px.bar(x=class_avg_scores.index, y=class_avg_scores.values,
                       title=f"Performance in {selected_class}", labels={'x': 'Test', 'y': 'Average Score'})
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No class column detected in the dataset.")

# =============================
# Individual Insights
# =============================
elif st.session_state.current_view == "individual":
    st.markdown('<h2 class="sub-header">📈 Individual Insights</h2>', unsafe_allow_html=True)
    df = st.session_state.df
    detected_cols = st.session_state.detected_cols
    
    if detected_cols['name_col']:
        student_names = df[detected_cols['name_col']].unique()
        selected_student = st.selectbox("Select Student", student_names)
        student_data = df[df[detected_cols['name_col']] == selected_student].iloc[0]
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Student", selected_student)
        with col2:
            if detected_cols['class_col']:
                st.metric("Class", student_data[detected_cols['class_col']])
        with col3:
            if detected_cols['attendance_col']:
                st.metric("Attendance", f"{student_data[detected_cols['attendance_col']]}%")
        
        if detected_cols['score_cols']:
            scores = student_data[detected_cols['score_cols']]
            fig = px.line(x=detected_cols['score_cols'], y=scores.values,
                        title=f"Performance Trend - {selected_student}",
                        labels={'x': 'Test', 'y': 'Score'})
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No student name column detected in the dataset.")

# =============================
# Advanced Analysis
# =============================
elif st.session_state.current_view == "advanced":
    st.markdown('<h2 class="sub-header">🔍 Advanced Analysis</h2>', unsafe_allow_html=True)
    df = st.session_state.df
    detected_cols = st.session_state.detected_cols
    
    if detected_cols['attendance_col'] and detected_cols['score_cols']:
        st.subheader("📈 Attendance vs Performance Correlation")
        df['average_score'] = df[detected_cols['score_cols']].mean(axis=1)
        fig = px.scatter(df, x=detected_cols['attendance_col'], y='average_score',
                       title="Attendance vs Average Score Correlation",
                       labels={detected_cols['attendance_col']: 'Attendance (%)', 'average_score': 'Average Score (%)'},
                       trendline="ols")
        st.plotly_chart(fig, use_container_width=True)
        
        correlation = df[detected_cols['attendance_col']].corr(df['average_score'])
        st.metric("Correlation Coefficient", f"{correlation:.3f}")
        
        if correlation > 0.7:
            st.success("Strong positive correlation: Higher attendance strongly correlates with better performance")
        elif correlation > 0.3:
            st.info("Moderate positive correlation: Attendance has some positive impact on performance")
        elif correlation > -0.3:
            st.warning("Weak correlation: Little relationship between attendance and performance")
        else:
            st.error("Negative correlation: Unexpected relationship detected")
    else:
        st.warning("Need both attendance and score columns for correlation analysis.")

# =============================
# Documentation / Help Section
# =============================
elif st.session_state.current_view == "documentation":
    st.markdown('<h2 class="sub-header">📖 Documentation</h2>', unsafe_allow_html=True)
    st.markdown("""
    Welcome to **Right iTech Educational Analytics Platform**! 🎓
    
    This tool helps you analyze student data efficiently. Below are the sections explained:
    
    - **📤 Data Upload**: Upload CSV/Excel files. The system auto-detects ID, Name, Class, Attendance, and Score columns.
    - **📊 Overview Dashboard**: See high-level statistics like total students, average attendance, performance, and distribution plots.
    - **👥 Student Comparison**: Select two students to compare attendance and scores side by side, including radar charts.
    - **🏫 Class Analytics**: Focus on a specific class to see attendance rates and test averages.
    - **📈 Individual Insights**: Choose one student to track their performance trends across tests.
    - **🔍 Advanced Analysis**: Correlation study between attendance and performance with visual interpretation.
    
 ### 📌 Notes
    - Ensure your dataset has properly labeled columns (like `Name`, `Class`, `Attendance`, `Score`).
    - Missing values may affect the accuracy of statistics.
    - Use the sidebar navigation to switch between views.
    """)   # 👈 This was missing
