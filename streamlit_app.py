import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from io import StringIO, BytesIO
import warnings
import re
from fpdf import FPDF
import base64
from datetime import datetime
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="WMT Analytics Pro",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced styling
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
        transition: transform 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-5px);
    }
    .stButton>button {
        background: linear-gradient(45deg, #FF4B2B, #FF416C);
        color: white;
        border: none;
        padding: 1rem 2rem;
        border-radius: 10px;
        font-weight: bold;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    .stButton>button:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 16px rgba(0,0,0,0.3);
    }
    .student-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 2rem;
        border-radius: 15px;
        border-left: 6px solid #667eea;
        margin: 1.5rem 0;
        box-shadow: 0 6px 12px rgba(0,0,0,0.1);
    }
    .warning-box {
        background: linear-gradient(135deg, #ff9966 0%, #ff5e62 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .success-box {
        background: linear-gradient(135deg, #56ab2f 0%, #a8e6cf 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'df' not in st.session_state:
    st.session_state.df = None
if 'file_uploaded' not in st.session_state:
    st.session_state.file_uploaded = False
if 'current_view' not in st.session_state:
    st.session_state.current_view = "data_upload"
if 'detected_cols' not in st.session_state:
    st.session_state.detected_cols = {}
if 'attendance_df' not in st.session_state:
    st.session_state.attendance_df = None
if 'merged_df' not in st.session_state:
    st.session_state.merged_df = None

# WMT Data Processing Functions
def process_wmt_data(df):
    """Process WMT dataset with multiple test columns"""
    processed_df = df.copy()
    
    # Identify WMT columns
    wmt_columns = [col for col in df.columns if 'WMT' in col.upper()]
    info_columns = [col for col in df.columns if col.upper() in ['ID', 'ROLL', 'NAME']]
    
    # Convert 'Ab' and blanks to NaN for numeric processing
    for col in wmt_columns:
        processed_df[col] = pd.to_numeric(processed_df[col], errors='coerce')
    
    # Extract test information from column names
    test_info = []
    for col in wmt_columns:
        match = re.match(r'WMT\s*W(\d+)\s*\[(\d+)\]', col, re.IGNORECASE)
        if match:
            week = f"Week {match.group(1)}"
            max_marks = int(match.group(2))
            test_info.append({'column': col, 'week': week, 'max_marks': max_marks, 'subject': 'Unknown'})
    
    return processed_df, wmt_columns, info_columns, test_info

def detect_subjects_from_columns(wmt_columns):
    """Try to detect subjects from column patterns"""
    subjects = set()
    subject_patterns = {
        'Math': ['MATH', 'MATHEMATICS', 'CALCULUS'],
        'Science': ['SCIENCE', 'PHYSICS', 'CHEMISTRY', 'BIOLOGY'],
        'English': ['ENGLISH', 'LANGUAGE', 'GRAMMAR'],
        'Social': ['SOCIAL', 'HISTORY', 'GEOGRAPHY', 'CIVICS'],
        'Computer': ['COMPUTER', 'IT', 'INFORMATICS'],
        'Hindi': ['HINDI', 'REGIONAL'],
        'Sanskrit': ['SANSKRIT'],
        'Physics': ['PHYSICS'],
        'Chemistry': ['CHEMISTRY'],
        'Biology': ['BIOLOGY']
    }
    
    for col in wmt_columns:
        col_upper = col.upper()
        for subject, patterns in subject_patterns.items():
            if any(pattern in col_upper for pattern in patterns):
                subjects.add(subject)
                break
    
    return sorted(list(subjects)) if subjects else ['Subject 1', 'Subject 2', 'Subject 3']

# PDF Generation Functions
def create_pdf_report(df, detected_cols, analysis_type="full"):
    """Create PDF report for WMT data"""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, "WMT Analytics Pro - Student Performance Report", ln=True, align='C')
    pdf.ln(10)
    pdf.set_font("Arial", '', 12)
    pdf.cell(200, 10, f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True, align='C')
    pdf.ln(15)
    
    # Add your PDF content here based on WMT data structure
    return pdf.output(dest='S').encode('latin1')

# App title and description
st.markdown('<h1 class="main-header">📊 WMT Analytics Pro</h1>', unsafe_allow_html=True)
st.markdown("### Weekly Model Test Performance Analysis Platform")

# Sidebar navigation
with st.sidebar:
    st.markdown("## 🧭 Navigation")
    
    view_options = {
        "📤 Data Upload": "data_upload",
        "📊 Overview Dashboard": "overview",
        "👥 Student Comparison": "comparison",
        "📈 Test Analysis": "test_analysis",
        "📋 Student Insights": "individual",
        "🔍 Advanced Analysis": "advanced"
    }
    
    selected_view = st.radio("Select Section", list(view_options.keys()))
    st.session_state.current_view = view_options[selected_view]
    
    st.markdown("---")
    st.markdown("### ⚙️ Settings")
    if st.session_state.file_uploaded:
        analysis_mode = st.selectbox("Analysis Mode", ["Basic", "Detailed", "Comprehensive"])
    
    st.markdown("---")
    if st.button("🔄 Reset Data", type="secondary"):
        st.session_state.df = None
        st.session_state.file_uploaded = False
        st.session_state.detected_cols = {}
        st.session_state.attendance_df = None
        st.session_state.merged_df = None
        st.rerun()

# Data Upload Section
if st.session_state.current_view == "data_upload":
    st.markdown('<h2 class="sub-header">📤 WMT Data Upload</h2>', unsafe_allow_html=True)
    
    # Main WMT data upload
    uploaded_file = st.file_uploader(
        "Upload WMT Scores File", 
        type=['csv', 'xlsx', 'xls'],
        help="Upload your Weekly Model Test scores file with columns like 'WMT W1 [30]', 'WMT W2 [30]' etc."
    )
    
    # Optional attendance file upload
    attendance_file = st.file_uploader(
        "Upload Attendance File (Optional)",
        type=['csv', 'xlsx', 'xls'],
        help="Upload attendance data if available. Should have Student ID and attendance percentage."
    )
    
    if uploaded_file is not None:
        try:
            # Read main WMT file
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            st.session_state.df = df
            
            # Process WMT data
            processed_df, wmt_columns, info_columns, test_info = process_wmt_data(df)
            subjects = detect_subjects_from_columns(wmt_columns)
            
            # Read attendance file if provided
            if attendance_file:
                if attendance_file.name.endswith('.csv'):
                    attendance_df = pd.read_csv(attendance_file)
                else:
                    attendance_df = pd.read_excel(attendance_file)
                st.session_state.attendance_df = attendance_df
            
            # Store detected information
            st.session_state.detected_cols = {
                'wmt_columns': wmt_columns,
                'info_columns': info_columns,
                'test_info': test_info,
                'subjects': subjects,
                'processed_df': processed_df
            }
            
            st.session_state.file_uploaded = True
            
            st.success(f"✅ Successfully loaded {df.shape[0]} students with {len(wmt_columns)} test columns")
            st.info(f"🔍 Detected {len(subjects)} subjects: {', '.join(subjects)}")
            st.info(f"📊 Found {len(test_info)} test instances across different weeks")
            
            # Show data preview
            st.subheader("📋 Dataset Preview")
            st.dataframe(df.head(), use_container_width=True)
            
            # Show test information
            with st.expander("📊 Test Structure Analysis"):
                test_df = pd.DataFrame(test_info)
                st.dataframe(test_df, use_container_width=True)
                
                # Data quality check
                st.subheader("📈 Data Quality Report")
                missing_data = processed_df[wmt_columns].isnull().sum().sum()
                total_cells = len(processed_df) * len(wmt_columns)
                st.write(f"Missing/Abent scores: {missing_data} out of {total_cells} ({missing_data/total_cells*100:.1f}%)")
            
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
            st.exception(e)

# Check if data is loaded before showing other sections
elif not st.session_state.file_uploaded:
    st.warning("📝 Please upload your WMT data first in the 'Data Upload' section.")
    st.stop()

else:
    # Get data from session state
    df = st.session_state.df
    detected_cols = st.session_state.detected_cols
    processed_df = detected_cols['processed_df']
    wmt_columns = detected_cols['wmt_columns']
    subjects = detected_cols['subjects']
    test_info = detected_cols['test_info']

    # Overview Dashboard
    if st.session_state.current_view == "overview":
        st.markdown('<h2 class="sub-header">📊 WMT Performance Overview</h2>', unsafe_allow_html=True)
        
        # Calculate overall metrics
        total_students = len(processed_df)
        avg_scores = processed_df[wmt_columns].mean().mean()
        participation_rate = (1 - processed_df[wmt_columns].isnull().sum().sum() / (total_students * len(wmt_columns))) * 100
        
        # Metrics dashboard
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Students", total_students)
        with col2:
            st.metric("Average Score", f"{avg_scores:.1f}%")
        with col3:
            st.metric("Participation Rate", f"{participation_rate:.1f}%")
        with col4:
            st.metric("Total Tests", len(wmt_columns))
        
        # Visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            # Weekly performance trend
            weekly_avg = processed_df[wmt_columns].mean()
            weekly_dates = [f"W{i+1}" for i in range(len(weekly_avg))]
            fig = px.line(x=weekly_dates, y=weekly_avg.values, 
                         title="Average Performance by Week",
                         labels={'x': 'Test Week', 'y': 'Average Score'})
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Score distribution
            all_scores = processed_df[wmt_columns].values.flatten()
            all_scores = all_scores[~np.isnan(all_scores)]  # Remove NaN values
            fig = px.histogram(x=all_scores, nbins=20, 
                             title="Overall Score Distribution",
                             labels={'x': 'Score', 'y': 'Frequency'})
            st.plotly_chart(fig, use_container_width=True)
        
        # Subject-wise analysis (if we can detect subjects)
        if len(subjects) > 1:
            st.subheader("📚 Subject-wise Performance")
            # This would require mapping columns to subjects - simplified version
            subject_avgs = []
            for subject in subjects:
                # Simple pattern matching for subject columns
                subject_cols = [col for col in wmt_columns if subject.upper() in col.upper()]
                if subject_cols:
                    subject_avg = processed_df[subject_cols].mean().mean()
                    subject_avgs.append({'Subject': subject, 'Average Score': subject_avg})
            
            if subject_avgs:
                subject_df = pd.DataFrame(subject_avgs)
                fig = px.bar(subject_df, x='Subject', y='Average Score', 
                           title="Average Scores by Subject Area")
                st.plotly_chart(fig, use_container_width=True)

    # Student Comparison View
    elif st.session_state.current_view == "comparison":
        st.markdown('<h2 class="sub-header">👥 Student Comparison</h2>', unsafe_allow_html=True)
        
        student_names = processed_df['Name'].tolist() if 'Name' in processed_df.columns else [f"Student {i+1}" for i in range(len(processed_df))]
        
        col1, col2 = st.columns(2)
        with col1:
            student1 = st.selectbox("Select First Student", student_names, key="student1")
        with col2:
            student2 = st.selectbox("Select Second Student", 
                                  [s for s in student_names if s != student1], 
                                  key="student2")
        
        if student1 and student2:
            # Get student data
            student1_data = processed_df[processed_df['Name'] == student1].iloc[0]
            student2_data = processed_df[processed_df['Name'] == student2].iloc[0]
            
            # Display comparison
            col1, col2 = st.columns(2)
            with col1:
                st.subheader(f"👤 {student1}")
                st.metric("Average Score", f"{student1_data[wmt_columns].mean():.1f}%")
                st.metric("Tests Taken", f"{student1_data[wmt_columns].count()}/{len(wmt_columns)}")
            
            with col2:
                st.subheader(f"👤 {student2}")
                st.metric("Average Score", f"{student2_data[wmt_columns].mean():.1f}%")
                st.metric("Tests Taken", f"{student2_data[wmt_columns].count()}/{len(wmt_columns)}")
            
            # Weekly comparison chart
            comparison_data = []
            for i, col in enumerate(wmt_columns):
                week = f"Week {i+1}"
                comparison_data.append({
                    'Week': week,
                    student1: student1_data[col],
                    student2: student2_data[col]
                })
            
            comparison_df = pd.DataFrame(comparison_data)
            fig = px.line(comparison_df, x='Week', y=[student1, student2],
                         title="Weekly Score Comparison", markers=True)
            st.plotly_chart(fig, use_container_width=True)

    # Test Analysis View
    elif st.session_state.current_view == "test_analysis":
        st.markdown('<h2 class="sub-header">📈 Test Performance Analysis</h2>', unsafe_allow_html=True)
        
        # Select test to analyze
        test_to_analyze = st.selectbox("Select Test to Analyze", wmt_columns)
        
        if test_to_analyze:
            test_scores = processed_df[test_to_analyze].dropna()
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Average Score", f"{test_scores.mean():.1f}%")
            with col2:
                st.metric("Participation", f"{len(test_scores)}/{len(processed_df)} students")
            with col3:
                st.metric("Top Score", f"{test_scores.max():.1f}%")
            
            # Score distribution for selected test
            fig = px.histogram(x=test_scores, nbins=15, 
                             title=f"Score Distribution - {test_to_analyze}",
                             labels={'x': 'Score', 'y': 'Number of Students'})
            st.plotly_chart(fig, use_container_width=True)
            
            # Top performers for this test
            top_performers = processed_df.nlargest(10, test_to_analyze)[['Name', test_to_analyze]]
            st.dataframe(top_performers, use_container_width=True)

    # Individual Student Insights
    elif st.session_state.current_view == "individual":
        st.markdown('<h2 class="sub-header">📋 Student Performance Profile</h2>', unsafe_allow_html=True)
        
        student_names = processed_df['Name'].tolist() if 'Name' in processed_df.columns else [f"Student {i+1}" for i in range(len(processed_df))]
        selected_student = st.selectbox("Select Student", student_names)
        
        if selected_student:
            student_data = processed_df[processed_df['Name'] == selected_student].iloc[0]
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Student", selected_student)
            with col2:
                st.metric("Overall Average", f"{student_data[wmt_columns].mean():.1f}%")
            with col3:
                st.metric("Tests Completed", f"{student_data[wmt_columns].count()}/{len(wmt_columns)}")
            
            # Progress chart
            scores = student_data[wmt_columns].values
            weeks = [f"Week {i+1}" for i in range(len(scores))]
            valid_scores = score[~np.isnan(scores)]
            valid_weeks = [weeks[i] for i in range(len(scores)) if not np.isnan(scores[i])]
            
            if len(valid_scores) > 0:
                fig = px.line(x=valid_weeks, y=valid_scores, 
                            title=f"Performance Trend - {selected_student}",
                            labels={'x': 'Test Week', 'y': 'Score'},
                            markers=True)
                st.plotly_chart(fig, use_container_width=True)
            
            # Attendance integration (if available)
            if st.session_state.attendance_df is not None:
                st.subheader("📊 Attendance Information")
                # Add attendance analysis here

    # Advanced Analysis View
    elif st.session_state.current_view == "advanced":
        st.markdown('<h2 class="sub-header">🔍 Advanced Analytics</h2>', unsafe_allow_html=True)
        
        # Correlation analysis between different tests
        st.subheader("📈 Test Correlation Analysis")
        
        # Select tests for correlation
        selected_tests = st.multiselect("Select tests for correlation analysis", 
                                      wmt_columns, 
                                      default=wmt_columns[:min(5, len(wmt_columns))])
        
        if len(selected_tests) >= 2:
            correlation_matrix = processed_df[selected_tests].corr()
            fig = px.imshow(correlation_matrix, 
                          title="Correlation Between Tests",
                          color_continuous_scale='RdBu_r',
                          aspect='auto')
            st.plotly_chart(fig, use_container_width=True)
            
            # Highlight strongest correlations
            st.subheader("💡 Key Insights")
            corr_values = correlation_matrix.unstack().sort_values(ascending=False)
            # Remove diagonal and duplicates
            corr_values = corr_values[corr_values < 0.99]
            if len(corr_values) > 0:
                strongest_corr = corr_values.iloc[0]
                test1, test2 = corr_values.index[0]
                st.info(f"Strongest correlation: {test1} ↔ {test2} ({strongest_corr:.3f})")

# Footer
st.markdown("---")
st.markdown("### 🎓 WMT Analytics Pro - Weekly Model Test Analysis Platform")

# Add download functionality
with st.sidebar:
    st.markdown("---")
    st.markdown("### 💾 Export Options")
    if st.session_state.file_uploaded:
        if st.button("📄 Generate PDF Report", use_container_width=True):
            pdf_bytes = create_pdf_report(processed_df, detected_cols)
            st.download_button(
                label="📥 Download PDF Report",
                data=pdf_bytes,
                file_name=f"wmt_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                mime="application/pdf",
                use_container_width=True
            )
        
        if st.button("📊 Export Processed Data", use_container_width=True):
            csv = processed_df.to_csv(index=False)
            st.download_button(
                label="📥 Download CSV",
                data=csv,
                file_name="wmt_processed_data.csv",
                mime="text/csv",
                use_container_width=True
            )
