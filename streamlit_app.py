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


# Initialize session state for data persistence
if 'df' not in st.session_state:
    st.session_state.df = None
if 'file_uploaded' not in st.session_state:
    st.session_state.file_uploaded = False
if 'selected_students' not in st.session_state:
    st.session_state.selected_students = []
if 'current_view' not in st.session_state:
    st.session_state.current_view = "overview"
# REMOVE any theme-related session state lines

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.8rem;
        color: #ff7f0e;
        border-bottom: 3px solid #ff7f0e;
        padding-bottom: 0.5rem;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 1rem;
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.2);
        text-align: center;
        margin: 0.5rem;
    }
    .stButton>button {
        background: linear-gradient(45deg, #FF4B2B, #FF416C);
        color: white;
        border: none;
        padding: 0.8rem 1.5rem;
        border-radius: 0.5rem;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.2);
    }
    .student-card {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 1rem;
        border-left: 5px solid #667eea;
        margin: 1rem 0;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# App title and description
st.markdown('<h1 class="main-header">🎓 Student Analytics Dashboard Pro</h1>', unsafe_allow_html=True)
st.markdown("""
### Advanced Student Performance Analysis Platform
Analyze student performance with interactive visualizations, detailed comparisons, and comprehensive insights.
""")

# Initialize session state for data persistence
if 'df' not in st.session_state:
    st.session_state.df = None
if 'file_uploaded' not in st.session_state:
    st.session_state.file_uploaded = False
if 'selected_students' not in st.session_state:
    st.session_state.selected_students = []
if 'current_view' not in st.session_state:
    st.session_state.current_view = "overview"
# ADD THIS LINE FOR THEME:
if 'current_theme' not in st.session_state:
    st.session_state.current_theme = "Default"

# File upload section in sidebar
with st.sidebar:
    st.header("📁 Data Upload")
    uploaded_file = st.file_uploader(
        "Choose a CSV or Excel file", 
        type=['csv', 'xlsx'],
        help="Upload your student data file with attendance and test scores."
    )
    
    if uploaded_file is not None:
        try:
            # Read the file
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            st.session_state.df = df
            st.session_state.file_uploaded = True
            st.success("✅ File uploaded successfully!")
            
            # Show basic info
            st.subheader("📊 Dataset Info")
            st.write(f"**Shape:** {df.shape[0]} rows, {df.shape[1]} columns")
            
        except Exception as e:
            st.error(f"❌ Error reading file: {e}")
    
    # Navigation
    st.header("🧭 Navigation")
    view_options = {
        "📊 Overview": "overview",
        "👥 Student Comparison": "comparison",
        "🏫 Class Performance": "class_performance",
        "📈 Individual Dashboards": "individual",
        "⚙️ Advanced Analysis": "advanced"
    }
    
    selected_view = st.radio("Select View", list(view_options.keys()))
    st.session_state.current_view = view_options[selected_view]

       # Display options (remove the broken theme selector)
    st.header("📊 Display Options")
    show_animations = st.checkbox("Show animations", value=True)
    data_points = st.slider("Max data points to show", 10, 200, 50)
        
    # Data sampling
    st.header("📋 Data Options")
    sample_size = st.slider("Sample Size", 10, 100, 50)
    show_raw_data = st.checkbox("Show Raw Data")

# If no file uploaded yet, show instructions
if not st.session_state.file_uploaded:
    st.info("📝 Please upload a data file using the sidebar to begin analysis.")
    
    # Sample data structure guidance
    with st.expander("💡 What should my data look like?"):
        st.markdown("""
        Your dataset should include:
        - **Student Identification**: ID, Name, Class
        - **Attendance Data**: Percentage or daily records
        - **Test Scores**: Multiple tests for different subjects
        
        Example structure:
        """)
        
        sample_data = pd.DataFrame({
            'Student_ID': ['S001', 'S002', 'S003'],
            'Name': ['John Doe', 'Jane Smith', 'Bob Johnson'],
            'Class': ['10A', '10A', '10B'],
            'Attendance_Pct': [95, 88, 92],
            'Math_Test1': [85, 92, 78],
            'Science_Test1': [88, 94, 82],
            'English_Test1': [92, 88, 85],
        })
        st.dataframe(sample_data)
    
    st.stop()

# Get the dataframe from session state
df = st.session_state.df

# Data preprocessing
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
attendance_cols = [col for col in df.columns if 'attendance' in col.lower() or 'pct' in col.lower()]
test_score_cols = [col for col in df.columns if 'test' in col.lower()]

# Overview Dashboard
if st.session_state.current_view == "overview":
    st.markdown('<h2 class="sub-header">📊 Overview Dashboard</h2>', unsafe_allow_html=True)
    
    # Quick filters
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        filter_class = st.selectbox("Filter by Class", ["All"] + list(df['Class'].unique()) if 'Class' in df.columns else ["All"])
    with col2:
        min_attendance = st.slider("Min Attendance", 0, 100, 60) if attendance_cols else st.empty()
    with col3:
        min_score = st.slider("Min Avg Score", 0, 100, 50) if test_score_cols else st.empty()
    with col4:
        st.write("")  # Spacer
        apply_filters = st.button("Apply Filters", type="primary")
    
    # Apply filters
    filtered_df = df.copy()
    if filter_class != "All" and 'Class' in df.columns:
        filtered_df = filtered_df[filtered_df['Class'] == filter_class]
    if attendance_cols:
        filtered_df = filtered_df[filtered_df[attendance_cols[0]] >= min_attendance]
    
    # Key metrics
    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    with metric_col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Total Students", filtered_df.shape[0], f"{filtered_df.shape[0] - df.shape[0]} after filter")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with metric_col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        if attendance_cols:
            avg_attendance = filtered_df[attendance_cols[0]].mean()
            st.metric("Avg Attendance", f"{avg_attendance:.1f}%", f"{avg_attendance - df[attendance_cols[0]].mean():.1f}%")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with metric_col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        if test_score_cols:
            avg_score = filtered_df[test_score_cols].mean().mean()
            st.metric("Avg Test Score", f"{avg_score:.1f}%", f"{avg_score - df[test_score_cols].mean().mean():.1f}%")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with metric_col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        if attendance_cols and test_score_cols:
            correlation = filtered_df[attendance_cols[0]].corr(filtered_df[test_score_cols].mean(axis=1))
            st.metric("Correlation", f"{correlation:.2f}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Visualizations
    tab1, tab2, tab3 = st.tabs(["📈 Performance Trends", "📊 Distribution", "📋 Top Performers"])
    
    with tab1:
        if test_score_cols and len(test_score_cols) > 1:
            test_avgs = filtered_df[test_score_cols].mean()
            fig = px.line(
                x=range(len(test_avgs)),
                y=test_avgs.values,
                title="Test Score Trends Over Time",
                labels={'x': 'Test Number', 'y': 'Average Score'}
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            if attendance_cols:
                fig = px.histogram(filtered_df, x=attendance_cols[0], title="Attendance Distribution")
                st.plotly_chart(fig, use_container_width=True)
        with col2:
            if test_score_cols:
                avg_scores = filtered_df[test_score_cols].mean(axis=1)
                fig = px.histogram(x=avg_scores, title="Average Score Distribution")
                st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        if test_score_cols:
            filtered_df['Average_Score'] = filtered_df[test_score_cols].mean(axis=1)
            top_students = filtered_df.nlargest(5, 'Average_Score')[['Name', 'Average_Score'] + (['Class'] if 'Class' in df.columns else [])]
            st.dataframe(top_students.style.highlight_max(axis=0))

# Student Comparison View
elif st.session_state.current_view == "comparison":
    st.markdown('<h2 class="sub-header">👥 Student Comparison</h2>', unsafe_allow_html=True)
    
    # Student selection
    student_names = df['Name'].tolist() if 'Name' in df.columns else [f"Student {i+1}" for i in range(len(df))]
    
    col1, col2 = st.columns(2)
    with col1:
        student1 = st.selectbox("Select First Student", student_names, key="student1")
    with col2:
        student2 = st.selectbox("Select Second Student", student_names, key="student2")
    
    if student1 and student2:
        # Get student data
        if 'Name' in df.columns:
            student1_data = df[df['Name'] == student1].iloc[0]
            student2_data = df[df['Name'] == student2].iloc[0]
        else:
            idx1 = student_names.index(student1)
            idx2 = student_names.index(student2)
            student1_data = df.iloc[idx1]
            student2_data = df.iloc[idx2]
        
        # Comparison metrics
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"### {student1}")
            st.markdown('<div class="student-card">', unsafe_allow_html=True)
            if 'Class' in df.columns:
                st.write(f"**Class:** {student1_data['Class']}")
            if attendance_cols:
                st.write(f"**Attendance:** {student1_data[attendance_cols[0]]}%")
            if test_score_cols:
                avg_score = student1_data[test_score_cols].mean()
                st.write(f"**Average Score:** {avg_score:.1f}%")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"### {student2}")
            st.markdown('<div class="student-card">', unsafe_allow_html=True)
            if 'Class' in df.columns:
                st.write(f"**Class:** {student2_data['Class']}")
            if attendance_cols:
                st.write(f"**Attendance:** {student2_data[attendance_cols[0]]}%")
            if test_score_cols:
                avg_score = student2_data[test_score_cols].mean()
                st.write(f"**Average Score:** {avg_score:.1f}%")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Radar chart comparison
        if test_score_cols:
            fig = go.Figure()
            
            fig.add_trace(go.Scatterpolar(
                r=student1_data[test_score_cols].values,
                theta=test_score_cols,
                fill='toself',
                name=student1
            ))
            
            fig.add_trace(go.Scatterpolar(
                r=student2_data[test_score_cols].values,
                theta=test_score_cols,
                fill='toself',
                name=student2
            ))
            
            fig.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                title="Test Score Comparison - Radar Chart",
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Bar chart comparison
        if test_score_cols:
            comparison_data = []
            for test in test_score_cols:
                comparison_data.append({
                    'Test': test,
                    student1: student1_data[test],
                    student2: student2_data[test]
                })
            
            comparison_df = pd.DataFrame(comparison_data)
            fig = px.bar(
                comparison_df,
                x='Test',
                y=[student1, student2],
                barmode='group',
                title="Test Score Comparison - Bar Chart"
            )
            st.plotly_chart(fig, use_container_width=True)

# Class Performance View
elif st.session_state.current_view == "class_performance":
    st.markdown('<h2 class="sub-header">🏫 Class Performance Analysis</h2>', unsafe_allow_html=True)
    
    if 'Class' in df.columns:
        # Class selection
        selected_class = st.selectbox("Select Class", df['Class'].unique())
        
        class_data = df[df['Class'] == selected_class]
        
        # Class statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Students in Class", len(class_data))
        with col2:
            if attendance_cols:
                st.metric("Class Attendance", f"{class_data[attendance_cols[0]].mean():.1f}%")
        with col3:
            if test_score_cols:
                st.metric("Class Average", f"{class_data[test_score_cols].mean().mean():.1f}%")
        with col4:
            if attendance_cols and test_score_cols:
                st.metric("Correlation", f"{class_data[attendance_cols[0]].corr(class_data[test_score_cols].mean(axis=1)):.2f}")
        
        # Class performance charts
        tab1, tab2 = st.tabs(["📊 Performance Distribution", "📈 Progress Over Time"])
        
        with tab1:
            col1, col2 = st.columns(2)
            with col1:
                if attendance_cols:
                    fig = px.box(class_data, y=attendance_cols[0], title=f"Attendance Distribution - {selected_class}")
                    st.plotly_chart(fig, use_container_width=True)
            with col2:
                if test_score_cols:
                    avg_scores = class_data[test_score_cols].mean(axis=1)
                    fig = px.histogram(x=avg_scores, title=f"Score Distribution - {selected_class}")
                    st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            if test_score_cols and len(test_score_cols) > 1:
                class_avgs = class_data[test_score_cols].mean()
                fig = px.line(
                    x=range(len(class_avgs)),
                    y=class_avgs.values,
                    title=f"Test Score Trends - {selected_class}",
                    labels={'x': 'Test Number', 'y': 'Average Score'}
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Class ranking
        if test_score_cols:
            class_data['Average_Score'] = class_data[test_score_cols].mean(axis=1)
            ranked_students = class_data.sort_values('Average_Score', ascending=False)[['Name', 'Average_Score', attendance_cols[0] if attendance_cols else '']]
            st.dataframe(ranked_students.head(10).style.highlight_max(axis=0))
    
    else:
        st.warning("No class information available in the dataset.")

# Individual Student Dashboards
elif st.session_state.current_view == "individual":
    st.markdown('<h2 class="sub-header">📈 Individual Student Dashboard</h2>', unsafe_allow_html=True)
    
    student_names = df['Name'].tolist() if 'Name' in df.columns else [f"Student {i+1}" for i in range(len(df))]
    selected_student = st.selectbox("Select Student", student_names)
    
    if selected_student:
        # Get student data
        if 'Name' in df.columns:
            student_data = df[df['Name'] == selected_student].iloc[0]
        else:
            idx = student_names.index(selected_student)
            student_data = df.iloc[idx]
        
        # Student info card
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown('<div class="student-card">', unsafe_allow_html=True)
            st.write(f"**Student:** {selected_student}")
            if 'Class' in df.columns:
                st.write(f"**Class:** {student_data['Class']}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="student-card">', unsafe_allow_html=True)
            if attendance_cols:
                st.write(f"**Attendance:** {student_data[attendance_cols[0]]}%")
                # Attendance status
                attendance = student_data[attendance_cols[0]]
                if attendance >= 90:
                    st.success("Excellent Attendance 🎯")
                elif attendance >= 75:
                    st.info("Good Attendance 👍")
                else:
                    st.warning("Needs Improvement ⚠️")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="student-card">', unsafe_allow_html=True)
            if test_score_cols:
                avg_score = student_data[test_score_cols].mean()
                st.write(f"**Average Score:** {avg_score:.1f}%")
                # Performance status
                if avg_score >= 90:
                    st.success("Top Performer 🌟")
                elif avg_score >= 75:
                    st.info("Good Performance 📈")
                else:
                    st.warning("Needs Support 📚")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Performance charts
        if test_score_cols:
            # Progress chart
            test_values = [student_data[test] for test in test_score_cols]
            fig = px.line(
                x=test_score_cols,
                y=test_values,
                title=f"Test Scores for {selected_student}",
                labels={'x': 'Test', 'y': 'Score'}
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Subject performance
            subject_scores = {}
            for test in test_score_cols:
                subject = test.split('_')[0]  # Extract subject from test name
                if subject not in subject_scores:
                    subject_scores[subject] = []
                subject_scores[subject].append(student_data[test])
            
            # Calculate average per subject
            subject_avgs = {subject: np.mean(scores) for subject, scores in subject_scores.items()}
            fig = px.bar(
                x=list(subject_avgs.keys()),
                y=list(subject_avgs.values()),
                title=f"Subject-wise Performance - {selected_student}",
                labels={'x': 'Subject', 'y': 'Average Score'}
            )
            st.plotly_chart(fig, use_container_width=True)

# Advanced Analysis View
elif st.session_state.current_view == "advanced":
    st.markdown('<h2 class="sub-header">⚙️ Advanced Analysis</h2>', unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["📈 Correlation Matrix", "📊 Cluster Analysis", "🔮 Predictive Insights"])
    
    with tab1:
        if len(numeric_cols) > 1:
            corr_matrix = df[numeric_cols].corr()
            fig = px.imshow(
                corr_matrix,
                title="Correlation Matrix",
                aspect="auto",
                color_continuous_scale='RdBu_r'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.info("Cluster analysis would group students with similar performance patterns.")
        if st.button("Run Cluster Analysis", type="primary"):
            st.warning("This feature requires additional data preprocessing and would typically use K-Means or DBSCAN clustering.")
    
    with tab3:
        st.info("Predictive insights could forecast future performance based on historical data.")
        if st.button("Generate Predictions", type="primary"):
            st.warning("This would typically involve machine learning models like linear regression or time series forecasting.")

# Footer
st.markdown("---")
st.markdown("### 🎓 About This Advanced Dashboard")
st.markdown("""
This enhanced Student Analytics Dashboard provides comprehensive insights into student performance with:
- **Interactive filters** and **sliders** for dynamic data exploration
- **Side-by-side student comparison** with radar and bar charts
- **Detailed class performance** analysis with rankings
- **Individual student dashboards** with performance metrics
- **Advanced analytics** including correlation matrices
- **Beautiful visualizations** with custom styling

Upload your student data to unlock all these features!
""")

# Data export section
with st.sidebar:
    st.header("💾 Export Data")
    if st.button("Export Processed Data"):
        csv = df.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name="student_analytics_data.csv",
            mime="text/csv"
        )
