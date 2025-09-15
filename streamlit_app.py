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

# Set page configuration
st.set_page_config(
    page_title="Student Analytics Dashboard",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        border-bottom: 2px solid #ff7f0e;
        padding-bottom: 0.3rem;
        margin-top: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
</style>
""", unsafe_allow_html=True)

# App title and description
st.markdown('<h1 class="main-header">🎓 Student Analytics Dashboard</h1>', unsafe_allow_html=True)
st.markdown("""
This application helps you analyze student performance data including attendance and test scores.
Upload your data file (CSV or XLSX) to get started.
""")

# Initialize session state for data persistence
if 'df' not in st.session_state:
    st.session_state.df = None
if 'file_uploaded' not in st.session_state:
    st.session_state.file_uploaded = False

# File upload section in sidebar
with st.sidebar:
    st.header("Data Upload")
    uploaded_file = st.file_uploader(
        "Choose a CSV or Excel file",
        type=['csv', 'xlsx'],
        help="Upload your student data file. It should contain columns for student names, attendance, and test scores."
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
            st.success("File uploaded successfully!")

            # Show basic info about the dataset
            st.subheader("Dataset Info")
            st.write(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns")

            # Show column names
            st.write("Columns:")
            st.write(list(df.columns))

        except Exception as e:
            st.error(f"Error reading file: {e}")

    # Add some useful information in the sidebar
    st.header("Instructions")
    st.info("""
    1. Upload a CSV or Excel file with student data
    2. Ensure your data includes columns for:
       - Student names/IDs
       - Attendance records
       - Test scores by subject
    3. Use the filters to analyze specific data segments
    4. Explore the visualizations to gain insights
    """)

# If no file uploaded yet, show instructions
if not st.session_state.file_uploaded:
    st.info("Please upload a data file using the sidebar to begin analysis.")

    # Sample data structure guidance
    with st.expander("What should my data look like?"):
        st.markdown("""
        Your dataset should ideally include:

        - **Student Identification**: Student ID, Name, Class/Section
        - **Attendance Data**: Dates with attendance status (Present/Absent) or percentage
        - **Test Scores**: Weekly test scores for different subjects

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
            'Math_Test2': [82, 96, 80],
            'Science_Test2': [90, 92, 84],
            'English_Test2': [90, 86, 88]
        })
        st.dataframe(sample_data)

    st.stop()

# Get the dataframe from session state
df = st.session_state.df

# Data preprocessing and setup
st.markdown('<h2 class="sub-header">Data Overview</h2>', unsafe_allow_html=True)

# Show the first few rows of the dataframe
if st.checkbox("Show raw data"):
    st.dataframe(df)

# Basic statistics
st.subheader("Basic Statistics")
st.dataframe(df.describe())

# Identify numeric columns for analysis
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

# Check if we have attendance and test score columns
attendance_cols = [col for col in df.columns if 'attendance' in col.lower() or 'pct' in col.lower()]
test_score_cols = [col for col in df.columns if 'test' in col.lower()]

# If we can't automatically detect columns, let the user specify
if not attendance_cols or not test_score_cols:
    st.warning("Could not automatically identify attendance and test score columns.")

    with st.expander("Manual Column Identification"):
        st.write("Please specify which columns contain the following data:")

        # Let user map columns to data types
        attendance_col = st.selectbox(
            "Attendance column",
            options=df.columns,
            index=0 if not attendance_cols else df.columns.get_loc(attendance_cols[0])
        )

        # Let user select test score columns
        test_cols = st.multiselect(
            "Test score columns",
            options=df.columns,
            default=test_score_cols if test_score_cols else numeric_cols[:3]
        )

        # Update our column lists
        attendance_cols = [attendance_col]
        test_score_cols = test_cols

# Filters section
st.markdown('<h2 class="sub-header">Data Filters</h2>', unsafe_allow_html=True)

# Create filters based on available columns
col1, col2, col3 = st.columns(3)

# Filter by class/section if available
class_filter = None
if any('class' in col.lower() for col in df.columns):
    class_col = [col for col in df.columns if 'class' in col.lower()][0]
    class_options = ['All'] + list(df[class_col].unique())
    class_filter = col1.selectbox("Filter by Class", options=class_options)

# Filter by attendance range
if attendance_cols:
    attendance_min = df[attendance_cols[0]].min()
    attendance_max = df[attendance_cols[0]].max()
    attendance_range = col2.slider(
        "Attendance Range",
        min_value=float(attendance_min),
        max_value=float(attendance_max),
        value=(float(attendance_min), float(attendance_max))
    )

# Apply filters
filtered_df = df.copy()

if class_filter and class_filter != 'All' and 'class' in [col.lower() for col in df.columns]:
    class_col = [col for col in df.columns if 'class' in col.lower()][0]
    filtered_df = filtered_df[filtered_df[class_col] == class_filter]

if attendance_cols:
    filtered_df = filtered_df[
        (filtered_df[attendance_cols[0]] >= attendance_range[0]) &
        (filtered_df[attendance_cols[0]] <= attendance_range[1])
    ]

# Display filtered dataset info
st.write(f"Filtered dataset: {filtered_df.shape[0]} rows, {filtered_df.shape[1]} columns")

# Key metrics
st.markdown('<h2 class="sub-header">Key Metrics</h2>', unsafe_allow_html=True)

# Calculate and display important metrics
metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)

with metric_col1:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric("Total Students", filtered_df.shape[0])
    st.markdown('</div>', unsafe_allow_html=True)

with metric_col2:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    if attendance_cols:
        avg_attendance = filtered_df[attendance_cols[0]].mean()
        st.metric("Average Attendance", f"{avg_attendance:.1f}%")
    else:
        st.metric("Average Attendance", "N/A")
    st.markdown('</div>', unsafe_allow_html=True)

with metric_col3:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    if test_score_cols:
        avg_score = filtered_df[test_score_cols].mean().mean()
        st.metric("Average Test Score", f"{avg_score:.1f}%")
    else:
        st.metric("Average Test Score", "N/A")
    st.markdown('</div>', unsafe_allow_html=True)

with metric_col4:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    if attendance_cols and test_score_cols:
        correlation = filtered_df[attendance_cols[0]].corr(filtered_df[test_score_cols].mean(axis=1))
        st.metric("Attendance-Score Correlation", f"{correlation:.2f}")
    else:
        st.metric("Attendance-Score Correlation", "N/A")
    st.markdown('</div>', unsafe_allow_html=True)

# Visualization section
st.markdown('<h2 class="sub-header">Data Visualizations</h2>', unsafe_allow_html=True)

# Create tabs for different visualizations
viz_tabs = st.tabs(["Attendance Analysis", "Test Performance", "Correlation Analysis", "Student Comparison"])

# Tab 1: Attendance Analysis
with viz_tabs[0]:
    st.subheader("Attendance Analysis")

    if attendance_cols:
        # Attendance distribution
        fig = px.histogram(
            filtered_df,
            x=attendance_cols[0],
            title="Distribution of Attendance",
            nbins=20
        )
        st.plotly_chart(fig, use_container_width=True)

        # Attendance by class (if available)
        if any('class' in col.lower() for col in filtered_df.columns):
            class_col = [col for col in filtered_df.columns if 'class' in col.lower()][0]
            fig = px.box(
                filtered_df,
                x=class_col,
                y=attendance_cols[0],
                title="Attendance by Class"
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No attendance data available for visualization.")

# Tab 2: Test Performance
with viz_tabs[1]:
    st.subheader("Test Performance Analysis")

    if test_score_cols:
        # Average scores by test
        test_means = filtered_df[test_score_cols].mean()
        fig = px.bar(
            x=test_means.index,
            y=test_means.values,
            title="Average Scores by Test",
            labels={'x': 'Test', 'y': 'Average Score'}
        )
        st.plotly_chart(fig, use_container_width=True)

        # Test score distribution
        selected_test = st.selectbox("Select test to view distribution", test_score_cols)
        fig = px.histogram(
            filtered_df,
            x=selected_test,
            title=f"Distribution of Scores for {selected_test}",
            nbins=20
        )
        st.plotly_chart(fig, use_container_width=True)

        # Test scores over time (if tests are ordered)
        if len(test_score_cols) > 1:
            # Calculate average for each test
            test_avgs = filtered_df[test_score_cols].mean()
            fig = px.line(
                x=range(len(test_avgs)),
                y=test_avgs.values,
                title="Test Score Trends Over Time",
                labels={'x': 'Test Number', 'y': 'Average Score'}
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No test score data available for visualization.")

# Tab 3: Correlation Analysis
with viz_tabs[2]:
    st.subheader("Correlation Analysis")

    if attendance_cols and test_score_cols:
        # Scatter plot of attendance vs average test score
        filtered_df['avg_test_score'] = filtered_df[test_score_cols].mean(axis=1)
        fig = px.scatter(
            filtered_df,
            x=attendance_cols[0],
            y='avg_test_score',
            title="Attendance vs Test Scores",
            trendline="ols"
        )
        st.plotly_chart(fig, use_container_width=True)

        # Correlation heatmap
        corr_data = filtered_df[test_score_cols + attendance_cols]
        corr_matrix = corr_data.corr()

        fig = px.imshow(
            corr_matrix,
            title="Correlation Matrix",
            aspect="auto",
            color_continuous_scale='RdBu_r'
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Need both attendance and test score data for correlation analysis.")

# Tab 4: Student Comparison
with viz_tabs[3]:
    st.subheader("Student Comparison")

    # Select students to compare
    if 'name' in [col.lower() for col in filtered_df.columns]:
        name_col = [col for col in filtered_df.columns if 'name' in col.lower()][0]
        student_options = filtered_df[name_col].tolist()
    else:
        # If no name column, use index
        name_col = None
        student_options = [f"Student {i+1}" for i in range(filtered_df.shape[0])]

    selected_students = st.multiselect(
        "Select students to compare",
        options=student_options,
        default=student_options[:2] if len(student_options) >= 2 else student_options
    )

    if selected_students:
        # Get indices of selected students
        if name_col:
            student_indices = [student_options.index(student) for student in selected_students]
        else:
            student_indices = [int(student.split(" ")[1]) - 1 for student in selected_students]

        # Create comparison data
        if test_score_cols:
            comparison_data = []
            for i, student_idx in enumerate(student_indices):
                for test in test_score_cols:
                    comparison_data.append({
                        'Student': selected_students[i],
                        'Test': test,
                        'Score': filtered_df.iloc[student_idx][test]
                    })

            comparison_df = pd.DataFrame(comparison_data)

            # Create radar chart for comparison
            fig = go.Figure()

            for student in selected_students:
                student_scores = comparison_df[comparison_df['Student'] == student]['Score']
                fig.add_trace(go.Scatterpolar(
                    r=student_scores,
                    theta=test_score_cols,
                    fill='toself',
                    name=student
                ))

            fig.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                title="Student Comparison - Test Scores"
            )

            st.plotly_chart(fig, use_container_width=True)

        # Also show bar chart comparison
        if test_score_cols:
            fig = px.bar(
                comparison_df,
                x='Test',
                y='Score',
                color='Student',
                barmode='group',
                title="Test Score Comparison by Student"
            )
            st.plotly_chart(fig, use_container_width=True)

# Student details section
st.markdown('<h2 class="sub-header">Individual Student Details</h2>', unsafe_allow_html=True)

# Select a student to view detailed information
if name_col:
    selected_student = st.selectbox("Select a student to view details", options=student_options)

    if selected_student:
        student_data = filtered_df[filtered_df[name_col] == selected_student].iloc[0]

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Basic Information")
            st.write(f"**Name**: {student_data[name_col]}")

            if 'class' in [col.lower() for col in filtered_df.columns]:
                class_col = [col for col in filtered_df.columns if 'class' in col.lower()][0]
                st.write(f"**Class**: {student_data[class_col]}")

            if attendance_cols:
                st.write(f"**Attendance**: {student_data[attendance_cols[0]]}%")

        with col2:
            st.subheader("Test Scores")
            if test_score_cols:
                for test in test_score_cols:
                    st.write(f"**{test}**: {student_data[test]}%")

                avg_score = student_data[test_score_cols].mean()
                st.write(f"**Average Score**: {avg_score:.1f}%")

        # Progress chart for the selected student
        if test_score_cols and len(test_score_cols) > 1:
            st.subheader("Progress Over Tests")
            test_values = [student_data[test] for test in test_score_cols]

            fig = px.line(
                x=test_score_cols,
                y=test_values,
                title=f"Test Scores for {selected_student}",
                labels={'x': 'Test', 'y': 'Score'}
            )
            st.plotly_chart(fig, use_container_width=True)

# Data export section
st.markdown('<h2 class="sub-header">Export Results</h2>', unsafe_allow_html=True)

# Allow users to download processed data
csv = filtered_df.to_csv(index=False)
st.download_button(
    label="Download filtered data as CSV",
    data=csv,
    file_name="filtered_student_data.csv",
    mime="text/csv"
)

# Footer
st.markdown("---")
st.markdown("### About This App")
st.markdown("""
This Student Analytics Dashboard helps educators analyze student performance data.
It provides insights into attendance patterns, test performance, and correlations between different metrics.
""")
