# =============================
# Session State Initialization
# =============================
if "df" not in st.session_state:
    st.session_state.df = None
if "file_uploaded" not in st.session_state:
    st.session_state.file_uploaded = False
if "detected_cols" not in st.session_state:
    st.session_state.detected_cols = {}
if "current_view" not in st.session_state:
    st.session_state.current_view = "overview"  # Default view

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
# Main Content Switcher
# =============================
if st.session_state.current_view == "data_upload":
    st.header("📤 Data Upload")
    # (your upload logic goes here...)

elif st.session_state.current_view == "overview":
    st.header("📊 Overview Dashboard")
    if not st.session_state.file_uploaded:
        st.info("Upload data first to see overview.")
    else:
        # (overview logic goes here...)
        pass

elif st.session_state.current_view == "comparison":
    st.header("👥 Student Comparison")
    # (comparison logic...)

elif st.session_state.current_view == "class_analytics":
    st.header("🏫 Class Analytics")
    # (class analysis logic...)

elif st.session_state.current_view == "individual":
    st.header("📈 Individual Insights")
    # (individual insights logic...)

elif st.session_state.current_view == "advanced":
    st.header("🔍 Advanced Analysis")
    # (advanced analysis logic...)

elif st.session_state.current_view == "documentation":
    st.header("📖 Documentation")
    st.markdown("""
    Welcome to **Right iTech Educational Analytics Platform**! 🎓  

    Use the sidebar to navigate between sections:
    - **Data Upload** → Load your dataset.
    - **Overview Dashboard** → Quick stats & trends.
    - **Student Comparison** → Compare two students.
    - **Class Analytics** → Deep dive into a class.
    - **Individual Insights** → Track one student.
    - **Advanced Analysis** → Correlations & patterns.
    """)

# Advanced column detection with machine learning-like pattern matching
def advanced_column_detection(df):
    detected = {
        'id_cols': [],
        'name_cols': [],
        'class_cols': [],
        'attendance_cols': [],
        'score_cols': [],
        'date_cols': [],
        'demographic_cols': []
    }
    
    for col in df.columns:
        col_lower = str(col).lower()
        col_dtype = str(df[col].dtype)
        
        # Multi-factor column detection
        score = 0

        # ID columns detection
        if any(x in col_lower for x in ['id', 'roll', 'number', 'code', 'studentid', 'reg']):
            if col_dtype in ['int64', 'float64'] or (df[col].nunique() == len(df)):
                detected['id_cols'].append(col)

        # Name columns detection  
        elif any(x in col_lower for x in ['name', 'student', 'fullname', 'first', 'last']):
            if col_dtype == 'object' and df[col].nunique() > 1:
                detected['name_cols'].append(col)
        
        # Class columns detection
        elif any(x in col_lower for x in ['class', 'section', 'grade', 'form', 'batch', 'group']):
            detected['class_cols'].append(col)
        
        # Attendance columns detection
        elif any(x in col_lower for x in ['attendance', 'present', 'absent', 'pct', '%', 'rate']):
            if col_dtype in ['int64', 'float64']:
                detected['attendance_cols'].append(col)
        
        # Score columns detection
        elif any(x in col_lower for x in ['test', 'exam', 'score', 'mark', 'assessment', 'quiz', 'assignment']):
            if col_dtype in ['int64', 'float64']:
                detected['score_cols'].append(col)
        
        # Date columns detection
        elif any(x in col_lower for x in ['date', 'time', 'day', 'month', 'year']):
            detected['date_cols'].append(col)
        
        # Demographic columns
        elif any(x in col_lower for x in ['age', 'gender', 'dob', 'background', 'category']):
            detected['demographic_cols'].append(col)
    
    # Auto-select best candidates
    final_mapping = {
        'primary_id': detected['id_cols'][0] if detected['id_cols'] else None,
        'primary_name': detected['name_cols'][0] if detected['name_cols'] else None,
        'primary_class': detected['class_cols'][0] if detected['class_cols'] else None,
        'primary_attendance': detected['attendance_cols'][0] if detected['attendance_cols'] else None,
        'score_columns': detected['score_cols'],
        'demographic_columns': detected['demographic_cols']
    }
    
    return final_mapping
