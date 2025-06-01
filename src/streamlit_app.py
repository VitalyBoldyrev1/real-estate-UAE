from catboost import CatBoostRegressor
from pathlib import Path
import datetime
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Dubai Real Estate Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(90deg, #1f4e79, #2e8bc0);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .metric-container {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #2e8bc0;
        margin: 0.5rem 0;
    }
    .success-box {
        background: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .info-box {
        background: #e3f2fd;
        border: 1px solid #90caf9;
        border-radius: 5px;
        padding: 0.75rem;
        margin: 0.5rem 0;
        font-size: 0.9rem;
        color: #1565c0;
        font-weight: 500;
    }
    .stButton > button {
        background: linear-gradient(90deg, #1f4e79, #2e8bc0);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.5rem 2rem;
        font-weight: bold;
        transition: all 0.3s;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.2);
    }
</style>
""", unsafe_allow_html=True)

# Constants and configuration
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "dubai_model_v11.cbm"
PROJECT_NAME_PATH = BASE_DIR / "data" / "project_name_en.txt"
MASTER_PROJECT_PATH = BASE_DIR / "data" / "master_project_en.txt"

# Currency conversion rates
CURRENCY_RATES = {
    "AED_TO_RUB": 21.739,  # 1 AED = 21.739 RUB (1 RUB = 0.046 AED)
    "AED_TO_USD": 0.2725,  # 1 AED = 0.2725 USD (1 USD = 3.67 AED)
}
EXCHANGE_RATE_DATE = "May 28, 2025"
DATASET_DATE = "March 14, 2025"

# Options for dropdowns
TRANS_GROUP_EN_OPTIONS = sorted(["Sales", "Mortgages", "Gifts"])
REG_TYPE_EN_OPTIONS = sorted(["Existing Properties", "Off-Plan Properties"])
PROCEDURE_NAME_EN_GROUPED_OPTIONS = sorted([
    "Standard Sale", "Mortgage", "Grant", "Development", 
    "Lease Agreement", "Portfolio", "Other_Transaction"
])
DISTRICT_OPTIONS = sorted([
    "Dubai Marina & JBR", "Al Barsha & Al Quoz", "Eastern Dubai",
    "Meydan & Nad Al Shiba", "Palm Jumeirah", "Dubai South / New Developments",
    "TECOM, Greens & Emirates Hills Area", "Airport & Nearby Areas",
    "Downtown Dubai & Business Bay", "Jebel Ali & Dubai South West",
    "Bur Dubai", "Coastal Strip (Jumeirah/Umm Suqeim)",
    "North-Eastern Dubai", "Deira", "Islands & Special Zones",
    "Industrial Areas (Central/East)", "Hatta"
])

UNKNOWN_VALUE_PLACEHOLDER = "Unknown"

@st.cache_resource
def load_model_and_artifacts():
    """Load ML model and project data with enhanced error handling"""
    model = None
    known_projects = {UNKNOWN_VALUE_PLACEHOLDER}
    known_master_projects = {UNKNOWN_VALUE_PLACEHOLDER}

    # Load model with progress indicator
    with st.spinner("Loading prediction model..."):
        try:
            if not MODEL_PATH.exists():
                st.error(f"Model file not found: {MODEL_PATH}")
                return None, known_projects, known_master_projects
                
            model_loader = CatBoostRegressor()
            model_loader.load_model(MODEL_PATH)
            model = model_loader
            st.success("✅ Model loaded successfully!")
        except Exception as e:
            st.error(f"❌ Error loading model from '{MODEL_PATH}': {e}")
            return None, known_projects, known_master_projects

    def load_known_values_from_file(path, default_set):
        """Load known values with better error handling"""
        try:
            if not path.exists():
                st.warning(f"⚠️ File not found: {path}")
                return default_set
                
            with open(path, "r", encoding="utf-8") as f:
                values = set(line.strip() for line in f if line.strip())
                values.add(UNKNOWN_VALUE_PLACEHOLDER)
                return values
        except Exception as e:
            st.warning(f"⚠️ Error loading {path}: {e}")
            return default_set

    # Load project data
    known_projects.update(load_known_values_from_file(PROJECT_NAME_PATH, known_projects))
    known_master_projects.update(load_known_values_from_file(MASTER_PROJECT_PATH, known_master_projects))

    return model, known_projects, known_master_projects

def validate_project_name(name, known_set, field_name):
    """Validate and process project names with user feedback"""
    final_name = name.strip()
    if not final_name:
        return UNKNOWN_VALUE_PLACEHOLDER
    elif final_name not in known_set:
        st.warning(f"⚠️ {field_name} '{final_name}' not found in database. Using '{UNKNOWN_VALUE_PLACEHOLDER}' instead.")
        return UNKNOWN_VALUE_PLACEHOLDER
    else:
        st.success(f"✅ {field_name} '{final_name}' found in database.")
        return final_name

def create_input_dataframe(trans_group, date_val, reg_type, project_name, master_project, 
                         area, proc_name_grouped, district_val, known_projects_set, 
                         known_master_projects_set, unknown_placeholder):
    """Create input DataFrame with enhanced validation"""
    
    final_project_name = validate_project_name(
        project_name, known_projects_set, "Project name"
    )
    final_master_project_name = validate_project_name(
        master_project, known_master_projects_set, "Developer name"
    )

    input_data = {
        "trans_group_en": trans_group,
        "date": pd.to_datetime(date_val),
        "reg_type_en": reg_type,
        "project_name_en": final_project_name,
        "master_project_en": final_master_project_name,
        "procedure_area": float(area),
        "procedure_name_en_grouped": proc_name_grouped,
        "district": district_val,
    }

    feature_order = [
        "trans_group_en", "date", "reg_type_en", "project_name_en",
        "master_project_en", "procedure_area", "procedure_name_en_grouped", "district"
    ]

    return pd.DataFrame([input_data])[feature_order]

def convert_currency(amount_aed):
    """Convert AED to other currencies"""
    return {
        "AED": amount_aed,
        "RUB": amount_aed * CURRENCY_RATES["AED_TO_RUB"],
        "USD": amount_aed * CURRENCY_RATES["AED_TO_USD"]
    }

# Load model and data
model, KNOWN_PROJECT_NAMES, KNOWN_MASTER_PROJECT_NAMES = load_model_and_artifacts()

# Main UI
st.markdown("""
<div class="main-header">
    <h1>🏠 Dubai Real Estate Price Predictor 💰</h1>
    <p>Advanced ML-powered property valuation for Dubai real estate market</p>
</div>
""", unsafe_allow_html=True)

# Information about the app
with st.expander("ℹ️ About This Application", expanded=False):
    st.markdown("""
    **What it does:** This application uses advanced machine learning (CatBoost) to predict real estate prices in Dubai.
    
    **How it works:** 
    - Enter property details in the form below
    - Our AI model analyzes market trends and comparable properties
    - Get instant price predictions in multiple currencies
    
    **Data Source:** Open data from Dubai Pulse (Government of Dubai)
    """)

if model is None:
    st.error("🚫 The prediction model failed to load. Please contact the administrator.")
    st.stop()

# Create form for better UX
with st.form("prediction_form"):
    col1, col2, col3 = st.columns([1, 1, 0.8])
    
    with col1:
        st.markdown("### 📋 Transaction Details")
        trans_group_en_input = st.selectbox(
            "Transaction Type",
            options=TRANS_GROUP_EN_OPTIONS,
            help="Type of real estate transaction"
        )
        
        procedure_name_en_grouped_input = st.selectbox(
            "Procedure Type",
            options=PROCEDURE_NAME_EN_GROUPED_OPTIONS,
            help="Specific procedure category"
        )
        
        reg_type_en_input = st.selectbox(
            "Registration Type",
            options=REG_TYPE_EN_OPTIONS,
            help="Property registration category"
        )

    with col2:
        st.markdown("### 🏢 Property Details")
        procedure_area_input = st.number_input(
            "Property Area (sq.m.)",
            min_value=10.0,
            max_value=10000.0,
            value=100.0,
            step=5.0,
            help="Total area of the property in square meters"
        )
        
        district_input = st.selectbox(
            "District",
            options=DISTRICT_OPTIONS,
            help="Dubai district where the property is located"
        )

    with col3:
        st.markdown("### 🏗️ Project Information")
        st.markdown('<div class="info-box">💡 <strong>Tip:</strong> Leave fields empty to use default values or use the search options below</div>', 
                   unsafe_allow_html=True)
        
        project_name_input_str = st.text_input(
            "Project Name (Optional)",
            placeholder="e.g., Marina Gate, Burj Khalifa",
            help="Specific project name (optional)"
        )
        
        master_project_name_input_str = st.text_input(
            "Developer Name (Optional)",
            placeholder="e.g., Emaar, DAMAC",
            help="Developer or master project name (optional)"
        )

    # Help sections in expandable format
    col_help1, col_help2 = st.columns(2)
    
    with col_help1:
        with st.expander("🔍 Search Project Names"):
            selected_project = st.selectbox(
                "Available Projects",
                options=[""] + sorted(list(KNOWN_PROJECT_NAMES - {UNKNOWN_VALUE_PLACEHOLDER})),
                help="Select from known projects in our database"
            )
            if selected_project:
                st.info(f"Selected: **{selected_project}** - Copy this to the Project Name field above")

    with col_help2:
        with st.expander("🏢 Search Developer Names"):
            selected_developer = st.selectbox(
                "Available Developers",
                options=[""] + sorted(list(KNOWN_MASTER_PROJECT_NAMES - {UNKNOWN_VALUE_PLACEHOLDER})),
                help="Select from known developers in our database"
            )
            if selected_developer:
                st.info(f"Selected: **{selected_developer}** - Copy this to the Developer Name field above")

    # Prediction button
    predict_button = st.form_submit_button(
        "🚀 Predict Property Price",
        use_container_width=True,
        type="primary"
    )

# Process prediction
if predict_button:
    with st.spinner("🔄 Analyzing market data and generating prediction..."):
        try:
            current_date_val = datetime.date.today()
            input_df = create_input_dataframe(
                trans_group=trans_group_en_input,
                date_val=current_date_val,
                reg_type=reg_type_en_input,
                project_name=project_name_input_str,
                master_project=master_project_name_input_str,
                area=procedure_area_input,
                proc_name_grouped=procedure_name_en_grouped_input,
                district_val=district_input,
                known_projects_set=KNOWN_PROJECT_NAMES,
                known_master_projects_set=KNOWN_MASTER_PROJECT_NAMES,
                unknown_placeholder=UNKNOWN_VALUE_PLACEHOLDER,
            )

            # Make prediction
            prediction = model.predict(input_df)
            predicted_price_per_sqm_aed = np.expm1(prediction[0])
            total_price_aed = predicted_price_per_sqm_aed * procedure_area_input

            # Convert to multiple currencies
            price_per_sqm_currencies = convert_currency(predicted_price_per_sqm_aed)
            total_price_currencies = convert_currency(total_price_aed)

            # Display results
            st.markdown("## 🎯 Prediction Results")

            # Price metrics in columns
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("#### 🇦🇪 AED (Dirhams)")
                st.metric(
                    "Price per sq.m.",
                    f"{price_per_sqm_currencies['AED']:,.0f} AED",
                    help="Price per square meter in UAE Dirhams"
                )
                st.metric(
                    "Total Property Price",
                    f"{total_price_currencies['AED']:,.0f} AED",
                    help="Total estimated property value in UAE Dirhams"
                )

            with col2:
                st.markdown("#### 🇺🇸 USD (Dollars)")
                st.metric(
                    "Price per sq.m.",
                    f"${price_per_sqm_currencies['USD']:,.0f}",
                    help="Price per square meter in US Dollars"
                )
                st.metric(
                    "Total Property Price",
                    f"${total_price_currencies['USD']:,.0f}",
                    help="Total estimated property value in US Dollars"
                )

            with col3:
                st.markdown("#### 🇷🇺 RUB (Rubles)")
                st.metric(
                    "Price per sq.m.",
                    f"{price_per_sqm_currencies['RUB']:,.0f} ₽",
                    help="Price per square meter in Russian Rubles"
                )
                st.metric(
                    "Total Property Price",
                    f"{total_price_currencies['RUB']:,.0f} ₽",
                    help="Total estimated property value in Russian Rubles"
                )

        except Exception as e:
            st.error(f"❌ Error during prediction: {str(e)}")
            with st.expander("Technical Details"):
                st.exception(e)

# Enhanced sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Model Information")
st.sidebar.info(f"""
**Model Type:** CatBoost Regressor  
**Data Source:** Dubai Pulse (Gov. of Dubai)  
**Current Projects in Dubai:** {len(KNOWN_PROJECT_NAMES):,}  
**Current Developers in Dubai:** {len(KNOWN_MASTER_PROJECT_NAMES):,}  
**Dataset Date:** {DATASET_DATE}
""")

st.sidebar.markdown("### 💱 Currency Rates")
st.sidebar.caption(f"Fixed rates as of {EXCHANGE_RATE_DATE}")
st.sidebar.text("1 AED = 0.27 USD")
st.sidebar.text("1 AED = 21.74 RUB")

st.sidebar.markdown("### ℹ️ Disclaimer")
st.sidebar.caption("""
Predictions are estimates based on historical data and market trends. 
Actual prices may vary based on specific property conditions, 
market fluctuations, and other factors.
""")
