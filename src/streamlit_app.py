from catboost import CatBoostRegressor
import datetime
import numpy as np
import pandas as pd
import streamlit as st
st.set_page_config(layout="wide")
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "dubai_model_v11.cbm"
PROJECT_NAME_PATH = BASE_DIR / "data" / "project_name_en.txt"
MASTER_PROJECT_PATH = BASE_DIR / "data" / "master_project_en.txt"


TRANS_GROUP_EN_OPTIONS = sorted(['Sales', 'Mortgages', 'Gifts'])
REG_TYPE_EN_OPTIONS = sorted(['Existing Properties', 'Off-Plan Properties'])
PROCEDURE_NAME_EN_GROUPED_OPTIONS = sorted(
    [
        'Standard Sale',
        'Mortgage',
        'Grant',
        'Development',
        'Lease Agreement',
        'Portfolio',
        'Other_Transaction'])
DISTRICT_OPTIONS = sorted(['Dubai Marina & JBR',
                           'Al Barsha & Al Quoz',
                           'Eastern Dubai',
                           'Meydan & Nad Al Shiba',
                           'Palm Jumeirah',
                           'Dubai South / New Developments',
                           'TECOM, Greens & Emirates Hills Area',
                           'Airport & Nearby Areas',
                           'Downtown Dubai & Business Bay',
                           'Jebel Ali & Dubai South West',
                           'Bur Dubai',
                           'Coastal Strip (Jumeirah/Umm Suqeim)',
                           'North-Eastern Dubai',
                           'Deira',
                           'Islands & Special Zones',
                           'Industrial Areas (Central/East)',
                           'Hatta'])

UNKNOWN_VALUE_PLACEHOLDER = "Unknown"


@st.cache_resource
def load_model_and_artifacts():
    model = None
    known_projects = {UNKNOWN_VALUE_PLACEHOLDER}
    known_master_projects = {UNKNOWN_VALUE_PLACEHOLDER}

    try:
        model_loader = CatBoostRegressor()
        model_loader.load_model(MODEL_PATH)
        model = model_loader
        print("Success!")
    except Exception as e:
        st.error(f"Error with '{MODEL_PATH}': {e}")

    def load_known_values_from_file(path, default_set):
        try:
            with open(path, 'r') as f:
                values = set(line.strip() for line in f if line.strip())
                values.add(UNKNOWN_VALUE_PLACEHOLDER)
                return values
        except Exception as e:
            st.warning(f"Error {path}: {e}")
            return default_set

    known_projects.update(
        load_known_values_from_file(
            PROJECT_NAME_PATH,
            known_projects))
    known_master_projects.update(
        load_known_values_from_file(
            MASTER_PROJECT_PATH,
            known_master_projects))

    print(f"Loaded {len(known_projects)} known project_name_en.")
    print(f"Loaded {len(known_master_projects)} known master_project_en.")

    return model, known_projects, known_master_projects


model, KNOWN_PROJECT_NAMES, KNOWN_MASTER_PROJECT_NAMES = load_model_and_artifacts()


def create_input_dataframe(
        trans_group,
        date_val,
        reg_type,
        project_name,
        master_project,
        area,
        proc_name_grouped,
        district_val,
        known_projects_set,
        known_master_projects_set,
        unknown_placeholder):

    final_project_name = project_name.strip()
    if not final_project_name:
        final_project_name = unknown_placeholder
    elif final_project_name not in known_projects_set:
        st.info(f"Name of '{final_project_name}' not found. '{
            unknown_placeholder}' will be used instead.")
        final_project_name = unknown_placeholder

    final_master_project_name = master_project.strip()
    if not final_master_project_name:
        final_master_project_name = unknown_placeholder
    elif final_master_project_name not in known_master_projects_set:
        st.info(f"Name of '{final_master_project_name}' not found. '{
            unknown_placeholder}' will be used instead.")
        final_master_project_name = unknown_placeholder

    input_data = {
        'trans_group_en': trans_group,
        'date': pd.to_datetime(date_val),
        'reg_type_en': reg_type,
        'project_name_en': final_project_name,
        'master_project_en': final_master_project_name,
        'procedure_area': float(area),
        'procedure_name_en_grouped': proc_name_grouped,
        'district': district_val
    }

    feature_order = [
        'trans_group_en',
        'date',
        'reg_type_en',
        'project_name_en',
        'master_project_en',
        'procedure_area',
        'procedure_name_en_grouped',
        'district']

    return pd.DataFrame([input_data])[feature_order]


# --- Streamlit UI ---
st.title("Price Prediction for Real Estate in Dubai :house::moneybag:")
st.markdown(
    "Enter known parameters of the property to get an approximate price per square meter and total estimated value.")

if model is None:
    st.error(
        "The model failed to load. Please check the model path and restart the app.")
else:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Transaction Parameters")
        trans_group_en_input = st.selectbox("Transaction Type:", options=TRANS_GROUP_EN_OPTIONS, index=TRANS_GROUP_EN_OPTIONS.index(
            UNKNOWN_VALUE_PLACEHOLDER) if UNKNOWN_VALUE_PLACEHOLDER in TRANS_GROUP_EN_OPTIONS else 0)
        procedure_name_en_grouped_input = st.selectbox(
            "Procedure Type (Grouped):",
            options=PROCEDURE_NAME_EN_GROUPED_OPTIONS,
            index=PROCEDURE_NAME_EN_GROUPED_OPTIONS.index(UNKNOWN_VALUE_PLACEHOLDER) if UNKNOWN_VALUE_PLACEHOLDER in PROCEDURE_NAME_EN_GROUPED_OPTIONS else 0)
        reg_type_en_input = st.selectbox("Registration Type:", options=REG_TYPE_EN_OPTIONS, index=REG_TYPE_EN_OPTIONS.index(
            UNKNOWN_VALUE_PLACEHOLDER) if UNKNOWN_VALUE_PLACEHOLDER in REG_TYPE_EN_OPTIONS else 0)

        st.subheader("Property Parameters")
        procedure_area_input = st.number_input(
            "Property Area (sq.m.):",
            min_value=10.0,
            max_value=10000.0,
            value=100.0,
            step=10.0)

    with col2:
        st.subheader("Location and Project")
        district_input = st.selectbox("District:", options=DISTRICT_OPTIONS, index=DISTRICT_OPTIONS.index(
            UNKNOWN_VALUE_PLACEHOLDER if "Unknown_District" not in DISTRICT_OPTIONS else "Unknown_District") if (
            UNKNOWN_VALUE_PLACEHOLDER if "Unknown_District" not in DISTRICT_OPTIONS else "Unknown_District") in DISTRICT_OPTIONS else 0)

        project_name_input_str = st.text_input(
            "Project Name:",
            placeholder=f"e.g., 'Marina Gate' or leave blank to use '{UNKNOWN_VALUE_PLACEHOLDER}'")

        with st.expander("Help: Select Project Name (optional)"):
            selected_project_from_list = st.selectbox("Search or choose a project:", options=[
                                                      ""] + sorted(list(KNOWN_PROJECT_NAMES)))
            if selected_project_from_list and not project_name_input_str:
                st.caption(f"Selected project: {selected_project_from_list}. You can copy it to the field above or leave it blank to use '{
                    UNKNOWN_VALUE_PLACEHOLDER}'.")

        master_project_name_input_str = st.text_input(
            "Developer Name:",
            placeholder=f"e.g., 'Dubai Marina' or leave blank to use '{UNKNOWN_VALUE_PLACEHOLDER}'")

        with st.expander("Help: Select Developer Name (optional)"):
            selected_master_project_from_list = st.selectbox(
                "Search or choose a developer project:",
                options=[""] +
                sorted(
                    list(KNOWN_MASTER_PROJECT_NAMES)))
            if selected_master_project_from_list and not master_project_name_input_str:
                st.caption(f"Selected developer project: {
                    selected_master_project_from_list}. You can copy it to the field above or leave it blank to use '{UNKNOWN_VALUE_PLACEHOLDER}'.")

    current_date_val = datetime.date.today()
    if st.button("Predict Price", type="primary", use_container_width=True):

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
            unknown_placeholder=UNKNOWN_VALUE_PLACEHOLDER
        )

        try:
            prediction = model.predict(input_df)
            predicted_price_per_sqm = np.expm1(prediction[0])
            total_price = predicted_price_per_sqm * procedure_area_input
            st.subheader("Prediction Results:")
            st.metric(label="Predicted Price per sq.m. (AED)",
                      value=f"{predicted_price_per_sqm:,.2f}")
            st.metric(
                label="Estimated Total Property Price (AED)",
                value=f"{
                    total_price:,.0f}")

        except Exception as e:
            st.error(f"Error during prediction: {e}")
            st.exception(e)

    st.sidebar.markdown("---")
    st.sidebar.markdown("**About this App**")
    st.sidebar.markdown(f"Model: CatBoost (loaded from `{MODEL_PATH}`)")
    st.sidebar.markdown("Data source: Open data from Dubai Pulse")
