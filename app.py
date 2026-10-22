
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

st.set_page_config(page_title="Employee Salary Prediction", page_icon="💼")

st.title("💼 Employee Salary Prediction System")
st.write("Predict an estimated salary using employee details.")

@st.cache_data
def load_data():
    return pd.read_csv("salary.csv")

df = load_data()

required_columns = [
    "experience_years", "education_level", "job_role", "salary"
]

if not all(col in df.columns for col in required_columns):
    st.error("salary.csv is missing required columns.")
    st.stop()

X = df[["experience_years", "education_level", "job_role"]]
y = df["salary"]

def create_model():
    prep = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"),
         ["education_level", "job_role"])
    ], remainder="passthrough")

    return make_pipeline(
        prep,
        RandomForestRegressor(random_state=42)
    )

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

evaluation_model = create_model()
evaluation_model.fit(X_train, y_train)
test_predictions = evaluation_model.predict(X_test)

col1, col2 = st.columns(2)
col1.metric(
    "Test MAE",
    f"₹{mean_absolute_error(y_test, test_predictions):,.2f}"
)
col2.metric(
    "Test R² Score",
    f"{r2_score(y_test, test_predictions):.3f}"
)

# Train the final prediction model using all available records
model = create_model()
model.fit(X, y)

st.subheader("Enter Employee Details")

with st.form("salary_form"):
    experience = st.number_input(
        "Experience (years)",
        min_value=0.0,
        max_value=50.0,
        value=3.0,
        step=1.0
    )

    education = st.selectbox(
        "Education Level",
        sorted(df["education_level"].dropna().unique().tolist())
    )

    role = st.selectbox(
        "Job Role",
        sorted(df["job_role"].dropna().unique().tolist())
    )

    submitted = st.form_submit_button("Predict Salary")

if submitted:
    employee = pd.DataFrame([{
        "experience_years": experience,
        "education_level": education,
        "job_role": role
    }])

    prediction = model.predict(employee)[0]

    st.success(f"Estimated Salary: ₹{prediction:,.2f}")
    st.caption(
        "This is an estimate based on the supplied dataset, "
        "not a guaranteed market salary or job offer."
    )
