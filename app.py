# ============================================================
#  ZION TECH HUB - Breast Cancer Detection Project
#  File: app.py (Streamlit App)
# ============================================================

import  streamlit as st
import  pandas as pd
import  joblib
import  plotly.express as px

# --- Load model and top features ---
model = joblib.load("breast_cancer_ml_project/models/best_rf_model.pkl")
top_features = joblib.load("breast_cancer_ml_project/models/top_features.pkl")

# --- Load dataset for example values ---
df = pd.read_csv("breast_cancer.csv")
df.drop(columns=['id'], errors='ignore', inplace=True)
X = df.drop("diagnosis", axis=1)
example_values = X[top_features].mean().round(3).tolist()

# --- Streamlit Page Config ---
st.set_page_config(
    page_title="Breast Cancer Predictor",
    page_icon="🧬",
    layout="wide"
)

# --- Styling ---
st.markdown("""
<style>
h1, h2, h3 {
    font-family: 'Segoe UI', sans-serif;
    color: #3f51b5;
}
.stButton>button {
    background-color: #6200ea;
    color: white;
}
.stNumberInput>div>input {
    background-color: #f3e5f5;
}
.subtitle {
    text-align: center;
    font-size: 18px;
    font-weight: 600;
    color: #e91e63; /* Pink awareness color */
    font-family: 'Segoe UI', sans-serif;
}
</style>
""", unsafe_allow_html=True)

# --- Sidebar Info ---
with st.sidebar:
    st.title("📘 Project Info")
    st.markdown("""
    **Breast Cancer Detection App**

    **Objective:** Predict whether a tumor is benign or malignant using the **top 10 most important features**.  

    **Purpose:** Assist early diagnosis while keeping the app efficient.  

    *Note:* This is for educational purposes and not a medical diagnostic tool.  

    **Built With:**  
    - Random Forest Classifier  
    - Streamlit  
    - Python & Machine Learning  

    ❓ **Help / FAQ**  

    **Q: Why only 10 features?**  
    - Using the most important features makes the app faster and more focused.  

    **Q: Can I trust this result?**  
    - This model is accurate, but it should **never** replace a medical professional’s judgment.  
    """)

# --- Header with Centered Logo, Title & Subtitle ---
st.image("assets/breast_cancer_awareness.png", width=200)
st.markdown(
    "<h1 style='text-align: center; color: #3f51b5; font-family: Segoe UI, sans-serif;'>🧬 Breast Cancer Diagnosis Predictor</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<p class='subtitle'>AI-Powered Early Detection Assistant</p>",
    unsafe_allow_html=True
)

# --- What is Breast Cancer Section ---
st.markdown("## What is Breast Cancer?")
st.markdown("""
Breast cancer happens when cells in the breast grow uncontrollably, forming a tumor.  
Tumors can be benign (non-cancerous) or malignant (cancerous).  

👉 Here, the **10 most important tumor features** are used to keep predictions efficient and reliable.
""")

# --- Example values ---
example_df = pd.DataFrame({"Feature": top_features, "Example Value": example_values})
with st.expander(" Example Values for Testing"):
    st.markdown("These values are based on the average from the **Wisconsin Breast Cancer Dataset**.")
    st.dataframe(example_df)

# --- Input section ---
st.header(" Enter Tumor Measurements")
user_input = []
columns = st.columns(2)

for i, name in enumerate(top_features):
    col = columns[i % 2]
    val = col.number_input(name, min_value=0.0, value=float(example_values[i]), key=name)
    user_input.append(val)

X_input = pd.DataFrame([user_input], columns=top_features)

# --- Prediction ---
if st.button("🔍 Predict Diagnosis"):
    prediction = model.predict(X_input)[0]
    confidence = model.predict_proba(X_input)[0][prediction]
    result = "🟢 Benign (Not Cancer)" if prediction == 0 else "🔴 Malignant (Cancer Detected)"
    
    st.success(f"**Diagnosis:** {result}")
    st.info(f"**Confidence:** {confidence:.2%}")

    # --- Feature impact chart ---
    top_input = pd.DataFrame(user_input, index=top_features, columns=["Value"]).sort_values("Value", ascending=False)
    fig = px.bar(top_input, x="Value", y=top_input.index, orientation="h", title="Entered Feature Values")
    st.plotly_chart(fig, use_container_width=True)

    # --- Findings & Recommendations ---
    first, second = top_input.index[0], top_input.index[1]
    st.markdown(f"""
    ###  Findings
    - The features **{first}** and **{second}** had the strongest influence in this case.  
    - These values suggest important tumor characteristics for diagnosis.  

    ###  Recommendations
    - Clinicians should pay extra attention to these features during analysis.  
    - Combining model output with medical imaging/testing increases accuracy.  

    ###  Conclusion
    - Using only the most important features keeps the app efficient, without losing accuracy.  
    - This tool supports, but never replaces, professional medical judgment.  
    """)
