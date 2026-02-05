import streamlit as st
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="Glioma Grade Classification",
    page_icon="🧠",
    layout="centered"
)

# =====================================================
# GLOBAL UI STYLE (UI ONLY)
# =====================================================
st.markdown("""
<style>
/* ---------- HERO ---------- */
.hero {
    background: linear-gradient(135deg, #0f172a, #020617);
    padding: 2rem 2rem;
    border-radius: 14px;
    margin-bottom: 1.5rem;
}
.hero h1 {
    font-size: 2.2rem;
    font-weight: 700;
    color: #f8fafc;
    margin-bottom: 0.4rem;
}
.hero h3 {
    font-size: 1rem;
    font-weight: 500;
    color: #c7d2fe;
    margin-bottom: 0.6rem;
}
.hero p {
    color: #94a3b8;
    font-size: 0.9rem;
}
.badge {
    display: inline-block;
    background-color: rgba(148,163,184,0.15);
    color: #e5e7eb;
    padding: 0.3rem 0.65rem;
    border-radius: 999px;
    font-size: 0.72rem;
    margin-top: 0.7rem;
}

/* ---------- TABS (NAVY DIKECILIN JADI GARIS) ---------- */
div[data-baseweb="tab-list"] {
    background: transparent !important;
    border-bottom: 1px solid rgba(148,163,184,0.25);
}
div[data-baseweb="tab"] {
    padding: 0.35rem 0.75rem;
    font-size: 0.95rem;
}
div[data-baseweb="tab"][aria-selected="true"] {
    border-bottom: 2px solid #6366f1;
    color: #6366f1;
}

/* ---------- HIGHLIGHT BOX ---------- */
.highlight-box {
    background: rgba(148,163,184,0.08);
    border: 1px solid rgba(148,163,184,0.25);
    padding: 0.9rem 1rem;
    border-radius: 10px;
    margin-bottom: 1rem;
    font-size: 0.9rem;
}

/* ---------- LINK ---------- */
a {
    color: #6366f1;
    text-decoration: none;
}
a:hover {
    text-decoration: underline;
}
</style>
""", unsafe_allow_html=True)

# =====================================================
# LOAD MODEL & PSO FEATURES
# =====================================================
@st.cache_resource
def load_artifacts():
    model = load_model("cnn1d_pso_model.h5", compile=False)
    with open("pso_selected_features.json", "r") as f:
        selected_features = json.load(f)
    return model, selected_features

model, selected_features = load_artifacts()

# =====================================================
# AGE NORMALIZATION (FROM TCGA TRAINING DATA)
# =====================================================
MEAN_AGE = 50.9354
STD_AGE = 15.7023

# =====================================================
# UI HEADER
# =====================================================
st.markdown("""
<div class="hero">
    <h1>🧠 Glioma Grade Classification</h1>
    <h3>Multi-Strategy Feature Selection for Glioma Grading Using Classical and Deep Models</h3>
    <p>
        Academic deployment of a CNN-1D model integrated with Particle Swarm Optimization (PSO),
        trained on TCGA molecular and clinical features.
    </p>
    <span class="badge">Research Demo • Non-Clinical Use</span>
</div>
""", unsafe_allow_html=True)

# =====================================================
# TABS
# =====================================================
tab1, tab2 = st.tabs(["🧠 Overview & Model", "🔍 Prediction"])

# =====================================================
# TAB 1 — OVERVIEW (UNTOUCHED CONTENT)
# =====================================================
with tab1:
    st.subheader("📌 Overview")
    st.markdown("""
        This web application presents an **academic demonstration** of the study entitled  
        **_“Multi-Strategy Feature Selection for Glioma Grading Using Classical and Deep Models.”_**

        The deployed system implements a **Convolutional Neural Network (CNN-1D)** optimized using
        **Particle Swarm Optimization (PSO)** for feature selection, ensuring that only the most
        discriminative TCGA molecular features are utilized during inference.

        The objective of this application is to illustrate how **feature selection strategies**
        can enhance classification robustness when applied to both classical and deep learning
        models in glioma grading tasks.
    """)

    st.subheader("🧪 How to Use This Demo")
    st.markdown("""
        1. Select a prediction mode:
            - **Manual Input**: Enter molecular feature values aligned with PSO-selected TCGA features.
            - **CSV Upload**: Upload a TCGA-formatted patient sample file.

        2. Ensure that all input features conform to the **PSO feature subset** used during model training.

        3. Click **Predict** to obtain:
            - Glioma grade classification (LGG or HGG)
            - Prediction probability distribution
            - Visual summary of class confidence

        This demo is designed for **methodological illustration** and **academic evaluation** only.
    """)

# =====================================================
# TAB 2 — PREDICTION
# =====================================================
with tab2:
    mode = st.radio(
        "Choose prediction mode:",
        ["📤 Upload CSV File", "✍️ Manual Input (Single Patient)"]
    )

    # =================================================
    # UPLOAD MODE
    # =================================================
    if mode == "📤 Upload CSV File":
        st.markdown("""
        <div class="highlight-box">
        Upload a CSV file containing <b>PSO-selected TCGA features only</b>.
        All values must follow the original TCGA/UCI encoding scheme.<br><br>
        📎 <b>Testing dataset (Google Drive):</b><br>
        <a href="https://drive.google.com/drive/folders/1sLmHIbFyaF6AQdQPmVut81BplGxaKKKG?usp=sharing" target="_blank">
            https://drive.google.com/your-testing-dataset-link
        </a>
        </div>
        """, unsafe_allow_html=True)

        uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            st.dataframe(df, use_container_width=True)

            missing = set(selected_features) - set(df.columns)
            if missing:
                st.error(f"Missing required PSO features: {missing}")
                st.stop()

            df = df[selected_features]

            for col in selected_features:
                if col == "Age_at_diagnosis":
                    df[col] = df[col]
                elif col == "Race":
                    if not df[col].isin([0, 1, 2, 3]).all():
                        st.error("Invalid Race encoding detected.")
                        st.stop()
                else:
                    if not df[col].isin([0, 1]).all():
                        st.error(f"Feature '{col}' must be binary (0 or 1).")
                        st.stop()

            if st.button("🔍 Predict"):
                X = np.expand_dims(df.values, axis=-1)
                probs = model.predict(X, verbose=0).flatten()
                preds = (probs > 0.5).astype(int)

                results = pd.DataFrame({
                    "Patient": [f"Patient {i+1}" for i in range(len(df))],
                    "Predicted Grade": ["High Grade" if p == 1 else "Low Grade" for p in preds],
                    "Probability (High Grade)": np.round(probs, 3)
                })

                st.subheader("🧠 Prediction Results")
                st.dataframe(results, use_container_width=True)

                high = (preds == 1).sum()
                low = (preds == 0).sum()
                avg_prob = probs.mean()

                st.markdown(f"""
                **Conclusion**

                From **{len(df)} evaluated samples**, the CNN-1D model with PSO feature selection
                classified **{high} cases as High-Grade Glioma** and **{low} cases as Low-Grade Glioma**.
                The average predicted probability for the High-Grade class was **{avg_prob:.2f}**,
                reflecting model confidence under TCGA-consistent molecular input.
                """)

    # =================================================
    # MANUAL INPUT MODE
    # =================================================
    else:
        st.markdown("""
        <div class="highlight-box">
        Manual input is initialized using a <b>TCGA testing reference sample</b>
        to preserve molecular feature distribution consistency during inference.
        </div>
        """, unsafe_allow_html=True)

        @st.cache_data
        def load_tcga_reference():
            df_ref = pd.read_csv("sample_tcga_patients_demo_1.csv")
            return df_ref[selected_features]

        tcga_ref = load_tcga_reference()
        base_row = tcga_ref.sample(1, random_state=42).iloc[0]

        input_data = {}

        st.info(
            "Only selected key features are editable. "
            "Other PSO-selected features remain fixed to TCGA reference values."
        )

        editable_features = [
            "Age_at_diagnosis",
            "IDH1",
            "TP53",
            "MUC16",
            "NOTCH1",
            "IDH2",
            "PDGFRA"
        ]

        for feat in selected_features:
            if feat == "Age_at_diagnosis":
                input_data[feat] = st.number_input(
                    "Age at Diagnosis", 0, 120, int(base_row[feat])
                )
            elif feat == "Race":
                race_map = {
                    0: "White",
                    1: "Black or African American",
                    2: "Asian",
                    3: "American Indian / Alaska Native"
                }
                input_data[feat] = st.selectbox(
                    "Race",
                    race_map.keys(),
                    index=int(base_row[feat]),
                    format_func=lambda x: race_map[x]
                )
            elif feat in editable_features:
                input_data[feat] = st.selectbox(
                    f"{feat} (Mutation)", [0, 1], index=int(base_row[feat])
                )
            else:
                input_data[feat] = base_row[feat]

        if st.button("🔍 Predict"):
            X = np.expand_dims(pd.DataFrame([input_data]).values, axis=-1)
            prob = model.predict(X, verbose=0).flatten()[0]
            pred = "High Grade Glioma" if prob > 0.5 else "Low Grade Glioma"

            st.subheader("🧠 Prediction Result")
            st.metric("Predicted Grade", pred)
            st.metric("Probability (High Grade)", f"{prob:.3f}")

            st.markdown("""
            **Conclusion**

            The prediction was generated using a hybrid approach in which
            manual inputs were anchored to a TCGA-derived reference sample.
            This strategy preserves molecular feature distribution consistency
            and prevents out-of-distribution inference during CNN-1D prediction.
            """)

# =====================================================
# FOOTER
# =====================================================
st.divider()
st.caption("⚠️ Academic demonstration only. Not intended for clinical diagnosis.")
