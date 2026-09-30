import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide"
)

# --------------------------------------------------
# CUSTOM DASHBOARD DESIGN
# --------------------------------------------------

st.markdown("""
<style>

/* Main background */
.stApp {
    background: linear-gradient(135deg, #0f172a, #1e1b4b, #312e81);
    color: white;
}

/* Main heading */
h1 {
    color: #ffffff !important;
    font-size: 38px !important;
    font-weight: 800 !important;
}

/* Subheadings */
h2, h3 {
    color: #c4b5fd !important;
}

/* Paragraphs */
p, label {
    color: #e2e8f0 !important;
}

/* Metric cards */
[data-testid="stMetric"] {
    background: linear-gradient(135deg, #312e81, #4c1d95);
    border: 1px solid #818cf8;
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0 5px 18px rgba(0,0,0,0.25);
}

/* Metric values */
[data-testid="stMetricValue"] {
    color: #67e8f9 !important;
    font-weight: bold;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 12px;
    background-color: #1e1b4b;
    padding: 10px;
    border-radius: 12px;
}

.stTabs [data-baseweb="tab"] {
    background-color: #312e81;
    color: white;
    border-radius: 10px;
    padding: 10px 18px;
}

.stTabs [aria-selected="true"] {
    background-color: #7c3aed !important;
    color: white !important;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(90deg, #7c3aed, #06b6d4);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 10px 22px;
    font-weight: bold;
}

.stButton > button:hover {
    background: linear-gradient(90deg, #06b6d4, #7c3aed);
    color: white;
    border: none;
}

/* Dataframes */
[data-testid="stDataFrame"] {
    border: 1px solid #818cf8;
    border-radius: 12px;
}

/* Divider */
hr {
    border-color: #6366f1;
}

</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# LOAD MODEL AND DATASET
# --------------------------------------------------

model = joblib.load("fraud_detection_model.pkl")

df = pd.read_csv("data/creditcard.csv")


# --------------------------------------------------
# PREPARE DATA
# --------------------------------------------------

feature_columns = [
    "Time",
    "V1", "V2", "V3", "V4", "V5",
    "V6", "V7", "V8", "V9", "V10",
    "V11", "V12", "V13", "V14", "V15",
    "V16", "V17", "V18", "V19", "V20",
    "V21", "V22", "V23", "V24", "V25",
    "V26", "V27", "V28",
    "Amount"
]

X = df[feature_columns].copy()
y = df["Class"]

# Scale Amount
scaler = StandardScaler()
X["Amount"] = scaler.fit_transform(X[["Amount"]])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Model predictions
y_pred = model.predict(X_test)


# --------------------------------------------------
# CALCULATE PERFORMANCE
# --------------------------------------------------

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)

cm = confusion_matrix(y_test, y_pred)


# --------------------------------------------------
# DASHBOARD TITLE
# --------------------------------------------------

st.title("🛡️ AI-Powered Credit Card Fraud Detection")

st.markdown(
    "### Intelligent Transaction Monitoring & Fraud Risk Analysis"
)

st.write(
    "A Machine Learning based application for detecting "
    "potentially fraudulent credit card transactions."
)

st.divider()


# --------------------------------------------------
# DATASET STATISTICS
# --------------------------------------------------

total = len(df)
fraud = int(df["Class"].sum())
genuine = total - fraud

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("💳 Total Transactions", f"{total:,}")

with col2:
    st.metric("✅ Genuine Transactions", f"{genuine:,}")

with col3:
    st.metric("🚨 Fraud Transactions", f"{fraud:,}")

st.divider()


# --------------------------------------------------
# TABS
# --------------------------------------------------

tab1, tab2, tab3 = st.tabs([
    "🔍 Test Transaction",
    "📊 Model Performance",
    "📈 Data Visualization"
])


# ==================================================
# TAB 1 - TEST TRANSACTION
# ==================================================

with tab1:

    st.subheader("🔍 Test an Actual Transaction")

    transaction_type = st.radio(
        "Choose transaction type:",
        ["Fraud Transaction", "Genuine Transaction"]
    )

    if transaction_type == "Fraud Transaction":
        filtered_df = df[df["Class"] == 1]
    else:
        filtered_df = df[df["Class"] == 0]

    row_number = st.number_input(
        "Enter transaction row number",
        min_value=0,
        max_value=len(filtered_df) - 1,
        value=0,
        step=1
    )

    selected_row = filtered_df.iloc[int(row_number)]

    st.subheader("📄 Transaction Details")

    col1, col2 = st.columns(2)

    with col1:
        st.write(
            "Transaction Time:",
            selected_row["Time"]
        )

        st.write(
            "Transaction Amount:",
            f"₹{selected_row['Amount']:.2f}"
        )

    with col2:

        actual_class = int(selected_row["Class"])

        if actual_class == 1:
            st.write("Actual Class: 🚨 Fraud")
        else:
            st.write("Actual Class: ✅ Genuine")

    if st.button(
        "🤖 Predict This Transaction",
        use_container_width=True
    ):

        input_data = selected_row[
            feature_columns
        ].to_frame().T.copy()

        # Scale Amount
        input_data["Amount"] = scaler.transform(
            input_data[["Amount"]]
        )

        prediction = model.predict(input_data)[0]

        if hasattr(model, "predict_proba"):
            probability = model.predict_proba(
                input_data
            )[0][1]
        else:
            probability = None

        st.divider()

        if prediction == 1:
            st.error("🚨 MODEL PREDICTION: FRAUD")
        else:
            st.success("✅ MODEL PREDICTION: GENUINE")

        if probability is not None:
            st.metric(
                "Fraud Probability",
                f"{probability * 100:.2f}%"
            )

        if prediction == actual_class:
            st.success(
                "✅ Model prediction matches the actual class!"
            )
        else:
            st.warning(
                "⚠️ Model prediction does not match the actual class."
            )


# ==================================================
# TAB 2 - MODEL PERFORMANCE
# ==================================================

with tab2:

    st.subheader("📊 Model Performance")

    st.write(
        "The following metrics are calculated using the test dataset."
    )

    # Performance Metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Accuracy",
            f"{accuracy * 100:.2f}%"
        )

    with col2:
        st.metric(
            "Precision",
            f"{precision * 100:.2f}%"
        )

    with col3:
        st.metric(
            "Recall",
            f"{recall * 100:.2f}%"
        )

    with col4:
        st.metric(
            "F1 Score",
            f"{f1 * 100:.2f}%"
        )

    st.divider()

    # Explain Metrics
    st.subheader("📖 What do these metrics mean?")

    st.write(
        "**Accuracy:** Overall percentage of correct predictions."
    )

    st.write(
        "**Precision:** Out of transactions predicted as fraud, "
        "how many were actually fraud."
    )

    st.write(
        "**Recall:** Out of actual fraud transactions, "
        "how many were detected by the model."
    )

    st.write(
        "**F1 Score:** Combined measure of precision and recall."
    )

    st.divider()

    # Confusion Matrix
    st.subheader("🔲 Confusion Matrix")

    fig, ax = plt.subplots()

    ax.imshow(cm, cmap="Purples")

    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted Class")
    ax.set_ylabel("Actual Class")

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])

    ax.set_xticklabels(["Genuine", "Fraud"])
    ax.set_yticklabels(["Genuine", "Fraud"])

    for i in range(2):
        for j in range(2):
            ax.text(
                j,
                i,
                cm[i, j],
                ha="center",
                va="center",
                color="black",
                fontsize=12,
                fontweight="bold"
            )

    st.pyplot(fig)

    st.write(
        "The confusion matrix shows how many genuine and "
        "fraudulent transactions were correctly or incorrectly classified."
    )

    st.divider()

    # Performance Bar Chart
    st.subheader("📈 Performance Metrics Chart")

    metrics_df = pd.DataFrame({
        "Metric": [
            "Accuracy",
            "Precision",
            "Recall",
            "F1 Score"
        ],
        "Score": [
            accuracy,
            precision,
            recall,
            f1
        ]
    })

    chart_data = metrics_df.set_index("Metric")

    st.bar_chart(chart_data)

    st.divider()

    # Fraud Detection Summary
    st.subheader("🚨 Fraud Detection Summary")

    true_negative = cm[0][0]
    false_positive = cm[0][1]
    false_negative = cm[1][0]
    true_positive = cm[1][1]

    col1, col2 = st.columns(2)

    with col1:
        st.write(
            f"✅ Correct Genuine Predictions: **{true_negative:,}**"
        )

        st.write(
            f"🚨 Correct Fraud Predictions: **{true_positive:,}**"
        )

    with col2:
        st.write(
            f"⚠️ Genuine classified as Fraud: **{false_positive:,}**"
        )

        st.write(
            f"❌ Fraud classified as Genuine: **{false_negative:,}**"
        )


# ==================================================
# TAB 3 - DATA VISUALIZATION
# ==================================================

with tab3:

    st.subheader("📈 Transaction Data Visualization")

    # Genuine vs Fraud
    st.write("### Genuine vs Fraud Transactions")

    class_counts = df["Class"].value_counts()

    chart_data = pd.DataFrame({
        "Transaction Type": ["Genuine", "Fraud"],
        "Count": [
            class_counts.get(0, 0),
            class_counts.get(1, 0)
        ]
    })

    st.bar_chart(
        chart_data.set_index("Transaction Type")
    )

    # Fraud percentage
    fraud_count = int((df["Class"] == 1).sum())
    total_count = len(df)
    fraud_percentage = (fraud_count / total_count) * 100

    st.metric(
        "🚨 Fraud Percentage",
        f"{fraud_percentage:.2f}%"
    )

    # Transaction amount
    st.write("### 💰 Transaction Amount Distribution")

    st.bar_chart(
        df[["Amount"]].head(500)
    )

    # Fraud over time
    st.write("### 🚨 Fraud Transactions Over Time")

    fraud_time = df[df["Class"] == 1].groupby("Time").size()

    st.line_chart(fraud_time)