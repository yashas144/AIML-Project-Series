import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="Linear Regression Explorer",
    page_icon="📈",
    layout="wide",
)


# -----------------------------
# Helper functions
# -----------------------------
@st.cache_data
def generate_sample_data() -> pd.DataFrame:
    """Generate a simple dataset relating study hours to exam scores."""
    rng = np.random.default_rng(42)

    study_hours = np.round(rng.uniform(1, 10, 80), 1)
    noise = rng.normal(0, 6, 80)
    exam_score = np.clip(42 + 5.2 * study_hours + noise, 0, 100)

    return pd.DataFrame(
        {
            "Study Hours": study_hours,
            "Exam Score": np.round(exam_score, 1),
        }
    )


def train_model(
    dataframe: pd.DataFrame,
    feature: str,
    target: str,
    test_size: float,
):
    """Prepare the data, split it, and train a linear regression model."""
    model_data = dataframe[[feature, target]].copy()

    model_data[feature] = pd.to_numeric(
        model_data[feature],
        errors="coerce",
    )
    model_data[target] = pd.to_numeric(
        model_data[target],
        errors="coerce",
    )

    model_data = model_data.dropna()

    if len(model_data) < 5:
        raise ValueError(
            "At least five valid numeric rows are required."
        )

    X = model_data[[feature]]
    y = model_data[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=42,
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    test_predictions = model.predict(X_test)

    metrics = {
        "r2": r2_score(y_test, test_predictions),
        "mae": mean_absolute_error(y_test, test_predictions),
        "rmse": np.sqrt(
            mean_squared_error(y_test, test_predictions)
        ),
    }

    return model, model_data, X_train, X_test, y_train, y_test, metrics


# -----------------------------
# Header
# -----------------------------
st.title("📈 Simple Linear Regression Explorer")

st.markdown(
    """
    Build and explore a simple linear regression model using your own CSV
    file or the included sample dataset.
    """
)


# -----------------------------
# Sidebar controls
# -----------------------------
with st.sidebar:
    st.header("⚙️ Model Settings")

    data_source = st.radio(
        "Choose a data source",
        options=["Sample Dataset", "Upload CSV"],
    )

    uploaded_file = None

    if data_source == "Upload CSV":
        uploaded_file = st.file_uploader(
            "Upload a CSV file",
            type=["csv"],
        )

    test_size = st.slider(
        "Test dataset percentage",
        min_value=10,
        max_value=40,
        value=20,
        step=5,
    ) / 100

    st.info(
        "Simple linear regression uses one input feature to predict "
        "one numeric target."
    )


# -----------------------------
# Load data
# -----------------------------
try:
    if data_source == "Upload CSV":
        if uploaded_file is None:
            st.warning("Upload a CSV file to continue.")
            st.stop()

        df = pd.read_csv(uploaded_file)
    else:
        df = generate_sample_data()

except Exception as error:
    st.error(f"Unable to load the data: {error}")
    st.stop()


# -----------------------------
# Dataset preview
# -----------------------------
st.subheader("1. Explore the Dataset")

left_column, right_column = st.columns([2, 1])

with left_column:
    st.dataframe(
        df.head(15),
        use_container_width=True,
    )

with right_column:
    st.metric("Number of Rows", f"{df.shape[0]:,}")
    st.metric("Number of Columns", df.shape[1])
    st.metric(
        "Missing Values",
        int(df.isna().sum().sum()),
    )


# -----------------------------
# Select model columns
# -----------------------------
numeric_columns = df.select_dtypes(
    include=np.number
).columns.tolist()

if len(numeric_columns) < 2:
    st.error(
        "The dataset must contain at least two numeric columns."
    )
    st.stop()

st.subheader("2. Select the Variables")

selection_col1, selection_col2 = st.columns(2)

with selection_col1:
    feature_column = st.selectbox(
        "Independent variable — X",
        options=numeric_columns,
        index=0,
    )

available_targets = [
    column
    for column in numeric_columns
    if column != feature_column
]

with selection_col2:
    target_column = st.selectbox(
        "Target variable — y",
        options=available_targets,
        index=0,
    )


# -----------------------------
# Train the model
# -----------------------------
try:
    (
        model,
        clean_data,
        X_train,
        X_test,
        y_train,
        y_test,
        metrics,
    ) = train_model(
        dataframe=df,
        feature=feature_column,
        target=target_column,
        test_size=test_size,
    )

except ValueError as error:
    st.error(str(error))
    st.stop()

except Exception as error:
    st.error(f"Model training failed: {error}")
    st.stop()


# -----------------------------
# Model results
# -----------------------------
st.subheader("3. Model Performance")

metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)

with metric_col1:
    st.metric("R² Score", f"{metrics['r2']:.3f}")

with metric_col2:
    st.metric("Mean Absolute Error", f"{metrics['mae']:.2f}")

with metric_col3:
    st.metric("RMSE", f"{metrics['rmse']:.2f}")

with metric_col4:
    st.metric("Training Rows", len(X_train))


slope = float(model.coef_[0])
intercept = float(model.intercept_)

st.markdown("#### Regression Equation")

st.code(
    f"{target_column} = {intercept:.3f} "
    f"+ ({slope:.3f} × {feature_column})",
    language="text",
)

if slope > 0:
    relationship = "positive"
elif slope < 0:
    relationship = "negative"
else:
    relationship = "neutral"

st.success(
    f"The model found a **{relationship} relationship**. "
    f"For every one-unit increase in **{feature_column}**, "
    f"the predicted **{target_column}** changes by "
    f"approximately **{slope:.2f} units**."
)


# -----------------------------
# Interactive regression chart
# -----------------------------
st.subheader("4. Interactive Regression Chart")

chart = px.scatter(
    clean_data,
    x=feature_column,
    y=target_column,
    trendline="ols",
    title=f"{target_column} vs. {feature_column}",
    opacity=0.75,
    hover_data={
        feature_column: ":.2f",
        target_column: ":.2f",
    },
)

chart.update_layout(
    title_x=0.5,
    hovermode="closest",
)

st.plotly_chart(
    chart,
    use_container_width=True,
)


# -----------------------------
# Prediction interface
# -----------------------------
st.subheader("5. Make a Prediction")

minimum_value = float(clean_data[feature_column].min())
maximum_value = float(clean_data[feature_column].max())
average_value = float(clean_data[feature_column].mean())

prediction_col1, prediction_col2 = st.columns([2, 1])

with prediction_col1:
    input_value = st.number_input(
        f"Enter a value for {feature_column}",
        min_value=minimum_value,
        max_value=maximum_value,
        value=average_value,
        step=max((maximum_value - minimum_value) / 100, 0.01),
    )

with prediction_col2:
    prediction = model.predict(
        pd.DataFrame(
            {feature_column: [input_value]}
        )
    )[0]

    st.metric(
        f"Predicted {target_column}",
        f"{prediction:.2f}",
    )


# -----------------------------
# Actual versus predicted results
# -----------------------------
st.subheader("6. Actual vs. Predicted Values")

results = X_test.copy()
results[f"Actual {target_column}"] = y_test
results[f"Predicted {target_column}"] = model.predict(X_test)
results["Residual"] = (
    results[f"Actual {target_column}"]
    - results[f"Predicted {target_column}"]
)

st.dataframe(
    results.round(2),
    use_container_width=True,
)

comparison_chart = px.scatter(
    results,
    x=f"Actual {target_column}",
    y=f"Predicted {target_column}",
    title="Actual vs. Predicted Values",
)

minimum_comparison = min(
    results[f"Actual {target_column}"].min(),
    results[f"Predicted {target_column}"].min(),
)

maximum_comparison = max(
    results[f"Actual {target_column}"].max(),
    results[f"Predicted {target_column}"].max(),
)

comparison_chart.add_shape(
    type="line",
    x0=minimum_comparison,
    y0=minimum_comparison,
    x1=maximum_comparison,
    y1=maximum_comparison,
    line={"dash": "dash"},
)

comparison_chart.update_layout(title_x=0.5)

st.plotly_chart(
    comparison_chart,
    use_container_width=True,
)


# -----------------------------
# Download results
# -----------------------------
csv_results = results.to_csv(index=False).encode("utf-8")

st.download_button(
    label="⬇️ Download Prediction Results",
    data=csv_results,
    file_name="linear_regression_results.csv",
    mime="text/csv",
)