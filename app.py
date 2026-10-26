
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score


st.set_page_config(page_title="GemWorth")

st.title("GemWorth")
st.write("Explore diamond prices and estimate a reference price.")

st.info(
    "This model estimates diamond prices for educational comparison; "
    "inaccurate predictions could lead users to misjudge whether "
    "a listed price is reasonable."
)

st.caption(
    "Prices are in US dollars. This dataset is not a live market feed, "
    "so estimates should not be treated as current appraisals."
)


#Load the saved data or an uploaded replacement
st.header("1. Data preview")

uploaded_file = st.file_uploader(
    "Optional: upload a diamonds CSV",
    type=["csv"]
)

data_path = Path(__file__).parent / "diamonds_cleaned.csv"

try:
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_csv(data_path)

except Exception:
    st.error(
        "Upload a valid CSV, or make sure diamonds_cleaned.csv "
        "is in the same folder as app.py."
    )
    st.stop()


#Prepare the data
df.columns = df.columns.str.strip()

required = ["carat", "depth", "price"]

if not set(required).issubset(df.columns):
    st.error("The CSV must contain carat, depth, and price.")
    st.stop()

df = df.drop(columns=["Unnamed: 0"], errors="ignore")

for column in required:
    df[column] = pd.to_numeric(df[column], errors="coerce")

df[required] = df[required].replace(
    [np.inf, -np.inf], np.nan
)

df = df.dropna(subset=required)
df = df.drop_duplicates()

df = df[
    (df["carat"] > 0)
    & (df["depth"] > 0)
    & (df["price"] > 0)
]

st.write(f"Usable rows: {len(df):,}")
st.dataframe(df.head(10))

if len(df) < 20:
    st.warning("At least 20 usable rows are needed.")
    st.stop()

columns = ["carat", "depth", "price"]

labels = {
    "carat": "Carat weight",
    "depth": "Depth (%)",
    "price": "Price (USD)"
}


#Interactive charts
st.header("2. Explore the data")

chart_type = st.selectbox(
    "Chart type",
    ["Scatter", "Histogram", "Line"]
)

x_column = st.selectbox(
    "Column / X-axis",
    columns,
    format_func=lambda c: labels[c]
)

fig, ax = plt.subplots(figsize=(8, 5))

if chart_type == "Histogram":
    ax.hist(df[x_column], bins=30)
    ax.set_ylabel("Number of diamonds")

else:
    y_options = [c for c in columns if c != x_column]

    y_column = st.selectbox(
        "Y-axis",
        y_options,
        index=y_options.index("price") if "price" in y_options else 0,
        format_func=lambda c: labels[c]
    )

    if chart_type == "Scatter":
        ax.scatter(df[x_column], df[y_column], alpha=0.2)

    else:
        averages = (
            df.groupby(x_column)[y_column]
            .mean()
            .sort_index()
        )

        ax.plot(averages.index, averages.values)

        st.caption(
            "The line connects averages in numeric order. "
            "It does not show prices changing over time."
        )

    ax.set_ylabel(labels[y_column])

ax.set_xlabel(labels[x_column])
fig.tight_layout()

st.pyplot(fig)
plt.close(fig)


#Choose two inputs and a target
st.header("3. Make a prediction")

st.write(
    "For a price estimate, keep price as the target "
    "and use carat and depth as inputs."
)

target = st.selectbox(
    "Target",
    columns,
    index=2,
    format_func=lambda c: labels[c]
)

input_options = [c for c in columns if c != target]

input_1 = st.selectbox(
    "First input",
    input_options,
    format_func=lambda c: labels[c]
)

input_2 = st.selectbox(
    "Second input",
    [c for c in input_options if c != input_1],
    format_func=lambda c: labels[c]
)

features = [input_1, input_2]

if any(df[c].nunique() < 2 for c in features + [target]):
    st.warning("Each selected column needs varying values.")
    st.stop()


#Train and evaluate
X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

baseline_predictions = [y_train.median()] * len(y_test)
baseline_mae = mean_absolute_error(y_test, baseline_predictions)

prefix = "$" if target == "price" else ""

with st.expander("View model performance"):
    st.write(f"Model MAE: {prefix}{mae:,.2f}")
    st.write(f"Median baseline MAE: {prefix}{baseline_mae:,.2f}")
    st.write(f"R-squared: {r2:.3f}")

    st.caption(
        "MAE is average absolute error on held-out rows, "
        "not a guaranteed error range for each prediction."
    )

    if mae >= baseline_mae:
        st.warning("The model did not beat the median baseline.")


#Get user inputs
if target != "price":
    st.caption(
        "Exploration mode: predicting a measurement rather than price. "
        "Verify estimates against the diamond's specifications."
    )

st.caption(
    "Carat is diamond weight. Depth is a percentage listed "
    "in diamond specifications."
)

with st.form("prediction_form"):
    value_1 = st.number_input(
        labels[input_1],
        min_value=float(df[input_1].min()),
        max_value=float(df[input_1].max()),
        value=float(df[input_1].median()),
        step=0.01,
        key=f"value_{input_1}"
    )

    value_2 = st.number_input(
        labels[input_2],
        min_value=float(df[input_2].min()),
        max_value=float(df[input_2].max()),
        value=float(df[input_2].median()),
        step=0.01,
        key=f"value_{input_2}"
    )

    predict_button = st.form_submit_button("Estimate")


#Predict with the same column names and order
if predict_button:
    new_diamond = pd.DataFrame(
        [[value_1, value_2]],
        columns=features
    )

    prediction = model.predict(new_diamond)[0]

    st.metric(
        f"Estimated {labels[target]}",
        f"{prefix}{prediction:,.2f}"
    )

    if prediction <= 0:
        st.warning(
            "This estimate is not realistic. "
            "The linear model is unsuitable for this input combination."
        )

st.caption(
    "The model uses only two inputs. Cut, color, clarity, "
    "and other omitted factors affect price."
)
