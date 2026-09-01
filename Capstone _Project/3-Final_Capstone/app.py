import io
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import streamlit as st
from xgboost import XGBRegressor

# ----------------------------------------------------------------------
# Page config & Styling
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="🏡 House Price Predictor | Enterprise Edition",
    page_icon="🏡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .stApp { background: linear-gradient(180deg, #f7f3ee 0%, #eef2f7 100%); }
    .hero {
        background: linear-gradient(120deg, #2c3e50 0%, #4a6572 100%);
        padding: 2rem; border-radius: 18px; margin-bottom: 1.5rem;
        color: white; box-shadow: 0 10px 30px rgba(0,0,0,0.15);
    }
    .hero h1 { margin: 0; font-size: 2.2rem; }
    .hero p { opacity: 0.9; margin-top: .4rem; font-size: 1.05rem; }
    .metric-card {
        background: white; border-radius: 16px; padding: 1.2rem;
        box-shadow: 0 4px 18px rgba(0,0,0,0.06); text-align: center; border: 1px solid #eee;
    }
    .price-box {
        background: linear-gradient(120deg, #16a085 0%, #1abc9c 100%);
        border-radius: 18px; padding: 1.8rem; color: white; text-align: center;
        box-shadow: 0 10px 30px rgba(22,160,133,0.35);
        margin-bottom: 1rem;
    }
    .price-box h2 { font-size: 2.8rem; margin: 0; }
    section[data-testid="stSidebar"] { background: #1f2d3d; }
    section[data-testid="stSidebar"] * { color: #eee !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# Header Section
st.markdown(
    """
    <div class="hero">
        <h1>🏡 Advanced House Price Predictor</h1>
        <p>Interactive Real Estate Valuation and Geospatial Analytics powered by Ensemble Stacking Model (XGBoost + LightGBM + CatBoost).</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------
# House Imagery Section
# ----------------------------------------------------------------------
col_img1, col_img2, col_img3 = st.columns(3)
house_images = [
    "https://images.unsplash.com/photo-1568605114967-8130f3a36994?w=600&q=80",
    "https://images.unsplash.com/photo-1570129477492-45c003edd2be?w=600&q=80",
    "https://images.unsplash.com/photo-1600585154340-be6161a56a0c?w=600&q=80",
]
for col, url in zip([col_img1, col_img2, col_img3], house_images):
  with col:
    st.image(url, use_column_width=True)


# ----------------------------------------------------------------------
# Data Loading & Model Setup
# ----------------------------------------------------------------------
@st.cache_data
def load_data():
  try:
    df = pd.read_csv("kc_house_data.csv")
    source = "Kaggle King County dataset (kc_house_data.csv)"
  except FileNotFoundError:
    rng = np.random.default_rng(42)
    n = 4000
    sqft_living = rng.normal(2100, 900, n).clip(400, 8000)
    bedrooms = rng.integers(1, 7, n)
    bathrooms = rng.choice([1, 1.5, 2, 2.5, 3, 3.5, 4], n)
    floors = rng.choice([1, 1.5, 2, 2.5, 3], n)
    grade = rng.integers(4, 13, n)
    condition = rng.integers(1, 6, n)
    waterfront = rng.choice([0, 1], n, p=[0.99, 0.01])
    view = rng.integers(0, 5, n)
    yr_built = rng.integers(1900, 2016, n)
    zipcode = rng.choice(range(98001, 98120), n)
    sqft_lot = (sqft_living * rng.uniform(1.2, 4.0, n)).clip(500, 40000)
    lat = rng.uniform(47.15, 47.78, n)
    long = rng.uniform(-122.5, -121.7, n)

    price = (
        50000
        + sqft_living * 180
        + bedrooms * 8000
        + bathrooms * 15000
        + grade * 25000
        + waterfront * 300000
        + view * 20000
        + (condition - 3) * 8000
        - (2024 - yr_built) * 400
        + rng.normal(0, 40000, n)
    ).clip(75000, 3_500_000)

    df = pd.DataFrame({
        "price": price,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "sqft_living": sqft_living.astype(int),
        "sqft_lot": sqft_lot.astype(int),
        "floors": floors,
        "waterfront": waterfront,
        "view": view,
        "condition": condition,
        "grade": grade,
        "yr_built": yr_built,
        "zipcode": zipcode,
        "lat": lat,
        "long": long,
    })
    source = "synthetic sample data (add kc_house_data.csv for real data)"

  df["house_age"] = 2024 - df["yr_built"]
  return df, source


df, data_source = load_data()

FEATURES = [
    "bedrooms",
    "bathrooms",
    "sqft_living",
    "sqft_lot",
    "floors",
    "waterfront",
    "view",
    "condition",
    "grade",
    "house_age",
    "zipcode",
    "lat",
    "long",
]
FEATURES = [f for f in FEATURES if f in df.columns]


@st.cache_resource
def train_model(data: pd.DataFrame, features: list):
  X = data[features].fillna(data[features].median())
  y = np.log1p(data["price"])

  X_train, X_test, y_train, y_test = train_test_split(
      X, y, test_size=0.2, random_state=42
  )

  # Stacking Regressor setup
  xgb = XGBRegressor(n_estimators=250, learning_rate=0.07, random_state=42)
  lgb = LGBMRegressor(
      n_estimators=250, learning_rate=0.07, random_state=42, verbose=-1
  )
  cat = CatBoostRegressor(
      iterations=250, learning_rate=0.07, verbose=0, random_seed=42
  )

  stacking_model = StackingRegressor(
      estimators=[("xgb", xgb), ("lgb", lgb), ("cat", cat)],
      final_estimator=RidgeCV(),
  )

  stacking_model.fit(X_train, y_train)

  preds_log = stacking_model.predict(X_test)
  preds = np.expm1(preds_log)
  y_test_orig = np.expm1(y_test)

  mae = mean_absolute_error(y_test_orig, preds)
  r2 = r2_score(y_test_orig, preds)

  return stacking_model, mae, r2


model, mae, r2 = train_model(df, FEATURES)

# ----------------------------------------------------------------------
# Sidebar Inputs
# ----------------------------------------------------------------------
st.sidebar.header("🛠️ Property Details")

bedrooms = st.sidebar.slider("Bedrooms", 1, 8, 3)
bathrooms = st.sidebar.slider("Bathrooms", 1.0, 5.0, 2.0, step=0.25)
sqft_living = st.sidebar.slider("Living Area (sqft)", 400, 6000, 1800, step=50)
sqft_lot = st.sidebar.slider("Lot Size (sqft)", 500, 20000, 5000, step=100)
floors = st.sidebar.select_slider("Floors", options=[1, 1.5, 2, 2.5, 3], value=1.0)
waterfront = st.sidebar.checkbox("Waterfront View 🌊")
view = st.sidebar.slider("View Quality (0–4)", 0, 4, 0)
condition = st.sidebar.slider("Condition (1–5)", 1, 5, 3)
grade = st.sidebar.slider("Construction Grade (1–13)", 1, 13, 7)
yr_built = st.sidebar.slider("Year Built", 1900, 2024, 1990)
zipcode_val = st.sidebar.selectbox("Zipcode Area", sorted(df["zipcode"].unique()))

avg_lat = df[df["zipcode"] == zipcode_val]["lat"].mean()
avg_long = df[df["zipcode"] == zipcode_val]["long"].mean()

st.sidebar.markdown("---")
st.sidebar.header("🏦 Mortgage Estimator")
down_payment_pct = st.sidebar.slider("Down Payment (%)", 0, 50, 20)
interest_rate = st.sidebar.slider("Interest Rate (%)", 2.0, 10.0, 6.5, step=0.25)
loan_years = st.sidebar.selectbox("Loan Duration (Years)", [15, 20, 30], index=2)

user_input = pd.DataFrame([{
    "bedrooms": bedrooms,
    "bathrooms": bathrooms,
    "sqft_living": sqft_living,
    "sqft_lot": sqft_lot,
    "floors": floors,
    "waterfront": int(waterfront),
    "view": view,
    "condition": condition,
    "grade": grade,
    "house_age": 2024 - yr_built,
    "zipcode": zipcode_val,
    "lat": avg_lat,
    "long": avg_long,
}])[FEATURES]

log_pred = model.predict(user_input)[0]
predicted_price = np.expm1(log_pred)

# Margin Error Calculation
margin_pct = (mae / predicted_price) * 100

# Mortgage Calculation
loan_amount = predicted_price * (1 - down_payment_pct / 100)
monthly_rate = (interest_rate / 100) / 12
num_payments = loan_years * 12
monthly_payment = (
    (
        loan_amount
        * (monthly_rate * (1 + monthly_rate) ** num_payments)
        / ((1 + monthly_rate) ** num_payments - 1)
    )
    if monthly_rate > 0
    else loan_amount / num_payments
)

# ----------------------------------------------------------------------
# Application Navigation (TABS)
# ----------------------------------------------------------------------
tab1, tab2 = st.tabs([
    "🏡 Valuation & Mortgage",
    "🗺️ Geospatial & Market Analysis",
])

# ----------------------------------------------------------------------
# TAB 1: Valuation & Mortgage
# ----------------------------------------------------------------------
with tab1:
  col_center1, col_center2 = st.columns([1.5, 1])

  with col_center1:
    st.markdown(
        f"""
            <div class="price-box">
                <p style="margin-bottom:0px; font-size: 1.1rem;">Estimated Valuation</p>
                <h2>${predicted_price:,.0f}</h2>
                <p style="font-size:1rem; margin-top:8px;">Est. Error Margin: <b>±{margin_pct:.1f}%</b></p>
            </div>
            """,
        unsafe_allow_html=True,
    )

    m1, m2 = st.columns(2)
    with m1:
      st.markdown(
          f'<div class="metric-card"><h3>${monthly_payment:,.0f}/mo</h3><p>Est.'
          ' Monthly Mortgage</p></div>',
          unsafe_allow_html=True,
      )
    with m2:
      st.markdown(
          f'<div class="metric-card"><h3>{r2:.2%}</h3><p>Model R²'
          ' Accuracy</p></div>',
          unsafe_allow_html=True,
      )

  with col_center2:
    st.write("### 📋 Valuation Summary")
    st.write(f"- **Property Zipcode:** {zipcode_val}")
    st.write(f"- **Living Area:** {sqft_living:,} sqft")
    st.write(f"- **Bedrooms / Bathrooms:** {bedrooms} beds / {bathrooms} baths")
    st.write(f"- **Estimated Monthly Payment:** `${monthly_payment:,.2f}`")

    st.write("")
    report_df = pd.DataFrame([{
        "Predicted Price": f"${predicted_price:,.2f}",
        "Error Margin": f"±{margin_pct:.1f}%",
        "Est. Monthly Payment": f"${monthly_payment:,.2f}",
        "Bedrooms": bedrooms,
        "Bathrooms": bathrooms,
        "Sqft Living": sqft_living,
        "Zipcode": zipcode_val,
    }])
    csv_buffer = io.BytesIO()
    report_df.to_csv(csv_buffer, index=False)

    st.download_button(
        label="📥 Download Official Valuation Report",
        data=csv_buffer.getvalue(),
        file_name="property_valuation_report.csv",
        mime="text/csv",
        use_container_width=True,
    )

# ----------------------------------------------------------------------
# TAB 2: Geospatial & Market
# ----------------------------------------------------------------------
with tab2:
  st.subheader("🗺️ Geographic Price Distribution & Selected Location")
  sample_map_df = df.sample(min(1000, len(df)), random_state=42)

  fig_map = px.scatter_mapbox(
      sample_map_df,
      lat="lat",
      lon="long",
      color="price",
      size="sqft_living",
      color_continuous_scale="Viridis",
      size_max=12,
      zoom=9,
      hover_name="zipcode",
      hover_data=["price", "bedrooms", "bathrooms"],
      mapbox_style="carto-positron",
  )

  fig_map.add_trace(
      go.Scattermapbox(
          lat=[avg_lat],
          lon=[avg_long],
          mode="markers",
          marker=dict(size=18, color="crimson"),
          name="Selected Location",
          hoverinfo="text",
          text=f"Selected Property (Zipcode {zipcode_val})",
      )
  )
  fig_map.update_layout(height=420, margin={"r": 0, "t": 10, "l": 0, "b": 0})
  st.plotly_chart(fig_map, use_container_width=True)

  c1, c2 = st.columns(2)
  with c1:
    fig_scatter = px.scatter(
        df.sample(min(1200, len(df)), random_state=1),
        x="sqft_living",
        y="price",
        color="grade",
        title="Price vs. Living Area",
        color_continuous_scale="Teal",
        opacity=0.6,
    )
    fig_scatter.add_trace(
        go.Scatter(
            x=[sqft_living],
            y=[predicted_price],
            mode="markers",
            marker=dict(size=16, color="crimson", symbol="star"),
            name="Your Property",
        )
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

  with c2:
    fig_hist = px.histogram(
        df,
        x="price",
        nbins=40,
        title="Overall Market Price Distribution",
        color_discrete_sequence=["#1abc9c"],
    )
    fig_hist.add_vline(
        x=predicted_price,
        line_dash="dash",
        line_color="crimson",
        annotation_text="Your Property",
    )
    st.plotly_chart(fig_hist, use_container_width=True)

st.markdown("---")
st.caption(
    f"Data Source: {data_source} · Model: Stacking Regressor (XGBoost + LightGBM"
    " + CatBoost) · Capstone Project."
)
