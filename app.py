import streamlit as st
import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings("ignore")

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Car Price Predictor",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .stApp { background-color: #0f1117; }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #1a1d27;
        border-right: 1px solid #2e3040;
    }

    /* Metric cards */
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, #1e2130, #252840);
        border: 1px solid #3a3d55;
        border-radius: 12px;
        padding: 16px 20px;
    }
    [data-testid="stMetricValue"] {
        font-size: 2rem !important;
        font-weight: 700 !important;
        color: #7c83fd !important;
    }
    [data-testid="stMetricLabel"] {
        color: #9095b0 !important;
        font-size: 0.8rem !important;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }

    /* Prediction box */
    .prediction-box {
        background: linear-gradient(135deg, #1f2040, #2a2d52);
        border: 2px solid #5c62e8;
        border-radius: 16px;
        padding: 32px;
        text-align: center;
        margin: 20px 0;
    }
    .prediction-label {
        color: #9095b0;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        margin-bottom: 8px;
    }
    .prediction-value {
        color: #7c83fd;
        font-size: 3.2rem;
        font-weight: 800;
        line-height: 1.1;
    }
    .prediction-sub {
        color: #6b7280;
        font-size: 0.85rem;
        margin-top: 10px;
    }

    /* Section headers */
    .section-title {
        color: #e2e4f0;
        font-size: 1.1rem;
        font-weight: 600;
        margin-bottom: 4px;
        padding-bottom: 6px;
        border-bottom: 1px solid #2e3040;
    }

    /* Input labels */
    label { color: #c4c8e0 !important; font-size: 0.9rem !important; }

    /* Selectbox & Slider */
    .stSelectbox > div > div { background-color: #1e2130 !important; border-color: #3a3d55 !important; }
    .stSlider [data-baseweb="slider"] { margin-top: 0; }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #5c62e8, #7c83fd);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 12px 28px;
        font-size: 1rem;
        font-weight: 600;
        width: 100%;
        letter-spacing: 0.03em;
        transition: opacity 0.2s;
    }
    .stButton > button:hover { opacity: 0.88; }

    /* Feature importance bars */
    .feat-row {
        display: flex;
        align-items: center;
        margin-bottom: 10px;
        gap: 10px;
    }
    .feat-label { color: #c4c8e0; font-size: 0.82rem; width: 110px; flex-shrink: 0; }
    .feat-bar-bg { background: #1e2130; border-radius: 4px; flex: 1; height: 8px; }
    .feat-bar { background: linear-gradient(90deg, #5c62e8, #7c83fd); border-radius: 4px; height: 8px; }
    .feat-pct { color: #9095b0; font-size: 0.78rem; width: 38px; text-align: right; }

    /* Divider */
    hr { border-color: #2e3040 !important; }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] { background-color: #1a1d27; border-radius: 8px; padding: 4px; }
    .stTabs [data-baseweb="tab"] { color: #9095b0; font-size: 0.88rem; }
    .stTabs [aria-selected="true"] { background-color: #5c62e8 !important; color: white !important; border-radius: 6px; }
</style>
""", unsafe_allow_html=True)


# ── Load model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    with open("car_price_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("model_columns.pkl", "rb") as f:
        cols = pickle.load(f)
    return model, cols

@st.cache_data
def load_data():
    return pd.read_csv("car_price_dataset.csv")

model, model_cols = load_model()
df = load_data()


# ── Sidebar – Inputs ──────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🚗 Car Details")
    st.markdown("Fill in the specs to get an instant price estimate.")
    st.markdown("---")

    st.markdown('<p class="section-title">Brand & Type</p>', unsafe_allow_html=True)
    brand = st.selectbox("Brand", ["BMW", "Ford", "Honda", "Hyundai", "Tesla", "Toyota"])
    fuel  = st.selectbox("Fuel Type", ["Diesel", "Electric", "Hybrid", "Petrol"])
    trans = st.selectbox("Transmission", ["Automatic", "Manual"])

    st.markdown("---")
    st.markdown('<p class="section-title">Specifications</p>', unsafe_allow_html=True)
    year        = st.slider("Model Year",    2005, 2023, 2018)
    engine_size = st.slider("Engine Size (L)", 1.0, 5.0, 2.0, step=0.1)
    horsepower  = st.slider("Horsepower",     70,  399,  150)
    doors       = st.selectbox("Doors", [2, 3, 4], index=2)

    st.markdown("---")
    st.markdown('<p class="section-title">History</p>', unsafe_allow_html=True)
    mileage     = st.slider("Mileage (km)", 5000, 200000, 50000, step=1000)
    owner_count = st.selectbox("Number of Previous Owners", [1, 2, 3, 4])

    st.markdown("---")
    predict_btn = st.button("🔍  Predict Price", use_container_width=True)


# ── Prediction logic ──────────────────────────────────────────────────────────
def make_prediction(brand, fuel, trans, year, engine_size, hp, doors, mileage, owners):
    row = {c: 0 for c in model_cols}
    row["Model_Year"]    = year
    row["Engine_Size"]   = engine_size
    row["Mileage"]       = mileage
    row["Doors"]         = doors
    row["Owner_Count"]   = owners
    row["Horsepower"]    = hp
    if f"Brand_{brand}"       in row: row[f"Brand_{brand}"]       = 1
    if f"Fuel_Type_{fuel}"    in row: row[f"Fuel_Type_{fuel}"]    = 1
    if trans == "Manual"            : row["Transmission_Manual"]   = 1
    X = pd.DataFrame([row])
    pred = model.predict(X)[0]
    return round(pred, 2)

# Run prediction (also on load with defaults)
price = make_prediction(brand, fuel, trans, year, engine_size, horsepower, doors, mileage, owner_count)

# Price range estimate (±8%)
low  = price * 0.92
high = price * 1.08


# ── Main content ──────────────────────────────────────────────────────────────
st.markdown("# 🚗 Car Price Predictor")
st.markdown("*Machine Learning model trained on 2,000 cars — Random Forest Regressor*")
st.markdown("---")

tab1, tab2, tab3 = st.tabs(["  💰  Price Estimate  ", "  📊  Data Insights  ", "  🤖  Model Info  "])

# ─ Tab 1: Prediction ─────────────────────────────────────────────────────────
with tab1:
    col_left, col_right = st.columns([1.1, 1], gap="large")

    with col_left:
        # Main prediction card
        st.markdown(f"""
        <div class="prediction-box">
            <div class="prediction-label">Estimated Market Price</div>
            <div class="prediction-value">${price:,.0f}</div>
            <div class="prediction-sub">Range: ${low:,.0f} – ${high:,.0f}</div>
        </div>
        """, unsafe_allow_html=True)

        # Key metrics
        m1, m2, m3 = st.columns(3)
        age = 2024 - year
        price_per_km = price / mileage if mileage > 0 else 0
        with m1: st.metric("Vehicle Age",     f"{age} yrs")
        with m2: st.metric("Engine",          f"{engine_size}L / {horsepower}hp")
        with m3: st.metric("Price / km",      f"${price_per_km:.2f}")

        st.markdown("---")
        st.markdown("#### 📋 Summary")
        summary_data = {
            "Feature": ["Brand", "Year", "Fuel", "Transmission", "Mileage", "Horsepower", "Engine Size", "Doors", "Owners"],
            "Value":   [brand, year, fuel, trans, f"{mileage:,} km", f"{horsepower} hp", f"{engine_size} L", doors, owner_count]
        }
        st.dataframe(pd.DataFrame(summary_data), hide_index=True, use_container_width=True)

    with col_right:
        st.markdown("#### 🔑 Feature Importance")
        st.markdown("*What drives price the most in this model*")

        importances = model.feature_importances_
        feat_imp = sorted(zip(model_cols, importances), key=lambda x: x[1], reverse=True)

        # Clean names
        name_map = {
            "Engine_Size": "Engine Size", "Horsepower": "Horsepower",
            "Model_Year": "Model Year", "Mileage": "Mileage",
            "Owner_Count": "Owners", "Doors": "Doors",
            "Transmission_Manual": "Manual Trans",
            "Brand_Ford": "Brand: Ford", "Brand_Honda": "Brand: Honda",
            "Brand_Hyundai": "Brand: Hyundai", "Brand_Tesla": "Brand: Tesla",
            "Brand_Toyota": "Brand: Toyota",
            "Fuel_Type_Electric": "Fuel: Electric", "Fuel_Type_Hybrid": "Fuel: Hybrid",
            "Fuel_Type_Petrol": "Fuel: Petrol",
        }
        max_imp = feat_imp[0][1]
        bars_html = ""
        for feat, imp in feat_imp[:10]:
            label = name_map.get(feat, feat)
            pct   = imp * 100
            width = (imp / max_imp) * 100
            bars_html += f"""
            <div class="feat-row">
                <span class="feat-label">{label}</span>
                <div class="feat-bar-bg"><div class="feat-bar" style="width:{width:.1f}%"></div></div>
                <span class="feat-pct">{pct:.1f}%</span>
            </div>"""
        st.markdown(bars_html, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("#### 💡 Price Factors")
        if year >= 2020:
            st.success("✅ Recent model year — adds value")
        elif year < 2010:
            st.warning("⚠️ Older model year — reduces value")
        if mileage < 30000:
            st.success("✅ Low mileage — premium factor")
        elif mileage > 150000:
            st.warning("⚠️ High mileage — reduces value")
        if horsepower >= 250:
            st.info("⚡ High performance engine")
        if fuel == "Electric":
            st.info("🔋 Electric — growing demand")
        if owner_count == 1:
            st.success("✅ Single owner — better resale")
        elif owner_count >= 3:
            st.warning("⚠️ Multiple owners — lower value")


# ─ Tab 2: Data Insights ───────────────────────────────────────────────────────
with tab2:
    st.markdown("### 📊 Dataset Overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Records",  f"{len(df):,}")
    c2.metric("Avg Price",      f"${df['Price'].mean():,.0f}")
    c3.metric("Min Price",      f"${df['Price'].min():,.0f}")
    c4.metric("Max Price",      f"${df['Price'].max():,.0f}")

    st.markdown("---")
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("#### Avg Price by Brand")
        brand_avg = df.groupby("Brand")["Price"].mean().sort_values(ascending=False).reset_index()
        brand_avg.columns = ["Brand", "Avg Price ($)"]
        brand_avg["Avg Price ($)"] = brand_avg["Avg Price ($)"].round(0).astype(int)
        st.bar_chart(brand_avg.set_index("Brand"))

    with col_b:
        st.markdown("#### Avg Price by Fuel Type")
        fuel_avg = df.groupby("Fuel_Type")["Price"].mean().sort_values(ascending=False).reset_index()
        fuel_avg.columns = ["Fuel Type", "Avg Price ($)"]
        fuel_avg["Avg Price ($)"] = fuel_avg["Avg Price ($)"].round(0).astype(int)
        st.bar_chart(fuel_avg.set_index("Fuel Type"))

    st.markdown("---")
    st.markdown("#### Price Distribution by Brand")
    pivot = df.pivot_table(index="Brand", values="Price", aggfunc=["mean", "min", "max"]).round(0)
    pivot.columns = ["Avg Price", "Min Price", "Max Price"]
    st.dataframe(pivot.style.format("${:,.0f}"), use_container_width=True)

    st.markdown("---")
    st.markdown("#### Sample Data (first 10 rows)")
    st.dataframe(df.head(10), use_container_width=True, hide_index=True)


# ─ Tab 3: Model Info ─────────────────────────────────────────────────────────
with tab3:
    st.markdown("### 🤖 About the Model")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        #### Algorithm
        **Random Forest Regressor** — an ensemble of decision trees that averages
        predictions to reduce overfitting and improve accuracy.

        #### Training Data
        - **2,000 car records** across 6 brands
        - Features: year, engine, fuel, transmission, mileage, horsepower, doors, owners
        - Target: Car price in USD

        #### Model Configuration
        ```
        RandomForestRegressor(random_state=42)
        ```
        """)

    with col2:
        st.markdown("#### Top Features by Importance")
        feat_df = pd.DataFrame(
            [(name_map.get(f, f), round(i * 100, 2)) for f, i in feat_imp],
            columns=["Feature", "Importance (%)"]
        ).head(8)
        st.dataframe(feat_df, hide_index=True, use_container_width=True)

    st.markdown("---")
    st.markdown("""
    #### 📁 Project Files
    | File | Description |
    |------|-------------|
    | `car_price_model.pkl` | Trained Random Forest model |
    | `model_columns.pkl`   | One-hot encoded feature columns |
    | `car_price_dataset.csv` | Training dataset (2,000 rows) |
    | `app.py`              | Streamlit web application |

    #### 🚀 Deployment
    This app is deployed free on **[Streamlit Community Cloud](https://streamlit.io/cloud)**.
    > Push to GitHub → Connect repo on share.streamlit.io → Live in 2 minutes.
    """)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:#555; font-size:0.8rem;'>"
    "Built with Streamlit · Random Forest ML · Portfolio Project</p>",
    unsafe_allow_html=True
)
