import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import plotly.graph_objects as go
import time
import os
from dotenv import load_dotenv
from pymongo import MongoClient

# --- DATABASE CONNECTION ---
load_dotenv()

mongo_uri = os.getenv("MONGO_URI")

if not mongo_uri:
    mongo_uri = st.secrets["MONGO_URI"]

@st.cache_resource
def init_connection():
    return MongoClient(mongo_uri)

client = init_connection()
db = client['smartsole_db']
collection = db['sensor_logs']

def get_latest_sensor_data():
    """Fetches the single most recent reading."""
    latest = collection.find_one(sort=[("timestamp", -1)])
    if latest:
        return latest
    return {"heel": 0, "arch": 0, "meta1": 0, "meta3": 0, "meta5": 0, "toe": 0}

def get_historical_data(limit=30):
    """Fetches the last N readings for the time-series chart."""
    cursor = collection.find({}, {"_id": 0, "timestamp": 1, "heel": 1, "arch": 1, "meta1": 1, "meta3": 1, "meta5": 1, "toe": 1}).sort("timestamp", -1).limit(limit)
    data = list(cursor)[::-1]  # Reverse to get chronological order
    return data

# --- PAGE CONFIGURATION & ADVANCED CSS ---
st.set_page_config(page_title="Smart Sole Pro", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&display=swap');
    
    .stApp {
        background: radial-gradient(circle at top left, #1a2a3a 0%, #0d1117 100%);
        color: #e6edf3;
        font-family: 'Inter', sans-serif;
    }
    
    header, #MainMenu, footer {visibility: hidden;}
    
    /* Advanced Glass Cards */
    .glass-card {
        background: rgba(33, 38, 45, 0.65);
        border-radius: 12px;
        border: 1px solid rgba(240, 246, 252, 0.1);
        box-shadow: 0 8px 24px rgba(0,0,0,0.5);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        padding: 20px;
        margin-bottom: 20px;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .glass-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 32px rgba(0, 229, 255, 0.15);
        border: 1px solid rgba(0, 229, 255, 0.3);
    }
    
    /* Typography */
    .section-header { color: #00e5ff; font-weight: 800; font-size: 1.4rem; letter-spacing: 1.5px; text-transform: uppercase; margin-bottom: 15px; border-bottom: 1px solid rgba(0, 229, 255, 0.2); padding-bottom: 5px; }
    .card-title { color: #8b949e; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 1.5px; font-weight: 600; }
    .card-value { font-size: 2.8rem; font-weight: 800; line-height: 1.1; margin: 8px 0; }
    .text-cyan { background: linear-gradient(90deg, #00e5ff, #0077ff); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    .text-warning { background: linear-gradient(90deg, #ff416c, #ff4b2b); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    .text-green { background: linear-gradient(90deg, #00ff88, #00b359); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    
    /* Sensor Grid (S1-S6) */
    .sensor-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 15px; }
    .sensor-box { background: rgba(0,0,0,0.3); border-radius: 8px; padding: 15px; text-align: center; border: 1px solid rgba(255,255,255,0.05); }
    .sensor-id { color: #00e5ff; font-size: 0.8rem; font-weight: bold; letter-spacing: 1px; }
    .sensor-val { font-size: 1.8rem; font-weight: bold; color: #fff; }
    .sensor-loc { color: #8b949e; font-size: 0.75rem; }
    
    /* Progress Bars Customization */
    .stProgress > div > div > div > div { background-image: linear-gradient(to right, #00e5ff, #0077ff); }
    .warning-bar > div > div > div > div { background-image: linear-gradient(to right, #ff416c, #ff4b2b) !important; }
</style>
""", unsafe_allow_html=True)

# --- MACHINE LEARNING MODEL ---
@st.cache_resource
def train_ml_model():
    np.random.seed(42)
    n = 1000
    h_heel, h_mid, h_fore, h_toes = np.random.uniform(35, 40, n), np.random.uniform(14, 18, n), np.random.uniform(32, 38, n), np.random.uniform(5, 7, n)
    nh_heel, nh_mid, nh_fore, nh_toes = np.random.uniform(25, 30, n), np.random.uniform(20, 26, n), np.random.uniform(40, 48, n), np.random.uniform(7, 10, n)
    
    df_h = pd.DataFrame({'Heel': h_heel, 'Midfoot': h_mid, 'Forefoot': h_fore, 'Toes': h_toes, 'Label': 'Healthy Pattern'})
    df_nh = pd.DataFrame({'Heel': nh_heel, 'Midfoot': nh_mid, 'Forefoot': nh_fore, 'Toes': nh_toes, 'Label': 'Non-Healthy Pattern'})
    
    df = pd.concat([df_h, df_nh], ignore_index=True)
    X, y = df[['Heel', 'Midfoot', 'Forefoot', 'Toes']], df['Label']
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

model = train_ml_model()

# --- SIDEBAR CONTROLS ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/5029/5029141.png", width=60)
    st.markdown("<h2 style='color:#00e5ff; margin-top:0;'>Smart Sole UI</h2>", unsafe_allow_html=True)
    st.markdown("---")
    test_mode = st.radio("Select Test Condition:", ["Standing (Full Weight)", "Seated (Partial Load)"])
    st.markdown("<p style='font-size:0.8rem; color:#8b949e;'>Toggle to adjust load expectations based on Reference Load Measurements.</p>", unsafe_allow_html=True)

# --- DATA PROCESSING ---
live_data = get_latest_sensor_data()
hist_data = get_historical_data()

s_heel = live_data.get('heel', 0)
s_arch = live_data.get('arch', 0)
s_meta1 = live_data.get('meta1', 0)
s_meta3 = live_data.get('meta3', 0)
s_meta5 = live_data.get('meta5', 0)
s_toe = live_data.get('toe', 0)

total_load = sum([s_heel, s_arch, s_meta1, s_meta3, s_meta5, s_toe])
safe_total = total_load if total_load > 0 else 1

heel_pct = (s_heel / safe_total) * 100
midfoot_pct = ((s_arch + s_meta1) / safe_total) * 100
forefoot_pct = ((s_meta3 + s_meta5) / safe_total) * 100
toes_pct = (s_toe / safe_total) * 100

input_features = pd.DataFrame([[heel_pct, midfoot_pct, forefoot_pct, toes_pct]], columns=['Heel', 'Midfoot', 'Forefoot', 'Toes'])
prediction = model.predict(input_features)[0]
confidence = max(model.predict_proba(input_features)[0]) * 100

# Require minimum load to classify (10kg standing, 2kg seated)
min_load = 10000 if test_mode == "Standing (Full Weight)" else 2000
if total_load < min_load:
    prediction = "Awaiting Load..."
    confidence = 0.0

# --- MAIN UI RENDER ---
st.markdown("<div class='section-header'>Real-Time Clinical Dashboard</div>", unsafe_allow_html=True)

# ROW 1: KPI CARDS
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(f"""
    <div class="glass-card">
        <div class="card-title">Total Measured Load</div>
        <div class="card-value text-cyan">{total_load/1000:.2f} <span style='font-size:1.5rem; color:#8b949e;'>kg</span></div>
        <div style="color: #8b949e; font-size: 0.8rem;">Sum of all channels (S1-S6)</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    val_class = "text-warning" if "Non-Healthy" in prediction else ("text-green" if "Healthy" in prediction else "text-cyan")
    st.markdown(f"""
    <div class="glass-card">
        <div class="card-title">Pattern Classification (ML)</div>
        <div class="card-value {val_class}" style="font-size: 2.2rem;">{prediction}</div>
        <div style="color: #8b949e; font-size: 0.8rem;">Based on regional reference rules</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="glass-card">
        <div class="card-title">Model Confidence</div>
        <div class="card-value text-cyan">{confidence:.1f}%</div>
        <div style="color: #8b949e; font-size: 0.8rem;">Random Forest Probability</div>
    </div>
    """, unsafe_allow_html=True)

# ROW 2: LIVE RAW SENSORS (S1-S6)
st.markdown("<div class='section-header' style='margin-top:20px;'>Hardware Telemetry (Live S1-S6)</div>", unsafe_allow_html=True)
st.markdown("""
<div class="glass-card">
    <div class="sensor-grid">
        <div class="sensor-box"><div class="sensor-id">S1</div><div class="sensor-val">""" + str(int(s_meta1)) + """g</div><div class="sensor-loc">Midfoot (Lateral)</div></div>
        <div class="sensor-box"><div class="sensor-id">S2</div><div class="sensor-val">""" + str(int(s_arch)) + """g</div><div class="sensor-loc">Midfoot (Medial)</div></div>
        <div class="sensor-box"><div class="sensor-id">S3</div><div class="sensor-val">""" + str(int(s_meta3)) + """g</div><div class="sensor-loc">Forefoot (Medial)</div></div>
        <div class="sensor-box"><div class="sensor-id">S4</div><div class="sensor-val">""" + str(int(s_meta5)) + """g</div><div class="sensor-loc">Forefoot (Lateral)</div></div>
        <div class="sensor-box"><div class="sensor-id">S5</div><div class="sensor-val">""" + str(int(s_toe)) + """g</div><div class="sensor-loc">Toes</div></div>
        <div class="sensor-box"><div class="sensor-id">S6</div><div class="sensor-val">""" + str(int(s_heel)) + """g</div><div class="sensor-loc">Heel</div></div>
    </div>
</div>
""", unsafe_allow_html=True)

# ROW 3: REGIONAL ANALYSIS & TIME SERIES
col_regions, col_history = st.columns([1, 1.5])

with col_regions:
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.markdown("<div class='card-title' style='margin-bottom:15px;'>Regional Percentage Breakdown</div>", unsafe_allow_html=True)
    
    regions = {
        "Heel (S6)": (heel_pct, 35, 40),
        "Forefoot (S3, S4)": (forefoot_pct, 32, 38),
        "Midfoot (S1, S2)": (midfoot_pct, 14, 18),
        "Toes (S5)": (toes_pct, 5, 7)
    }
    
    for name, (pct, ref_min, ref_max) in regions.items():
        is_warning = not (ref_min <= pct <= ref_max) and total_load > min_load
        color = "#ff416c" if is_warning else "#00e5ff"
        
        st.markdown(f"""
        <div style="display:flex; justify-content:space-between; align-items:flex-end;">
            <strong style="color:#e6edf3;">{name}</strong>
            <span style='color:{color}; font-size:1.2rem; font-weight:800;'>{pct:.1f}%</span>
        </div>
        """, unsafe_allow_html=True)
        
        # Streamlit progress bar injected with custom CSS class if out of bounds
        if is_warning:
            st.markdown('<div class="warning-bar">', unsafe_allow_html=True)
        st.progress(min(pct / 100.0, 1.0))
        if is_warning:
            st.markdown('</div>', unsafe_allow_html=True)
            
        st.markdown(f"<span style='font-size:0.75rem; color:#8b949e;'>Healthy Reference: {ref_min}% - {ref_max}%</span><br><br>", unsafe_allow_html=True)
        
    st.markdown("</div>", unsafe_allow_html=True)

with col_history:
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.markdown("<div class='card-title'>Trial Comparison: Load Over Time</div>", unsafe_allow_html=True)
    
    if hist_data:
        # Calculate total load history
        history_totals = [sum([d.get('heel',0), d.get('arch',0), d.get('meta1',0), d.get('meta3',0), d.get('meta5',0), d.get('toe',0)]) / 1000 for d in hist_data]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            y=history_totals,
            mode='lines+markers',
            line=dict(color='#00e5ff', width=3, shape='spline'),
            marker=dict(size=6, color='#0077ff'),
            fill='tozeroy',
            fillcolor='rgba(0, 229, 255, 0.1)'
        ))
        
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#8b949e'),
            margin=dict(t=10, b=20, l=30, r=10),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(gridcolor='rgba(255,255,255,0.05)', title='Total Load (kg)')
        )
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
    else:
        st.markdown("<p style='text-align:center; color:#8b949e; margin-top:50px;'>Waiting for data points...</p>", unsafe_allow_html=True)
        
    st.markdown("</div>", unsafe_allow_html=True)

# Auto-refresh loop
time.sleep(1)
st.rerun()