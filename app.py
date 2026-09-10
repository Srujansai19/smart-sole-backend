from flask import Flask, jsonify, render_template_string, request
from pymongo import MongoClient
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
import os
from threading import Lock

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

app = Flask(__name__)

# -------------------------------------------------------------------
# DATABASE CONNECTION
# -------------------------------------------------------------------
mongo_uri = os.getenv("MONGO_URI")

_client = None
_collection = None
_db_lock = Lock()


def get_collection():
    """Create/reuse the MongoDB connection used by the Vercel function."""
    global _client, _collection

    if not mongo_uri:
        return None

    if _collection is None:
        with _db_lock:
            if _collection is None:
                _client = MongoClient(
                    mongo_uri,
                    serverSelectionTimeoutMS=3000,
                    connectTimeoutMS=3000,
                    socketTimeoutMS=3000,
                    maxPoolSize=10,
                )
                _collection = _client["smartsole_db"]["sensor_logs"]

    return _collection


def get_latest_sensor_data():
    """Fetches the single most recent reading."""
    collection = get_collection()

    if collection is None:
        return {
            "heel": 0,
            "arch": 0,
            "meta1": 0,
            "meta3": 0,
            "meta5": 0,
            "toe": 0,
        }

    try:
        latest = collection.find_one(sort=[("timestamp", -1)])
        if latest:
            latest["_id"] = str(latest.get("_id", ""))
            return latest
    except Exception:
        pass

    return {
        "heel": 0,
        "arch": 0,
        "meta1": 0,
        "meta3": 0,
        "meta5": 0,
        "toe": 0,
    }


def get_historical_data(limit=30):
    """Fetches the last N readings for the time-series chart."""
    collection = get_collection()

    if collection is None:
        return []

    try:
        cursor = collection.find(
            {},
            {
                "_id": 0,
                "timestamp": 1,
                "heel": 1,
                "arch": 1,
                "meta1": 1,
                "meta3": 1,
                "meta5": 1,
                "toe": 1,
            },
        ).sort("timestamp", -1).limit(limit)

        data = list(cursor)
        data.reverse()
        return data
    except Exception:
        return []


# -------------------------------------------------------------------
# MACHINE LEARNING MODEL
# -------------------------------------------------------------------
_model = None
_model_lock = Lock()


def train_ml_model():
    np.random.seed(42)
    n = 1000

    h_heel = np.random.uniform(35, 40, n)
    h_mid = np.random.uniform(14, 18, n)
    h_fore = np.random.uniform(32, 38, n)
    h_toes = np.random.uniform(5, 7, n)

    nh_heel = np.random.uniform(25, 30, n)
    nh_mid = np.random.uniform(20, 26, n)
    nh_fore = np.random.uniform(40, 48, n)
    nh_toes = np.random.uniform(7, 10, n)

    df_h = pd.DataFrame(
        {
            "Heel": h_heel,
            "Midfoot": h_mid,
            "Forefoot": h_fore,
            "Toes": h_toes,
            "Label": "Healthy Pattern",
        }
    )

    df_nh = pd.DataFrame(
        {
            "Heel": nh_heel,
            "Midfoot": nh_mid,
            "Forefoot": nh_fore,
            "Toes": nh_toes,
            "Label": "Non-Healthy Pattern",
        }
    )

    df = pd.concat([df_h, df_nh], ignore_index=True)
    X = df[["Heel", "Midfoot", "Forefoot", "Toes"]]
    y = df["Label"]

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model


def get_model():
    global _model

    if _model is None:
        with _model_lock:
            if _model is None:
                _model = train_ml_model()

    return _model


# -------------------------------------------------------------------
# DATA PROCESSING
# -------------------------------------------------------------------
def make_dashboard_data(test_mode):
    live_data = get_latest_sensor_data()
    hist_data = get_historical_data()

    s_heel = float(live_data.get("heel", 0) or 0)
    s_arch = float(live_data.get("arch", 0) or 0)
    s_meta1 = float(live_data.get("meta1", 0) or 0)
    s_meta3 = float(live_data.get("meta3", 0) or 0)
    s_meta5 = float(live_data.get("meta5", 0) or 0)
    s_toe = float(live_data.get("toe", 0) or 0)

    total_load = sum(
        [s_heel, s_arch, s_meta1, s_meta3, s_meta5, s_toe]
    )
    safe_total = total_load if total_load > 0 else 1

    heel_pct = (s_heel / safe_total) * 100
    midfoot_pct = ((s_arch + s_meta1) / safe_total) * 100
    forefoot_pct = ((s_meta3 + s_meta5) / safe_total) * 100
    toes_pct = (s_toe / safe_total) * 100

    input_features = pd.DataFrame(
        [[heel_pct, midfoot_pct, forefoot_pct, toes_pct]],
        columns=["Heel", "Midfoot", "Forefoot", "Toes"],
    )

    model = get_model()
    prediction = model.predict(input_features)[0]
    confidence = max(model.predict_proba(input_features)[0]) * 100

    # Same thresholds as the original Streamlit application.
    min_load = 10000 if test_mode == "Standing (Full Weight)" else 2000

    if total_load < min_load:
        prediction = "Awaiting Load..."
        confidence = 0.0

    history_totals = []
    for d in hist_data:
        history_totals.append(
            sum(
                [
                    float(d.get("heel", 0) or 0),
                    float(d.get("arch", 0) or 0),
                    float(d.get("meta1", 0) or 0),
                    float(d.get("meta3", 0) or 0),
                    float(d.get("meta5", 0) or 0),
                    float(d.get("toe", 0) or 0),
                ]
            )
            / 1000
        )

    return {
        "sensors": {
            "heel": s_heel,
            "arch": s_arch,
            "meta1": s_meta1,
            "meta3": s_meta3,
            "meta5": s_meta5,
            "toe": s_toe,
        },
        "total_load": total_load,
        "heel_pct": heel_pct,
        "midfoot_pct": midfoot_pct,
        "forefoot_pct": forefoot_pct,
        "toes_pct": toes_pct,
        "prediction": prediction,
        "confidence": confidence,
        "history_totals": history_totals,
        "db_configured": bool(mongo_uri),
    }


# -------------------------------------------------------------------
# EXACT-SAME VISUAL UI, IMPLEMENTED AS FLASK HTML/JS
# -------------------------------------------------------------------
HTML = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Smart Sole Pro</title>

<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>

<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&display=swap');

    * {
        box-sizing: border-box;
    }

    body {
        margin: 0;
        background: radial-gradient(circle at top left, #1a2a3a 0%, #0d1117 100%);
        color: #e6edf3;
        font-family: 'Inter', sans-serif;
        min-height: 100vh;
    }

    header, #MainMenu, footer {
        visibility: hidden;
    }

    .app-shell {
        min-height: 100vh;
        display: flex;
    }

    .sidebar {
        width: 270px;
        min-width: 270px;
        padding: 25px 18px;
        border-right: 1px solid rgba(240, 246, 252, 0.08);
        background: rgba(13, 17, 23, 0.72);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
    }

    .sidebar-logo {
        width: 60px;
        height: 60px;
        object-fit: contain;
        display: block;
        margin-bottom: 2px;
    }

    .sidebar-title {
        color: #00e5ff;
        margin-top: 0;
        margin-bottom: 16px;
        font-size: 1.5rem;
        font-weight: 800;
    }

    .sidebar-line {
        border: 0;
        border-top: 1px solid rgba(255,255,255,0.1);
        margin: 14px 0 20px;
    }

    .sidebar-label {
        font-weight: 600;
        font-size: 0.95rem;
        display: block;
        margin-bottom: 10px;
    }

    .radio-option {
        display: flex;
        align-items: center;
        gap: 9px;
        padding: 7px 0;
        color: #e6edf3;
        cursor: pointer;
    }

    .radio-option input {
        accent-color: #00e5ff;
    }

    .sidebar-help {
        font-size: 0.8rem;
        color: #8b949e;
        line-height: 1.45;
        margin-top: 8px;
    }

    .main {
        flex: 1;
        min-width: 0;
        padding: 28px 32px 35px;
    }

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
    .section-header {
        color: #00e5ff;
        font-weight: 800;
        font-size: 1.4rem;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        margin-bottom: 15px;
        border-bottom: 1px solid rgba(0, 229, 255, 0.2);
        padding-bottom: 5px;
    }

    .card-title {
        color: #8b949e;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        font-weight: 600;
    }

    .card-value {
        font-size: 2.8rem;
        font-weight: 800;
        line-height: 1.1;
        margin: 8px 0;
    }

    .text-cyan {
        background: linear-gradient(90deg, #00e5ff, #0077ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .text-warning {
        background: linear-gradient(90deg, #ff416c, #ff4b2b);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .text-green {
        background: linear-gradient(90deg, #00ff88, #00b359);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    /* Sensor Grid (S1-S6) */
    .sensor-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 15px;
    }

    .sensor-box {
        background: rgba(0,0,0,0.3);
        border-radius: 8px;
        padding: 15px;
        text-align: center;
        border: 1px solid rgba(255,255,255,0.05);
    }

    .sensor-id {
        color: #00e5ff;
        font-size: 0.8rem;
        font-weight: bold;
        letter-spacing: 1px;
    }

    .sensor-val {
        font-size: 1.8rem;
        font-weight: bold;
        color: #fff;
    }

    .sensor-loc {
        color: #8b949e;
        font-size: 0.75rem;
    }

    /* Progress Bars Customization */
    .progress-track {
        height: 8px;
        background: rgba(255,255,255,0.08);
        border-radius: 999px;
        overflow: hidden;
        margin-top: 8px;
    }

    .progress-fill {
        height: 100%;
        background-image: linear-gradient(to right, #00e5ff, #0077ff);
        border-radius: 999px;
        transition: width 0.35s ease;
    }

    .progress-fill.warning {
        background-image: linear-gradient(to right, #ff416c, #ff4b2b);
    }

    .region-row {
        display: flex;
        justify-content: space-between;
        align-items: flex-end;
    }

    .healthy-reference {
        font-size: 0.75rem;
        color: #8b949e;
        display: inline-block;
        margin-top: 5px;
        margin-bottom: 18px;
    }

    .history-container {
        min-height: 390px;
    }

    #history-chart {
        width: 100%;
        height: 340px;
    }

    .waiting {
        text-align: center;
        color: #8b949e;
        margin-top: 50px;
    }

    .top-row {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 20px;
    }

    .analysis-row {
        display: grid;
        grid-template-columns: 1fr 1.5fr;
        gap: 20px;
    }

    @media (max-width: 1000px) {
        .top-row,
        .analysis-row {
            grid-template-columns: 1fr;
        }

        .sidebar {
            width: 220px;
            min-width: 220px;
        }
    }

    @media (max-width: 700px) {
        .app-shell {
            display: block;
        }

        .sidebar {
            width: 100%;
            min-width: 0;
        }

        .main {
            padding: 20px 15px;
        }

        .sensor-grid {
            grid-template-columns: repeat(2, 1fr);
        }

        .top-row {
            grid-template-columns: 1fr;
        }
    }
</style>
</head>

<body>
<div class="app-shell">

    <aside class="sidebar">
        <img
            class="sidebar-logo"
            src="https://cdn-icons-png.flaticon.com/512/5029/5029141.png"
            width="60"
            alt="Smart Sole"
        >

        <h2 class="sidebar-title">Smart Sole UI</h2>

        <hr class="sidebar-line">

        <label class="sidebar-label">Select Test Condition:</label>

        <label class="radio-option">
            <input
                type="radio"
                name="test_mode"
                value="Standing (Full Weight)"
                checked
            >
            <span>Standing (Full Weight)</span>
        </label>

        <label class="radio-option">
            <input
                type="radio"
                name="test_mode"
                value="Seated (Partial Load)"
            >
            <span>Seated (Partial Load)</span>
        </label>

        <p class="sidebar-help">
            Toggle to adjust load expectations based on Reference Load Measurements.
        </p>
    </aside>

    <main class="main">

        <div class="section-header">Real-Time Clinical Dashboard</div>

        <!-- ROW 1: KPI CARDS -->
        <div class="top-row">

            <div class="glass-card">
                <div class="card-title">Total Measured Load</div>
                <div class="card-value text-cyan">
                    <span id="total-load">0.00</span>
                    <span style="font-size:1.5rem; color:#8b949e;">kg</span>
                </div>
                <div style="color:#8b949e; font-size:0.8rem;">
                    Sum of all channels (S1-S6)
                </div>
            </div>

            <div class="glass-card">
                <div class="card-title">Pattern Classification (ML)</div>
                <div
                    id="prediction"
                    class="card-value text-cyan"
                    style="font-size:2.2rem;"
                >
                    Awaiting Load...
                </div>
                <div style="color:#8b949e; font-size:0.8rem;">
                    Based on regional reference rules
                </div>
            </div>

            <div class="glass-card">
                <div class="card-title">Model Confidence</div>
                <div class="card-value text-cyan">
                    <span id="confidence">0.0</span>%
                </div>
                <div style="color:#8b949e; font-size:0.8rem;">
                    Random Forest Probability
                </div>
            </div>

        </div>

        <!-- ROW 2 -->
        <div class="section-header" style="margin-top:20px;">
            Hardware Telemetry (Live S1-S6)
        </div>

        <div class="glass-card">
            <div class="sensor-grid">

                <div class="sensor-box">
                    <div class="sensor-id">S1</div>
                    <div class="sensor-val" id="s1">0g</div>
                    <div class="sensor-loc">Midfoot (Lateral)</div>
                </div>

                <div class="sensor-box">
                    <div class="sensor-id">S2</div>
                    <div class="sensor-val" id="s2">0g</div>
                    <div class="sensor-loc">Midfoot (Medial)</div>
                </div>

                <div class="sensor-box">
                    <div class="sensor-id">S3</div>
                    <div class="sensor-val" id="s3">0g</div>
                    <div class="sensor-loc">Forefoot (Medial)</div>
                </div>

                <div class="sensor-box">
                    <div class="sensor-id">S4</div>
                    <div class="sensor-val" id="s4">0g</div>
                    <div class="sensor-loc">Forefoot (Lateral)</div>
                </div>

                <div class="sensor-box">
                    <div class="sensor-id">S5</div>
                    <div class="sensor-val" id="s5">0g</div>
                    <div class="sensor-loc">Toes</div>
                </div>

                <div class="sensor-box">
                    <div class="sensor-id">S6</div>
                    <div class="sensor-val" id="s6">0g</div>
                    <div class="sensor-loc">Heel</div>
                </div>

            </div>
        </div>

        <!-- ROW 3 -->
        <div class="analysis-row">

            <div class="glass-card">
                <div
                    class="card-title"
                    style="margin-bottom:15px;"
                >
                    Regional Percentage Breakdown
                </div>

                <div id="regions">

                    <div class="region">
                        <div class="region-row">
                            <strong>Heel (S6)</strong>
                            <span id="heel-pct" class="region-value">0.0%</span>
                        </div>
                        <div class="progress-track">
                            <div id="heel-bar" class="progress-fill" style="width:0%;"></div>
                        </div>
                        <span class="healthy-reference">
                            Healthy Reference: 35% - 40%
                        </span>
                    </div>

                    <div class="region">
                        <div class="region-row">
                            <strong>Forefoot (S3, S4)</strong>
                            <span id="forefoot-pct" class="region-value">0.0%</span>
                        </div>
                        <div class="progress-track">
                            <div id="forefoot-bar" class="progress-fill" style="width:0%;"></div>
                        </div>
                        <span class="healthy-reference">
                            Healthy Reference: 32% - 38%
                        </span>
                    </div>

                    <div class="region">
                        <div class="region-row">
                            <strong>Midfoot (S1, S2)</strong>
                            <span id="midfoot-pct" class="region-value">0.0%</span>
                        </div>
                        <div class="progress-track">
                            <div id="midfoot-bar" class="progress-fill" style="width:0%;"></div>
                        </div>
                        <span class="healthy-reference">
                            Healthy Reference: 14% - 18%
                        </span>
                    </div>

                    <div class="region">
                        <div class="region-row">
                            <strong>Toes (S5)</strong>
                            <span id="toes-pct" class="region-value">0.0%</span>
                        </div>
                        <div class="progress-track">
                            <div id="toes-bar" class="progress-fill" style="width:0%;"></div>
                        </div>
                        <span class="healthy-reference">
                            Healthy Reference: 5% - 7%
                        </span>
                    </div>

                </div>
            </div>

            <div class="glass-card history-container">
                <div class="card-title">
                    Trial Comparison: Load Over Time
                </div>

                <div id="history-chart"></div>
                <div id="waiting" class="waiting" style="display:none;">
                    Waiting for data points...
                </div>
            </div>

        </div>

    </main>
</div>

<script>
let chartInitialized = false;

function getTestMode() {
    const selected = document.querySelector(
        'input[name="test_mode"]:checked'
    );
    return selected ? selected.value : 'Standing (Full Weight)';
}

function setText(id, value) {
    const el = document.getElementById(id);
    if (el) el.textContent = value;
}

function updateRegion(pctId, barId, pct, refMin, refMax, totalLoad, minLoad) {
    const pctEl = document.getElementById(pctId);
    const barEl = document.getElementById(barId);

    const warning =
        !(pct >= refMin && pct <= refMax) &&
        totalLoad > minLoad;

    pctEl.textContent = pct.toFixed(1) + '%';
    pctEl.style.color = warning ? '#ff416c' : '#00e5ff';

    barEl.style.width = Math.min(Math.max(pct, 0), 100) + '%';

    if (warning) {
        barEl.classList.add('warning');
    } else {
        barEl.classList.remove('warning');
    }
}

function updateChart(history) {
    const chart = document.getElementById('history-chart');
    const waiting = document.getElementById('waiting');

    if (!history || history.length === 0) {
        chart.style.display = 'none';
        waiting.style.display = 'block';
        return;
    }

    chart.style.display = 'block';
    waiting.style.display = 'none';

    const trace = {
        y: history,
        mode: 'lines+markers',
        line: {
            color: '#00e5ff',
            width: 3,
            shape: 'spline'
        },
        marker: {
            size: 6,
            color: '#0077ff'
        },
        fill: 'tozeroy',
        fillcolor: 'rgba(0, 229, 255, 0.1)'
    };

    const layout = {
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        font: {
            color: '#8b949e'
        },
        margin: {
            t: 10,
            b: 20,
            l: 45,
            r: 10
        },
        xaxis: {
            showgrid: false,
            zeroline: false,
            showticklabels: false
        },
        yaxis: {
            gridcolor: 'rgba(255,255,255,0.05)',
            title: 'Total Load (kg)'
        }
    };

    const config = {
        displayModeBar: false,
        responsive: true
    };

    Plotly.react(
        'history-chart',
        [trace],
        layout,
        config
    );
}

function updateDashboard(data) {
    const s = data.sensors || {};

    setText('total-load', (data.total_load / 1000).toFixed(2));

    setText('s1', Math.round(s.meta1 || 0) + 'g');
    setText('s2', Math.round(s.arch || 0) + 'g');
    setText('s3', Math.round(s.meta3 || 0) + 'g');
    setText('s4', Math.round(s.meta5 || 0) + 'g');
    setText('s5', Math.round(s.toe || 0) + 'g');
    setText('s6', Math.round(s.heel || 0) + 'g');

    setText('confidence', Number(data.confidence || 0).toFixed(1));

    const predictionEl = document.getElementById('prediction');
    predictionEl.textContent = data.prediction;

    predictionEl.classList.remove(
        'text-cyan',
        'text-warning',
        'text-green'
    );

    if (data.prediction.includes('Non-Healthy')) {
        predictionEl.classList.add('text-warning');
    } else if (data.prediction.includes('Healthy')) {
        predictionEl.classList.add('text-green');
    } else {
        predictionEl.classList.add('text-cyan');
    }

    const mode = getTestMode();
    const minLoad =
        mode === 'Standing (Full Weight)' ? 10000 : 2000;

    updateRegion(
        'heel-pct',
        'heel-bar',
        data.heel_pct,
        35,
        40,
        data.total_load,
        minLoad
    );

    updateRegion(
        'forefoot-pct',
        'forefoot-bar',
        data.forefoot_pct,
        32,
        38,
        data.total_load,
        minLoad
    );

    updateRegion(
        'midfoot-pct',
        'midfoot-bar',
        data.midfoot_pct,
        14,
        18,
        data.total_load,
        minLoad
    );

    updateRegion(
        'toes-pct',
        'toes-bar',
        data.toes_pct,
        5,
        7,
        data.total_load,
        minLoad
    );

    updateChart(data.history_totals || []);
}

let requestInProgress = false;

async function refreshDashboard() {
    if (requestInProgress) return;

    requestInProgress = true;

    try {
        const mode = encodeURIComponent(getTestMode());

        const response = await fetch(
            '/api/data?test_mode=' + mode,
            {
                cache: 'no-store'
            }
        );

        if (!response.ok) {
            throw new Error('Dashboard request failed');
        }

        const data = await response.json();
        updateDashboard(data);

    } catch (error) {
        console.error(error);
    } finally {
        requestInProgress = false;
    }
}

document
    .querySelectorAll('input[name="test_mode"]')
    .forEach(input => {
        input.addEventListener('change', refreshDashboard);
    });

refreshDashboard();

// Same one-second live refresh behavior as the original app.
setInterval(refreshDashboard, 1000);
</script>

</body>
</html>
"""


# -------------------------------------------------------------------
# FLASK ROUTES
# -------------------------------------------------------------------
@app.route("/")
def home():
    return render_template_string(HTML)


@app.route("/api/data")
def api_data():
    test_mode = request.args.get(
        "test_mode",
        "Standing (Full Weight)"
    )

    if test_mode not in [
        "Standing (Full Weight)",
        "Seated (Partial Load)",
    ]:
        test_mode = "Standing (Full Weight)"

    return jsonify(make_dashboard_data(test_mode))


@app.route("/api/health")
def health():
    return jsonify({
        "status": "online",
        "mongo_uri_configured": bool(mongo_uri),
        "service": "Smart Sole Pro"
    })


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000)),
        debug=True
    )
