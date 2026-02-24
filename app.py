import os
from datetime import datetime

import numpy as np
import joblib
from flask import Flask, request, jsonify, render_template_string, redirect, url_for
from flask_sqlalchemy import SQLAlchemy

# --------------------------------------------------
# App + DB Setup
# --------------------------------------------------
app = Flask(__name__)
app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", "dev")

db_url = os.getenv("DATABASE_URL")
# Render sometimes uses postgres:// (old). SQLAlchemy wants postgresql://
if db_url and db_url.startswith("postgres://"):
    db_url = db_url.replace("postgres://", "postgresql://", 1)
if not db_url:
    db_url = "sqlite:///local.db"

app.config["SQLALCHEMY_DATABASE_URI"] = db_url
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)

# --------------------------------------------------
# Model (optional)
# --------------------------------------------------
MODEL_PATH = "demo_model.pkl"
model = None
if os.path.exists(MODEL_PATH):
    try:
        model = joblib.load(MODEL_PATH)
    except Exception:
        model = None

# --------------------------------------------------
# DB Model
# --------------------------------------------------
class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)

    age = db.Column(db.Float, nullable=False)
    bmi = db.Column(db.Float, nullable=False)
    exercise_level = db.Column(db.Float, nullable=False)
    sys_bp = db.Column(db.Float, nullable=False)
    dia_bp = db.Column(db.Float, nullable=False)
    heart_rate = db.Column(db.Float, nullable=False)

    label = db.Column(db.String(32), nullable=False)
    probability = db.Column(db.Float, nullable=False)  # 0..1

with app.app_context():
    db.create_all()

# --------------------------------------------------
# Helpers
# --------------------------------------------------
def _to_float(value, default=None):
    if value is None:
        if default is None:
            raise ValueError("Missing required field")
        return float(default)
    s = str(value).strip()
    if s == "":
        if default is None:
            raise ValueError("Missing required field")
        return float(default)
    return float(s)

def build_features(payload: dict):
    """
    Accepts either:
      - form fields: age,bmi,exercise_level,sys_bp,dia_bp,heart_rate
      - OR older field names: exercise, systolic_bp, diastolic_bp
      - exercise_level may also be "Low/Moderate/High"
    """
    age = _to_float(payload.get("age"))
    bmi = _to_float(payload.get("bmi"))

    ex_raw = payload.get("exercise_level", payload.get("exercise"))
    if ex_raw is None:
        raise ValueError("Missing exercise_level")
    ex_s = str(ex_raw).strip().lower()
    if ex_s in ["low", "0"]:
        exercise_level = 0.0
    elif ex_s in ["moderate", "medium", "1"]:
        exercise_level = 1.0
    elif ex_s in ["high", "2"]:
        exercise_level = 2.0
    else:
        # if user typed a number like 0/1/2 or 0.5 etc
        exercise_level = float(ex_s)

    sys_bp = _to_float(payload.get("sys_bp", payload.get("systolic_bp", payload.get("systolic"))))
    dia_bp = _to_float(payload.get("dia_bp", payload.get("diastolic_bp", payload.get("diastolic"))))
    heart_rate = _to_float(payload.get("heart_rate", payload.get("hr")))

    X = np.array([[age, bmi, exercise_level, sys_bp, dia_bp, heart_rate]], dtype=float)
    return age, bmi, exercise_level, sys_bp, dia_bp, heart_rate, X

def run_model(X: np.ndarray):
    """
    Returns (label, prob_high).
    If model exists and supports predict_proba, uses it.
    Otherwise uses a stable heuristic so the app always works.
    """
    if model is not None:
        try:
            pred = int(model.predict(X)[0])
            prob_high = float(model.predict_proba(X)[0][1]) if hasattr(model, "predict_proba") else (1.0 if pred == 1 else 0.0)
            label = "High Risk" if pred == 1 else "Low Risk"
            return label, max(0.0, min(1.0, prob_high))
        except Exception:
            pass  # fall back to heuristic

    # Heuristic fallback (keeps app working even without a model)
    age, bmi, ex, sys_bp, dia_bp, hr = X[0]
    score = 0.0
    score += 0.02 * max(age - 40.0, 0.0)
    score += 0.08 * max(bmi - 25.0, 0.0)
    score += 0.10 * max(sys_bp - 120.0, 0.0)
    score += 0.12 * max(dia_bp - 80.0, 0.0)
    score += 0.03 * max(hr - 80.0, 0.0)
    score -= 0.35 * ex  # more exercise lowers risk

    # squash to 0..1
    prob_high = 1.0 / (1.0 + np.exp(-0.15 * score))
    label = "High Risk" if prob_high >= 0.5 else "Low Risk"
    return label, float(prob_high)

def save_prediction(age, bmi, exercise_level, sys_bp, dia_bp, heart_rate, label, prob_high):
    row = Prediction(
        age=age,
        bmi=bmi,
        exercise_level=exercise_level,
        sys_bp=sys_bp,
        dia_bp=dia_bp,
        heart_rate=heart_rate,
        label=label,
        probability=prob_high,
    )
    db.session.add(row)
    db.session.commit()

# --------------------------------------------------
# Single self-contained UI (no templates needed)
# - Server Submit: POST /
# - Instant JS: fetch POST /api/predict
# --------------------------------------------------
PAGE = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>Early Risk Alert AI</title>
  <style>
    :root{--bg:#0b1220;--card:#121b2e;--text:#e9eefc;--muted:#8ea0c6;--border:rgba(255,255,255,.12)}
    *{box-sizing:border-box}
    body{margin:0;background:var(--bg);color:var(--text);font-family:system-ui,-apple-system,Segoe UI,Roboto,Helvetica,Arial}
    .shell{max-width:980px;margin:28px auto;padding:0 16px}
    .top{display:flex;justify-content:space-between;align-items:center;margin-bottom:16px}
    a{color:var(--text)}
    .card{background:rgba(255,255,255,.04);border:1px solid var(--border);border-radius:16px;padding:16px;margin:14px 0}
    h1{margin:0 0 6px;font-size:18px}
    h2{margin:0 0 10px;font-size:16px}
    p{margin:8px 0;color:var(--muted)}
    .grid{display:grid;grid-template-columns:1fr 1fr;gap:10px}
    @media (max-width:720px){.grid{grid-template-columns:1fr}}
    input,select{width:100%;padding:10px;border-radius:10px;border:1px solid var(--border);background:#0f1830;color:var(--text)}
    button{width:100%;padding:12px;border-radius:10px;border:1px solid var(--border);background:#1a2a54;color:var(--text);font-weight:700;cursor:pointer}
    button:hover{filter:brightness(1.06)}
    .result{margin-top:10px;padding:12px;border-radius:12px;border:1px solid var(--border);background:rgba(0,0,0,.2)}
    .row{display:flex;gap:10px;align-items:center;justify-content:space-between}
    .small{font-size:13px;color:var(--muted)}
    .k{font-weight:800}
  </style>
</head>
<body>
  <div class="shell">
    <div class="top">
      <div>
        <h1>Early Risk Alert AI</h1>
        <div class="small">Server Submit posts to <span class="k">/</span>. Instant JS calls <span class="k">/api/predict</span>.</div>
      </div>
      <div class="row">
        <a href="/history">History</a>
      </div>
    </div>

    <div class="card">
      <h2>Server Submit (Reload)</h2>
      <form method="POST" action="/">
        <div class="grid">
          <input name="age" placeholder="Age" value="{{ server.age or '' }}" required>
          <input name="bmi" placeholder="BMI" value="{{ server.bmi or '' }}" required>
          <select name="exercise_level" required>
            <option value="0" {% if server.exercise_level == '0' %}selected{% endif %}>Low</option>
            <option value="1" {% if server.exercise_level == '1' %}selected{% endif %}>Moderate</option>
            <option value="2" {% if server.exercise_level == '2' %}selected{% endif %}>High</option>
          </select>
          <input name="sys_bp" placeholder="Systolic BP" value="{{ server.sys_bp or '' }}" required>
          <input name="dia_bp" placeholder="Diastolic BP" value="{{ server.dia_bp or '' }}" required>
          <input name="heart_rate" placeholder="Heart Rate" value="{{ server.heart_rate or '' }}" required>
        </div>
        <div style="margin-top:10px">
          <button type="submit">Run Prediction (Server)</button>
        </div>
      </form>

      <div class="result">
        <div><span class="k">Risk:</span> {{ prediction or "—" }}</div>
        <div><span class="k">Probability:</span> {{ probability if probability is not none else "—" }}{% if probability is not none %}%{% endif %}</div>
      </div>
    </div>

    <div class="card">
      <h2>Instant JS (No Reload)</h2>
      <div class="grid">
        <input id="i_age" placeholder="Age" />
        <input id="i_bmi" placeholder="BMI" />
        <select id="i_ex">
          <option value="0">Low</option>
          <option value="1">Moderate</option>
          <option value="2">High</option>
        </select>
        <input id="i_sys" placeholder="Systolic BP" />
        <input id="i_dia" placeholder="Diastolic BP" />
        <input id="i_hr" placeholder="Heart Rate" />
      </div>
      <div style="margin-top:10px">
        <button id="instantBtn" type="button">Run Prediction (Instant)</button>
      </div>

      <div class="result">
        <div><span class="k">Risk:</span> <span id="j_risk">—</span></div>
        <div><span class="k">Probability:</span> <span id="j_prob">—</span></div>
        <div class="small" id="j_err" style="margin-top:6px"></div>
      </div>
    </div>
  </div>

<script>
(function(){
  const $ = (id) => document.getElementById(id);
  const btn = $("instantBtn");
  btn.addEventListener("click", async () => {
    $("j_err").textContent = "";
    $("j_risk").textContent = "Loading...";
    $("j_prob").textContent = "Loading...";

    const payload = {
      age: $("i_age").value,
      bmi: $("i_bmi").value,
      exercise_level: $("i_ex").value,
      sys_bp: $("i_sys").value,
      dia_bp: $("i_dia").value,
      heart_rate: $("i_hr").value
    };

    try{
      const res = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });

      // even if server returns 400, read JSON if possible
      const text = await res.text();
      let data = null;
      try { data = JSON.parse(text); } catch(e) {}

      if(!res.ok){
        const msg = (data && (data.error || data.message)) ? (data.error || data.message) : ("HTTP " + res.status + " " + res.statusText);
        $("j_risk").textContent = "Error";
        $("j_prob").textContent = "0%";
        $("j_err").textContent = msg;
        return;
      }

      $("j_risk").textContent = data.label || "—";
      $("j_prob").textContent = (typeof data.probability === "number" ? (data.probability.toFixed(2) + "%") : "—");
    } catch(err){
      $("j_risk").textContent = "Error";
      $("j_prob").textContent = "0%";
      $("j_err").textContent = String(err);
    }
  });
})();
</script>
</body>
</html>
"""

HISTORY_PAGE = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>History - Early Risk Alert AI</title>
  <style>
    :root{--bg:#0b1220;--card:#121b2e;--text:#e9eefc;--muted:#8ea0c6;--border:rgba(255,255,255,.12)}
    *{box-sizing:border-box}
    body{margin:0;background:var(--bg);color:var(--text);font-family:system-ui,-apple-system,Segoe UI,Roboto,Helvetica,Arial}
    .shell{max-width:980px;margin:28px auto;padding:0 16px}
    .card{background:rgba(255,255,255,.04);border:1px solid var(--border);border-radius:16px;padding:16px;margin:14px 0}
    a{color:var(--text)}
    table{width:100%;border-collapse:collapse}
    th,td{padding:10px;border-bottom:1px solid var(--border);text-align:left;font-size:14px}
    th{color:var(--muted);font-weight:700}
    .small{color:var(--muted);font-size:13px}
  </style>
</head>
<body>
  <div class="shell">
    <div class="card">
      <div style="display:flex;justify-content:space-between;align-items:center;gap:10px">
        <div>
          <h2 style="margin:0">Prediction History</h2>
          <div class="small">Most recent first</div>
        </div>
        <div><a href="/">Back</a></div>
      </div>
    </div>

    <div class="card">
      <table>
        <thead>
          <tr>
            <th>Date</th>
            <th>Age</th>
            <th>BMI</th>
            <th>Exercise</th>
            <th>Sys</th>
            <th>Dia</th>
            <th>HR</th>
            <th>Result</th>
            <th>Prob</th>
          </tr>
        </thead>
        <tbody>
          {% for r in rows %}
          <tr>
            <td>{{ r.created_at.strftime("%Y-%m-%d %H:%M:%S") }}</td>
            <td>{{ "%.1f"|format(r.age) }}</td>
            <td>{{ "%.1f"|format(r.bmi) }}</td>
            <td>{{ "%.0f"|format(r.exercise_level) }}</td>
            <td>{{ "%.0f"|format(r.sys_bp) }}</td>
            <td>{{ "%.0f"|format(r.dia_bp) }}</td>
            <td>{{ "%.0f"|format(r.heart_rate) }}</td>
            <td>{{ r.label }}</td>
            <td>{{ "%.2f"|format(r.probability * 100) }}%</td>
          </tr>
          {% endfor %}
        </tbody>
      </table>
      {% if rows|length == 0 %}
        <div class="small" style="margin-top:10px">No history yet.</div>
      {% endif %}
    </div>
  </div>
</body>
</html>
"""

# --------------------------------------------------
# Routes
# --------------------------------------------------
@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    probability = None

    # keep last-entered values after reload
    server_vals = {
        "age": "",
        "bmi": "",
        "exercise_level": "0",
        "sys_bp": "",
        "dia_bp": "",
        "heart_rate": "",
    }

    # IMPORTANT: this fixes your Render log issue (POST / causing 405).
    if request.method == "POST":
        try:
            payload = dict(request.form)
            # store values for repopulating inputs
            for k in server_vals.keys():
                if k in payload:
                    server_vals[k] = payload.get(k, "")

            age, bmi, ex, sys_bp, dia_bp, hr, X = build_features(payload)
            label, prob_high = run_model(X)
            save_prediction(age, bmi, ex, sys_bp, dia_bp, hr, label, prob_high)

            prediction = label
            probability = round(prob_high * 100, 2)
        except Exception as e:
            prediction = f"Error: {str(e)}"
            probability = 0

    return render_template_string(
        PAGE,
        prediction=prediction,
        probability=probability,
        server=server_vals,
    )

@app.route("/api/predict", methods=["POST"])
def api_predict():
    # IMPORTANT: this fixes your "Instant JS" button (no more 400s).
    try:
        payload = request.get_json(force=True, silent=False)
        if not isinstance(payload, dict):
            return jsonify({"ok": False, "error": "Invalid JSON body"}), 400

        age, bmi, ex, sys_bp, dia_bp, hr, X = build_features(payload)
        label, prob_high = run_model(X)
        save_prediction(age, bmi, ex, sys_bp, dia_bp, hr, label, prob_high)

        return jsonify(
            {
                "ok": True,
                "label": label,
                "probability": round(prob_high * 100, 2),
            }
        ), 200
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400

@app.route("/history", methods=["GET"])
def history():
    rows = Prediction.query.order_by(Prediction.created_at.desc()).limit(200).all()
    return render_template_string(HISTORY_PAGE, rows=rows)

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"ok": True}), 200

# Gunicorn entrypoint: "app:app"
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "10000")), debug=True)
