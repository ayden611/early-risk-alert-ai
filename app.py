import os
import datetime as dt

import numpy as np
import joblib

from flask import Flask, request, render_template_string, url_for
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import text

# ----------------------------
# App / DB setup
# ----------------------------
app = Flask(__name__)
app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", "dev")

db_url = os.getenv("DATABASE_URL")
if db_url and db_url.startswith("postgres://"):
    db_url = db_url.replace("postgres://", "postgresql://", 1)
if not db_url:
    db_url = "sqlite:///local.db"

app.config["SQLALCHEMY_DATABASE_URI"] = db_url
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)

# ----------------------------
# Model load
# ----------------------------
MODEL_PATH = "demo_model.pkl"
model = joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None

# ----------------------------
# DB Model
# ----------------------------
class Prediction(db.Model):
    __tablename__ = "prediction"

    id = db.Column(db.Integer, primary_key=True)
    created_at = db.Column(db.DateTime, nullable=False, default=dt.datetime.utcnow)

    age = db.Column(db.Float, nullable=False)
    bmi = db.Column(db.Float, nullable=False)
    exercise_level = db.Column(db.Float, nullable=False)

    # IMPORTANT:
    # Existing DB may have systolic_bp/diastolic_bp (NOT NULL)
    # Your newer code may use sys_bp/dia_bp
    # So we store BOTH to always satisfy the DB.
    systolic_bp = db.Column(db.Float, nullable=True)
    diastolic_bp = db.Column(db.Float, nullable=True)

    sys_bp = db.Column(db.Float, nullable=True)
    dia_bp = db.Column(db.Float, nullable=True)

    heart_rate = db.Column(db.Float, nullable=False)

    label = db.Column(db.String(32), nullable=False)
    probability = db.Column(db.Float, nullable=False)


def _ensure_db_schema():
    """
    Makes sure the table exists and adds any missing columns.
    This fixes BOTH styles:
      - systolic_bp / diastolic_bp
      - sys_bp / dia_bp
    """
    db.create_all()

    alters = [
        # core columns
        "ALTER TABLE prediction ADD COLUMN IF NOT EXISTS created_at TIMESTAMP",
        "ALTER TABLE prediction ADD COLUMN IF NOT EXISTS age DOUBLE PRECISION",
        "ALTER TABLE prediction ADD COLUMN IF NOT EXISTS bmi DOUBLE PRECISION",
        "ALTER TABLE prediction ADD COLUMN IF NOT EXISTS exercise_level DOUBLE PRECISION",
        "ALTER TABLE prediction ADD COLUMN IF NOT EXISTS heart_rate DOUBLE PRECISION",
        "ALTER TABLE prediction ADD COLUMN IF NOT EXISTS label VARCHAR(32)",
        "ALTER TABLE prediction ADD COLUMN IF NOT EXISTS probability DOUBLE PRECISION",

        # BOTH naming styles
        "ALTER TABLE prediction ADD COLUMN IF NOT EXISTS systolic_bp DOUBLE PRECISION",
        "ALTER TABLE prediction ADD COLUMN IF NOT EXISTS diastolic_bp DOUBLE PRECISION",
        "ALTER TABLE prediction ADD COLUMN IF NOT EXISTS sys_bp DOUBLE PRECISION",
        "ALTER TABLE prediction ADD COLUMN IF NOT EXISTS dia_bp DOUBLE PRECISION",
    ]

    try:
        with db.engine.begin() as conn:
            for sql in alters:
                try:
                    conn.execute(text(sql))
                except Exception:
                    # ignore dialect differences (sqlite / older engines)
                    pass
    except Exception:
        pass


with app.app_context():
    _ensure_db_schema()

# ----------------------------
# Helpers
# ----------------------------
def _to_float(v, default=0.0):
    try:
        return float(v)
    except Exception:
        return default


def _parse_exercise(v):
    if v is None:
        return 0.0
    s = str(v).strip().lower()
    if s in ("low", "l"):
        return 0.0
    if s in ("moderate", "med", "m"):
        return 1.0
    if s in ("high", "h"):
        return 2.0
    return _to_float(v, 0.0)


def _predict(form):
    if model is None:
        return "Model not loaded", 0.0

    age = _to_float(form.get("age"))
    bmi = _to_float(form.get("bmi"))
    ex = _parse_exercise(form.get("exercise_level"))

    sys_bp = _to_float(form.get("sys_bp"))
    dia_bp = _to_float(form.get("dia_bp"))
    heart_rate = _to_float(form.get("heart_rate"))

    X = np.array([[age, bmi, ex, sys_bp, dia_bp, heart_rate]], dtype=float)

    pred = int(model.predict(X)[0])
    prob_high = float(model.predict_proba(X)[0][1])

    label = "High Risk" if pred == 1 else "Low Risk"
    return label, prob_high


# ----------------------------
# Inline UI (SERVER SUBMIT ONLY)
# ----------------------------
PAGE = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Early Risk Alert AI</title>
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <style>
    :root{
      --bg:#0b1220; --text:#e9eefc; --muted:#8ea0c6;
      --border:rgba(255,255,255,.12);
    }
    *{box-sizing:border-box}
    body{margin:0;background:var(--bg);color:var(--text);font-family:system-ui,-apple-system,Segoe UI,Roboto,Arial}
    .shell{max-width:980px;margin:28px auto;padding:0 16px}
    a{color:#b9c8ff;text-decoration:none}
    .top{display:flex;align-items:center;justify-content:space-between;margin-bottom:16px}
    .card{background:rgba(255,255,255,.04);border:1px solid var(--border);border-radius:14px;padding:16px}
    .grid{display:grid;grid-template-columns:1fr 1fr;gap:12px}
    label{font-size:13px;color:var(--muted)}
    input,select{width:100%;padding:12px;border-radius:10px;border:1px solid var(--border);background:rgba(255,255,255,.06);color:var(--text)}
    button{width:100%;padding:12px;border-radius:12px;border:1px solid var(--border);background:rgba(255,255,255,.12);color:var(--text);font-weight:800;cursor:pointer}
    .result{margin-top:14px;padding:12px;border-radius:12px;border:1px solid var(--border);background:rgba(0,0,0,.22)}
    .mono{white-space:pre-wrap;font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12px;color:#ffb9b9}
    @media(max-width:720px){.grid{grid-template-columns:1fr}}
  </style>
</head>
<body>
  <div class="shell">
    <div class="top">
      <div>
        <div style="font-size:18px;font-weight:900">Early Risk Alert AI</div>
        <div style="font-size:13px;color:var(--muted)">Server Submit posts to <b>/</b> and saves to Postgres.</div>
      </div>
      <div><a href="{{ url_for('history') }}">History</a></div>
    </div>

    <div class="card">
      <div style="font-size:16px;font-weight:900;margin-bottom:10px">Server Submit (Reload)</div>

      <form method="POST" action="/">
        <div class="grid">
          <div>
            <label>Age</label>
            <input name="age" type="number" step="0.1" required value="{{ form.age }}">
          </div>
          <div>
            <label>BMI</label>
            <input name="bmi" type="number" step="0.1" required value="{{ form.bmi }}">
          </div>
          <div>
            <label>Exercise Level</label>
            <select name="exercise_level">
              <option value="0" {% if form.exercise_level == "0" %}selected{% endif %}>Low</option>
              <option value="1" {% if form.exercise_level == "1" %}selected{% endif %}>Moderate</option>
              <option value="2" {% if form.exercise_level == "2" %}selected{% endif %}>High</option>
            </select>
          </div>
          <div>
            <label>Systolic BP</label>
            <input name="sys_bp" type="number" step="0.1" required value="{{ form.sys_bp }}">
          </div>
          <div>
            <label>Diastolic BP</label>
            <input name="dia_bp" type="number" step="0.1" required value="{{ form.dia_bp }}">
          </div>
          <div>
            <label>Heart Rate</label>
            <input name="heart_rate" type="number" step="0.1" required value="{{ form.heart_rate }}">
          </div>
        </div>

        <div style="margin-top:12px">
          <button type="submit">Run Prediction (Server)</button>
        </div>
      </form>

      <div class="result">
        <div><b>Risk:</b> {{ result.label }}</div>
        <div><b>Probability:</b> {{ result.prob }}%</div>
        {% if result.error %}
          <div class="mono" style="margin-top:10px">{{ result.error }}</div>
        {% endif %}
      </div>
    </div>
  </div>
</body>
</html>
"""

HISTORY_PAGE = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>History</title>
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <style>
    :root{--bg:#0b1220;--text:#e9eefc;--muted:#8ea0c6;--border:rgba(255,255,255,.12)}
    *{box-sizing:border-box}
    body{margin:0;background:var(--bg);color:var(--text);font-family:system-ui,-apple-system,Segoe UI,Roboto,Arial}
    .shell{max-width:980px;margin:28px auto;padding:0 16px}
    a{color:#b9c8ff;text-decoration:none}
    .top{display:flex;align-items:center;justify-content:space-between;margin-bottom:16px}
    .card{background:rgba(255,255,255,.04);border:1px solid var(--border);border-radius:14px;padding:16px}
    table{width:100%;border-collapse:collapse}
    th,td{padding:10px;border-bottom:1px solid var(--border);font-size:14px}
    th{color:var(--muted);text-align:left}
  </style>
</head>
<body>
  <div class="shell">
    <div class="top">
      <div style="font-size:18px;font-weight:900">Prediction History</div>
      <div><a href="{{ url_for('home') }}">Back</a></div>
    </div>

    <div class="card">
      <table>
        <thead>
          <tr>
            <th>Date</th><th>Age</th><th>BMI</th><th>Ex</th><th>Sys</th><th>Dia</th><th>HR</th><th>Result</th><th>Prob</th>
          </tr>
        </thead>
        <tbody>
          {% for r in rows %}
          <tr>
            <td>{{ r.created_at }}</td>
            <td>{{ "%.1f"|format(r.age) }}</td>
            <td>{{ "%.1f"|format(r.bmi) }}</td>
            <td>{{ "%.0f"|format(r.exercise_level) }}</td>
            <td>{{ "%.0f"|format((r.systolic_bp if r.systolic_bp is not none else r.sys_bp) or 0) }}</td>
            <td>{{ "%.0f"|format((r.diastolic_bp if r.diastolic_bp is not none else r.dia_bp) or 0) }}</td>
            <td>{{ "%.0f"|format(r.heart_rate) }}</td>
            <td>{{ r.label }}</td>
            <td>{{ "%.2f"|format(r.probability * 100) }}%</td>
          </tr>
          {% endfor %}
        </tbody>
      </table>
    </div>
  </div>
</body>
</html>
"""

# ----------------------------
# Routes
# ----------------------------
@app.route("/", methods=["GET", "POST"])
def home():
    result = {"label": "—", "prob": "—", "error": ""}
    form = {
        "age": "",
        "bmi": "",
        "exercise_level": "0",
        "sys_bp": "",
        "dia_bp": "",
        "heart_rate": "",
    }

    if request.method == "POST":
        form = {
            "age": request.form.get("age", ""),
            "bmi": request.form.get("bmi", ""),
            "exercise_level": request.form.get("exercise_level", "0"),
            "sys_bp": request.form.get("sys_bp", ""),
            "dia_bp": request.form.get("dia_bp", ""),
            "heart_rate": request.form.get("heart_rate", ""),
        }

        try:
            # make sure schema is good
            _ensure_db_schema()

            label, prob_high = _predict(form)

            sys_bp = _to_float(form["sys_bp"])
            dia_bp = _to_float(form["dia_bp"])

            p = Prediction(
                created_at=dt.datetime.utcnow(),
                age=_to_float(form["age"]),
                bmi=_to_float(form["bmi"]),
                exercise_level=_parse_exercise(form["exercise_level"]),
                heart_rate=_to_float(form["heart_rate"]),
                label=label,
                probability=prob_high,

                # WRITE BOTH NAMING STYLES (this fixes your error)
                systolic_bp=sys_bp,
                diastolic_bp=dia_bp,
                sys_bp=sys_bp,
                dia_bp=dia_bp,
            )

            db.session.add(p)
            db.session.commit()

            result["label"] = label
            result["prob"] = f"{prob_high * 100:.2f}"

        except Exception as e:
            result["label"] = "Error"
            result["prob"] = "0"
            result["error"] = str(e)

    return render_template_string(PAGE, result=result, form=form)


@app.route("/history", methods=["GET"])
def history():
    rows = Prediction.query.order_by(Prediction.created_at.desc()).limit(50).all()
    return render_template_string(HISTORY_PAGE, rows=rows)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")))
