let mode = "server"; // "server" or "instant"

function setMode(newMode) {
  mode = newMode;
  const note = document.getElementById("resultNote");
  if (mode === "server") {
    note.textContent = "Server mode: form POST renders HTML result.";
  } else {
    note.textContent = "Instant mode: calls /api/v1/predict (mobile-ready JSON).";
  }
}

document.addEventListener("DOMContentLoaded", () => {
  const form = document.getElementById("riskForm");
  const btnServer = document.getElementById("btnServer");
  const btnInstant = document.getElementById("btnInstant");

  const riskLabel = document.getElementById("riskLabel");
  const riskProb = document.getElementById("riskProb");

  btnServer?.addEventListener("click", () => setMode("server"));
  btnInstant?.addEventListener("click", () => setMode("instant"));

  // default
  setMode("server");

  form?.addEventListener("submit", async (e) => {
    if (mode === "server") return; // let normal POST happen

    // instant mode
    e.preventDefault();

    const fd = new FormData(form);
    const payload = {
      age: Number(fd.get("age")),
      bmi: Number(fd.get("bmi")),
      exercise_level: String(fd.get("exercise_level")),
      systolic_bp: Number(fd.get("systolic_bp")),
      diastolic_bp: Number(fd.get("diastolic_bp")),
      heart_rate: Number(fd.get("heart_rate")),
    };

    try {
      const res = await fetch("/api/v1/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      const data = await res.json();
      if (!res.ok) {
        throw new Error(data?.message || data?.error || "API error");
      }

      riskLabel.textContent = data.risk_label ?? "—";
      const p = data.probability;
      riskProb.textContent =
        typeof p === "number" ? `${(p * 100).toFixed(2)}%` : "—";
    } catch (err) {
      riskLabel.textContent = "Error";
      riskProb.textContent = "—";
      alert(err.message || String(err));
    }
  });
});
