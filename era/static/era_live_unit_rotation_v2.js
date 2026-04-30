(function(){
  function ready(fn){
    if(document.readyState === "loading"){ document.addEventListener("DOMContentLoaded", fn); }
    else{ fn(); }
  }

  const snapshots = [
    [
      {
        patient:"ICU-12", unit:"ICU", tier:"Critical", score:8.6, driver:"SpO₂ decline", trend:"Worsening", lead:"~4.6 hrs",
        hr:"122 bpm", spo2:"90%", bp:"98/61", rr:"27", temp:"100.3°F",
        note:"High review score due to oxygenation decline plus worsening trend. Not a patient-risk percentage."
      },
      {
        patient:"TEL-18", unit:"Telemetry", tier:"Elevated", score:8.1, driver:"HR instability", trend:"Worsening", lead:"~3.8 hrs",
        hr:"132 bpm", spo2:"94%", bp:"118/73", rr:"23", temp:"99.2°F",
        note:"Telemetry case rises because HR instability is persistent in this snapshot."
      },
      {
        patient:"SDU-04", unit:"Stepdown", tier:"Elevated", score:7.3, driver:"RR elevation", trend:"Stable / Watch", lead:"~3.0 hrs",
        hr:"104 bpm", spo2:"93%", bp:"126/77", rr:"26", temp:"98.9°F",
        note:"Moderate review priority with respiratory-rate elevation."
      },
      {
        patient:"WARD-09", unit:"Ward", tier:"Watch", score:6.4, driver:"BP trend", trend:"Stable", lead:"~2.4 hrs",
        hr:"96 bpm", spo2:"96%", bp:"106/66", rr:"21", temp:"98.7°F",
        note:"Watch-tier case with lower review score."
      }
    ],
    [
      {
        patient:"ICU-21", unit:"ICU", tier:"Critical", score:8.8, driver:"BP instability", trend:"Worsening", lead:"~4.4 hrs",
        hr:"128 bpm", spo2:"92%", bp:"88/54", rr:"25", temp:"99.8°F",
        note:"ICU-21 moves to the top in this snapshot so ICU-12 is not permanently first."
      },
      {
        patient:"ICU-12", unit:"ICU", tier:"Critical", score:8.2, driver:"SpO₂ decline", trend:"Worsening", lead:"~4.1 hrs",
        hr:"116 bpm", spo2:"91%", bp:"101/63", rr:"26", temp:"99.9°F",
        note:"ICU-12 remains visible but the score and vitals change."
      },
      {
        patient:"TEL-18", unit:"Telemetry", tier:"Elevated", score:7.7, driver:"HR instability", trend:"Watch", lead:"~3.5 hrs",
        hr:"126 bpm", spo2:"95%", bp:"120/75", rr:"22", temp:"98.8°F",
        note:"Still elevated, but below the ICU critical cases."
      },
      {
        patient:"SDU-04", unit:"Stepdown", tier:"Watch", score:6.2, driver:"RR elevation", trend:"Stable", lead:"~2.8 hrs",
        hr:"101 bpm", spo2:"94%", bp:"124/78", rr:"24", temp:"98.6°F",
        note:"Stable monitoring case."
      }
    ],
    [
      {
        patient:"TEL-18", unit:"Telemetry", tier:"Critical", score:8.4, driver:"HR instability", trend:"Worsening", lead:"~3.9 hrs",
        hr:"138 bpm", spo2:"92%", bp:"112/70", rr:"24", temp:"99.4°F",
        note:"A non-ICU case can rise to the top when the review score is highest."
      },
      {
        patient:"ICU-12", unit:"ICU", tier:"Elevated", score:7.4, driver:"SpO₂ recovery watch", trend:"Stable / Watch", lead:"~3.2 hrs",
        hr:"108 bpm", spo2:"93%", bp:"108/66", rr:"23", temp:"99.1°F",
        note:"ICU-12 is not always Critical. This makes the demo more realistic."
      },
      {
        patient:"ICU-21", unit:"ICU", tier:"Elevated", score:7.2, driver:"BP trend", trend:"Stable / Watch", lead:"~3.0 hrs",
        hr:"112 bpm", spo2:"94%", bp:"96/60", rr:"22", temp:"99.0°F",
        note:"Borderline hemodynamic trend under review."
      },
      {
        patient:"WARD-09", unit:"Ward", tier:"Watch", score:5.9, driver:"RR elevation", trend:"Stable", lead:"~2.5 hrs",
        hr:"94 bpm", spo2:"95%", bp:"118/70", rr:"23", temp:"98.4°F",
        note:"Lower-priority watch state."
      }
    ]
  ];

  let idx = 0;
  let selected = null;
  let timer = null;

  function tierClass(tier){
    const t = (tier || "").toLowerCase();
    if(t.includes("critical")) return "era-tier-critical";
    if(t.includes("elevated")) return "era-tier-elevated";
    if(t.includes("watch")) return "era-tier-watch";
    return "era-tier-low";
  }

  function trendClass(trend){
    const t = (trend || "").toLowerCase();
    if(t.includes("worsening")) return "era-trend-worse";
    if(t.includes("watch")) return "era-trend-watch";
    if(t.includes("improv")) return "era-trend-improve";
    return "era-trend-stable";
  }

  function rows(){
    const r = (snapshots[idx] || []).slice().sort((a,b) => b.score - a.score);
    r.forEach((x,i)=>x.rank=i+1);
    return r;
  }

  function hideOldStaticLiveUnitCards(){
    const root = document.getElementById("era-live-unit-rotation-root");
    if(!root) return;

    const candidates = Array.from(document.querySelectorAll("section, article, div")).filter(el => {
      if(el === root || root.contains(el) || el.contains(root)) return false;
      const text = (el.textContent || "").replace(/\s+/g, " ").trim().toLowerCase();
      return (
        text.includes("clinical review attention surfaced") &&
        text.includes("current monitored review summary") &&
        (text.includes("risk 99%") || text.includes("risk") || text.includes("icu-12"))
      );
    });

    candidates.sort((a,b) => (a.textContent || "").length - (b.textContent || "").length);

    if(candidates.length){
      const target = candidates[candidates.length - 1];
      target.style.display = "none";
      target.setAttribute("data-era-hidden-static-live-units", "true");
    }

    Array.from(document.querySelectorAll("section, article, div")).forEach(el => {
      if(el === root || root.contains(el) || el.contains(root)) return;
      const text = (el.textContent || "").replace(/\s+/g, " ").trim().toLowerCase();
      if(text.includes("icu-12") && text.includes("risk 99%")){
        el.style.display = "none";
        el.setAttribute("data-era-hidden-static-risk-card", "true");
      }
    });
  }

  function removeDuplicateMetricClarityBars(){
    const bars = Array.from(document.querySelectorAll(".era-metric-guardrail-box, .era-command-score-guardrail")).filter(el => {
      return (el.textContent || "").toLowerCase().includes("metric clarity");
    });
    bars.forEach((el, i) => {
      if(i > 0) el.style.display = "none";
    });
  }

  function render(){
    const root = document.getElementById("era-live-unit-rotation-root");
    if(!root) return;

    hideOldStaticLiveUnitCards();
    removeDuplicateMetricClarityBars();

    const data = rows();
    const chosen = data.find(x => x.patient === selected) || data[0];
    selected = chosen ? chosen.patient : null;

    root.innerHTML = `
      <div class="era-live-unit-shell">
        <div class="era-live-unit-head">
          <div>
            <span class="era-live-kicker">Live monitored review snapshots</span>
            <h2 class="era-live-unit-title">Unit view with rotating review context</h2>
            <p class="era-live-unit-sub">
              The visible patient cards now rotate through controlled synthetic snapshots so ICU-12 is not frozen at one value.
              All case-level values use a 0-10 Review Score. Aggregate validation percentages stay separate on evidence pages.
            </p>
          </div>
          <div class="era-live-controls">
            <button class="era-live-btn" id="era-live-prev">Previous</button>
            <button class="era-live-btn" id="era-live-next">Rotate snapshot</button>
          </div>
        </div>

        <div class="era-live-note">
          <strong>Metric clarity:</strong> Review Score is a case-level queue prioritization value on a 0-10 scale.
          It is not a patient-risk percentage. No card should show ICU-12 as a fixed 99% risk.
        </div>

        <div class="era-live-grid">
          ${data.map(item => `
            <div class="era-live-card ${item.tier.toLowerCase().includes("critical") ? "critical" : ""}" data-patient="${item.patient}">
              <div class="era-live-card-top">
                <div>
                  <div class="era-live-patient">#${item.rank} ${item.patient}</div>
                  <div class="era-live-unit">${item.unit}</div>
                </div>
                <span class="era-tier-chip ${tierClass(item.tier)}">${item.tier}</span>
              </div>

              <div class="era-mini-wave"></div>

              <div class="era-score-big">
                <strong>${item.score.toFixed(1)}</strong><span>/10</span>
              </div>
              <div class="era-score-label">Review Score</div>

              <div class="era-vital-grid">
                <div class="era-vital"><span>HR</span><strong>${item.hr}</strong></div>
                <div class="era-vital"><span>SpO₂</span><strong>${item.spo2}</strong></div>
                <div class="era-vital"><span>BP</span><strong>${item.bp}</strong></div>
                <div class="era-vital"><span>RR</span><strong>${item.rr}</strong></div>
              </div>

              <div class="era-live-meta">
                <div><strong>Driver:</strong> ${item.driver}</div>
                <div><strong>Trend:</strong> <span class="${trendClass(item.trend)}">${item.trend}</span></div>
                <div><strong>Lead time:</strong> ${item.lead}</div>
              </div>
            </div>
          `).join("")}
        </div>

        ${chosen ? `
          <div class="era-live-selected">
            <strong>Selected review basis:</strong> ${chosen.patient} is shown with Review Score ${chosen.score.toFixed(1)}/10.
            Primary driver: ${chosen.driver}. Trend: ${chosen.trend}. Lead-time context: ${chosen.lead}.
            ${chosen.note}
          </div>
        ` : ""}

        <div class="era-live-guardrail">
          <strong>Guardrail:</strong> simulated de-identified demonstration queue. Workflow actions are review-state examples only.
          Decision support only. Not intended to diagnose, direct treatment, replace clinician judgment, or independently trigger escalation.
        </div>
      </div>
    `;

    root.querySelectorAll("[data-patient]").forEach(card => {
      card.addEventListener("click", function(){
        selected = this.getAttribute("data-patient");
        render();
      });
    });

    const next = document.getElementById("era-live-next");
    if(next){
      next.addEventListener("click", function(){
        idx = (idx + 1) % snapshots.length;
        selected = null;
        render();
      });
    }

    const prev = document.getElementById("era-live-prev");
    if(prev){
      prev.addEventListener("click", function(){
        idx = (idx - 1 + snapshots.length) % snapshots.length;
        selected = null;
        render();
      });
    }
  }

  ready(function(){
    render();
    if(timer) clearInterval(timer);
    timer = setInterval(function(){
      idx = (idx + 1) % snapshots.length;
      selected = null;
      render();
    }, 9000);

    setTimeout(hideOldStaticLiveUnitCards, 500);
    setTimeout(hideOldStaticLiveUnitCards, 1500);
    setTimeout(hideOldStaticLiveUnitCards, 3000);
  });
})();
