(function(){
  function ready(fn){
    if(document.readyState !== "loading"){ fn(); }
    else{ document.addEventListener("DOMContentLoaded", fn); }
  }

  const demoSnapshots = [
    {
      label: "Snapshot A",
      top: {
        rank:"#1", patient:"Case-014", unit:"ICU", tier:"Critical", score:"8.7",
        driver:"SpO2 decline", trend:"Worsening ↑", lead:"~4.1 hrs",
        vitals:{spo2:"90%", hr:"112", bp:"94/58", rr:"24"}
      },
      rows: [
        {rank:"#1", patient:"Case-014", unit:"ICU", tier:"Critical", score:"8.7", driver:"SpO2 decline", trend:"Worsening ↑", lead:"~4.1 hrs", state:"Needs review"},
        {rank:"#2", patient:"Case-021", unit:"Stepdown", tier:"Elevated", score:"8.1", driver:"BP instability", trend:"Worsening ↑", lead:"~3.6 hrs", state:"Acknowledged"},
        {rank:"#3", patient:"Case-009", unit:"Telemetry", tier:"Elevated", score:"7.4", driver:"HR variability", trend:"Stable / Watch", lead:"~3.2 hrs", state:"Assigned"},
        {rank:"#4", patient:"Case-032", unit:"MedSurg", tier:"Watch", score:"6.5", driver:"RR elevation", trend:"Stable →", lead:"—", state:"Monitoring"}
      ]
    },
    {
      label: "Snapshot B",
      top: {
        rank:"#1", patient:"Case-027", unit:"Stepdown", tier:"Elevated", score:"8.3",
        driver:"BP trend", trend:"Worsening ↑", lead:"~3.8 hrs",
        vitals:{spo2:"93%", hr:"105", bp:"90/56", rr:"22"}
      },
      rows: [
        {rank:"#1", patient:"Case-027", unit:"Stepdown", tier:"Elevated", score:"8.3", driver:"BP trend", trend:"Worsening ↑", lead:"~3.8 hrs", state:"Needs review"},
        {rank:"#2", patient:"Case-018", unit:"ICU", tier:"Critical", score:"8.2", driver:"SpO2 decline", trend:"Worsening ↑", lead:"~4.3 hrs", state:"Acknowledged"},
        {rank:"#3", patient:"Case-041", unit:"Telemetry", tier:"Elevated", score:"7.2", driver:"HR instability", trend:"Stable / Watch", lead:"~2.9 hrs", state:"Assigned"},
        {rank:"#4", patient:"Case-006", unit:"MedSurg", tier:"Watch", score:"6.4", driver:"Respiratory rate", trend:"Stable →", lead:"—", state:"Monitoring"}
      ]
    },
    {
      label: "Snapshot C",
      top: {
        rank:"#1", patient:"Case-033", unit:"ICU", tier:"Critical", score:"8.9",
        driver:"Respiratory pattern", trend:"Worsening ↑", lead:"~4.6 hrs",
        vitals:{spo2:"91%", hr:"118", bp:"98/61", rr:"27"}
      },
      rows: [
        {rank:"#1", patient:"Case-033", unit:"ICU", tier:"Critical", score:"8.9", driver:"Respiratory pattern", trend:"Worsening ↑", lead:"~4.6 hrs", state:"Needs review"},
        {rank:"#2", patient:"Case-012", unit:"Stepdown", tier:"Elevated", score:"7.9", driver:"SpO2 / HR pattern", trend:"Worsening ↑", lead:"~3.4 hrs", state:"Acknowledged"},
        {rank:"#3", patient:"Case-020", unit:"Telemetry", tier:"Elevated", score:"7.5", driver:"BP variability", trend:"Stable / Watch", lead:"~3.1 hrs", state:"Assigned"},
        {rank:"#4", patient:"Case-044", unit:"MedSurg", tier:"Watch", score:"6.6", driver:"HR trend", trend:"Stable →", lead:"—", state:"Monitoring"}
      ]
    }
  ];

  function snapshotIndex(){
    const now = new Date();
    const bucket = Math.floor(now.getMinutes() / 2);
    return bucket % demoSnapshots.length;
  }

  function tierClass(tier){
    const t = String(tier || "").toLowerCase();
    if(t.includes("critical")) return "critical";
    if(t.includes("elevated")) return "elevated";
    return "watch";
  }

  function rowHtml(r, isTop){
    const cls = tierClass(r.tier);
    return `
      <tr class="${isTop ? "era-row-top" : ""}" data-era-realistic-row data-tier="${cls}">
        <td>${r.rank}</td>
        <td><strong>${r.patient}</strong><br><span style="opacity:.72">${r.unit}</span></td>
        <td><span class="era-tier-pill era-tier-${cls}">${r.tier}</span></td>
        <td class="era-driver-value">${r.driver}</td>
        <td>${r.trend}</td>
        <td class="era-lead-value">${r.lead}</td>
        <td><strong>${r.score}</strong><br><span style="opacity:.65;font-size:11px;">review score</span></td>
        <td>${r.state}</td>
      </tr>
    `;
  }

  function removeOldDemoBlocks(){
    const oldIds = [
      "era-tool-command-center",
      "era-compact-review-queue",
      "explainability-review-queue",
      "era-realistic-command-center-queue"
    ];
    oldIds.forEach(function(id){
      const el = document.getElementById(id);
      if(el) el.remove();
    });
  }

  function buildQueue(){
    removeOldDemoBlocks();

    const snap = demoSnapshots[snapshotIndex()];
    const top = snap.top;
    const cls = tierClass(top.tier);

    const section = document.createElement("section");
    section.id = "era-realistic-command-center-queue";
    section.className = "era-realistic-queue";

    section.innerHTML = `
      <div class="era-realistic-head">
        <div>
          <span class="era-realistic-kicker">Live-style review queue</span>
          <h2 class="era-realistic-title">Explainable review prioritization</h2>
          <p class="era-realistic-subtitle">
            Compact command-center view showing queue rank, priority tier, primary driver,
            trend direction, review score, and retrospective lead-time context. Demo values rotate in
            controlled snapshots so the display feels realistic without making unsupported clinical claims.
          </p>
        </div>
        <div class="era-demo-snapshot">${snap.label} • synthetic demo fixture</div>
      </div>

      <div class="era-realistic-grid">
        <aside class="era-priority-card ${cls}" aria-label="Highest priority review item">
          <div class="era-priority-rank">
            <span class="era-rank-pill">${top.rank} review item</span>
            <span class="era-rank-pill">${top.patient} • ${top.unit}</span>
          </div>

          <div class="era-score-realistic">${top.score}</div>
          <span class="era-score-caption">Review score, not a diagnosis</span>

          <div class="era-badge-row">
            <span class="era-mini-badge ${cls}">${top.tier}</span>
            <span class="era-mini-badge">${top.driver}</span>
            <span class="era-mini-badge elevated">${top.trend}</span>
            <span class="era-mini-badge lead">${top.lead}</span>
          </div>

          <div class="era-vital-grid">
            <div class="era-vital"><span>SpO2</span><strong>${top.vitals.spo2}</strong></div>
            <div class="era-vital"><span>HR</span><strong>${top.vitals.hr}</strong></div>
            <div class="era-vital"><span>BP</span><strong>${top.vitals.bp}</strong></div>
            <div class="era-vital"><span>RR</span><strong>${top.vitals.rr}</strong></div>
          </div>

          <div class="era-safe-note">
            Synthetic demo fixture only. Values are intentionally plausible and varied;
            they are not patient data and do not direct treatment.
          </div>
        </aside>

        <div class="era-queue-panel">
          <div class="era-queue-panel-head">
            <h3>Top review queue</h3>
            <div class="era-queue-controls">
              <button class="era-queue-filter active" data-era-realistic-filter="all">All</button>
              <button class="era-queue-filter" data-era-realistic-filter="critical">Critical</button>
              <button class="era-queue-filter" data-era-realistic-filter="elevated">Elevated</button>
              <button class="era-queue-filter" data-era-realistic-filter="watch">Watch</button>
            </div>
          </div>

          <table class="era-realistic-table">
            <thead>
              <tr>
                <th>Rank</th>
                <th>Case / unit</th>
                <th>Tier</th>
                <th>Primary driver</th>
                <th>Trend</th>
                <th>Lead context</th>
                <th>Score</th>
                <th>Workflow</th>
              </tr>
            </thead>
            <tbody>
              ${snap.rows.map(function(r, idx){ return rowHtml(r, idx === 0); }).join("")}
            </tbody>
          </table>
        </div>
      </div>

      <div class="era-safe-note">
        Decision support only. Retrospective aggregate evidence only. This display supports review
        prioritization and workflow awareness; it is not intended to diagnose, direct treatment,
        replace clinician judgment, or independently trigger escalation.
      </div>
    `;

    const root = document.querySelector("main") ||
                 document.querySelector(".main-content") ||
                 document.querySelector(".container") ||
                 document.body;

    const header = Array.from(document.querySelectorAll("h1, .hero, .hero-section, header")).find(function(el){
      const t = (el.textContent || "").toLowerCase();
      return t.includes("command center") || t.includes("early risk alert");
    });

    if(header && header.parentNode){
      if(header.nextSibling) header.parentNode.insertBefore(section, header.nextSibling);
      else header.parentNode.appendChild(section);
    } else if(root.firstChild){
      root.insertBefore(section, root.firstChild);
    } else {
      root.appendChild(section);
    }

    attachFilters();
  }

  function attachFilters(){
    Array.from(document.querySelectorAll("[data-era-realistic-filter]")).forEach(function(btn){
      btn.addEventListener("click", function(){
        const filter = btn.getAttribute("data-era-realistic-filter");

        Array.from(document.querySelectorAll("[data-era-realistic-filter]")).forEach(function(b){
          b.classList.remove("active");
        });
        btn.classList.add("active");

        Array.from(document.querySelectorAll("[data-era-realistic-row]")).forEach(function(row){
          const tier = row.getAttribute("data-tier");
          row.style.display = (filter === "all" || tier === filter) ? "" : "none";
        });
      });
    });
  }

  function removeExaggeratedLeafText(){
    Array.from(document.querySelectorAll("body *")).forEach(function(el){
      if(el.children.length) return;
      const t = (el.textContent || "").trim();
      if(t === "99%" || t === "99") {
        el.textContent = "8.7";
      }
    });
  }

  function run(){
    removeExaggeratedLeafText();
    buildQueue();
  }

  ready(function(){
    run();
    setTimeout(run, 300);
    setTimeout(run, 1200);
  });
})();
