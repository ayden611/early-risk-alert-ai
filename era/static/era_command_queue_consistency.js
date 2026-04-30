(function(){
  let locked = false;
  let manualSnapshot = null;

  const snapshots = [
    {
      label:"Snapshot A",
      summary:"Oxygenation-driven queue",
      rows:[
        {rank:"#1", patient:"ICU-12", unit:"ICU", tier:"Critical", score:8.7, driver:"SpO2 decline", trend:"Worsening", lead:"~4.8 hrs", workflow:"Needs review"},
        {rank:"#2", patient:"ICU-07", unit:"ICU", tier:"Critical", score:8.3, driver:"BP instability", trend:"Worsening", lead:"~4.1 hrs", workflow:"Acknowledged"},
        {rank:"#3", patient:"TEL-18", unit:"Telemetry", tier:"Elevated", score:7.6, driver:"HR instability", trend:"Stable / Watch", lead:"~3.6 hrs", workflow:"Assigned"},
        {rank:"#4", patient:"SDU-04", unit:"Stepdown", tier:"Watch", score:6.9, driver:"RR elevation", trend:"Stable", lead:"~2.9 hrs", workflow:"Monitoring"}
      ]
    },
    {
      label:"Snapshot B",
      summary:"Blood-pressure instability queue",
      rows:[
        {rank:"#1", patient:"ICU-07", unit:"ICU", tier:"Critical", score:8.5, driver:"BP instability", trend:"Worsening", lead:"~4.2 hrs", workflow:"Needs review"},
        {rank:"#2", patient:"SDU-16", unit:"Stepdown", tier:"Elevated", score:7.9, driver:"SpO2 / RR pattern", trend:"Worsening", lead:"~3.7 hrs", workflow:"Acknowledged"},
        {rank:"#3", patient:"TEL-18", unit:"Telemetry", tier:"Elevated", score:7.4, driver:"HR variability", trend:"Stable / Watch", lead:"~3.1 hrs", workflow:"Assigned"},
        {rank:"#4", patient:"ICU-12", unit:"ICU", tier:"Watch", score:6.8, driver:"Oxygenation improving", trend:"Improving / Watch", lead:"—", workflow:"Monitoring"}
      ]
    },
    {
      label:"Snapshot C",
      summary:"Telemetry instability queue",
      rows:[
        {rank:"#1", patient:"TEL-18", unit:"Telemetry", tier:"Elevated", score:8.1, driver:"HR instability", trend:"Worsening", lead:"~3.8 hrs", workflow:"Needs review"},
        {rank:"#2", patient:"ICU-21", unit:"ICU", tier:"Critical", score:8.0, driver:"SpO2 decline", trend:"Worsening", lead:"~4.4 hrs", workflow:"Acknowledged"},
        {rank:"#3", patient:"SDU-04", unit:"Stepdown", tier:"Elevated", score:7.3, driver:"RR elevation", trend:"Stable / Watch", lead:"~3.0 hrs", workflow:"Assigned"},
        {rank:"#4", patient:"ICU-12", unit:"ICU", tier:"Watch", score:6.7, driver:"SpO2 stable", trend:"Stable", lead:"—", workflow:"Monitoring"}
      ]
    },
    {
      label:"Snapshot D",
      summary:"Stepdown respiratory queue",
      rows:[
        {rank:"#1", patient:"SDU-04", unit:"Stepdown", tier:"Elevated", score:8.0, driver:"RR elevation", trend:"Worsening", lead:"~3.5 hrs", workflow:"Needs review"},
        {rank:"#2", patient:"ICU-12", unit:"ICU", tier:"Elevated", score:7.8, driver:"SpO2 trend", trend:"Stable / Watch", lead:"~3.2 hrs", workflow:"Acknowledged"},
        {rank:"#3", patient:"ICU-07", unit:"ICU", tier:"Watch", score:6.9, driver:"BP stabilized", trend:"Stable", lead:"—", workflow:"Monitoring"},
        {rank:"#4", patient:"TEL-18", unit:"Telemetry", tier:"Watch", score:6.6, driver:"HR trend", trend:"Stable", lead:"—", workflow:"Monitoring"}
      ]
    }
  ];

  function ready(fn){
    if(document.readyState !== "loading"){ fn(); }
    else{ document.addEventListener("DOMContentLoaded", fn); }
  }

  function text(el){
    return (el && el.textContent ? el.textContent : "").trim();
  }

  function isCommandCenter(){
    const path = window.location.pathname || "";
    return path.includes("command-center") || path === "/" || path === "";
  }

  function currentSnapshotIndex(){
    if(manualSnapshot !== null) return manualSnapshot;
    const bucket = Math.floor(Date.now() / 60000);
    return bucket % snapshots.length;
  }

  function tierClass(tier){
    const t = String(tier || "").toLowerCase();
    if(t.includes("critical")) return "critical";
    if(t.includes("elevated")) return "elevated";
    return "watch";
  }

  function findOldQueueRoot(){
    const headings = Array.from(document.querySelectorAll("h1,h2,h3,h4")).filter(function(h){
      const t = text(h).toLowerCase();
      return t.includes("prioritized patient review queue") ||
             t.includes("top review queue") ||
             t.includes("hospital command wall");
    });

    for(const h of headings){
      let node = h;
      for(let i=0; i<8 && node && node !== document.body; i++){
        const t = text(node).toLowerCase();
        const hasTable = !!(node.querySelector && node.querySelector("table"));
        const hasQueue = t.includes("review queue") || t.includes("primary driver") || t.includes("lead time");
        const hasNav = t.includes("hospital demo investor access executive walkthrough");
        if(hasTable && hasQueue && !hasNav){
          return node;
        }
        node = node.parentElement;
      }
    }

    return null;
  }

  function removeDuplicateMetricClarity(root){
    if(!root) return;
    const seen = new Set();
    Array.from(root.querySelectorAll("*")).forEach(function(el){
      const t = text(el).toLowerCase();
      if(t.startsWith("metric clarity:")){
        if(seen.has(t)){
          el.remove();
        } else {
          seen.add(t);
        }
      }
    });
  }

  function normalizeOldScoreText(root){
    if(!root) return;

    Array.from(root.querySelectorAll("*")).forEach(function(el){
      if(el.children && el.children.length) return;
      const t = text(el);
      if(!t) return;

      let n = t;

      n = n.replace(/^Risk$/i, "Review Score");
      n = n.replace(/^Risk Score$/i, "Review Score");
      n = n.replace(/^Risk %$/i, "Review Score");
      n = n.replace(/^Current Risk$/i, "Review Score");
      n = n.replace(/\brisk score\b/ig, "review score");
      n = n.replace(/\bpatient risk\b/ig, "review score");

      n = n.replace(/\b99%\s*risk\b/ig, "8.7 / 10 review score");
      n = n.replace(/\b([0-9]{2})%\b/g, function(match, value){
        const num = parseInt(value, 10);
        if(num >= 50 && num <= 99){
          return (num / 10).toFixed(1) + " / 10";
        }
        return match;
      });

      if(n !== t){
        el.textContent = n;
      }
    });
  }

  function cardHtml(row, index){
    const cls = tierClass(row.tier);
    return `
      <article class="era-cq-card ${index === 0 ? "top" : ""}">
        <div class="era-cq-card-row">
          <div>
            <div class="era-cq-rank">${row.rank} ${row.patient}</div>
            <div class="era-cq-case">${row.unit}</div>
          </div>
          <span class="era-cq-pill era-cq-${cls}">${row.tier}</span>
        </div>
        <span class="era-cq-score">${row.score.toFixed(1)} <small>/ 10</small></span>
        <div class="era-cq-meta">
          <div><strong>Driver:</strong> ${row.driver}</div>
          <div><strong>Trend:</strong> ${row.trend}</div>
          <div><strong>Lead time:</strong> ${row.lead}</div>
        </div>
      </article>
    `;
  }

  function rowHtml(row, index){
    const cls = tierClass(row.tier);
    return `
      <tr class="${index === 0 ? "top-row" : ""}" data-era-cq-row data-tier="${cls}">
        <td>${row.rank}</td>
        <td><strong>${row.patient}</strong></td>
        <td>${row.unit}</td>
        <td><span class="era-cq-pill era-cq-${cls}">${row.tier}</span></td>
        <td class="era-cq-review-score">${row.score.toFixed(1)} / 10</td>
        <td><strong>${row.driver}</strong></td>
        <td>${row.trend}</td>
        <td class="era-cq-lead">${row.lead}</td>
        <td>${row.workflow}</td>
      </tr>
    `;
  }

  function buildQueue(){
    const snap = snapshots[currentSnapshotIndex()];
    const top = snap.rows[0];

    return `
      <div class="era-cq-head">
        <div>
          <span class="era-cq-kicker">Live-style demo queue · 0–10 review score</span>
          <h2 class="era-cq-title">Prioritized patient review queue</h2>
          <p class="era-cq-subtitle">
            Compact command-center queue showing rank, priority tier, primary driver, trend, lead-time context,
            and a single consistent Review Score scale. Values rotate across controlled synthetic snapshots so
            the demo does not appear static or repetitive.
          </p>
        </div>
        <div class="era-cq-snapshot">
          ${snap.label} · ${snap.summary}
        </div>
      </div>

      <div class="era-cq-actions">
        <button class="era-cq-btn active" data-era-cq-filter="all">All</button>
        <button class="era-cq-btn" data-era-cq-filter="critical">Critical</button>
        <button class="era-cq-btn" data-era-cq-filter="elevated">Elevated</button>
        <button class="era-cq-btn" data-era-cq-filter="watch">Watch</button>
        <button class="era-cq-btn" data-era-cq-rotate="1">Rotate demo snapshot</button>
      </div>

      <div class="era-cq-grid">
        ${snap.rows.slice(0,3).map(cardHtml).join("")}
      </div>

      <div class="era-cq-table-wrap">
        <table class="era-cq-table">
          <thead>
            <tr>
              <th>Rank</th>
              <th>Patient</th>
              <th>Unit</th>
              <th>Tier</th>
              <th>Review Score</th>
              <th>Primary Driver</th>
              <th>Trend</th>
              <th>Lead Time</th>
              <th>Workflow</th>
            </tr>
          </thead>
          <tbody>
            ${snap.rows.map(rowHtml).join("")}
          </tbody>
        </table>
      </div>

      <div class="era-cq-note">
        Selected review basis: ${top.patient} is queue-ranked first in this snapshot because the Review Score is highest,
        the primary driver is <strong>${top.driver}</strong>, the trend is <strong>${top.trend}</strong>,
        and the lead-time context is <strong>${top.lead}</strong>.
      </div>

      <div class="era-cq-safe">
        Guardrail: simulated de-identified demonstration queue. Review Score is a 0–10 prioritization score,
        not a patient-risk percentage. Workflow actions are review-state examples only. No diagnosis,
        treatment direction, clinician replacement, or autonomous escalation.
      </div>
    `;
  }

  function attachControls(root){
    Array.from(root.querySelectorAll("[data-era-cq-filter]")).forEach(function(btn){
      btn.addEventListener("click", function(){
        const filter = btn.getAttribute("data-era-cq-filter");
        Array.from(root.querySelectorAll("[data-era-cq-filter]")).forEach(function(b){
          b.classList.remove("active");
        });
        btn.classList.add("active");

        Array.from(root.querySelectorAll("[data-era-cq-row]")).forEach(function(row){
          const tier = row.getAttribute("data-tier");
          row.style.display = (filter === "all" || tier === filter) ? "" : "none";
        });
      });
    });

    const rotate = root.querySelector("[data-era-cq-rotate]");
    if(rotate){
      rotate.addEventListener("click", function(){
        manualSnapshot = (currentSnapshotIndex() + 1) % snapshots.length;
        render(true);
      });
    }
  }

  function cleanupOldInjectedQueues(){
    [
      "era-realistic-command-center-queue",
      "era-tool-command-center",
      "era-compact-review-queue",
      "explainability-review-queue"
    ].forEach(function(id){
      const el = document.getElementById(id);
      if(el) el.remove();
    });
  }

  function render(force){
    if(!isCommandCenter()) return;
    if(locked && !force) return;

    cleanupOldInjectedQueues();

    let root = document.getElementById("era-command-queue-consistency");

    if(!root){
      const oldRoot = findOldQueueRoot();

      if(oldRoot){
        removeDuplicateMetricClarity(oldRoot);
        normalizeOldScoreText(oldRoot);
        oldRoot.classList.add("era-consistent-queue");
        oldRoot.id = "era-command-queue-consistency";
        root = oldRoot;
      } else {
        root = document.createElement("section");
        root.id = "era-command-queue-consistency";
        root.className = "era-consistent-queue";

        const anchor = Array.from(document.querySelectorAll("h1,h2,h3")).find(function(h){
          return text(h).toLowerCase().includes("command center");
        });

        if(anchor && anchor.parentNode){
          if(anchor.nextSibling) anchor.parentNode.insertBefore(root, anchor.nextSibling);
          else anchor.parentNode.appendChild(root);
        } else {
          const main = document.querySelector("main") || document.body;
          main.insertBefore(root, main.firstChild);
        }
      }
    }

    root.className = "era-consistent-queue";
    root.innerHTML = buildQueue();
    attachControls(root);
    locked = true;
  }

  function removeRiskPercentAnywhere(){
    if(!isCommandCenter()) return;

    Array.from(document.querySelectorAll("body *")).forEach(function(el){
      if(el.children && el.children.length) return;
      const t = text(el);
      if(!t) return;

      let n = t;
      n = n.replace(/\b99%\s*risk\b/ig, "8.7 / 10 review score");
      n = n.replace(/\b94%\b/g, "8.4 / 10");
      n = n.replace(/\b86%\b/g, "7.6 / 10");
      n = n.replace(/\b72%\b/g, "6.9 / 10");

      if(n !== t) el.textContent = n;
    });
  }

  ready(function(){
    render(false);
    removeRiskPercentAnywhere();

    setTimeout(function(){ render(true); removeRiskPercentAnywhere(); }, 400);
    setTimeout(function(){ render(true); removeRiskPercentAnywhere(); }, 1200);
    setTimeout(function(){ render(true); removeRiskPercentAnywhere(); }, 2600);

    setInterval(function(){
      manualSnapshot = (currentSnapshotIndex() + 1) % snapshots.length;
      render(true);
    }, 60000);
  });
})();
