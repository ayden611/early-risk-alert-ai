(function () {
  const SNAPSHOTS = [
    [
      {
        rank: 1,
        patient: "ICU-12",
        unit: "ICU",
        tier: "Critical",
        score: 8.6,
        driver: "SpO₂ decline",
        trend: "Worsening",
        lead: "~4.6 hrs",
        firstCross: "t-4.6h",
        workflow: "Needs review",
        vitals: { HR: "127 bpm", SpO2: "88%", BP: "168/101", RR: "29/min", Temp: "100.6°F" },
        rationale: "Oxygenation is the main driver, with worsening trend and lead-time context.",
        timeline: ["SpO₂ trend moved below review threshold.", "Score rose into Critical tier.", "Queue rank increased because trend is worsening."]
      },
      {
        rank: 2,
        patient: "TELEMETRY-04",
        unit: "Telemetry",
        tier: "Watch",
        score: 6.2,
        driver: "RR variability",
        trend: "Stable",
        lead: "~2.8 hrs",
        firstCross: "t-2.8h",
        workflow: "Monitoring",
        vitals: { HR: "109 bpm", SpO2: "93%", BP: "151/91", RR: "24/min", Temp: "99.4°F" },
        rationale: "Respiratory-rate variation remains visible but not top priority.",
        timeline: ["RR trend entered Watch range.", "No worsening acceleration detected.", "Maintained monitoring state."]
      },
      {
        rank: 3,
        patient: "STEPDOWN-09",
        unit: "Stepdown",
        tier: "Elevated",
        score: 7.4,
        driver: "RR elevation",
        trend: "Stable / Watch",
        lead: "~3.1 hrs",
        firstCross: "t-3.1h",
        workflow: "Assigned",
        vitals: { HR: "101 bpm", SpO2: "92%", BP: "142/88", RR: "27/min", Temp: "99.8°F" },
        rationale: "Respiratory trend supports elevated review priority.",
        timeline: ["RR increased above baseline.", "Trend stabilized after initial rise.", "Assigned for review workflow."]
      },
      {
        rank: 4,
        patient: "WARD-21",
        unit: "Ward",
        tier: "Watch",
        score: 5.8,
        driver: "BP trend",
        trend: "Stable",
        lead: "~2.2 hrs",
        firstCross: "t-2.2h",
        workflow: "Monitoring",
        vitals: { HR: "96 bpm", SpO2: "95%", BP: "146/86", RR: "21/min", Temp: "99.1°F" },
        rationale: "Blood-pressure trend remains watch-level.",
        timeline: ["BP trend moved into Watch band.", "No critical driver dominant.", "Continue monitoring workflow."]
      }
    ],
    [
      {
        rank: 1,
        patient: "TELEMETRY-04",
        unit: "Telemetry",
        tier: "Critical",
        score: 8.4,
        driver: "HR instability",
        trend: "Worsening",
        lead: "~4.0 hrs",
        firstCross: "t-4.0h",
        workflow: "Needs review",
        vitals: { HR: "132 bpm", SpO2: "94%", BP: "138/84", RR: "23/min", Temp: "99.7°F" },
        rationale: "Heart-rate instability is now the primary review driver.",
        timeline: ["HR instability increased.", "Trend moved to Worsening.", "Telemetry case moved to top of queue."]
      },
      {
        rank: 2,
        patient: "ICU-12",
        unit: "ICU",
        tier: "Elevated",
        score: 7.2,
        driver: "SpO₂ recovery watch",
        trend: "Stable / Watch",
        lead: "~3.2 hrs",
        firstCross: "t-3.2h",
        workflow: "Acknowledged",
        vitals: { HR: "118 bpm", SpO2: "91%", BP: "156/94", RR: "25/min", Temp: "99.9°F" },
        rationale: "ICU-12 is still visible, but no longer fixed as permanent 99% Critical.",
        timeline: ["SpO₂ improved from prior snapshot.", "Score moved from Critical to Elevated.", "Workflow state changed to Acknowledged."]
      },
      {
        rank: 3,
        patient: "STEPDOWN-09",
        unit: "Stepdown",
        tier: "Elevated",
        score: 7.0,
        driver: "RR elevation",
        trend: "Watch",
        lead: "~3.0 hrs",
        firstCross: "t-3.0h",
        workflow: "Assigned",
        vitals: { HR: "104 bpm", SpO2: "92%", BP: "144/88", RR: "26/min", Temp: "99.4°F" },
        rationale: "Respiratory trend remains elevated but not highest priority.",
        timeline: ["RR remains above baseline.", "Trend is watch-level.", "Assigned workflow remains open."]
      },
      {
        rank: 4,
        patient: "WARD-21",
        unit: "Ward",
        tier: "Watch",
        score: 6.1,
        driver: "BP trend",
        trend: "Stable",
        lead: "~2.5 hrs",
        firstCross: "t-2.5h",
        workflow: "Monitoring",
        vitals: { HR: "98 bpm", SpO2: "95%", BP: "149/88", RR: "20/min", Temp: "98.9°F" },
        rationale: "Ward review remains lower priority.",
        timeline: ["BP remains watch-level.", "No worsening trend.", "Monitoring continues."]
      }
    ],
    [
      {
        rank: 1,
        patient: "STEPDOWN-09",
        unit: "Stepdown",
        tier: "Critical",
        score: 8.1,
        driver: "RR elevation",
        trend: "Worsening",
        lead: "~3.8 hrs",
        firstCross: "t-3.8h",
        workflow: "Needs review",
        vitals: { HR: "112 bpm", SpO2: "90%", BP: "146/90", RR: "31/min", Temp: "100.1°F" },
        rationale: "Respiratory trend worsened, moving Stepdown to the top of the queue.",
        timeline: ["RR increased across the review window.", "Trend moved to Worsening.", "Stepdown case became top queue item."]
      },
      {
        rank: 2,
        patient: "ICU-12",
        unit: "ICU",
        tier: "Critical",
        score: 8.0,
        driver: "SpO₂ decline",
        trend: "Worsening",
        lead: "~4.1 hrs",
        firstCross: "t-4.1h",
        workflow: "Needs review",
        vitals: { HR: "124 bpm", SpO2: "89%", BP: "162/98", RR: "28/min", Temp: "100.3°F" },
        rationale: "ICU-12 remains high priority in this snapshot, but the score varies and is not shown as 99%.",
        timeline: ["SpO₂ decline persisted.", "Lead-time context remains visible.", "Rank changed based on current queue ordering."]
      },
      {
        rank: 3,
        patient: "TELEMETRY-04",
        unit: "Telemetry",
        tier: "Elevated",
        score: 7.3,
        driver: "HR instability",
        trend: "Stable / Watch",
        lead: "~3.4 hrs",
        firstCross: "t-3.4h",
        workflow: "Acknowledged",
        vitals: { HR: "118 bpm", SpO2: "94%", BP: "140/86", RR: "22/min", Temp: "99.2°F" },
        rationale: "HR instability remains visible but not top-ranked.",
        timeline: ["HR trend stabilized.", "Review score decreased.", "Workflow state remains acknowledged."]
      },
      {
        rank: 4,
        patient: "WARD-21",
        unit: "Ward",
        tier: "Watch",
        score: 5.7,
        driver: "BP trend",
        trend: "Stable",
        lead: "~2.3 hrs",
        firstCross: "t-2.3h",
        workflow: "Monitoring",
        vitals: { HR: "94 bpm", SpO2: "96%", BP: "142/84", RR: "19/min", Temp: "98.8°F" },
        rationale: "Stable lower-priority review context.",
        timeline: ["BP trend remains monitored.", "No high-acuity driver dominant.", "Monitoring state remains appropriate."]
      }
    ]
  ];

  let snapshotIndex = 0;
  let selectedPatient = null;
  let filterTier = "All";
  let sortMode = "rank";
  let paused = false;
  let intervalId = null;

  function ready(fn) {
    if (document.readyState === "loading") {
      document.addEventListener("DOMContentLoaded", fn);
    } else {
      fn();
    }
  }

  function isCommandCenter() {
    const path = window.location.pathname || "";
    const body = document.body ? document.body.innerText || "" : "";
    return path.includes("command-center") || body.includes("Command Center") || body.includes("Hospital Command Wall");
  }

  function tierClass(tier) {
    const t = String(tier || "").toLowerCase();
    if (t.includes("critical")) return "era-tier-critical";
    if (t.includes("elevated")) return "era-tier-elevated";
    if (t.includes("watch")) return "era-tier-watch";
    return "era-tier-low";
  }

  function currentSnapshot() {
    let rows = SNAPSHOTS[snapshotIndex].map(x => Object.assign({}, x));

    if (filterTier !== "All") {
      rows = rows.filter(r => r.tier === filterTier);
    }

    if (sortMode === "score") {
      rows.sort((a, b) => b.score - a.score);
    } else if (sortMode === "lead") {
      rows.sort((a, b) => leadNumber(b.lead) - leadNumber(a.lead));
    } else {
      rows.sort((a, b) => a.rank - b.rank);
    }

    return rows.map((r, i) => Object.assign({}, r, { viewRank: i + 1 }));
  }

  function leadNumber(v) {
    const m = String(v || "").match(/([0-9]+(\.[0-9]+)?)/);
    return m ? Number(m[1]) : 0;
  }

  function selectedRow() {
    const rows = currentSnapshot();
    if (selectedPatient) {
      const found = rows.find(r => r.patient === selectedPatient);
      if (found) return found;
    }
    return rows[0] || SNAPSHOTS[snapshotIndex][0];
  }

  function removeOldInjectedTopBlocks() {
    ["era-live-unit-rotation-root", "era-demo-review-queue-root"].forEach(id => {
      const node = document.getElementById(id);
      if (node) {
        const parent = node.closest("section") || node.closest("div") || node;
        parent.remove();
      }
    });

    Array.from(document.querySelectorAll("section, div, article")).forEach(el => {
      const t = (el.textContent || "").replace(/\s+/g, " ").trim().toLowerCase();
      if (
        t.includes("unit view with rotating review context") &&
        t.includes("live monitored review snapshots")
      ) {
        el.remove();
      }
    });
  }

  function findInsertionPoint() {
    const nav = document.querySelector("nav");
    if (nav && nav.parentElement) return nav;

    const brand = Array.from(document.querySelectorAll("h1, h2, div, header")).find(el => {
      const t = (el.textContent || "").replace(/\s+/g, " ").trim();
      return t.includes("Early Risk Alert AI") || t.includes("Command Center");
    });

    if (brand && brand.parentElement) return brand;

    return document.body.firstElementChild || document.body;
  }

  function ensureWorkboard() {
    let root = document.getElementById("era-command-tool-mode");
    if (root) return root;

    root = document.createElement("section");
    root.id = "era-command-tool-mode";

    const insertAfter = findInsertionPoint();
    if (insertAfter && insertAfter.parentElement) {
      insertAfter.parentElement.insertBefore(root, insertAfter.nextSibling);
    } else {
      document.body.prepend(root);
    }

    return root;
  }

  function renderWorkboard() {
    const root = ensureWorkboard();
    const rows = currentSnapshot();
    const sel = selectedRow();
    const top = rows[0] || sel;
    const criticalCount = rows.filter(r => r.tier === "Critical").length;
    const medianLead = median(rows.map(r => leadNumber(r.lead)));

    root.innerHTML = `
      <div class="era-cc-topbar">
        <div>
          <div class="era-cc-kicker">Live review workboard · demo mode</div>
          <h2 class="era-cc-title">Prioritized patient review queue</h2>
          <p class="era-cc-subtitle">
            Compact command-center view showing queue rank, priority tier, primary driver, trend, lead-time context,
            and workflow state. Review Score is a 0–10 prioritization score, not a patient-risk percentage.
          </p>
        </div>
        <div class="era-cc-controls">
          ${["All", "Critical", "Elevated", "Watch"].map(t => `
            <button type="button" class="era-cc-btn ${filterTier === t ? "active" : ""}" data-era-filter="${t}">${t}</button>
          `).join("")}
          <select class="era-cc-select" id="era-cc-sort">
            <option value="rank" ${sortMode === "rank" ? "selected" : ""}>Sort: Queue Rank</option>
            <option value="score" ${sortMode === "score" ? "selected" : ""}>Sort: Review Score</option>
            <option value="lead" ${sortMode === "lead" ? "selected" : ""}>Sort: Lead Time</option>
          </select>
          <button type="button" class="era-cc-btn" id="era-cc-rotate">Rotate snapshot</button>
          <button type="button" class="era-cc-btn" id="era-cc-pause">${paused ? "Resume" : "Pause"}</button>
        </div>
      </div>

      <div class="era-cc-metrics">
        <div class="era-cc-metric">
          <div class="era-cc-metric-label">Open review items</div>
          <div class="era-cc-metric-value">${rows.length}</div>
          <div class="era-cc-metric-note">Within current demo scope</div>
        </div>
        <div class="era-cc-metric">
          <div class="era-cc-metric-label">Critical tier</div>
          <div class="era-cc-metric-value">${criticalCount}</div>
          <div class="era-cc-metric-note">Tier shown visually, not buried in text</div>
        </div>
        <div class="era-cc-metric">
          <div class="era-cc-metric-label">Top review driver</div>
          <div class="era-cc-metric-value" style="font-size:22px">${escapeHtml(top.driver)}</div>
          <div class="era-cc-metric-note">${escapeHtml(top.patient)} · ${escapeHtml(top.trend)}</div>
        </div>
        <div class="era-cc-metric">
          <div class="era-cc-metric-label">Lead-time context</div>
          <div class="era-cc-metric-value">${medianLead.toFixed(1)}h</div>
          <div class="era-cc-metric-note">Median across visible queue</div>
        </div>
      </div>

      <div class="era-cc-body">
        <div class="era-cc-panel">
          <div class="era-cc-panel-head">
            <h3 class="era-cc-panel-title">Queue snapshot</h3>
            <div class="era-cc-timestamp">Snapshot ${snapshotIndex + 1}/${SNAPSHOTS.length} · ${new Date().toLocaleTimeString([], {hour: "2-digit", minute: "2-digit"})}</div>
          </div>

          <div class="era-cc-cards">
            ${rows.slice(0, 4).map(r => cardHtml(r, sel.patient)).join("")}
          </div>

          <div class="era-cc-table-wrap">
            <table class="era-cc-table">
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
                ${rows.map(r => `
                  <tr data-era-patient="${escapeAttr(r.patient)}">
                    <td>#${r.viewRank}</td>
                    <td><strong>${escapeHtml(r.patient)}</strong></td>
                    <td>${escapeHtml(r.unit)}</td>
                    <td><span class="era-tier ${tierClass(r.tier)}">${escapeHtml(r.tier)}</span></td>
                    <td class="era-cc-table-score">${r.score.toFixed(1)}/10</td>
                    <td>${escapeHtml(r.driver)}</td>
                    <td>${escapeHtml(r.trend)}</td>
                    <td>${escapeHtml(r.lead)}</td>
                    <td>${escapeHtml(r.workflow)}</td>
                  </tr>
                `).join("")}
              </tbody>
            </table>
          </div>
        </div>

        <div class="era-cc-panel">
          <div class="era-cc-panel-head">
            <h3 class="era-cc-panel-title">Selected review context</h3>
            <span class="era-tier ${tierClass(sel.tier)}">${escapeHtml(sel.tier)}</span>
          </div>
          ${detailHtml(sel)}
          <div class="era-cc-feed">
            <div class="era-cc-feed-item"><strong>Live review feed:</strong> ${escapeHtml(rows[0]?.patient || sel.patient)} moved to queue rank #1 based on review score, driver, trend, and lead-time context.</div>
            <div class="era-cc-feed-item"><strong>Metric clarity:</strong> case-level queue values use Review Score 0–10. Aggregate validation percentages remain separate evidence metrics.</div>
          </div>
        </div>
      </div>

      <div class="era-cc-guardrail">
        Guardrail: simulated de-identified demonstration queue. Decision support only. Review Score supports prioritization context and does not diagnose, direct treatment, replace clinician judgment, or independently trigger escalation.
      </div>
    `;

    bindEvents();
  }

  function cardHtml(r, selectedPatientId) {
    const selected = r.patient === selectedPatientId ? "selected" : "";
    return `
      <article class="era-cc-card ${selected}" data-era-patient="${escapeAttr(r.patient)}">
        <div class="era-cc-card-top">
          <div>
            <div class="era-cc-rank">#${r.viewRank}</div>
            <div class="era-cc-patient">${escapeHtml(r.patient)}</div>
            <div class="era-cc-unit">${escapeHtml(r.unit)}</div>
          </div>
          <span class="era-tier ${tierClass(r.tier)}">${escapeHtml(r.tier)}</span>
        </div>
        <div class="era-cc-score-row">
          <span class="era-cc-score">${r.score.toFixed(1)}</span>
          <span class="era-cc-score-scale">/10</span>
        </div>
        <div class="era-cc-driver-grid">
          <div class="era-cc-mini"><span>Driver:</span> ${escapeHtml(r.driver)}</div>
          <div class="era-cc-mini"><span>Trend:</span> ${escapeHtml(r.trend)}</div>
          <div class="era-cc-mini"><span>Lead:</span> ${escapeHtml(r.lead)}</div>
        </div>
      </article>
    `;
  }

  function detailHtml(r) {
    return `
      <div class="era-cc-detail">
        <div class="era-cc-detail-title">
          <div>
            <h3>${escapeHtml(r.patient)}</h3>
            <div class="era-cc-detail-sub">${escapeHtml(r.unit)} · Queue rank #${r.viewRank || r.rank} · First threshold context ${escapeHtml(r.firstCross)}</div>
          </div>
          <div class="era-cc-score-row" style="margin:0">
            <span class="era-cc-score">${r.score.toFixed(1)}</span>
            <span class="era-cc-score-scale">/10</span>
          </div>
        </div>

        <div class="era-cc-vitals">
          ${Object.entries(r.vitals).map(([k, v]) => `
            <div class="era-cc-vital">
              <div class="era-cc-vital-label">${escapeHtml(k)}</div>
              <div class="era-cc-vital-value">${escapeHtml(v)}</div>
            </div>
          `).join("")}
        </div>

        <div class="era-cc-mini"><span>Primary driver:</span> ${escapeHtml(r.driver)}</div>
        <div class="era-cc-mini"><span>Trend:</span> ${escapeHtml(r.trend)}</div>
        <div class="era-cc-mini"><span>Lead-time context:</span> ${escapeHtml(r.lead)}</div>
        <div class="era-cc-mini" style="margin-top:7px"><span>Review basis:</span> ${escapeHtml(r.rationale)}</div>

        <div class="era-cc-timeline">
          ${r.timeline.map(item => `<div class="era-cc-timeline-item"><strong>•</strong> ${escapeHtml(item)}</div>`).join("")}
        </div>

        <div class="era-cc-actions">
          <button type="button" class="era-cc-action">Acknowledge</button>
          <button type="button" class="era-cc-action">Assign</button>
          <button type="button" class="era-cc-action">Escalate</button>
          <button type="button" class="era-cc-action">Resolve</button>
        </div>
      </div>
    `;
  }

  function bindEvents() {
    document.querySelectorAll("[data-era-filter]").forEach(btn => {
      btn.addEventListener("click", () => {
        filterTier = btn.getAttribute("data-era-filter") || "All";
        selectedPatient = null;
        renderWorkboard();
      });
    });

    const sort = document.getElementById("era-cc-sort");
    if (sort) {
      sort.addEventListener("change", () => {
        sortMode = sort.value || "rank";
        selectedPatient = null;
        renderWorkboard();
      });
    }

    const rotate = document.getElementById("era-cc-rotate");
    if (rotate) {
      rotate.addEventListener("click", () => {
        nextSnapshot();
      });
    }

    const pause = document.getElementById("era-cc-pause");
    if (pause) {
      pause.addEventListener("click", () => {
        paused = !paused;
        renderWorkboard();
        setupInterval();
      });
    }

    document.querySelectorAll("[data-era-patient]").forEach(el => {
      el.addEventListener("click", () => {
        selectedPatient = el.getAttribute("data-era-patient");
        renderWorkboard();
      });
    });
  }

  function setupInterval() {
    if (intervalId) clearInterval(intervalId);
    if (!paused) {
      intervalId = setInterval(() => nextSnapshot(), 11000);
    }
  }

  function nextSnapshot() {
    snapshotIndex = (snapshotIndex + 1) % SNAPSHOTS.length;
    selectedPatient = null;
    renderWorkboard();
    normalizeExistingCommandCenter();
  }

  function median(values) {
    const nums = values.filter(v => !Number.isNaN(v)).sort((a, b) => a - b);
    if (!nums.length) return 0;
    const mid = Math.floor(nums.length / 2);
    return nums.length % 2 ? nums[mid] : (nums[mid - 1] + nums[mid]) / 2;
  }

  function normalizeExistingCommandCenter() {
    const map = {};
    SNAPSHOTS[snapshotIndex].forEach(r => {
      map[r.patient.toLowerCase()] = r;
      if (r.patient === "ICU-12") map["p101"] = r;
      if (r.patient === "TELEMETRY-04") map["p102"] = r;
      if (r.patient === "WARD-21") map["p104"] = r;
      if (r.patient === "STEPDOWN-09") map["p103"] = r;
    });

    // Fix old top cards and feed items without touching aggregate validation metrics.
    Array.from(document.querySelectorAll("section, article, div, li, tr")).forEach(el => {
      if (el.closest("#era-command-tool-mode")) return;

      const t = (el.textContent || "").toLowerCase();
      const key = Object.keys(map).find(k => t.includes(k));
      if (!key) return;

      const row = map[key];

      replaceTextNodes(el, row);
      upsertExistingNote(el, row);
    });

    compactLongExplainabilityText();
  }

  function replaceTextNodes(root, row) {
    const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT, null);
    const nodes = [];
    let n;
    while ((n = walker.nextNode())) nodes.push(n);

    nodes.forEach(node => {
      let s = node.nodeValue || "";

      s = s.replace(/\bRisk\b/g, "Review Score");
      s = s.replace(/\breview score\s+(99|98|97|96|95|94|93|92|91|90|89|88|87|86|85|84|83|82|81|80|79|78|77|76|75|74|73|72|71|70|69|68|67|66|65|64|63|62|61|60|59|58|57|56|55|54|53|52|51|50|49|48|47|46|45|44|43|42|41|40|39|38|37|36|35|34|33|32|31|30|29|28|27|26|25|24|23|22|21|20|19|18|17|16|15|14|13|12|11|10|9)%/gi, "Review score " + row.score.toFixed(1) + "/10");
      s = s.replace(/\b(99|98|97|96|95|94|93|92|91|90|89|88|87|86|85|84|83|82|81|80|79|78|77|76|75|74|73|72|71|70|69|68|67|66|65|64|63|62|61|60|59|58|57|56|55|54|53|52|51|50|49|48|47|46|45|44|43|42|41|40|39|38|37|36|35|34|33|32|31|30|29|28|27|26|25|24|23|22|21|20|19|18|17|16|15|14|13|12|11|10|9)%\b/g, row.score.toFixed(1) + "/10");

      if (node.nodeValue !== s) {
        node.nodeValue = s;
        if (node.parentElement) node.parentElement.classList.add("era-score-fixed");
      }
    });
  }

  function upsertExistingNote(root, row) {
    if (root.querySelector && root.querySelector(".era-cc-existing-note")) return;

    const text = (root.textContent || "").toLowerCase();
    if (!(text.includes("review score") || text.includes("clinical review") || text.includes("current monitored"))) return;

    const note = document.createElement("div");
    note.className = "era-cc-existing-note";
    note.textContent =
      "Current review context: " +
      row.tier +
      " · Review Score " +
      row.score.toFixed(1) +
      "/10 · Driver: " +
      row.driver +
      " · Trend: " +
      row.trend +
      " · Lead: " +
      row.lead +
      ".";

    if (root.firstElementChild) {
      root.insertBefore(note, root.firstElementChild.nextSibling);
    } else {
      root.prepend(note);
    }
  }

  function compactLongExplainabilityText() {
    Array.from(document.querySelectorAll("section, article, div")).forEach(el => {
      if (el.closest("#era-command-tool-mode")) return;

      const t = (el.textContent || "").replace(/\s+/g, " ").trim().toLowerCase();
      if (
        t.includes("explainable review basis") &&
        t.length > 900 &&
        !el.classList.contains("era-long-explainability-compact")
      ) {
        el.classList.add("era-long-explainability-compact");
      }
    });
  }

  function escapeHtml(value) {
    return String(value ?? "")
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;")
      .replaceAll("'", "&#039;");
  }

  function escapeAttr(value) {
    return escapeHtml(value).replaceAll("`", "&#096;");
  }

  ready(function () {
    if (!isCommandCenter()) return;

    removeOldInjectedTopBlocks();
    renderWorkboard();
    normalizeExistingCommandCenter();
    setupInterval();

    setTimeout(() => {
      removeOldInjectedTopBlocks();
      normalizeExistingCommandCenter();
    }, 800);

    setTimeout(() => {
      removeOldInjectedTopBlocks();
      normalizeExistingCommandCenter();
    }, 2000);
  });
})();
