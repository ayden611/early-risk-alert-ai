(function () {
  function ready(fn) {
    if (document.readyState === "loading") {
      document.addEventListener("DOMContentLoaded", fn);
    } else {
      fn();
    }
  }

  function isCommandCenter() {
    const p = window.location.pathname || "";
    return p.includes("command-center") || document.body.innerText.includes("Command Center");
  }

  const snapshots = [
    {
      "ICU-12": {
        score: "8.6/10",
        tier: "Critical",
        driver: "SpO₂ decline",
        trend: "Worsening",
        lead: "~4.6 hrs",
        note: "Review Score changed from a fixed patient-risk percent to a 0–10 prioritization score."
      },
      "TELEMETRY-04": {
        score: "6.4/10",
        tier: "Watch",
        driver: "RR instability",
        trend: "Stable",
        lead: "~2.8 hrs",
        note: "Stable review context; score remains moderate."
      },
      "WARD-21": {
        score: "5.9/10",
        tier: "Watch",
        driver: "BP trend",
        trend: "Stable / Watch",
        lead: "~2.4 hrs",
        note: "Lower-priority review context."
      },
      "STEPDOWN-09": {
        score: "6.8/10",
        tier: "Elevated",
        driver: "RR elevation",
        trend: "Stable / Watch",
        lead: "~3.1 hrs",
        note: "Respiratory-rate trend remains visible for review."
      }
    },
    {
      "ICU-12": {
        score: "7.4/10",
        tier: "Elevated",
        driver: "SpO₂ recovery watch",
        trend: "Stable / Watch",
        lead: "~3.2 hrs",
        note: "ICU-12 is no longer locked as permanent 99% Critical."
      },
      "TELEMETRY-04": {
        score: "8.2/10",
        tier: "Critical",
        driver: "HR instability",
        trend: "Worsening",
        lead: "~3.9 hrs",
        note: "Telemetry case can rise when its review score is highest."
      },
      "WARD-21": {
        score: "6.2/10",
        tier: "Watch",
        driver: "BP trend",
        trend: "Stable",
        lead: "~2.6 hrs",
        note: "Moderate watch-state context."
      },
      "STEPDOWN-09": {
        score: "7.0/10",
        tier: "Elevated",
        driver: "RR elevation",
        trend: "Watch",
        lead: "~3.0 hrs",
        note: "Elevated but not overstated."
      }
    },
    {
      "ICU-12": {
        score: "8.1/10",
        tier: "Critical",
        driver: "SpO₂ decline",
        trend: "Worsening",
        lead: "~4.1 hrs",
        note: "Still high-priority in this snapshot, but no longer static or percentage-based."
      },
      "TELEMETRY-04": {
        score: "7.1/10",
        tier: "Elevated",
        driver: "HR instability",
        trend: "Stable / Watch",
        lead: "~3.4 hrs",
        note: "Visible but not top-ranked."
      },
      "WARD-21": {
        score: "6.6/10",
        tier: "Watch",
        driver: "BP trend",
        trend: "Watch",
        lead: "~2.9 hrs",
        note: "Watch-tier review context."
      },
      "STEPDOWN-09": {
        score: "7.5/10",
        tier: "Elevated",
        driver: "RR elevation",
        trend: "Worsening",
        lead: "~3.5 hrs",
        note: "Stepdown case can rise when respiratory trend worsens."
      }
    }
  ];

  let snapshotIndex = 0;

  function textOf(el) {
    return (el && el.textContent ? el.textContent : "").replace(/\s+/g, " ").trim();
  }

  function removePriorTopInjectedBlocks() {
    ["era-live-unit-rotation-root", "era-demo-review-queue-root"].forEach(function (id) {
      const node = document.getElementById(id);
      if (!node) return;

      const shell =
        node.closest(".era-live-unit-shell") ||
        node.closest(".era-demo-shell") ||
        node.closest("section") ||
        node;

      shell.remove();
    });

    Array.from(document.querySelectorAll("section, div, article")).forEach(function (el) {
      const t = textOf(el).toLowerCase();

      if (
        t.includes("unit view with rotating review context") &&
        t.includes("live monitored review snapshots")
      ) {
        el.remove();
      }

      if (
        t.includes("prioritized patient review queue") &&
        t.includes("metric clarity") &&
        t.includes("rotate demo snapshot")
      ) {
        el.remove();
      }
    });
  }

  function findSmallestCard(patientKey) {
    const lower = patientKey.toLowerCase();
    const matches = Array.from(document.querySelectorAll("section, article, div, li")).filter(function (el) {
      const t = textOf(el).toLowerCase();
      return (
        t.includes(lower) &&
        (
          t.includes("risk") ||
          t.includes("review score") ||
          t.includes("clinical review") ||
          t.includes("current monitored")
        )
      );
    });

    matches.sort(function (a, b) {
      return textOf(a).length - textOf(b).length;
    });

    return matches[0] || null;
  }

  function textNodesUnder(el) {
    const nodes = [];
    const walker = document.createTreeWalker(el, NodeFilter.SHOW_TEXT, null);
    let node;
    while ((node = walker.nextNode())) {
      nodes.push(node);
    }
    return nodes;
  }

  function replaceRiskLabelAndScore(card, data) {
    const nodes = textNodesUnder(card);

    nodes.forEach(function (node) {
      const raw = node.nodeValue || "";
      const clean = raw.trim();

      if (/^risk$/i.test(clean)) {
        node.nodeValue = raw.replace(/risk/i, "Review Score");
        if (node.parentElement) node.parentElement.classList.add("era-risk-label-fixed");
      }

      if (/^(99|98|97|96|95|94|93|92|91|90|89|88|87|86|85|84|83|82|81|80|79|78|77|76|75|74|73|72|71|70|69|68|67|66|65|64|63|62|61|60|59|58|57|56|55|54|53|52|51|50|49|48|47|46|45|44|43|42|41|40|39|38|37)%$/.test(clean)) {
        const parentText = textOf(node.parentElement || card).toLowerCase();

        if (
          parentText.includes("risk") ||
          parentText.includes("review score") ||
          clean === "99%" ||
          clean === "37%"
        ) {
          node.nodeValue = raw.replace(clean, data.score);
          if (node.parentElement) node.parentElement.classList.add("era-score-text-fixed");
        }
      }

      if (/^[0-9]\.[0-9]\s*\/\s*10$/.test(clean)) {
        node.nodeValue = raw.replace(clean, data.score);
        if (node.parentElement) node.parentElement.classList.add("era-score-text-fixed");
      }
    });
  }

  function upsertBadge(card, patientKey, data) {
    let badge = card.querySelector(".era-inline-review-badge");
    if (!badge) {
      badge = document.createElement("div");
      badge.className = "era-inline-review-badge";

      const firstChild = card.firstElementChild;
      if (firstChild) {
        card.insertBefore(badge, firstChild.nextSibling);
      } else {
        card.prepend(badge);
      }
    }

    badge.innerHTML =
      "<strong>Review Score " + data.score + "</strong>" +
      " · " + data.tier +
      " · Driver: " + data.driver +
      " · Trend: " + data.trend +
      " · Lead: " + data.lead +
      "<br><span class='era-mini-muted'>" + data.note + "</span>";
  }

  function updateCards() {
    const current = snapshots[snapshotIndex];

    Object.keys(current).forEach(function (patientKey) {
      const card = findSmallestCard(patientKey);
      if (!card) return;

      replaceRiskLabelAndScore(card, current[patientKey]);
      upsertBadge(card, patientKey, current[patientKey]);
    });

    removeDuplicateMetricBars();
    addPageGuardrail();
  }

  function removeDuplicateMetricBars() {
    const seen = [];
    Array.from(document.querySelectorAll(".era-inline-review-warning")).forEach(function (el, i) {
      if (i > 0) el.remove();
    });

    Array.from(document.querySelectorAll("div, section, p")).forEach(function (el) {
      const t = textOf(el).toLowerCase();
      if (t.includes("metric clarity") && t.includes("review score") && t.includes("0-10")) {
        seen.push(el);
      }
    });

    seen.forEach(function (el, i) {
      if (i > 0 && !el.classList.contains("era-inline-review-warning")) {
        el.style.display = "none";
      }
    });
  }

  function addPageGuardrail() {
    if (document.querySelector(".era-inline-review-warning")) return;

    const anchor =
      document.querySelector("[href*='validation-intelligence']") ||
      document.querySelector("nav") ||
      document.body.firstElementChild;

    const guard = document.createElement("div");
    guard.className = "era-inline-review-warning";
    guard.textContent =
      "Metric clarity: Command Center case cards use Review Score on a 0-10 scale. Aggregate validation percentages remain evidence metrics, not patient-level risk percentages.";

    if (anchor && anchor.parentElement) {
      anchor.parentElement.appendChild(guard);
    }
  }

  function rotateSnapshot() {
    snapshotIndex = (snapshotIndex + 1) % snapshots.length;
    updateCards();
  }

  ready(function () {
    if (!isCommandCenter()) return;

    removePriorTopInjectedBlocks();
    updateCards();

    setTimeout(removePriorTopInjectedBlocks, 500);
    setTimeout(updateCards, 800);
    setTimeout(removePriorTopInjectedBlocks, 1600);
    setTimeout(updateCards, 2000);

    setInterval(rotateSnapshot, 9000);
  });
})();
