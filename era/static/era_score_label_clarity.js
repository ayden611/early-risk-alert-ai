(function(){
  function ready(fn){
    if(document.readyState !== "loading"){ fn(); }
    else{ document.addEventListener("DOMContentLoaded", fn); }
  }

  function textOf(el){
    return (el && el.textContent ? el.textContent : "").trim();
  }

  function walkTextLeaves(root, fn){
    if(!root) return;
    const all = Array.from(root.querySelectorAll("*"));
    all.forEach(function(el){
      if(el.children && el.children.length) return;
      const t = textOf(el);
      if(!t) return;
      fn(el, t);
    });
  }

  function findQueueRoots(){
    const headings = Array.from(document.querySelectorAll("h1,h2,h3,h4,.title,.section-title"));
    const roots = [];

    headings.forEach(function(h){
      const t = textOf(h).toLowerCase();
      if(
        t.includes("prioritized patient review queue") ||
        t.includes("top review queue") ||
        t.includes("hospital command wall") ||
        t.includes("review queue")
      ){
        const root = h.closest("section, article, .card, .panel, div") || h.parentElement;
        if(root && !roots.includes(root)) roots.push(root);
      }
    });

    const explicit = document.getElementById("era-realistic-command-center-queue");
    if(explicit && !roots.includes(explicit)) roots.unshift(explicit);

    return roots;
  }

  function findValidationRoots(){
    const headings = Array.from(document.querySelectorAll("h1,h2,h3,h4,.title,.section-title"));
    const roots = [];

    headings.forEach(function(h){
      const t = textOf(h).toLowerCase();
      if(
        t.includes("validation intelligence") ||
        t.includes("mimic validation") ||
        t.includes("alert reduction") ||
        t.includes("validation evidence") ||
        t.includes("validation run")
      ){
        const root = h.closest("section, article, .card, .panel, div") || h.parentElement;
        if(root && !roots.includes(root)) roots.push(root);
      }
    });

    return roots;
  }

  function addQueueBanner(root){
    if(!root || root.querySelector(".era-metric-clarity-banner")) return;

    const banner = document.createElement("div");
    banner.className = "era-metric-clarity-banner";
    banner.innerHTML =
      "<strong>Metric clarity:</strong> Review scores are case-level queue rankings on a 0–10 scale. " +
      "Validation percentages such as 94.3% are aggregate alert-reduction metrics, not patient risk.";

    const heading = Array.from(root.querySelectorAll("h1,h2,h3,h4")).find(function(h){
      return /review queue|command wall|patient/i.test(textOf(h));
    });

    if(heading && heading.parentNode){
      if(heading.nextSibling) heading.parentNode.insertBefore(banner, heading.nextSibling);
      else heading.parentNode.appendChild(banner);
    } else {
      root.insertBefore(banner, root.firstChild);
    }
  }

  function addValidationBanner(root){
    if(!root || root.querySelector(".era-validation-metric-clarity-banner")) return;

    const banner = document.createElement("div");
    banner.className = "era-metric-clarity-banner era-validation-metric-clarity-banner";
    banner.innerHTML =
      "<strong>Validation metric note:</strong> Alert reduction, FPR, detection, and lead-time are aggregate retrospective evidence metrics. " +
      "They should not be read as individual patient-risk percentages.";

    const heading = Array.from(root.querySelectorAll("h1,h2,h3,h4")).find(function(h){
      return /validation|alert reduction|mimic/i.test(textOf(h));
    });

    if(heading && heading.parentNode){
      if(heading.nextSibling) heading.parentNode.insertBefore(banner, heading.nextSibling);
      else heading.parentNode.appendChild(banner);
    }
  }

  function normalizeQueueLabels(root){
    walkTextLeaves(root, function(el, t){
      let n = t;

      n = n.replace(/^Risk$/i, "Review Score");
      n = n.replace(/^Risk Score$/i, "Review Score");
      n = n.replace(/^Current Risk$/i, "Review Score");
      n = n.replace(/^Patient Risk$/i, "Review Score");
      n = n.replace(/^Risk %$/i, "Review Score");
      n = n.replace(/\bpatient risk score\b/ig, "patient review score");
      n = n.replace(/\brisk score\b/ig, "review score");
      n = n.replace(/\bcurrent risk\b/ig, "review score");

      n = n.replace(/\b([0-9](?:\.[0-9])?)\s*%\b/g, function(match, num){
        const value = parseFloat(num);
        if(value > 0 && value < 10) return num + " / 10";
        return match;
      });

      n = n.replace(/\b([0-9](?:\.[0-9])?)\s*\/\s*10\b/g, function(match){
        return match;
      });

      if(n !== t){
        el.textContent = n;
      }

      const finalText = textOf(el);
      if(/^[0-9](?:\.[0-9])?\s*\/\s*10$/.test(finalText)){
        el.classList.add("era-review-score-chip");
      }
    });
  }

  function normalizeValidationLabels(root){
    walkTextLeaves(root, function(el, t){
      let n = t;

      n = n.replace(/\b([0-9]{2,3}(?:\.[0-9])?)\s*%\s*risk\b/ig, "$1% alert reduction");
      n = n.replace(/^Risk$/i, "Alert Reduction");
      n = n.replace(/^Risk Reduction$/i, "Alert Reduction");
      n = n.replace(/^Patient Risk$/i, "Aggregate Metric");

      if(n !== t){
        el.textContent = n;
      }

      const finalText = textOf(el);
      if(/^[0-9]{2,3}(?:\.[0-9])?%$/.test(finalText)){
        const nearby = textOf(el.closest("section,article,div,td,li"));
        if(/alert reduction|validation|mimic|eicu|fpr|detection|lead/i.test(nearby)){
          el.classList.add("era-validation-metric-chip");
        }
      }
    });
  }

  function removeMisleadingRiskNearPercent(){
    Array.from(document.querySelectorAll("body *")).forEach(function(el){
      if(el.children && el.children.length) return;

      const t = textOf(el);
      if(!t) return;

      if(/\b9[0-9](?:\.[0-9])?%\s*risk\b/i.test(t)){
        el.textContent = t.replace(/\brisk\b/ig, "alert reduction");
      }
    });
  }

  function run(){
    const queueRoots = findQueueRoots();
    queueRoots.forEach(function(root){
      addQueueBanner(root);
      normalizeQueueLabels(root);
    });

    const validationRoots = findValidationRoots();
    validationRoots.forEach(function(root){
      addValidationBanner(root);
      normalizeValidationLabels(root);
    });

    removeMisleadingRiskNearPercent();
  }

  ready(function(){
    run();
    setTimeout(run, 300);
    setTimeout(run, 1200);
    setTimeout(run, 2500);
  });
})();
