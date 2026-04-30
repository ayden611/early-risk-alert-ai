(function(){
  function ready(fn){
    if(document.readyState !== "loading"){ fn(); }
    else{ document.addEventListener("DOMContentLoaded", fn); }
  }

  function txt(el){
    return (el && el.textContent ? el.textContent : "").trim();
  }

  function path(){
    return window.location.pathname || "";
  }

  function isCommandCenter(){
    return path().includes("command-center") || path() === "/" || path() === "";
  }

  function isValidationPage(){
    return path().includes("validation-intelligence") ||
           path().includes("validation-evidence") ||
           path().includes("validation-runs");
  }

  function walkLeafText(root, fn){
    if(!root) return;
    Array.from(root.querySelectorAll("*")).forEach(function(el){
      if(el.children && el.children.length) return;
      var t = txt(el);
      if(!t) return;
      fn(el, t);
    });
  }

  function bestQueueRoot(){
    var candidates = Array.from(document.querySelectorAll("h1,h2,h3,h4,section,article,div")).filter(function(el){
      var t = txt(el).toLowerCase();
      return t.includes("prioritized patient review queue") ||
             t.includes("top review queue") ||
             t.includes("review queue") ||
             t.includes("hospital command wall");
    });

    var best = null;
    var bestScore = -1;

    candidates.forEach(function(el){
      var node = el;
      for(var i=0; i<7 && node; i++){
        var t = txt(node).toLowerCase();
        var hasTable = node.querySelector && node.querySelector("table");
        var hasQueueWords = t.includes("queue") || t.includes("review");
        var score = 0;
        if(hasQueueWords) score += 3;
        if(hasTable) score += 4;
        if(t.includes("primary driver")) score += 2;
        if(t.includes("lead")) score += 2;
        if(t.includes("workflow")) score += 1;
        if(score > bestScore){
          bestScore = score;
          best = node;
        }
        node = node.parentElement;
      }
    });

    return best;
  }

  function addCommandMetricBanner(root){
    if(!root || root.querySelector(".era-command-score-guardrail")) return;

    var banner = document.createElement("div");
    banner.className = "era-metric-guardrail-box era-command-score-guardrail";
    banner.innerHTML =
      "<strong>Metric clarity:</strong> Queue values use a 0–10 Review Score for prioritization. " +
      "Validation percentages such as 94.3% are aggregate alert-reduction metrics, not patient-risk percentages.";

    var heading = Array.from(root.querySelectorAll("h1,h2,h3,h4")).find(function(h){
      return /queue|command wall|patient review/i.test(txt(h));
    });

    if(heading && heading.parentNode){
      if(heading.nextSibling) heading.parentNode.insertBefore(banner, heading.nextSibling);
      else heading.parentNode.appendChild(banner);
    } else {
      root.insertBefore(banner, root.firstChild);
    }
  }

  function normalizeCommandQueue(root){
    if(!root) return;

    addCommandMetricBanner(root);

    walkLeafText(root, function(el, t){
      var n = t;

      n = n.replace(/^Risk$/i, "Review Score");
      n = n.replace(/^Risk %$/i, "Review Score");
      n = n.replace(/^Risk Score$/i, "Review Score");
      n = n.replace(/^Current Risk$/i, "Review Score");
      n = n.replace(/^Patient Risk$/i, "Review Score");
      n = n.replace(/\bpatient risk score\b/ig, "patient review score");
      n = n.replace(/\brisk score\b/ig, "review score");
      n = n.replace(/\bcurrent risk\b/ig, "review score");

      n = n.replace(/\b([0-9](?:\.[0-9])?)\s*%\s*(risk|review score)?\b/ig, function(match, num){
        return num + " / 10";
      });

      n = n.replace(/\b([0-9](?:\.[0-9])?)\s+risk\b/ig, function(match, num){
        return num + " / 10 review score";
      });

      if(n !== t){
        el.textContent = n;
      }

      var finalText = txt(el);
      if(/^[0-9](?:\.[0-9])?\s*\/\s*10$/.test(finalText)){
        el.classList.add("era-review-score-chip");
      }
    });
  }

  function normalizeMisleadingRiskPercentEverywhere(){
    Array.from(document.querySelectorAll("body *")).forEach(function(el){
      if(el.children && el.children.length) return;
      var t = txt(el);
      if(!t) return;

      var n = t;

      n = n.replace(/\b(9[0-9](?:\.[0-9])?|100)\s*%\s*risk\b/ig, "$1% alert reduction");
      n = n.replace(/\b(9[0-9](?:\.[0-9])?|100)\s*%\s*patient risk\b/ig, "$1% alert reduction");

      if(n !== t){
        el.textContent = n;
      }
    });
  }

  function addValidationMetricBanner(){
    var root = document.querySelector("main") ||
               document.querySelector(".main-content") ||
               document.querySelector(".container") ||
               document.body;

    if(!root || document.getElementById("era-validation-metric-note")) return;

    var banner = document.createElement("div");
    banner.id = "era-validation-metric-note";
    banner.className = "era-metric-guardrail-box";
    banner.innerHTML =
      "<strong>Validation metric note:</strong> Alert reduction, FPR, detection, and lead-time are aggregate retrospective evidence metrics. " +
      "They are not individual patient-risk percentages.";

    var heading = Array.from(document.querySelectorAll("h1,h2")).find(function(h){
      return /validation|evidence|runs/i.test(txt(h));
    });

    if(heading && heading.parentNode){
      if(heading.nextSibling) heading.parentNode.insertBefore(banner, heading.nextSibling);
      else heading.parentNode.appendChild(banner);
    } else if(root.firstChild){
      root.insertBefore(banner, root.firstChild);
    } else {
      root.appendChild(banner);
    }
  }

  function addEicuMetricSourceClarity(){
    if(!isValidationPage()) return;
    if(document.getElementById("era-eicu-metric-source-clarity")) return;

    var root = document.querySelector("main") ||
               document.querySelector(".main-content") ||
               document.querySelector(".container") ||
               document.body;

    var box = document.createElement("section");
    box.id = "era-eicu-metric-source-clarity";
    box.className = "era-eicu-clarity-box";

    box.innerHTML = `
      <span class="era-eicu-clarity-kicker">Metric source clarity</span>
      <h2>eICU results are separated by event-definition track</h2>
      <p>
        The platform currently has two eICU evidence tracks. They should not be merged into one unlabeled number.
        The outcome-proxy check and the harmonized clinical-event pass answer related but different validation questions.
      </p>

      <div class="era-eicu-clarity-grid">
        <div class="era-eicu-track">
          <h3>eICU outcome-proxy check</h3>
          <p>Earlier second-dataset check using mortality/discharge-derived outcome-proxy event context.</p>
          <div class="era-eicu-metrics">
            <div class="era-eicu-metric"><span>t=6 alert reduction</span><strong>96.8%</strong></div>
            <div class="era-eicu-metric"><span>t=6 FPR</span><strong>1.8%</strong></div>
            <div class="era-eicu-metric"><span>t=6 detection</span><strong>66.6%</strong></div>
            <div class="era-eicu-metric"><span>lead-time context</span><strong>3.41 hrs</strong></div>
          </div>
        </div>

        <div class="era-eicu-track">
          <h3>eICU harmonized clinical-event pass</h3>
          <p>Newer pass intended to better align eICU evaluation with the MIMIC clinical-event framework.</p>
          <div class="era-eicu-metrics">
            <div class="era-eicu-metric"><span>t=6 alert reduction</span><strong>94.25%</strong></div>
            <div class="era-eicu-metric"><span>t=6 FPR</span><strong>0.98%</strong></div>
            <div class="era-eicu-metric"><span>t=6 detection</span><strong>24.66%</strong></div>
            <div class="era-eicu-metric"><span>lead-time context</span><strong>4.83 hrs</strong></div>
          </div>
        </div>
      </div>

      <div class="era-eicu-warning">
        Guardrail: detection rates should be interpreted with event-definition context. 
        This is retrospective aggregate evidence only and is not a prospective clinical-performance claim.
      </div>
    `;

    var heading = Array.from(document.querySelectorAll("h1,h2")).find(function(h){
      return /validation|evidence|runs/i.test(txt(h));
    });

    if(heading && heading.parentNode){
      if(heading.nextSibling) heading.parentNode.insertBefore(box, heading.nextSibling);
      else heading.parentNode.appendChild(box);
    } else if(root.firstChild){
      root.insertBefore(box, root.firstChild);
    } else {
      root.appendChild(box);
    }
  }

  function tagValidationPercentages(){
    if(!isValidationPage()) return;

    Array.from(document.querySelectorAll("body *")).forEach(function(el){
      if(el.children && el.children.length) return;
      var t = txt(el);
      if(/^[0-9]{1,3}(?:\.[0-9])?%$/.test(t)){
        var nearby = txt(el.closest("section,article,div,td,li") || el);
        if(/alert reduction|validation|mimic|eicu|fpr|detection|lead/i.test(nearby)){
          el.classList.add("era-validation-metric-chip");
        }
      }
    });
  }

  function run(){
    normalizeMisleadingRiskPercentEverywhere();

    if(isCommandCenter()){
      var root = bestQueueRoot();
      normalizeCommandQueue(root);
    }

    if(isValidationPage()){
      addValidationMetricBanner();
      addEicuMetricSourceClarity();
      tagValidationPercentages();
    }
  }

  ready(function(){
    run();
    setTimeout(run, 300);
    setTimeout(run, 1000);
    setTimeout(run, 2500);
  });
})();
