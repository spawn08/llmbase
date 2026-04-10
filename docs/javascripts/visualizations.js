/**
 * LLMBase Interactive Visualizations
 * Zero-dependency, vanilla JS interactive widgets for the Foundations section.
 * All widgets auto-discover their container elements and self-initialize.
 */

(function () {
  "use strict";

  /* ─────────────────────────────────────────────────────────────
     Utility helpers
     ───────────────────────────────────────────────────────────── */
  const $ = (sel, ctx) => (ctx || document).querySelector(sel);
  const $$ = (sel, ctx) => [...(ctx || document).querySelectorAll(sel)];

  function isDark() {
    const scheme = document.body
      .closest("[data-md-color-scheme]")
      ?.getAttribute("data-md-color-scheme");
    return scheme === "slate";
  }

  function themeColors() {
    const dark = isDark();
    return {
      bg: dark ? "#141A22" : "#F6F7F9",
      fg: dark ? "#E4E8EE" : "#1B1F26",
      accent: dark ? "#6FA3B8" : "#2A6F88",
      accentSoft: dark ? "#8FA9C4" : "#3D7A92",
      grid: dark ? "rgba(255,255,255,0.06)" : "rgba(0,0,0,0.06)",
      barFill: dark ? "rgba(111,163,184,0.7)" : "rgba(42,111,136,0.7)",
      barFillAlt: dark ? "rgba(94,171,139,0.7)" : "rgba(60,140,100,0.7)",
      glass: dark ? "rgba(255,255,255,0.04)" : "rgba(0,0,0,0.02)",
      warn: dark ? "#c4975a" : "#b8860b",
      success: dark ? "#5eab8b" : "#2e8b57",
      danger: dark ? "#c66" : "#c33",
    };
  }

  function sliderHTML(id, label, min, max, step, value, unit) {
    return `<div class="viz-slider-group">
      <label for="${id}">${label}: <strong id="${id}-val">${value}${unit || ""}</strong></label>
      <input type="range" class="viz-slider" id="${id}" min="${min}" max="${max}" step="${step}" value="${value}">
    </div>`;
  }

  function softmax(arr, temperature) {
    temperature = temperature || 1;
    const maxVal = Math.max(...arr);
    const exps = arr.map((v) => Math.exp((v - maxVal) / temperature));
    const sum = exps.reduce((a, b) => a + b, 0);
    return exps.map((e) => e / sum);
  }

  /* ─────────────────────────────────────────────────────────────
     1. Softmax Temperature Visualizer
     ───────────────────────────────────────────────────────────── */
  function initSoftmaxViz(container) {
    const logits = [2.0, 1.0, 0.5, -0.5, -1.0];
    const labels = ["Token A", "Token B", "Token C", "Token D", "Token E"];

    container.innerHTML = `
      <div class="viz-controls">
        ${sliderHTML("softmax-temp", "Temperature (τ)", 0.1, 5, 0.1, 1.0, "")}
      </div>
      <canvas id="softmax-canvas" width="560" height="260"></canvas>
      <div class="viz-output" id="softmax-info">
        <span class="viz-tag">τ = 1.0</span>
        <span class="viz-tag">Max prob: —</span>
        <span class="viz-tag">Entropy: —</span>
      </div>`;

    const canvas = $("#softmax-canvas", container);
    const ctx = canvas.getContext("2d");
    const slider = $("#softmax-temp", container);
    const valSpan = $("#softmax-temp-val", container);
    const infoDiv = $("#softmax-info", container);

    function draw() {
      const t = parseFloat(slider.value);
      valSpan.textContent = t.toFixed(1);
      const probs = softmax(logits, t);
      const maxP = Math.max(...probs);
      const entropy = -probs.reduce(
        (s, p) => s + (p > 1e-12 ? p * Math.log2(p) : 0),
        0
      );

      const C = themeColors();
      const W = canvas.width,
        H = canvas.height;
      ctx.clearRect(0, 0, W, H);

      // Grid
      const pad = { l: 60, r: 20, t: 20, b: 50 };
      const plotW = W - pad.l - pad.r;
      const plotH = H - pad.t - pad.b;
      ctx.strokeStyle = C.grid;
      ctx.lineWidth = 1;
      for (let i = 0; i <= 5; i++) {
        const y = pad.t + (plotH * (5 - i)) / 5;
        ctx.beginPath();
        ctx.moveTo(pad.l, y);
        ctx.lineTo(W - pad.r, y);
        ctx.stroke();
        ctx.fillStyle = C.fg;
        ctx.font = "11px JetBrains Mono, monospace";
        ctx.textAlign = "right";
        ctx.fillText((i * 0.2).toFixed(1), pad.l - 8, y + 4);
      }

      // Bars
      const barW = plotW / probs.length - 12;
      probs.forEach((p, i) => {
        const x = pad.l + (plotW / probs.length) * i + 6;
        const h = p * plotH;
        const y = pad.t + plotH - h;
        // Gradient fill
        const grad = ctx.createLinearGradient(x, y, x, pad.t + plotH);
        grad.addColorStop(0, C.accent);
        grad.addColorStop(1, C.barFill);
        ctx.fillStyle = grad;
        ctx.beginPath();
        ctx.roundRect(x, y, barW, h, [4, 4, 0, 0]);
        ctx.fill();
        // Value on top
        ctx.fillStyle = C.fg;
        ctx.font = "bold 11px JetBrains Mono, monospace";
        ctx.textAlign = "center";
        ctx.fillText(p.toFixed(3), x + barW / 2, y - 6);
        // Label
        ctx.font = "11px Source Sans 3, sans-serif";
        ctx.fillText(labels[i], x + barW / 2, pad.t + plotH + 18);
        ctx.font = "10px JetBrains Mono, monospace";
        ctx.fillStyle = C.accentSoft;
        ctx.fillText(`(${logits[i].toFixed(1)})`, x + barW / 2, pad.t + plotH + 34);
      });

      // Info
      let behavior = "Balanced";
      if (t < 0.3) behavior = "Near one-hot (argmax)";
      else if (t < 0.8) behavior = "Peaked";
      else if (t > 2) behavior = "Near uniform";
      else if (t > 4) behavior = "Uniform";
      infoDiv.innerHTML = `
        <span class="viz-tag">τ = ${t.toFixed(1)}</span>
        <span class="viz-tag">Max prob: ${maxP.toFixed(3)}</span>
        <span class="viz-tag">Entropy: ${entropy.toFixed(3)} bits</span>
        <span class="viz-tag viz-tag-accent">${behavior}</span>`;
    }

    slider.addEventListener("input", draw);
    draw();
    // Redraw on theme change
    new MutationObserver(draw).observe(document.body.closest("[data-md-color-scheme]") || document.documentElement, {
      attributes: true,
      attributeFilter: ["data-md-color-scheme"],
    });
  }

  /* ─────────────────────────────────────────────────────────────
     2. Entropy Calculator
     ───────────────────────────────────────────────────────────── */
  function initEntropyCalc(container) {
    const n = 4;
    const initP = [0.4, 0.3, 0.2, 0.1];
    const eventLabels = ["A", "B", "C", "D"];

    let sliders = "";
    for (let i = 0; i < n; i++) {
      sliders += sliderHTML(
        `ent-p${i}`,
        `P(${eventLabels[i]})`,
        0, 1, 0.01, initP[i], ""
      );
    }

    container.innerHTML = `
      <div class="viz-controls">${sliders}
        <div class="viz-output" id="ent-norm-warn" style="display:none">
          <span class="viz-tag viz-tag-warn">⚠ Probabilities don't sum to 1 — auto-normalizing</span>
        </div>
      </div>
      <canvas id="ent-canvas" width="560" height="220"></canvas>
      <div class="viz-output" id="ent-info"></div>`;

    const canvas = $("#ent-canvas", container);
    const ctx = canvas.getContext("2d");
    const infoDiv = $("#ent-info", container);
    const warnDiv = $("#ent-norm-warn", container);

    function draw() {
      let probs = [];
      for (let i = 0; i < n; i++) {
        probs.push(parseFloat($(`#ent-p${i}`, container).value));
      }
      const sum = probs.reduce((a, b) => a + b, 0);
      const needsNorm = Math.abs(sum - 1.0) > 0.01;
      warnDiv.style.display = needsNorm ? "block" : "none";
      if (sum > 0) probs = probs.map((p) => p / sum);

      // Update displayed values
      for (let i = 0; i < n; i++) {
        $(`#ent-p${i}-val`, container).textContent = probs[i].toFixed(2);
      }

      const entropy = -probs.reduce(
        (s, p) => s + (p > 1e-12 ? p * Math.log2(p) : 0),
        0
      );
      const maxEntropy = Math.log2(n);
      const crossEntropyUniform = -probs.reduce(
        (s, p) => s + (p > 1e-12 ? p * Math.log2(1 / n) : 0),
        0
      );

      const C = themeColors();
      const W = canvas.width,
        H = canvas.height;
      ctx.clearRect(0, 0, W, H);

      const pad = { l: 55, r: 20, t: 20, b: 50 };
      const plotW = W - pad.l - pad.r;
      const plotH = H - pad.t - pad.b;

      // Surprise bars: -log2(p) for each event
      const surprises = probs.map((p) => (p > 1e-12 ? -Math.log2(p) : 0));
      const maxSurprise = Math.max(...surprises, maxEntropy);

      // Grid
      ctx.strokeStyle = C.grid;
      ctx.lineWidth = 1;
      const nTicks = 4;
      for (let i = 0; i <= nTicks; i++) {
        const y = pad.t + (plotH * (nTicks - i)) / nTicks;
        ctx.beginPath();
        ctx.moveTo(pad.l, y);
        ctx.lineTo(W - pad.r, y);
        ctx.stroke();
        ctx.fillStyle = C.fg;
        ctx.font = "11px JetBrains Mono, monospace";
        ctx.textAlign = "right";
        const val = (maxSurprise * i) / nTicks;
        ctx.fillText(val.toFixed(1), pad.l - 8, y + 4);
      }

      // Bars
      const barW = plotW / n - 16;
      probs.forEach((p, i) => {
        const x = pad.l + (plotW / n) * i + 8;
        const surprise = surprises[i];
        const h = maxSurprise > 0 ? (surprise / maxSurprise) * plotH : 0;
        const y = pad.t + plotH - h;

        const grad = ctx.createLinearGradient(x, y, x, pad.t + plotH);
        grad.addColorStop(0, C.warn);
        grad.addColorStop(1, C.barFill);
        ctx.fillStyle = grad;
        ctx.beginPath();
        ctx.roundRect(x, y, barW, h, [4, 4, 0, 0]);
        ctx.fill();

        // Surprise value
        ctx.fillStyle = C.fg;
        ctx.font = "bold 11px JetBrains Mono, monospace";
        ctx.textAlign = "center";
        if (surprise > 0) ctx.fillText(surprise.toFixed(2), x + barW / 2, y - 6);

        // Prob label
        ctx.font = "11px Source Sans 3, sans-serif";
        ctx.fillText(`${eventLabels[i]}`, x + barW / 2, pad.t + plotH + 16);
        ctx.font = "10px JetBrains Mono, monospace";
        ctx.fillStyle = C.accentSoft;
        ctx.fillText(`p=${probs[i].toFixed(2)}`, x + barW / 2, pad.t + plotH + 32);
      });

      // Y-axis label
      ctx.save();
      ctx.fillStyle = C.accentSoft;
      ctx.font = "11px Source Sans 3, sans-serif";
      ctx.textAlign = "center";
      ctx.translate(14, pad.t + plotH / 2);
      ctx.rotate(-Math.PI / 2);
      ctx.fillText("Surprise (bits)", 0, 0);
      ctx.restore();

      infoDiv.innerHTML = `
        <span class="viz-tag">H(X) = ${entropy.toFixed(4)} bits</span>
        <span class="viz-tag">Max entropy = ${maxEntropy.toFixed(4)} bits</span>
        <span class="viz-tag">Perplexity = 2^H = ${Math.pow(2, entropy).toFixed(2)}</span>
        <span class="viz-tag viz-tag-accent">${((entropy / maxEntropy) * 100).toFixed(0)}% of maximum</span>`;
    }

    for (let i = 0; i < n; i++) {
      $(`#ent-p${i}`, container).addEventListener("input", draw);
    }
    draw();
  }

  /* ─────────────────────────────────────────────────────────────
     3. Attention Heatmap (4×4 worked example)
     ───────────────────────────────────────────────────────────── */
  function initAttentionHeatmap(container) {
    const tokens = ["The", "cat", "sat", "."];
    const defaultQ = [[2,0,0,0],[0,2,0,0],[0,0,2,0],[0,0,0,2]];
    const defaultK = [[1,1,0,0],[1,0,1,0],[0,1,1,0],[0,0,1,1]];

    container.innerHTML = `
      <div class="viz-controls">
        <div class="viz-step-controls">
          <button class="viz-btn" id="attn-step-scores">1. Scores QK<sup>T</sup></button>
          <button class="viz-btn" id="attn-step-scale">2. Scale ÷√d<sub>k</sub></button>
          <button class="viz-btn" id="attn-step-softmax">3. Softmax</button>
          <button class="viz-btn viz-btn-active" id="attn-step-all">Show All</button>
        </div>
        <label class="viz-checkbox-label">
          <input type="checkbox" id="attn-causal"> Apply causal mask
        </label>
      </div>
      <canvas id="attn-canvas" width="580" height="320"></canvas>
      <div class="viz-output" id="attn-info"></div>`;

    const canvas = $("#attn-canvas", container);
    const ctx = canvas.getContext("2d");
    const infoDiv = $("#attn-info", container);
    const causalCheck = $("#attn-causal", container);
    let step = "all";

    function matmul(A, B) {
      const m = A.length, n = B[0].length, p = B.length;
      const C = Array.from({length: m}, () => Array(n).fill(0));
      for (let i = 0; i < m; i++)
        for (let j = 0; j < n; j++)
          for (let k = 0; k < p; k++)
            C[i][j] += A[i][k] * B[k][j];
      return C;
    }

    function transpose(M) {
      return M[0].map((_, j) => M.map(row => row[j]));
    }

    function draw() {
      const Q = defaultQ, K = defaultK;
      const dk = Q[0].length;
      const KT = transpose(K);
      const scores = matmul(Q, KT);
      const scaled = scores.map(row => row.map(v => v / Math.sqrt(dk)));

      const causal = causalCheck.checked;
      const masked = scaled.map((row, i) =>
        row.map((v, j) => (causal && j > i) ? -Infinity : v)
      );

      const weights = masked.map(row => {
        const finite = row.filter(v => isFinite(v));
        const maxV = finite.length > 0 ? Math.max(...finite) : 0;
        const exps = row.map(v => isFinite(v) ? Math.exp(v - maxV) : 0);
        const sum = exps.reduce((a,b) => a+b, 0);
        return exps.map(e => sum > 0 ? e / sum : 0);
      });

      // Determine which matrix to show
      let matrix, title, fmt;
      if (step === "scores") { matrix = scores; title = "Raw Scores QK^T"; fmt = v => v.toFixed(0); }
      else if (step === "scale") { matrix = causal ? masked : scaled; title = `Scaled${causal ? " + Masked" : ""}`; fmt = v => isFinite(v) ? v.toFixed(2) : "-∞"; }
      else if (step === "softmax") { matrix = weights; title = "Attention Weights (softmax)"; fmt = v => v.toFixed(3); }
      else { matrix = weights; title = "Attention Weights (softmax)"; fmt = v => v.toFixed(3); }

      const C = themeColors();
      const W = canvas.width, H = canvas.height;
      ctx.clearRect(0, 0, W, H);

      const cellSize = 58;
      const padL = 70, padT = 60;
      const n = tokens.length;

      // Title
      ctx.fillStyle = C.fg;
      ctx.font = "bold 13px Source Sans 3, sans-serif";
      ctx.textAlign = "center";
      ctx.fillText(title, padL + (n * cellSize) / 2, 18);

      // Header labels (Key positions)
      ctx.font = "12px Source Sans 3, sans-serif";
      ctx.fillStyle = C.accentSoft;
      ctx.textAlign = "center";
      for (let j = 0; j < n; j++) {
        ctx.fillText(tokens[j], padL + j * cellSize + cellSize / 2, padT - 10);
      }
      // Row labels (Query positions)
      ctx.textAlign = "right";
      for (let i = 0; i < n; i++) {
        ctx.fillText(tokens[i], padL - 10, padT + i * cellSize + cellSize / 2 + 4);
      }

      // Axis titles
      ctx.font = "10px Source Sans 3, sans-serif";
      ctx.fillStyle = C.accentSoft;
      ctx.textAlign = "center";
      ctx.fillText("← Key →", padL + (n * cellSize) / 2, padT - 26);
      ctx.save();
      ctx.translate(padL - 50, padT + (n * cellSize) / 2);
      ctx.rotate(-Math.PI / 2);
      ctx.fillText("← Query →", 0, 0);
      ctx.restore();

      // Draw cells
      const maxVal = Math.max(...matrix.flat().filter(isFinite));
      const minVal = Math.min(...matrix.flat().filter(isFinite));

      for (let i = 0; i < n; i++) {
        for (let j = 0; j < n; j++) {
          const x = padL + j * cellSize;
          const y = padT + i * cellSize;
          const v = matrix[i][j];

          // Color intensity
          let alpha = 0;
          if (isFinite(v) && maxVal !== minVal) {
            alpha = (v - minVal) / (maxVal - minVal);
          } else if (isFinite(v)) {
            alpha = 0.5;
          }

          if (!isFinite(v)) {
            ctx.fillStyle = C.danger + "22";
          } else {
            const r = Math.round(111 * alpha + 20 * (1 - alpha));
            const g = Math.round(163 * alpha + 26 * (1 - alpha));
            const b = Math.round(184 * alpha + 34 * (1 - alpha));
            ctx.fillStyle = `rgba(${r},${g},${b},${0.15 + alpha * 0.55})`;
          }

          ctx.beginPath();
          ctx.roundRect(x + 2, y + 2, cellSize - 4, cellSize - 4, 4);
          ctx.fill();

          // Border
          ctx.strokeStyle = C.grid;
          ctx.lineWidth = 1;
          ctx.stroke();

          // Value text
          ctx.fillStyle = alpha > 0.6 ? "#fff" : C.fg;
          ctx.font = "bold 12px JetBrains Mono, monospace";
          ctx.textAlign = "center";
          ctx.fillText(fmt(v), x + cellSize / 2, y + cellSize / 2 + 5);
        }
      }

      // Info
      const maxWeight = Math.max(...weights.flat());
      const maxIdx = weights.flat().indexOf(maxWeight);
      const qIdx = Math.floor(maxIdx / n);
      const kIdx = maxIdx % n;
      infoDiv.innerHTML = `
        <span class="viz-tag">d<sub>k</sub> = ${dk}</span>
        <span class="viz-tag">√d<sub>k</sub> = ${Math.sqrt(dk).toFixed(2)}</span>
        <span class="viz-tag">Strongest: "${tokens[qIdx]}" → "${tokens[kIdx]}" (${maxWeight.toFixed(3)})</span>
        ${causal ? '<span class="viz-tag viz-tag-warn">Causal mask active</span>' : ''}`;
    }

    ["scores", "scale", "softmax", "all"].forEach(s => {
      $(`#attn-step-${s}`, container).addEventListener("click", () => {
        step = s;
        $$(".viz-btn", container).forEach(b => b.classList.remove("viz-btn-active"));
        $(`#attn-step-${s}`, container).classList.add("viz-btn-active");
        draw();
      });
    });
    causalCheck.addEventListener("change", draw);
    draw();
  }

  /* ─────────────────────────────────────────────────────────────
     4. Embedding Explorer (2D interactive)
     ───────────────────────────────────────────────────────────── */
  function initEmbeddingExplorer(container) {
    const words = [
      { label: "king", x: 0.8, y: 0.7, color: "#6FA3B8" },
      { label: "queen", x: 0.8, y: 0.3, color: "#6FA3B8" },
      { label: "man", x: 0.4, y: 0.7, color: "#5eab8b" },
      { label: "woman", x: 0.4, y: 0.3, color: "#5eab8b" },
      { label: "prince", x: 0.65, y: 0.75, color: "#c4975a" },
      { label: "princess", x: 0.65, y: 0.35, color: "#c4975a" },
    ];

    container.innerHTML = `
      <div class="viz-controls">
        <div class="viz-output" id="emb-info">
          <span class="viz-tag">Click two words to see cosine similarity</span>
        </div>
        <label class="viz-checkbox-label">
          <input type="checkbox" id="emb-show-analogy" checked> Show king − man + woman = ?
        </label>
      </div>
      <canvas id="emb-canvas" width="560" height="360"></canvas>`;

    const canvas = $("#emb-canvas", container);
    const ctx = canvas.getContext("2d");
    const infoDiv = $("#emb-info", container);
    const analogyCheck = $("#emb-show-analogy", container);
    let selected = [];
    let dragging = -1;

    function toCanvas(wx, wy) {
      return { x: 50 + wx * (canvas.width - 100), y: canvas.height - 50 - wy * (canvas.height - 100) };
    }
    function fromCanvas(cx, cy) {
      return { x: (cx - 50) / (canvas.width - 100), y: (canvas.height - 50 - cy) / (canvas.height - 100) };
    }

    function cosine(a, b) {
      const dot = a.x * b.x + a.y * b.y;
      const na = Math.sqrt(a.x * a.x + a.y * a.y);
      const nb = Math.sqrt(b.x * b.x + b.y * b.y);
      return na > 0 && nb > 0 ? dot / (na * nb) : 0;
    }

    function draw() {
      const C = themeColors();
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      // Grid
      ctx.strokeStyle = C.grid;
      ctx.lineWidth = 1;
      for (let i = 0; i <= 10; i++) {
        const p = toCanvas(i / 10, 0);
        ctx.beginPath(); ctx.moveTo(p.x, 40); ctx.lineTo(p.x, canvas.height - 40); ctx.stroke();
        const p2 = toCanvas(0, i / 10);
        ctx.beginPath(); ctx.moveTo(40, p2.y); ctx.lineTo(canvas.width - 40, p2.y); ctx.stroke();
      }

      // Analogy arrow
      if (analogyCheck.checked) {
        const king = words.find(w => w.label === "king");
        const man = words.find(w => w.label === "man");
        const woman = words.find(w => w.label === "woman");
        if (king && man && woman) {
          const resultX = king.x - man.x + woman.x;
          const resultY = king.y - man.y + woman.y;
          const rp = toCanvas(resultX, resultY);

          // Draw offset arrows
          const kp = toCanvas(king.x, king.y);
          const mp = toCanvas(man.x, man.y);
          const wp = toCanvas(woman.x, woman.y);

          // man -> king arrow (royalty direction)
          ctx.strokeStyle = C.accent + "66";
          ctx.lineWidth = 2;
          ctx.setLineDash([6, 4]);
          ctx.beginPath(); ctx.moveTo(mp.x, mp.y); ctx.lineTo(kp.x, kp.y); ctx.stroke();
          // woman -> result arrow (same direction)
          ctx.beginPath(); ctx.moveTo(wp.x, wp.y); ctx.lineTo(rp.x, rp.y); ctx.stroke();
          ctx.setLineDash([]);

          // Result point
          ctx.beginPath();
          ctx.arc(rp.x, rp.y, 8, 0, Math.PI * 2);
          ctx.fillStyle = C.danger + "88";
          ctx.fill();
          ctx.strokeStyle = C.danger;
          ctx.lineWidth = 2;
          ctx.stroke();
          ctx.fillStyle = C.danger;
          ctx.font = "bold 11px Source Sans 3, sans-serif";
          ctx.textAlign = "left";
          ctx.fillText("≈ queen?", rp.x + 12, rp.y + 4);

          // Label the direction
          ctx.fillStyle = C.accentSoft;
          ctx.font = "10px Source Sans 3, sans-serif";
          ctx.textAlign = "center";
          ctx.fillText("royalty direction →", (mp.x + kp.x) / 2, (mp.y + kp.y) / 2 - 10);
        }
      }

      // Word points
      words.forEach((w, i) => {
        const p = toCanvas(w.x, w.y);
        const isSelected = selected.includes(i);

        ctx.beginPath();
        ctx.arc(p.x, p.y, isSelected ? 10 : 7, 0, Math.PI * 2);
        ctx.fillStyle = w.color + (isSelected ? "dd" : "99");
        ctx.fill();
        if (isSelected) {
          ctx.strokeStyle = w.color;
          ctx.lineWidth = 3;
          ctx.stroke();
        }

        ctx.fillStyle = C.fg;
        ctx.font = `${isSelected ? "bold " : ""}12px Source Sans 3, sans-serif`;
        ctx.textAlign = "center";
        ctx.fillText(w.label, p.x, p.y - 14);
      });

      // Similarity line
      if (selected.length === 2) {
        const a = words[selected[0]];
        const b = words[selected[1]];
        const pa = toCanvas(a.x, a.y);
        const pb = toCanvas(b.x, b.y);
        ctx.strokeStyle = C.accent + "88";
        ctx.lineWidth = 2;
        ctx.setLineDash([4, 4]);
        ctx.beginPath(); ctx.moveTo(pa.x, pa.y); ctx.lineTo(pb.x, pb.y); ctx.stroke();
        ctx.setLineDash([]);

        const sim = cosine(a, b);
        const eucl = Math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2);
        infoDiv.innerHTML = `
          <span class="viz-tag">cos(${a.label}, ${b.label}) = ${sim.toFixed(4)}</span>
          <span class="viz-tag">Euclidean dist = ${eucl.toFixed(4)}</span>
          <span class="viz-tag viz-tag-accent">${sim > 0.95 ? "Very similar" : sim > 0.8 ? "Similar" : sim > 0.5 ? "Somewhat similar" : "Different"}</span>`;
      }
    }

    canvas.addEventListener("click", (e) => {
      const rect = canvas.getBoundingClientRect();
      const cx = e.clientX - rect.left;
      const cy = e.clientY - rect.top;

      for (let i = 0; i < words.length; i++) {
        const p = toCanvas(words[i].x, words[i].y);
        if (Math.hypot(cx - p.x, cy - p.y) < 14) {
          if (selected.includes(i)) {
            selected = selected.filter(s => s !== i);
          } else {
            selected.push(i);
            if (selected.length > 2) selected.shift();
          }
          draw();
          return;
        }
      }
    });

    canvas.addEventListener("mousedown", (e) => {
      const rect = canvas.getBoundingClientRect();
      const cx = e.clientX - rect.left;
      const cy = e.clientY - rect.top;
      for (let i = 0; i < words.length; i++) {
        const p = toCanvas(words[i].x, words[i].y);
        if (Math.hypot(cx - p.x, cy - p.y) < 14) { dragging = i; return; }
      }
    });

    canvas.addEventListener("mousemove", (e) => {
      if (dragging < 0) return;
      const rect = canvas.getBoundingClientRect();
      const cx = e.clientX - rect.left;
      const cy = e.clientY - rect.top;
      const w = fromCanvas(cx, cy);
      words[dragging].x = Math.max(0, Math.min(1, w.x));
      words[dragging].y = Math.max(0, Math.min(1, w.y));
      draw();
    });

    canvas.addEventListener("mouseup", () => { dragging = -1; });
    canvas.addEventListener("mouseleave", () => { dragging = -1; });
    analogyCheck.addEventListener("change", draw);
    draw();
  }

  /* ─────────────────────────────────────────────────────────────
     5. N-gram Calculator
     ───────────────────────────────────────────────────────────── */
  function initNgramCalc(container) {
    const defaultCorpus = "the cat sat on the mat\nthe cat ate the fish\nthe dog sat on the log\nthe dog ate the bone";

    container.innerHTML = `
      <div class="viz-controls">
        <label for="ngram-corpus">Training Corpus (one sentence per line):</label>
        <textarea id="ngram-corpus" class="viz-textarea" rows="4">${defaultCorpus}</textarea>
        <div class="viz-row">
          ${sliderHTML("ngram-k", "Smoothing k", 0, 2, 0.1, 1.0, "")}
          <div class="viz-slider-group">
            <label>Context: <input type="text" id="ngram-ctx" class="viz-input" value="the" placeholder="context word"></label>
          </div>
        </div>
      </div>
      <div class="viz-output" id="ngram-results"></div>`;

    const corpusTA = $("#ngram-corpus", container);
    const kSlider = $("#ngram-k", container);
    const ctxInput = $("#ngram-ctx", container);
    const resultsDiv = $("#ngram-results", container);

    function compute() {
      const lines = corpusTA.value.trim().split("\n").filter(l => l.trim());
      const k = parseFloat(kSlider.value);
      $("#ngram-k-val", container).textContent = k.toFixed(1);

      // Build bigram counts
      const bigramCounts = {};
      const unigramCounts = {};
      const vocab = new Set();

      lines.forEach(line => {
        const toks = line.trim().toLowerCase().split(/\s+/);
        toks.forEach(t => { vocab.add(t); unigramCounts[t] = (unigramCounts[t] || 0) + 1; });
        for (let i = 0; i < toks.length - 1; i++) {
          const bg = `${toks[i]}|${toks[i+1]}`;
          bigramCounts[bg] = (bigramCounts[bg] || 0) + 1;
        }
      });

      const V = vocab.size;
      const ctx = ctxInput.value.trim().toLowerCase();

      if (!vocab.has(ctx)) {
        resultsDiv.innerHTML = `<span class="viz-tag viz-tag-warn">Word "${ctx}" not in corpus vocabulary</span>
          <br><span class="viz-tag">Vocabulary (${V} words): ${[...vocab].join(", ")}</span>`;
        return;
      }

      // Compute P(w | ctx) for all w
      const probs = [];
      const ctxCount = unigramCounts[ctx] || 0;
      vocab.forEach(w => {
        const bg = `${ctx}|${w}`;
        const count = bigramCounts[bg] || 0;
        const prob = (count + k) / (ctxCount + k * V);
        probs.push({ word: w, count, prob });
      });

      probs.sort((a, b) => b.prob - a.prob);
      const top = probs.slice(0, 8);

      let html = `<span class="viz-tag">Vocab size V = ${V}</span>
        <span class="viz-tag">count("${ctx}") = ${ctxCount}</span>
        <span class="viz-tag">k = ${k.toFixed(1)}</span>
        <table class="viz-table">
          <tr><th>Next word</th><th>count(${ctx}, w)</th><th>P(w | ${ctx})</th><th>Bar</th></tr>`;
      top.forEach(p => {
        const barPct = (p.prob * 100 / (top[0].prob || 1)) * top[0].prob > 0 ? (p.prob / top[0].prob * 100) : 0;
        html += `<tr>
          <td><strong>${p.word}</strong></td>
          <td>${p.count}</td>
          <td>${p.prob.toFixed(4)}</td>
          <td><div class="viz-inline-bar" style="width: ${barPct.toFixed(0)}%"></div></td>
        </tr>`;
      });
      html += "</table>";
      if (probs.length > 8) html += `<span class="viz-tag">...and ${probs.length - 8} more words</span>`;

      resultsDiv.innerHTML = html;
    }

    corpusTA.addEventListener("input", compute);
    kSlider.addEventListener("input", compute);
    ctxInput.addEventListener("input", compute);
    compute();
  }

  /* ─────────────────────────────────────────────────────────────
     6. LSTM Gate Visualizer
     ───────────────────────────────────────────────────────────── */
  function initLSTMViz(container) {
    container.innerHTML = `
      <div class="viz-controls">
        <div class="viz-row">
          ${sliderHTML("lstm-ht", "h(t-1) prev hidden", -2, 2, 0.1, 0.5, "")}
          ${sliderHTML("lstm-xt", "x(t) input", -2, 2, 0.1, 1.2, "")}
          ${sliderHTML("lstm-ct", "c(t-1) prev cell", -2, 2, 0.1, 0.8, "")}
        </div>
      </div>
      <canvas id="lstm-canvas" width="580" height="340"></canvas>
      <div class="viz-output" id="lstm-info"></div>`;

    const canvas = $("#lstm-canvas", container);
    const ctx = canvas.getContext("2d");
    const infoDiv = $("#lstm-info", container);

    function sigmoid(z) { return 1 / (1 + Math.exp(-z)); }
    function tanh_(z) { return Math.tanh(z); }

    function draw() {
      const ht1 = parseFloat($("#lstm-ht", container).value);
      const xt = parseFloat($("#lstm-xt", container).value);
      const ct1 = parseFloat($("#lstm-ct", container).value);

      $("#lstm-ht-val", container).textContent = ht1.toFixed(1);
      $("#lstm-xt-val", container).textContent = xt.toFixed(1);
      $("#lstm-ct-val", container).textContent = ct1.toFixed(1);

      // Simplified scalar LSTM with made-up weights
      const zf = 0.5 * ht1 + 0.3 * xt - 0.2;
      const zi = 0.4 * ht1 + 0.6 * xt + 0.1;
      const zc = 0.3 * ht1 + 0.5 * xt;
      const zo = 0.2 * ht1 + 0.4 * xt + 0.3;

      const ft = sigmoid(zf);
      const it = sigmoid(zi);
      const ct_cand = tanh_(zc);
      const ot = sigmoid(zo);
      const ct = ft * ct1 + it * ct_cand;
      const ht = ot * tanh_(ct);

      const C = themeColors();
      const W = canvas.width, H = canvas.height;
      ctx.clearRect(0, 0, W, H);

      // Draw gate pipeline
      const stages = [
        { name: "Forget Gate", sym: "f_t", val: ft, pre: zf, fn: "σ", color: C.danger },
        { name: "Input Gate", sym: "i_t", val: it, pre: zi, fn: "σ", color: C.success },
        { name: "Candidate", sym: "c̃_t", val: ct_cand, pre: zc, fn: "tanh", color: C.warn },
        { name: "Output Gate", sym: "o_t", val: ot, pre: zo, fn: "σ", color: C.accent },
      ];

      const boxW = 120, boxH = 60, gap = 16;
      const startX = 30, startY = 30;

      stages.forEach((s, i) => {
        const x = startX + i * (boxW + gap);
        const y = startY;

        // Box
        ctx.fillStyle = s.color + "18";
        ctx.strokeStyle = s.color + "66";
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.roundRect(x, y, boxW, boxH, 8);
        ctx.fill();
        ctx.stroke();

        // Title
        ctx.fillStyle = C.fg;
        ctx.font = "bold 11px Source Sans 3, sans-serif";
        ctx.textAlign = "center";
        ctx.fillText(s.name, x + boxW / 2, y + 18);

        // Value
        ctx.font = "bold 14px JetBrains Mono, monospace";
        ctx.fillStyle = s.color;
        ctx.fillText(s.val.toFixed(3), x + boxW / 2, y + 42);

        // Pre-activation
        ctx.font = "10px JetBrains Mono, monospace";
        ctx.fillStyle = C.accentSoft;
        ctx.fillText(`${s.fn}(${s.pre.toFixed(2)})`, x + boxW / 2, y + boxH + 16);
      });

      // Cell update visualization
      const cellY = startY + boxH + 50;
      ctx.fillStyle = C.fg;
      ctx.font = "bold 13px Source Sans 3, sans-serif";
      ctx.textAlign = "left";
      ctx.fillText("Cell Update:", startX, cellY);

      ctx.font = "13px JetBrains Mono, monospace";
      const eq1 = `c_t = ${ft.toFixed(3)} × ${ct1.toFixed(1)} + ${it.toFixed(3)} × ${ct_cand.toFixed(3)}`;
      ctx.fillStyle = C.accentSoft;
      ctx.fillText(eq1, startX + 10, cellY + 24);

      ctx.fillStyle = C.accent;
      ctx.font = "bold 13px JetBrains Mono, monospace";
      ctx.fillText(`    = ${(ft * ct1).toFixed(3)} + ${(it * ct_cand).toFixed(3)} = ${ct.toFixed(3)}`, startX + 10, cellY + 48);

      // Hidden output
      const hidY = cellY + 76;
      ctx.fillStyle = C.fg;
      ctx.font = "bold 13px Source Sans 3, sans-serif";
      ctx.fillText("Hidden Output:", startX, hidY);

      ctx.font = "13px JetBrains Mono, monospace";
      ctx.fillStyle = C.accentSoft;
      ctx.fillText(`h_t = ${ot.toFixed(3)} × tanh(${ct.toFixed(3)})`, startX + 10, hidY + 24);
      ctx.fillStyle = C.accent;
      ctx.font = "bold 13px JetBrains Mono, monospace";
      ctx.fillText(`    = ${ot.toFixed(3)} × ${tanh_(ct).toFixed(3)} = ${ht.toFixed(3)}`, startX + 10, hidY + 48);

      // Gate strength bars
      const barY = hidY + 70;
      ctx.fillStyle = C.fg;
      ctx.font = "bold 11px Source Sans 3, sans-serif";
      ctx.textAlign = "left";
      ctx.fillText("Gate Activations:", startX, barY);

      const barMaxW = 200;
      stages.forEach((s, i) => {
        const y = barY + 18 + i * 22;
        ctx.fillStyle = C.fg;
        ctx.font = "11px Source Sans 3, sans-serif";
        ctx.textAlign = "right";
        ctx.fillText(s.sym, startX + 30, y + 4);
        // Bar background
        ctx.fillStyle = C.glass;
        ctx.fillRect(startX + 40, y - 6, barMaxW, 14);
        // Bar fill
        const w = Math.abs(s.val) * barMaxW;
        ctx.fillStyle = s.color + "aa";
        ctx.beginPath();
        ctx.roundRect(startX + 40, y - 6, w, 14, 3);
        ctx.fill();
        // Value label
        ctx.fillStyle = C.fg;
        ctx.font = "10px JetBrains Mono, monospace";
        ctx.textAlign = "left";
        ctx.fillText(s.val.toFixed(3), startX + 40 + w + 6, y + 4);
      });

      infoDiv.innerHTML = `
        <span class="viz-tag">f_t = ${ft.toFixed(3)} → ${ft > 0.5 ? "Keeping" : "Forgetting"} old memory</span>
        <span class="viz-tag">i_t = ${it.toFixed(3)} → ${it > 0.5 ? "Writing" : "Blocking"} new info</span>
        <span class="viz-tag">c_t = ${ct.toFixed(3)}</span>
        <span class="viz-tag viz-tag-accent">h_t = ${ht.toFixed(3)}</span>`;
    }

    ["lstm-ht", "lstm-xt", "lstm-ct"].forEach(id => {
      $(`#${id}`, container).addEventListener("input", draw);
    });
    draw();
  }

  /* ─────────────────────────────────────────────────────────────
     7. Seq2Seq Attention Animator
     ───────────────────────────────────────────────────────────── */
  function initSeq2SeqViz(container) {
    const srcTokens = ["I", "am", "a", "student"];
    const tgtTokens = ["<sos>", "je", "suis", "un", "étudiant"];

    container.innerHTML = `
      <div class="viz-controls">
        ${sliderHTML("s2s-step", "Decoder step", 0, tgtTokens.length - 1, 1, 1, "")}
      </div>
      <canvas id="s2s-canvas" width="580" height="320"></canvas>
      <div class="viz-output" id="s2s-info"></div>`;

    const canvas = $("#s2s-canvas", container);
    const ctx = canvas.getContext("2d");
    const stepSlider = $("#s2s-step", container);
    const infoDiv = $("#s2s-info", container);

    // Fake attention weights for each decoder step
    const attentions = [
      [0.25, 0.25, 0.25, 0.25],  // <sos> - uniform
      [0.6, 0.15, 0.1, 0.15],    //  je -> I
      [0.1, 0.6, 0.15, 0.15],    // suis -> am
      [0.1, 0.1, 0.6, 0.2],      // un -> a
      [0.05, 0.05, 0.1, 0.8],    // étudiant -> student
    ];

    function draw() {
      const step = parseInt(stepSlider.value);
      $("#s2s-step-val", container).textContent = step;

      const C = themeColors();
      const W = canvas.width, H = canvas.height;
      ctx.clearRect(0, 0, W, H);

      const encY = 60, decY = 250;
      const encStartX = 60, decStartX = 40;
      const tokW = 110;

      // Title
      ctx.fillStyle = C.fg;
      ctx.font = "bold 12px Source Sans 3, sans-serif";
      ctx.textAlign = "left";
      ctx.fillText("ENCODER (English)", encStartX, 25);
      ctx.fillText("DECODER (French)", decStartX, decY - 20);

      // Encoder tokens
      srcTokens.forEach((t, i) => {
        const x = encStartX + i * tokW;
        ctx.fillStyle = C.accent + "20";
        ctx.strokeStyle = C.accent + "44";
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.roundRect(x, encY, tokW - 10, 36, 6);
        ctx.fill();
        ctx.stroke();
        ctx.fillStyle = C.fg;
        ctx.font = "bold 13px Source Sans 3, sans-serif";
        ctx.textAlign = "center";
        ctx.fillText(t, x + (tokW - 10) / 2, encY + 23);

        // Encoder hidden state indicator
        ctx.fillStyle = C.accentSoft;
        ctx.font = "10px JetBrains Mono, monospace";
        ctx.fillText(`h${i + 1}`, x + (tokW - 10) / 2, encY + 52);
      });

      // Decoder tokens
      tgtTokens.forEach((t, i) => {
        const x = decStartX + i * tokW;
        const isActive = i === step;
        const isPast = i < step;
        ctx.fillStyle = isActive ? C.success + "30" : isPast ? C.glass : "transparent";
        ctx.strokeStyle = isActive ? C.success : isPast ? C.accentSoft + "44" : C.grid;
        ctx.lineWidth = isActive ? 3 : 1;
        ctx.beginPath();
        ctx.roundRect(x, decY, tokW - 10, 36, 6);
        ctx.fill();
        ctx.stroke();
        ctx.fillStyle = isActive ? C.success : isPast ? C.fg : C.accentSoft;
        ctx.font = `${isActive ? "bold " : ""}13px Source Sans 3, sans-serif`;
        ctx.textAlign = "center";
        ctx.fillText(t, x + (tokW - 10) / 2, decY + 23);
      });

      // Attention lines from active decoder step to encoder
      if (step >= 0 && step < attentions.length) {
        const weights = attentions[step];
        const decX = decStartX + step * tokW + (tokW - 10) / 2;

        weights.forEach((w, j) => {
          const encX = encStartX + j * tokW + (tokW - 10) / 2;
          const alpha = 0.15 + w * 0.85;
          const width = 1 + w * 5;

          ctx.strokeStyle = C.accent;
          ctx.globalAlpha = alpha;
          ctx.lineWidth = width;
          ctx.beginPath();
          ctx.moveTo(encX, encY + 36);
          // Bezier curve for smooth lines
          const midY = (encY + 36 + decY) / 2;
          ctx.bezierCurveTo(encX, midY, decX, midY, decX, decY);
          ctx.stroke();

          // Weight label on the line
          if (w > 0.15) {
            ctx.globalAlpha = 1;
            ctx.fillStyle = C.accent;
            ctx.font = "bold 10px JetBrains Mono, monospace";
            ctx.textAlign = "center";
            ctx.fillText(w.toFixed(2), (encX + decX) / 2, midY);
          }
        });
        ctx.globalAlpha = 1;

        // Context vector indicator
        ctx.fillStyle = C.warn + "22";
        ctx.strokeStyle = C.warn;
        ctx.lineWidth = 2;
        ctx.beginPath();
        const cvX = W - 120, cvY = (encY + decY) / 2;
        ctx.roundRect(cvX, cvY - 16, 100, 32, 6);
        ctx.fill();
        ctx.stroke();
        ctx.fillStyle = C.warn;
        ctx.font = "bold 11px Source Sans 3, sans-serif";
        ctx.textAlign = "center";
        ctx.fillText(`c_${step}`, cvX + 50, cvY + 5);

        const maxW = Math.max(...weights);
        const maxIdx = weights.indexOf(maxW);
        infoDiv.innerHTML = `
          <span class="viz-tag">Step ${step}: generating "${tgtTokens[step]}"</span>
          <span class="viz-tag">Strongest attention: "${srcTokens[maxIdx]}" (α=${maxW.toFixed(2)})</span>
          <span class="viz-tag viz-tag-accent">${step === 0 ? "Uniform attention at start" : `Aligning with source word "${srcTokens[maxIdx]}"`}</span>`;
      }
    }

    stepSlider.addEventListener("input", draw);
    draw();
  }

  /* ─────────────────────────────────────────────────────────────
     Auto-Initialize: find containers and build widgets
     ───────────────────────────────────────────────────────────── */
  const registry = {
    "softmax-viz": initSoftmaxViz,
    "entropy-calc": initEntropyCalc,
    "attention-heatmap": initAttentionHeatmap,
    "embedding-explorer": initEmbeddingExplorer,
    "ngram-calc": initNgramCalc,
    "lstm-viz": initLSTMViz,
    "seq2seq-viz": initSeq2SeqViz,
  };

  function initAll() {
    Object.entries(registry).forEach(([id, initFn]) => {
      const el = document.getElementById(id);
      if (el && !el.dataset.vizInit) {
        el.dataset.vizInit = "1";
        initFn(el);
      }
    });
  }

  // MkDocs Material uses instant navigation — reinitialize on page change
  if (typeof document$ !== "undefined") {
    document$.subscribe(() => setTimeout(initAll, 100));
  } else {
    document.addEventListener("DOMContentLoaded", initAll);
  }
})();
