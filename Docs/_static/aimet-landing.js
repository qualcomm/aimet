// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

/*
 * Landing page hero: animates one real MobileNetV2 weight channel snapping
 * onto a 4-bit quantization grid.
 *
 * Data: torchvision mobilenet_v2 (MobileNet_V2_Weights.DEFAULT),
 * features.15.conv.0.0.weight, output channel 301 (160 values). The scale and
 * offset are what aimet_torch's QuantizeDequantize(bitwidth=4,
 * symmetric=False) computes for those values, so the integer grid shown is
 * the one AIMET would actually use.
 *
 * Loaded only on the landing page (see setup() in conf.py). Renders nothing
 * if the hero figure is absent.
 */
(function () {
  "use strict";

  var VALUES = [-0.01775,0.05598,0.07866,-0.01626,-0.03722,0.01672,0.00682,-0.0461,-0.02951,0.04859,-0.04749,0.00144,0.02453,-0.03016,0.00262,0.07945,-0.02729,0.02513,-0.02736,0.00106,-0.07165,0.00287,-0.06242,-0.06994,0.07492,0.03225,0.05705,-0.01933,-0.05303,0.06177,-0.03791,-0.04873,0.02745,0.02812,0.06439,0.04474,0.02391,-0.04194,-0.0068,-0.02958,-0.07337,0.018,-0.07152,0.06532,-0.00482,-0.04199,-0.02942,0.02372,0.06395,0.04557,0.0989,-0.01222,-0.0225,0.0071,0.07415,0.02963,0.0622,-0.03927,0.11096,-0.06046,0.07453,-0.05579,0.01879,0.04611,0.05923,0.06292,-0.05545,0.03798,0.02128,0.11707,0.11517,0.04307,0.01419,-0.00358,0.06223,0.0751,-0.03318,0.00384,-0.05811,-0.01556,0.0187,0.04356,0.00823,-0.0167,0.00393,0.04686,-0.05037,-0.04985,-0.05522,0.06126,0.01796,-0.02266,0.00932,-0.05364,-0.08207,0.03185,-0.01107,-0.0225,-0.03827,-0.07421,-0.01243,-0.02648,0.07365,0.04648,-0.03288,0.00532,-0.04672,-0.05328,0.03297,0.01688,0.07901,0.02582,0.03069,0.05946,-0.04695,-0.08677,0.11671,-0.05614,-0.10435,-0.02169,0.02617,-0.04986,0.05737,-0.08447,0.02348,0.02037,0.00511,-0.03906,-0.07065,-0.00106,0.01533,-0.04179,0.03658,-0.00489,-0.04826,-0.03551,0.06314,-0.00999,-0.00714,-0.08743,-0.04908,0.04915,0.08844,-0.00175,-0.05882,-0.02148,-0.01731,-0.03284,0.01724,0.07095,-0.08368,0.06341,0.0367,0.02727,-0.02821,-0.00544,-0.02842,-0.02912,0.03161,0.01441];
  var SCALE = 0.014761735685169697;
  var OFFSET = -7;
  var BITWIDTH = 4;
  var LEVELS = 1 << BITWIDTH;

  function init() {
    var fig = document.querySelector(".aimet-quant-viz");
    if (!fig) return;
    var canvas = fig.querySelector("canvas");
    var ctx = canvas && canvas.getContext("2d");
    if (!ctx) return;

    var out = {};
    ["state", "bw", "levels", "scale", "offset"].forEach(function (k) {
      out[k] = fig.querySelector('[data-aimet-viz="' + k + '"]');
    });

    var reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    var n = VALUES.length;
    var qmin = OFFSET * SCALE;
    var qmax = (OFFSET + LEVELS - 1) * SCALE;

    // Integer bin for each value, and its stacking position within the bin.
    var qidx = [], rank = [], counts = [];
    for (var b = 0; b < LEVELS; b++) counts.push(0);
    for (var i = 0; i < n; i++) {
      var q = Math.round(VALUES[i] / SCALE) - OFFSET;
      q = Math.max(0, Math.min(LEVELS - 1, q));
      qidx.push(q);
      rank.push(counts[q]++);
    }
    var maxCount = Math.max.apply(null, counts);

    // Deterministic jitter so the float "cloud" looks the same every load.
    var seed = 11, jitter = [];
    for (i = 0; i < n; i++) {
      seed = (seed * 16807) % 2147483647;
      jitter.push(seed / 2147483647);
    }

    var W = 0, H = 0, raf = null, t0 = 0, visible = true;

    function resize() {
      var r = canvas.getBoundingClientRect();
      var dpr = Math.min(2, window.devicePixelRatio || 1);
      W = r.width;
      H = r.height;
      canvas.width = Math.round(W * dpr);
      canvas.height = Math.round(H * dpr);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    }

    // Theme colors can be color-mix() expressions, which canvas fillStyle
    // doesn't reliably accept; resolve them to rgb() through a probe element.
    var probe = document.createElement("span");
    probe.hidden = true;
    fig.appendChild(probe);
    function color(name, fallback) {
      probe.style.color = fallback;
      probe.style.color = "var(" + name + ")";
      return getComputedStyle(probe).color || fallback;
    }
    function token(name, fallback) {
      var v = getComputedStyle(canvas).getPropertyValue(name).trim();
      return v || fallback;
    }

    function ease(x) {
      return x < 0.5 ? 4 * x * x * x : 1 - Math.pow(-2 * x + 2, 3) / 2;
    }

    function setText(el, s) {
      if (el && el.textContent !== s) el.textContent = s;
    }

    function draw(k) {
      if (!W) resize();
      var accent = color("--aimet-viz-accent", "#2a2aea");
      var muted = color("--lumina-text-muted", "#59635e");
      var line = color("--lumina-border", "#dfe5e1");

      ctx.clearRect(0, 0, W, H);
      var padX = 16, base = H - 26, top = 10;
      var x = function (v) {
        return padX + ((v - qmin) / (qmax - qmin)) * (W - 2 * padX);
      };
      var step = Math.min(9, (base - top) / maxCount);
      var r = Math.max(2, Math.min(3.4, step * 0.4));

      // Grid lines, one per integer level.
      ctx.lineWidth = 1;
      ctx.strokeStyle = line;
      ctx.globalAlpha = 0.35 + 0.65 * k;
      for (var q = 0; q < LEVELS; q++) {
        var gx = Math.round(x(qmin + q * SCALE)) + 0.5;
        ctx.beginPath();
        ctx.moveTo(gx, top);
        ctx.lineTo(gx, base + 4);
        ctx.stroke();
      }
      ctx.globalAlpha = 1;

      // Baseline and integer tick labels.
      ctx.strokeStyle = muted;
      ctx.globalAlpha = 0.5;
      ctx.beginPath();
      ctx.moveTo(padX, base + 4.5);
      ctx.lineTo(W - padX, base + 4.5);
      ctx.stroke();
      ctx.globalAlpha = 1;
      ctx.fillStyle = muted;
      ctx.font = "11px " + token("--lumina-font-mono", "monospace");
      ctx.textAlign = "center";
      [0, 5, 10, 15].forEach(function (q) {
        ctx.fillText(String(q + OFFSET), x(qmin + q * SCALE), H - 7);
      });

      // Points: float position (scattered) -> quantized position (stacked).
      for (var i = 0; i < n; i++) {
        var fx = x(VALUES[i]);
        var fy = base - r - 2 - jitter[i] * (base - top - 2 * r) * 0.9;
        var qx = x(qmin + qidx[i] * SCALE);
        var qy = base - r - 1 - rank[i] * step;
        ctx.fillStyle = k > 0.5 ? accent : muted;
        ctx.globalAlpha = 0.55 + 0.45 * k;
        ctx.beginPath();
        ctx.arc(fx + (qx - fx) * k, fy + (qy - fy) * k, r, 0, 2 * Math.PI);
        ctx.fill();
      }
      ctx.globalAlpha = 1;

      var quantized = k > 0.5;
      setText(out.state, quantized ? "int4 · quantized" : "fp32");
      setText(out.bw, quantized ? String(BITWIDTH) : "32");
      setText(out.levels, quantized ? String(LEVELS) : "—");
      setText(out.scale, quantized ? SCALE.toFixed(5) : "—");
      setText(out.offset, quantized ? String(OFFSET) : "—");
    }

    // 6.4 s loop: hold float, snap to grid, hold quantized, release.
    var PERIOD = 6.4;
    function progress(t) {
      var p = t % PERIOD;
      if (p < 1.6) return 0;
      if (p < 2.6) return ease((p - 1.6) / 1.0);
      if (p < 5.2) return 1;
      return 1 - ease((p - 5.2) / 1.2);
    }

    function frame(now) {
      raf = null;
      draw(progress((now - t0) / 1000));
      if (visible) raf = requestAnimationFrame(frame);
    }

    function start() {
      if (reduce) {
        draw(1); // Static, fully quantized frame.
        return;
      }
      if (!raf && visible) raf = requestAnimationFrame(frame);
    }

    // Pause when scrolled off-screen or the tab is hidden.
    if ("IntersectionObserver" in window) {
      new IntersectionObserver(function (entries) {
        visible = entries[0].isIntersecting && !document.hidden;
        start();
      }).observe(fig);
    }
    document.addEventListener("visibilitychange", function () {
      visible = !document.hidden;
      start();
    });

    window.addEventListener("resize", function () {
      W = 0;
      if (reduce) draw(1);
    });
    // Repaint with new token colors when the theme toggle flips data-theme.
    new MutationObserver(function () {
      if (reduce) draw(1);
    }).observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });

    t0 = performance.now();
    draw(reduce ? 1 : 0);
    start();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
