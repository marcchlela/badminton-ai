"use strict";

// ============================================================
// CONFIGURATION
// ============================================================

const $ = id => document.getElementById(id);

const COLORS = {
  bg: "#0c1422",
  court: "#176b61",
  lines: "#e3f5f0",
  net: "#e85d8e",
  near: "#68cff3",
  far: "#f6bc67",
  shuttle: "#8af6aa",
  white: "#ffffff",
  muted: "#9aafbc"
};

const COURT_W = 6.10;
const COURT_L = 13.40;

const videoCanvas = $("videoCanvas");
const vctx = videoCanvas.getContext("2d");

const courtCanvas = $("courtCanvas");
const ctx = courtCanvas.getContext("2d");

let currentRallyId = new URLSearchParams(
  window.location.search
).get("rally");

let rallyCatalog = [];

let timeline = [];
let metadata = {};
let videoInfo = {};

let players = {};
let shuttleDisplay = [];

let currentFrame = 0;
let isPlaying = false;
let playbackSpeed = 1;
let lastTick = null;
let frameAccumulator = 0;

let view = "split";

let frameImages = [];
let frameReady = [];
let frameFailed = [];

let sourceStart = 0;
let sourceEnd = 0;


// ============================================================
// HELPERS
// ============================================================

function clamp(x, a, b) {
  return Math.max(a, Math.min(b, x));
}

function point(value) {
  if (!Array.isArray(value) || value.length < 2) {
    return null;
  }

  if (value[0] == null || value[1] == null) {
    return null;
  }

  const x = Number(value[0]);
  const y = Number(value[1]);

  if (!Number.isFinite(x) || !Number.isFinite(y)) {
    return null;
  }

  return [x, y];
}

function distance(a, b) {
  return Math.hypot(a[0] - b[0], a[1] - b[1]);
}

function lerp(a, b, t) {
  return a.map((v, i) => v + (b[i] - v) * t);
}

function focusAllows(side) {
  const focus = $("focus").value;
  return focus === "Both" || focus === side;
}

function formatPosition(p) {
  if (!p) return "—";

  return `${p[0].toFixed(2)}, ${p[1].toFixed(2)}m`;
}


// ============================================================
// BUILD PLAYER DISPLAY DATA
// ============================================================

function buildSeries(side, property, visualOnly = false) {
  const n = timeline.length;

  const values = Array(n).fill(null);
  const measured = Array(n).fill(false);

  const samples = [];

  for (let i = 0; i < n; i++) {
    const row = timeline[i];

    if (!row.players_sampled) continue;

    const p = row.players?.[side];
    if (!p) continue;

    if (visualOnly && p.visual_detection !== true) {
      continue;
    }

    let value = null;

    if (property === "bbox") {
      if (Array.isArray(p.bbox) && p.bbox.length === 4) {
        value = p.bbox.map(Number);

        if (
          value.some(v => !Number.isFinite(v)) ||
          value[2] <= 0 ||
          value[3] <= 0
        ) {
          value = null;
        }
      }
    } else {
      value = point(p.court_m);

      if (value && (
        value[0] < 0 || value[0] > COURT_W ||
        value[1] < 0 || value[1] > COURT_L
      )) {
        value = null;
      }
    }

    if (!value) continue;

    values[i] = value;
    measured[i] = true;
    samples.push(i);
  }

  // Short-gap interpolation.
  for (let s = 1; s < samples.length; s++) {
    const a = samples[s - 1];
    const b = samples[s];

    if (b - a > 6) continue;

    for (let j = a + 1; j < b; j++) {
      values[j] = lerp(
        values[a],
        values[b],
        (j - a) / (b - a)
      );
    }
  }

  // Short edge holds.
  if (samples.length) {
    const first = samples[0];
    const last = samples[samples.length - 1];

    for (let i = Math.max(0, first - 5); i < first; i++) {
      values[i] = [...values[first]];
    }

    for (let i = last + 1; i < Math.min(n, last + 6); i++) {
      values[i] = [...values[last]];
    }
  }

  // Symmetric display-only smoothing.
  const smoothed = values.map((value, i) => {
    if (!value) return null;

    const sum = Array(value.length).fill(0);
    let total = 0;

    for (let offset = -2; offset <= 2; offset++) {
      const j = i + offset;

      if (j < 0 || j >= n || !values[j]) continue;

      // Never smooth across missing-data gaps.
      let connected = true;

      for (let k = Math.min(i, j); k <= Math.max(i, j); k++) {
        if (!values[k]) {
          connected = false;
          break;
        }
      }

      if (!connected) continue;

      const weight = 3 - Math.abs(offset);

      for (let d = 0; d < sum.length; d++) {
        sum[d] += values[j][d] * weight;
      }

      total += weight;
    }

    return total
      ? sum.map(x => x / total)
      : value;
  });

  return { values: smoothed, measured };
}

function preparePlayers() {
  for (const side of ["Near", "Far"]) {
    players[side] = {
      court: buildSeries(side, "court_m"),
      boxes: buildSeries(side, "bbox", true)
    };
  }
}


// ============================================================
// SCHEMATIC SHUTTLE DISPLAY
// ============================================================

function prepareShuttle() {
  const visiblePoints = timeline
    .filter(row => row.shuttle?.visible)
    .map(row => point(row.shuttle.pixel))
    .filter(Boolean);

  if (visiblePoints.length < 2) {
    shuttleDisplay = Array(timeline.length).fill(null);
    return;
  }

  const xs = visiblePoints.map(p => p[0]);
  const ys = visiblePoints.map(p => p[1]);

  const xmin = Math.min(...xs);
  const xmax = Math.max(...xs);
  const ymin = Math.min(...ys);
  const ymax = Math.max(...ys);

  const dx = Math.max(1, xmax - xmin);
  const dy = Math.max(1, ymax - ymin);

  const raw = timeline.map(row => {
    if (!row.shuttle?.visible) return null;

    const p = point(row.shuttle.pixel);
    if (!p) return null;

    const tx = (p[0] - xmin) / dx;
    const ty = (p[1] - ymin) / dy;

    return [
      0.65 + tx * (COURT_W - 1.30),
      1.00 + (1 - ty) * (COURT_L - 2.00)
    ];
  });

  // Smooth only the 2D schematic representation.
  // Missing detections stay missing.
  shuttleDisplay = raw.map((value, i) => {
    if (!value) return null;

    const sum = [0, 0];
    let total = 0;

    for (let offset = -2; offset <= 2; offset++) {
      const j = i + offset;

      if (j < 0 || j >= raw.length || !raw[j]) {
        continue;
      }

      let connected = true;

      for (let k = Math.min(i, j); k <= Math.max(i, j); k++) {
        if (!raw[k]) {
          connected = false;
          break;
        }
      }

      if (!connected) continue;

      const weight = 3 - Math.abs(offset);

      sum[0] += raw[j][0] * weight;
      sum[1] += raw[j][1] * weight;
      total += weight;
    }

    return total
      ? [sum[0] / total, sum[1] / total]
      : value;
  });
}


// ============================================================
// DRAW VIDEO OVERLAYS
// ============================================================

function drawVideoBox(box, color, label, estimated) {
  if (!box || box[2] <= 0 || box[3] <= 0) return;

  const [x, y, w, h] = box;

  vctx.save();

  vctx.strokeStyle = color;
  vctx.lineWidth = 2;

  if (estimated) {
    vctx.setLineDash([8, 6]);
  }

  vctx.strokeRect(x, y, w, h);
  vctx.setLineDash([]);

  vctx.fillStyle = color;
  vctx.font = "bold 15px Segoe UI";
  vctx.fillText(
    label.toUpperCase() + (estimated ? " ~" : ""),
    x,
    Math.max(18, y - 7)
  );

  vctx.restore();
}

function drawVideoOverlay() {
  vctx.clearRect(
    0, 0,
    videoCanvas.width, videoCanvas.height
  );

  for (const [side, checkbox, color] of [
    ["Near", "nearBox", COLORS.near],
    ["Far", "farBox", COLORS.far]
  ]) {
    if (!focusAllows(side) || !$(checkbox).checked) {
      continue;
    }

    const series = players[side].boxes;

    drawVideoBox(
      series.values[currentFrame],
      color,
      side,
      !series.measured[currentFrame]
    );
  }

  if ($("shuttleTrailVideo").checked) {
    const start = Math.max(1, currentFrame - 8);

    for (let i = start; i <= currentFrame; i++) {
      const a = timeline[i - 1].shuttle;
      const b = timeline[i].shuttle;

      if (!a.visible || !b.visible) continue;

      const p = point(a.pixel);
      const q = point(b.pixel);

      if (!p || !q || distance(p, q) > 140) {
        continue;
      }

      const alpha = 0.15 +
        0.7 * (1 - (currentFrame - i) / 8);

      vctx.beginPath();
      vctx.moveTo(...p);
      vctx.lineTo(...q);

      vctx.strokeStyle =
        `rgba(138,246,170,${alpha})`;

      vctx.lineWidth = 2;
      vctx.stroke();
    }
  }

  if ($("shuttleVideo").checked) {
    const shuttle = timeline[currentFrame].shuttle;

    if (shuttle.visible) {
      const p = point(shuttle.pixel);

      if (p) {
        vctx.beginPath();
        vctx.arc(p[0], p[1], 9, 0, Math.PI * 2);
        vctx.strokeStyle = "#06110d";
        vctx.lineWidth = 5;
        vctx.stroke();

        vctx.beginPath();
        vctx.arc(p[0], p[1], 9, 0, Math.PI * 2);
        vctx.strokeStyle = COLORS.shuttle;
        vctx.lineWidth = 2;
        vctx.stroke();
      }
    }
  }
}


// ============================================================
// DRAW TACTICAL COURT
// ============================================================

let toCourtScreen = null;

function drawCourtBase() {
  const w = courtCanvas.width;
  const h = courtCanvas.height;

  ctx.globalAlpha = 1;
  ctx.fillStyle = COLORS.bg;
  ctx.fillRect(0, 0, w, h);

  const ppm = Math.min(
    (w - 115) / COURT_W,
    (h - 145) / COURT_L
  );

  const courtWidth = COURT_W * ppm;
  const courtHeight = COURT_L * ppm;

  const left = (w - courtWidth) / 2;
  const top = (h - courtHeight) / 2;

  toCourtScreen = ([x, y]) => [
    left + x * ppm,
    top + (COURT_L - y) * ppm
  ];

  ctx.fillStyle = COLORS.court;
  ctx.fillRect(left, top, courtWidth, courtHeight);

  function line(a, b, color = COLORS.lines, width = 1) {
    const p = toCourtScreen(a);
    const q = toCourtScreen(b);

    ctx.beginPath();
    ctx.moveTo(...p);
    ctx.lineTo(...q);
    ctx.strokeStyle = color;
    ctx.lineWidth = width;
    ctx.stroke();
  }

  const side = (COURT_W - 5.18) / 2;
  const ns = 6.70 - 1.98;
  const fs = 6.70 + 1.98;

  ctx.strokeStyle = COLORS.lines;
  ctx.lineWidth = 2;
  ctx.strokeRect(left, top, courtWidth, courtHeight);

  line([side, 0], [side, COURT_L]);
  line([COURT_W-side, 0], [COURT_W-side, COURT_L]);

  line([0, .76], [COURT_W, .76]);
  line([0, COURT_L-.76], [COURT_W, COURT_L-.76]);

  line([0, ns], [COURT_W, ns]);
  line([0, fs], [COURT_W, fs]);

  line([3.05, 0], [3.05, ns]);
  line([3.05, fs], [3.05, COURT_L]);

  line([0, 6.70], [COURT_W, 6.70], COLORS.net, 2);

  ctx.fillStyle = COLORS.white;
  ctx.font = "bold 16px Segoe UI";
  ctx.fillText("TACTICAL VIEW", 23, 34);

  ctx.fillStyle = COLORS.muted;
  ctx.font = "12px Segoe UI";
  ctx.fillText("Player ground positions", 23, 53);
  ctx.fillText("Shuttle: schematic only", 23, 690);
}

function drawMovementTrail(values, color) {
  if (!$("playerTrails").checked) return;

  for (
    let i = Math.max(1, currentFrame - 13);
    i <= currentFrame;
    i++
  ) {
    const a = values[i - 1];
    const b = values[i];

    if (!a || !b || distance(a, b) > 1.1) {
      continue;
    }

    const alpha = 0.15 +
      0.7 * (1 - (currentFrame - i) / 13);

    ctx.globalAlpha = alpha;

    ctx.beginPath();
    ctx.moveTo(...toCourtScreen(a));
    ctx.lineTo(...toCourtScreen(b));
    ctx.strokeStyle = color;
    ctx.lineWidth = 3;
    ctx.stroke();
  }

  ctx.globalAlpha = 1;
}

function drawPlayerMarker(p, color) {
  if (!p) return;

  const [x, y] = toCourtScreen(p);

  ctx.beginPath();
  ctx.arc(x, y, 8, 0, Math.PI * 2);
  ctx.fillStyle = color;
  ctx.fill();

  ctx.strokeStyle = "#fff";
  ctx.lineWidth = 1.5;
  ctx.stroke();
}

function drawShuttleIcon(p, previous) {
  const [x, y] = toCourtScreen(p);

  let angle = -Math.PI / 2;

  if (previous) {
    const [px, py] = toCourtScreen(previous);
    const dx = x - px;
    const dy = y - py;

    if (Math.hypot(dx, dy) > 1) {
      angle = Math.atan2(dy, dx);
    }
  }

  ctx.save();
  ctx.translate(x, y);
  ctx.rotate(angle);

  // Cork points forward.
  ctx.fillStyle = "#fff";

  ctx.beginPath();
  ctx.moveTo(-3, 0);
  ctx.lineTo(-12, -6);
  ctx.lineTo(-12, 6);
  ctx.closePath();
  ctx.fill();

  ctx.beginPath();
  ctx.arc(5, 0, 3.5, 0, Math.PI * 2);
  ctx.fill();

  ctx.restore();
}

function drawTactical() {
  drawCourtBase();

  for (const [side, checkbox, color] of [
    ["Near", "nearMarker", COLORS.near],
    ["Far", "farMarker", COLORS.far]
  ]) {
    if (!focusAllows(side) || !$(checkbox).checked) {
      continue;
    }

    const values = players[side].court.values;

    drawMovementTrail(values, color);
    drawPlayerMarker(values[currentFrame], color);
  }

  if ($("courtShuttleTrail").checked) {
    const start = Math.max(1, currentFrame - 9);

    for (let i = start; i <= currentFrame; i++) {
      const a = shuttleDisplay[i - 1];
      const b = shuttleDisplay[i];

      if (!a || !b) continue;

      ctx.globalAlpha = 0.15 +
        0.65 * (1 - (currentFrame - i) / 9);

      ctx.beginPath();
      ctx.moveTo(...toCourtScreen(a));
      ctx.lineTo(...toCourtScreen(b));
      ctx.strokeStyle = COLORS.white;
      ctx.lineWidth = 2;
      ctx.stroke();
    }

    ctx.globalAlpha = 1;
  }

  if ($("courtShuttle").checked) {
    const p = shuttleDisplay[currentFrame];

    if (p) {
      let previous = null;

      // Only use continuous history for orientation.
      for (
        let i = currentFrame - 1;
        i >= Math.max(0, currentFrame - 5);
        i--
      ) {
        if (!shuttleDisplay[i]) break;

        previous = shuttleDisplay[i];

        if (distance(previous, p) > 0.03) {
          break;
        }
      }

      drawShuttleIcon(p, previous);
    }
  }
}


// ============================================================
// VIDEO FIT
// ============================================================

function fitVideo() {
  if (!videoInfo.width || !videoInfo.height) {
    return;
  }

  const area = $("videoArea");
  const availableWidth = area.clientWidth;
  const availableHeight = area.clientHeight;

  if (availableWidth <= 0 || availableHeight <= 0) {
    return;
  }

  const scale = Math.min(
    availableWidth / videoInfo.width,
    availableHeight / videoInfo.height
  );

  const box = $("videoBox");

  box.style.width =
    `${Math.floor(videoInfo.width * scale)}px`;

  box.style.height =
    `${Math.floor(videoInfo.height * scale)}px`;
}


// ============================================================
// FRAME DISPLAY
// ============================================================

function updateInspector() {
  const row = timeline[currentFrame];

  const lastTime = timeline[timeline.length - 1].time;

  $("timeLabel").textContent =
    `${row.time.toFixed(2)}s / ${lastTime.toFixed(2)}s`;

  $("frameLabel").textContent =
    `Frame ${row.frame} / ${timeline[timeline.length - 1].frame}`;

  $("seek").value = currentFrame;

  $("inspectTime").textContent =
    `${row.time.toFixed(2)}s`;

  $("inspectShuttle").textContent =
    row.shuttle.visible ? "Visible*" : "Missing";

  $("inspectNear").textContent = formatPosition(
    players.Near.court.values[currentFrame]
  );

  $("inspectFar").textContent = formatPosition(
    players.Far.court.values[currentFrame]
  );
}

function renderFrame() {
  if (!timeline.length) return;

  updateInspector();

  // Always redraw overlays so toggles apply immediately.
  drawVideoOverlay();

  if (view !== "video") {
    drawTactical();
  }

  const image = frameImages[currentFrame];

  if (frameReady[currentFrame] && image) {
    $("frameImage").src = image.src;
    $("frameImage").style.visibility = "visible";
    $("loading").hidden = true;
  } else {
    $("frameImage").style.visibility = "hidden";
    $("loading").hidden = false;

    $("loading").textContent =
      frameFailed[currentFrame]
        ? "Could not load this frame"
        : "Loading frame...";
  }
}

function setFrame(index) {
  currentFrame = clamp(
    Math.round(index),
    0,
    timeline.length - 1
  );

  renderFrame();
}


// ============================================================
// VIEW MODES
// ============================================================

function setView(nextView) {
  view = nextView;

  const workspace = $("workspace");

  workspace.className = "workspace " + (
    nextView === "video"
      ? "video-only"
      : nextView === "court"
        ? "court-only"
        : "split"
  );

  $("videoSurface").hidden = nextView === "court";
  $("courtSurface").hidden = nextView === "video";

  document.querySelectorAll("[data-view]").forEach(btn => {
    btn.classList.toggle(
      "active",
      btn.dataset.view === nextView
    );
  });

  requestAnimationFrame(() => {
    fitVideo();
    renderFrame();
  });
}


// ============================================================
// SETTINGS PANEL
// ============================================================

function showSettings(open) {
  $("settingsPanel").hidden = !open;

  $("settingsButton").classList.toggle("active", open);

  $("settingsButton").setAttribute(
    "aria-expanded",
    String(open)
  );
}


// ============================================================
// IMAGE PRELOADING
// ============================================================

function preloadFrames() {
  const n = timeline.length;

  frameImages = Array(n).fill(null);
  frameReady = Array(n).fill(false);
  frameFailed = Array(n).fill(false);

  let nextIndex = 0;
  let completed = 0;

  function loadNext() {
    if (nextIndex >= n) return;

    const index = nextIndex++;
    const image = new Image();

    image.onload = () => {
      frameReady[index] = true;
      completed++;

      $("status").textContent =
        `Frames ready: ${completed}/${n}`;

      if (index === currentFrame) {
        renderFrame();
      }

      loadNext();
    };

    image.onerror = () => {
      frameFailed[index] = true;
      completed++;

      $("status").textContent =
        `Frames checked: ${completed}/${n}`;

      if (index === currentFrame) {
        renderFrame();
      }

      loadNext();
    };

    frameImages[index] = image;

    image.src =
      `/frames/${encodeURIComponent(currentRallyId)}/` +
      `${String(index).padStart(4,"0")}.jpg`;
  }

  for (let i = 0; i < Math.min(6, n); i++) {
    loadNext();
  }
}


// ============================================================
// PLAYBACK ENGINE
// ============================================================

function updatePlaybackUI() {
  $("play").textContent =
    isPlaying ? "❚❚ Pause" : "▶ Play";

  $("playStatus").textContent =
    isPlaying
      ? `Playing ${playbackSpeed}×`
      : "Paused";
}

function pause() {
  isPlaying = false;
  frameAccumulator = 0;
  lastTick = null;
  updatePlaybackUI();
}

function togglePlay() {
  if (!timeline.length) return;

  if (isPlaying) {
    pause();
    return;
  }

  if (currentFrame === timeline.length - 1) {
    setFrame(0);
  }

  isPlaying = true;
  frameAccumulator = 0;
  lastTick = null;

  updatePlaybackUI();
}

function animationTick(timestamp) {
  if (isPlaying && timeline.length) {
    if (lastTick === null) {
      lastTick = timestamp;
    }

    const dt = Math.min(
      (timestamp - lastTick) / 1000,
      0.1
    );

    frameAccumulator += dt * playbackSpeed;

    const secondsPerFrame = 1 / metadata.fps;

    while (frameAccumulator >= secondsPerFrame) {
      if (currentFrame >= timeline.length - 1) {
        pause();
        break;
      }

      // Never jump ahead of frames still loading.
      if (!frameReady[currentFrame + 1]) {
        frameAccumulator = Math.min(
          frameAccumulator,
          secondsPerFrame
        );
        break;
      }

      frameAccumulator -= secondsPerFrame;
      setFrame(currentFrame + 1);
    }
  }

  lastTick = timestamp;

  requestAnimationFrame(animationTick);
}


// ============================================================
// EVENTS
// ============================================================

$("play").addEventListener("click", togglePlay);

$("back").addEventListener("click", () => {
  pause();
  setFrame(currentFrame - 1);
});

$("next").addEventListener("click", () => {
  pause();
  setFrame(currentFrame + 1);
});

$("restart").addEventListener("click", () => {
  pause();
  setFrame(0);
});

$("seek").addEventListener("input", event => {
  pause();
  setFrame(Number(event.target.value));
});

$("speed").addEventListener("change", event => {
  playbackSpeed = Number(event.target.value);
  updatePlaybackUI();
});

$("focus").addEventListener("change", renderFrame);

document.querySelectorAll("[data-view]").forEach(button => {
  button.addEventListener("click", () => {
    setView(button.dataset.view);
  });
});

$("settingsButton").addEventListener("click", () => {
  showSettings($("settingsPanel").hidden);
});

$("closeSettings").addEventListener("click", () => {
  showSettings(false);
});

document.querySelectorAll(
  '.settings input[type="checkbox"]'
).forEach(input => {
  input.addEventListener("change", renderFrame);
});

document.addEventListener("keydown", event => {
  const tag = document.activeElement?.tagName;

  if (
    ["INPUT", "SELECT", "TEXTAREA"].includes(tag)
  ) {
    return;
  }

  if (event.code === "Escape") {
    showSettings(false);
    return;
  }

  if (!timeline.length) return;

  if (event.code === "Space") {
    event.preventDefault();
    togglePlay();
  }

  if (event.code === "ArrowRight") {
    event.preventDefault();
    pause();
    setFrame(currentFrame + 1);
  }

  if (event.code === "ArrowLeft") {
    event.preventDefault();
    pause();
    setFrame(currentFrame - 1);
  }

  if (event.code === "Home") {
    event.preventDefault();
    pause();
    setFrame(0);
  }

  if (event.code === "End") {
    event.preventDefault();
    pause();
    setFrame(timeline.length - 1);
  }
});

window.addEventListener("resize", fitVideo);

const observer = new ResizeObserver(fitVideo);
observer.observe($("videoArea"));



// ============================================================
// MULTI-RALLY LIBRARY — V0.3
// ============================================================

async function loadRallyCatalog() {
  const response = await fetch("/api/rallies");

  if (!response.ok) {
    throw new Error(
      `Rally catalog request failed: ${response.status}`
    );
  }

  const data = await response.json();

  rallyCatalog = data.rallies;

  if (!Array.isArray(rallyCatalog) || !rallyCatalog.length) {
    throw new Error("No processed rallies registered");
  }

  const requested = rallyCatalog.find(
    rally => rally.id === currentRallyId
  );

  if (!requested) {
    currentRallyId = data.default_rally;
  }

  const selector = $("rallySelect");

  selector.replaceChildren();

  for (const rally of rallyCatalog) {
    const option = document.createElement("option");
    option.value = rally.id;
    option.textContent = rally.title;

    selector.appendChild(option);
  }

  selector.value = currentRallyId;

  selector.addEventListener("change", () => {
    const nextRally = selector.value;

    // Reloading resets playback and guarantees that
    // no data or image from the previous rally remains.
    window.location.assign(
      `/?rally=${encodeURIComponent(nextRally)}`
    );
  });

  // Keep the URL shareable and in sync.
  const url = new URL(window.location.href);
  url.searchParams.set("rally", currentRallyId);
  window.history.replaceState({}, "", url);

  const selected = rallyCatalog.find(
    rally => rally.id === currentRallyId
  );

  $("status").textContent =
    `Loading ${selected.title}...`;
}

// ============================================================
// INITIALIZATION
// ============================================================

async function initialize() {
  try {
    await loadRallyCatalog();

    const response = await fetch(
      `/api/timeline?rally=${encodeURIComponent(currentRallyId)}`
    );

    if (!response.ok) {
      throw new Error(
        `Timeline request failed: ${response.status}`
      );
    }

    const data = await response.json();

    timeline = data.timeline;
    metadata = data.metadata;
    videoInfo = data.video;

    if (!timeline.length || !metadata.fps) {
      throw new Error("Invalid replay dataset");
    }

    sourceStart = timeline[0].frame;
    sourceEnd = timeline[timeline.length - 1].frame;

    videoCanvas.width = videoInfo.width;
    videoCanvas.height = videoInfo.height;

    $("videoSize").textContent =
      `${videoInfo.width} × ${videoInfo.height}`;

    $("seek").max = timeline.length - 1;

    preparePlayers();
    prepareShuttle();

    fitVideo();
    setView("split");

    preloadFrames();
    renderFrame();
    updatePlaybackUI();

    requestAnimationFrame(animationTick);

  } catch (error) {
    console.error(error);

    $("status").textContent = "Replay unavailable";
    $("loading").textContent = error.message;
  }
}

initialize();
