
"use strict";

// ============================================================
// TACTICAL COURT — RESPONSIVE V2
// ============================================================

// Set false if you prefer a portrait court in every view.
const AUTO_LANDSCAPE_COURT_ONLY = true;

let toCourtScreen = null;
let courtLandscape = false;

// ============================================================
// CANVAS SIZE
// ============================================================

function prepareCourtCanvas() {
  const rect = courtCanvas.getBoundingClientRect();

  // Draw using CSS pixels. Scale the backing store separately
  // for sharp rendering on high-DPI displays.
  const w = Math.max(1, Math.round(rect.width));
  const h = Math.max(1, Math.round(rect.height));

  const dpr = Math.min(
    Math.max(window.devicePixelRatio || 1, 1),
    2
  );

  const pixelWidth = Math.round(w * dpr);
  const pixelHeight = Math.round(h * dpr);

  if (
    courtCanvas.width !== pixelWidth ||
    courtCanvas.height !== pixelHeight
  ) {
    courtCanvas.width = pixelWidth;
    courtCanvas.height = pixelHeight;
  }

  // All existing drawing functions can continue to use
  // logical CSS-pixel coordinates.
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

  return { w, h };
}

// ============================================================
// DRAW COURT GEOMETRY
// ============================================================

function drawCourtBase() {
  const { w, h } = prepareCourtCanvas();

  ctx.globalAlpha = 1;
  ctx.fillStyle = COLORS.bg;
  ctx.fillRect(0, 0, w, h);

  // Keep the normal portrait orientation in Split view.
  // Rotate only the screen representation when the court
  // has substantially more width than height.
  courtLandscape =
    AUTO_LANDSCAPE_COURT_ONLY &&
    view === "court" &&
    w > h * 1.35;

  // Dedicated space for labels that live in CSS rather
  // than inside the canvas.
  const topReserve = Math.min(50, h * 0.12);
  const bottomReserve = Math.min(34, h * 0.08);
  const sideReserve = Math.min(30, w * 0.06);

  const availableW = Math.max(1, w - sideReserve * 2);
  const availableH = Math.max(
    1,
    h - topReserve - bottomReserve
  );

  // The dimensions on screen depend on orientation,
  // but the actual court coordinates never change.
  const displayedWidth = courtLandscape
    ? COURT_L
    : COURT_W;

  const displayedHeight = courtLandscape
    ? COURT_W
    : COURT_L;

  const ppm = Math.min(
    availableW / displayedWidth,
    availableH / displayedHeight
  );

  const courtWidth = displayedWidth * ppm;
  const courtHeight = displayedHeight * ppm;

  const left = (w - courtWidth) / 2;

  const top =
    topReserve +
    (availableH - courtHeight) / 2;

  // Near player = small physical y coordinate.
  // Far player  = large physical y coordinate.
  //
  // Portrait: near at bottom, far at top.
  // Landscape: near at left, far at right.
  //
  // Every overlay continues using this same mapping.
  if (courtLandscape) {
  // Far player on the left, Near player on the right.
  // Rotate the court view 90 degrees clockwise.
  // Physical tracking coordinates stay unchanged.
  toCourtScreen = ([x, y]) => [
      left + (COURT_L - y) * ppm,
      top + (COURT_W - x) * ppm
    ];
  } else {
    // Standard portrait orientation.
    toCourtScreen = ([x, y]) => [
      left + x * ppm,
      top + (COURT_L - y) * ppm
    ];
  }

  ctx.fillStyle = COLORS.court;
  ctx.fillRect(
    left,
    top,
    courtWidth,
    courtHeight
  );

  function line(
    a,
    b,
    color = COLORS.lines,
    width = 1
  ) {
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

  ctx.strokeRect(
    left,
    top,
    courtWidth,
    courtHeight
  );

  // Singles sidelines.
  line([side, 0], [side, COURT_L]);

  line(
    [COURT_W - side, 0],
    [COURT_W - side, COURT_L]
  );

  // Doubles long service lines.
  line([0, .76], [COURT_W, .76]);

  line(
    [0, COURT_L - .76],
    [COURT_W, COURT_L - .76]
  );

  // Short service lines.
  line([0, ns], [COURT_W, ns]);
  line([0, fs], [COURT_W, fs]);

  // Centre service lines.
  line([3.05, 0], [3.05, ns]);
  line([3.05, fs], [3.05, COURT_L]);

  // Net.
  line(
    [0, 6.70],
    [COURT_W, 6.70],
    COLORS.net,
    2
  );
}

// ============================================================
// PLAYER MOVEMENT TRAILS
// ============================================================

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

    const alpha =
      0.15 +
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

// ============================================================
// PLAYER MARKERS
// ============================================================

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

// ============================================================
// SHUTTLE ICON
// ============================================================

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

// ============================================================
// FULL TACTICAL FRAME
// ============================================================

function drawTactical() {
  drawCourtBase();

  // Players.
  for (const [side, checkbox, color] of [
    ["Near", "nearMarker", COLORS.near],
    ["Far", "farMarker", COLORS.far]
  ]) {
    if (
      !focusAllows(side) ||
      !$(checkbox).checked
    ) {
      continue;
    }

    const values = players[side].court.values;

    drawMovementTrail(values, color);

    drawPlayerMarker(
      values[currentFrame],
      color
    );
  }

  // Shuttle trail.
  if ($("courtShuttleTrail").checked) {
    const start = Math.max(
      1,
      currentFrame - 9
    );

    for (
      let i = start;
      i <= currentFrame;
      i++
    ) {
      const a = shuttleDisplay[i - 1];
      const b = shuttleDisplay[i];

      if (!a || !b) continue;

      ctx.globalAlpha =
        0.15 +
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

  // Shuttle icon.
  if ($("courtShuttle").checked) {
    const p = shuttleDisplay[currentFrame];

    if (p) {
      let previous = null;

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
// REDRAW WHEN THE WORKSPACE RESIZES
// ============================================================

// Playback already redraws the court on every frame.
// ResizeObserver adds redraws when the panel changes size,
// including switching between Split and Court-only views.

let courtResizeScheduled = false;

function scheduleCourtRedraw() {
  if (courtResizeScheduled) return;

  courtResizeScheduled = true;

  requestAnimationFrame(() => {
    courtResizeScheduled = false;

    // Don't render before the rally data is initialized.
    if (
      typeof players === "undefined" ||
      !players?.Near?.court?.values ||
      !players?.Far?.court?.values ||
      typeof shuttleDisplay === "undefined" ||
      typeof currentFrame === "undefined"
    ) {
      return;
    }

    // Hidden canvases have no visible drawing area.
    if (
      courtCanvas.clientWidth === 0 ||
      courtCanvas.clientHeight === 0
    ) {
      return;
    }

    drawTactical();
  });
}

if (typeof ResizeObserver !== "undefined") {
  const courtResizeObserver = new ResizeObserver(
    scheduleCourtRedraw
  );

  courtResizeObserver.observe(
    courtCanvas.parentElement
  );
} else {
  window.addEventListener(
    "resize",
    scheduleCourtRedraw
  );
}
