"use strict";

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
