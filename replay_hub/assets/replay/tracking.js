"use strict";

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
