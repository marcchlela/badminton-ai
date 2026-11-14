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
