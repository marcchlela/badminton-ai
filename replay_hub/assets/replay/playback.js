"use strict";

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
