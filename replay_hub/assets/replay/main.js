"use strict";

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

    // Load manual annotations for the selected rally.
    // Annotation API errors must not block video playback.
    if (typeof initAnnotations === "function") {
      void initAnnotations();
    }

    requestAnimationFrame(animationTick);

  } catch (error) {
    console.error(error);

    $("status").textContent = "Replay unavailable";
    $("loading").textContent = error.message;
  }
}

initialize();
