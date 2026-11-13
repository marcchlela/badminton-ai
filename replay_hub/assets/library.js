"use strict";

const $ = id => document.getElementById(id);

let rallies = [];

function formatDuration(seconds) {
  return Number(seconds).toFixed(1) + "s";
}

function coverage(rally) {
  if (!rally.frame_count) return 0;

  return (
    100 * rally.shuttle_visible_frames /
    rally.frame_count
  );
}

function createMetric(label, value) {
  const wrap = document.createElement("div");
  wrap.className = "metric";

  const name = document.createElement("span");
  name.textContent = label;

  const number = document.createElement("strong");
  number.textContent = value;

  wrap.append(name, number);
  return wrap;
}

function createCard(rally) {
  const card = document.createElement("article");
  card.className = "card";

  const thumbnail = document.createElement("div");
  thumbnail.className = "thumbnail";

  const img = document.createElement("img");
  img.alt = `Preview of ${rally.title}`;
  img.loading = "lazy";
  img.src =
    `/frames/${encodeURIComponent(rally.id)}/0000.jpg`;

  const duration = document.createElement("span");
  duration.className = "duration";
  duration.textContent =
    formatDuration(rally.duration_seconds);

  thumbnail.append(img, duration);

  const body = document.createElement("div");
  body.className = "card-body";

  const title = document.createElement("h3");
  title.className = "card-title";
  title.textContent = rally.title;

  const description = document.createElement("p");
  description.className = "description";
  description.textContent =
    rally.description || "Processed badminton rally";

  const metrics = document.createElement("div");
  metrics.className = "metrics";

  metrics.append(
    createMetric("Frames", rally.frame_count),
    createMetric("Frame rate", `${rally.fps} FPS`),
    createMetric(
      "Shuttle reported",
      `${rally.shuttle_visible_frames}/${rally.frame_count}`
    ),
    createMetric(
      "Shuttle coverage",
      `${coverage(rally).toFixed(1)}%`
    )
  );

  const footer = document.createElement("div");
  footer.className = "card-footer";

  const tag = document.createElement("span");
  tag.className = "tag";
  tag.textContent = rally.has_players
    ? "Players + Shuttle"
    : "Shuttle only";

  const link = document.createElement("a");
  link.className = "open-link";
  link.href =
    `/?rally=${encodeURIComponent(rally.id)}`;
  link.textContent = "Open Replay";

  footer.append(tag, link);
  body.append(title, description, metrics, footer);
  card.append(thumbnail, body);

  return card;
}

function renderLibrary() {
  const search = $("search").value
    .trim().toLowerCase();

  const sort = $("sort").value;

  let filtered = rallies.filter(rally => {
    return (
      rally.title.toLowerCase().includes(search) ||
      rally.id.toLowerCase().includes(search) ||
      (rally.description || "")
        .toLowerCase().includes(search)
    );
  });

  filtered = [...filtered];

  if (sort === "title") {
    filtered.sort((a, b) =>
      a.title.localeCompare(b.title)
    );
  } else if (sort === "duration-desc") {
    filtered.sort((a, b) =>
      b.duration_seconds - a.duration_seconds
    );
  } else if (sort === "duration-asc") {
    filtered.sort((a, b) =>
      a.duration_seconds - b.duration_seconds
    );
  } else if (sort === "coverage-desc") {
    filtered.sort((a, b) =>
      coverage(b) - coverage(a)
    );
  }

  const grid = $("rallyGrid");
  grid.replaceChildren();

  filtered.forEach(rally => {
    grid.appendChild(createCard(rally));
  });

  $("resultCount").textContent =
    `${filtered.length} of ${rallies.length} rallies`;

  $("emptyState").hidden = filtered.length !== 0;
}

async function initialize() {
  try {
    const response = await fetch("/api/rallies");

    if (!response.ok) {
      throw new Error(
        `Catalog request failed: ${response.status}`
      );
    }

    const data = await response.json();

    rallies = data.rallies || [];

    $("totalRallies").textContent = rallies.length;

    const duration = rallies.reduce(
      (total, rally) =>
        total + rally.duration_seconds,
      0
    );

    $("totalDuration").textContent =
      formatDuration(duration);

    renderLibrary();

  } catch (error) {
    console.error(error);
    $("resultCount").textContent =
      "Library unavailable";

    $("emptyState").hidden = false;
    $("emptyState").textContent =
      "Failed to load rallies: " + error.message;
  }
}

$("search").addEventListener("input", renderLibrary);
$("sort").addEventListener("change", renderLibrary);

initialize();
