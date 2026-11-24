
"use strict";

// ============================================================
// BADMINTON AI — EVENT ANNOTATIONS V0.4.2
// ============================================================

let rallyEvents = [];
let annotationsReady = false;
let editingEventId = null;
let eventsBusy = false;

// ============================================================
// API
// ============================================================

async function eventRequest(url, options = {}) {
  const response = await fetch(url, {
    cache: "no-store",
    ...options
  });

  let result;

  try {
    result = await response.json();
  } catch {
    throw new Error(
      `Invalid server response (${response.status})`
    );
  }

  if (!response.ok) {
    throw new Error(
      result.error ||
      `Request failed (${response.status})`
    );
  }

  return result;
}

function eventsUrl() {
  return (
    "/api/events?rally=" +
    encodeURIComponent(currentRallyId)
  );
}

// ============================================================
// UTILITY FUNCTIONS
// ============================================================

const EVENT_NAMES = {
  contact: "Racket contact",
  bookmark: "Bookmark",
  rally_event: "Rally event"
};

const CONFIDENCE_NAMES = {
  confirmed: "Confirmed",
  approximate: "Approximate",
  uncertain: "Uncertain"
};

function normalizedConfidence(event) {
  return CONFIDENCE_NAMES[event.confidence]
    ? event.confidence
    : "uncertain";
}

function eventLabel(event) {
  const parts = [
    EVENT_NAMES[event.kind] || event.kind
  ];

  if (event.player !== "Unknown") {
    parts.push(event.player);
  }

  if (
    event.kind === "contact" &&
    event.shot !== "unknown"
  ) {
    parts.push(
      event.shot.charAt(0).toUpperCase() +
      event.shot.slice(1)
    );
  }

  return parts.join(" · ");
}

function showEventMessage(message, error = false) {
  const element = $("eventMessage");

  element.textContent = message;
  element.classList.toggle("error", error);
}

function getEventById(id) {
  return rallyEvents.find(
    event => event.id === id
  ) || null;
}

function eventFrameText() {
  if (!timeline.length) return "—";

  const row = timeline[currentFrame];

  return (
    `Frame ${row.frame} · ${row.time.toFixed(2)}s`
  );
}

function updateEventFrameLabel() {
  if (!annotationsReady || !timeline.length) {
    return;
  }

  $("eventCurrentFrame").textContent =
    eventFrameText();

  updateCurrentEventHighlight();
}

// ============================================================
// FILTERING
// ============================================================

function filteredEvents() {
  const kind = $("eventFilterKind").value;
  const player = $("eventFilterPlayer").value;
  const shot = $("eventFilterShot").value;

  return rallyEvents.filter(event => {
    if (kind !== "all" && event.kind !== kind) {
      return false;
    }

    if (
      player !== "all" &&
      event.player !== player
    ) {
      return false;
    }

    if (
      shot !== "all" &&
      event.shot !== shot
    ) {
      return false;
    }

    return true;
  });
}

function createFilterUI() {
  const filters = document.createElement("div");
  filters.className = "event-filters";

  filters.innerHTML = `
    <label>
      Event
      <select id="eventFilterKind">
        <option value="all">All events</option>
        <option value="contact">Racket contacts</option>
        <option value="bookmark">Bookmarks</option>
        <option value="rally_event">Rally events</option>
      </select>
    </label>

    <label>
      Player
      <select id="eventFilterPlayer">
        <option value="all">All players</option>
        <option value="Near">Near</option>
        <option value="Far">Far</option>
        <option value="Unknown">Unknown</option>
      </select>
    </label>

    <label>
      Shot
      <select id="eventFilterShot">
        <option value="all">All shot types</option>
        <option value="unknown">Unknown</option>
        <option value="serve">Serve</option>
        <option value="clear">Clear</option>
        <option value="drop">Drop</option>
        <option value="smash">Smash</option>
        <option value="lift">Lift</option>
        <option value="drive">Drive</option>
        <option value="net">Net shot</option>
        <option value="push">Push</option>
        <option value="other">Other</option>
      </select>
    </label>

    <div class="event-filter-summary"
         id="eventFilterSummary">
      Showing all events
    </div>
  `;

  const heading = document.querySelector(
    ".event-list-heading"
  );

  heading.before(filters);

  for (const id of [
    "eventFilterKind",
    "eventFilterPlayer",
    "eventFilterShot"
  ]) {
    $(id).addEventListener(
      "change",
      renderAllEvents
    );
  }
}

// ============================================================
// CONFIDENCE AND FORM CONTROLS
// ============================================================

function createConfidenceUI() {
  const field = document.createElement("label");
  field.className = "event-confidence-field";

  field.innerHTML = `
    Annotation confidence
    <select id="eventConfidence">
      <option value="uncertain">
        Uncertain — needs review
      </option>
      <option value="approximate">
        Approximate — frame or shot uncertain
      </option>
      <option value="confirmed">
        Confirmed — manually reviewed
      </option>
    </select>
    <span class="event-confidence-note">
      Confirmed means you have checked the evidence.
      It does not mean an AI model verified the event.
    </span>
  `;

  const notes = $("eventNotes").closest("label");
  notes.before(field);

  // Turn the existing save button into an action row.
  const save = $("saveEvent");

  const actions = document.createElement("div");
  actions.className = "event-form-actions";

  save.before(actions);
  actions.appendChild(save);

  const cancel = document.createElement("button");
  cancel.type = "button";
  cancel.id = "cancelEventEdit";
  cancel.className = "event-cancel";
  cancel.textContent = "Cancel edit";
  cancel.hidden = true;

  actions.appendChild(cancel);

  cancel.addEventListener("click", () => {
    resetEventForm();
    showEventMessage("Edit cancelled.");
  });

  const shortcuts = document.createElement("div");
  shortcuts.className = "event-shortcuts";

  shortcuts.innerHTML = `
    <strong>Keyboard shortcuts</strong>
    <div>
      <kbd>B</kbd> Bookmark
      <kbd>C</kbd> Contact
      <kbd>[</kbd> Previous event
      <kbd>]</kbd> Next event
    </div>
  `;

  $("eventList").after(shortcuts);
}

function updateShotField() {
  const contact =
    $("eventKind").value === "contact";

  $("eventShotField").hidden = !contact;

  if (!contact) {
    $("eventShot").value = "unknown";
  }
}

function resetEventForm() {
  editingEventId = null;

  $("eventKind").value = "contact";
  $("eventPlayer").value = "Near";
  $("eventShot").value = "unknown";
  $("eventConfidence").value = "uncertain";
  $("eventNotes").value = "";

  $("saveEvent").textContent =
    "Save annotation";

  $("cancelEventEdit").hidden = true;

  updateShotField();
}

function loadEventForEditing(event) {
  if (!event) return;

  pause();
  setFrame(event.local_frame);

  editingEventId = event.id;

  $("eventKind").value = event.kind;
  $("eventPlayer").value = event.player;
  $("eventShot").value = event.shot;
  $("eventConfidence").value =
    normalizedConfidence(event);

  $("eventNotes").value = event.notes || "";

  $("saveEvent").textContent =
    "Update annotation";

  $("cancelEventEdit").hidden = false;

  updateShotField();
  showEventPanel(true);

  showEventMessage(
    "Editing annotation. Seek to another frame " +
    "if you want to correct its timing."
  );
}

// ============================================================
// PLAYBACK NAVIGATION
// ============================================================

function jumpToEvent(event) {
  if (!event || !timeline.length) return;

  pause();
  setFrame(event.local_frame);
  updateEventFrameLabel();
}

function navigateEvents(direction) {
  const events = filteredEvents();

  if (!events.length) {
    showEventMessage(
      "No events match the current filters.",
      true
    );
    return;
  }

  let target;

  if (direction > 0) {
    target = events.find(
      event => event.local_frame > currentFrame
    ) || events[0];

  } else {
    target = [...events].reverse().find(
      event => event.local_frame < currentFrame
    ) || events[events.length - 1];
  }

  jumpToEvent(target);
}

// ============================================================
// TIMELINE MARKERS
// ============================================================

function renderEventMarkers() {
  const track = $("eventTrack");
  track.replaceChildren();

  if (!timeline.length) return;

  const maximum = Math.max(
    1,
    timeline.length - 1
  );

  const events = filteredEvents();
  const overlaps = new Map();

  for (const event of events) {
    const marker = document.createElement("button");

    marker.type = "button";

    const confidence =
      normalizedConfidence(event);

    marker.className = [
      "event-marker",
      event.kind,
      event.player === "Far" ? "far" : "",
      confidence
    ].filter(Boolean).join(" ");

    const sameFrameNumber =
      overlaps.get(event.local_frame) || 0;

    overlaps.set(
      event.local_frame,
      sameFrameNumber + 1
    );

    marker.style.left =
      `${100 * event.local_frame / maximum}%`;

    // Stagger markers sharing the same frame slightly.
    marker.style.top =
      `${1 + (sameFrameNumber % 2) * 2}px`;

    marker.title =
      `${eventLabel(event)} | ` +
      `${event.time_seconds.toFixed(2)}s | ` +
      confidence;

    marker.dataset.eventId = event.id;
    marker.dataset.frame = event.local_frame;

    marker.setAttribute(
      "aria-label",
      `Go to ${eventLabel(event)} ` +
      `at frame ${event.source_frame}`
    );

    marker.addEventListener("click", () => {
      jumpToEvent(event);
    });

    track.appendChild(marker);
  }

  updateCurrentEventHighlight();
}

// ============================================================
// ACTIVE EVENT HIGHLIGHT
// ============================================================

function updateCurrentEventHighlight() {
  document.querySelectorAll(
    ".event-marker"
  ).forEach(marker => {
    marker.classList.toggle(
      "current",
      Number(marker.dataset.frame) === currentFrame
    );
  });

  document.querySelectorAll(
    ".event-item"
  ).forEach(item => {
    item.classList.toggle(
      "current",
      Number(item.dataset.frame) === currentFrame
    );
  });
}

// ============================================================
// EVENT LIST
// ============================================================

function createEventListItem(event) {
  const item = document.createElement("div");
  item.className = "event-item";
  item.dataset.frame = event.local_frame;

  const jump = document.createElement("button");
  jump.type = "button";
  jump.className = "event-jump";

  const title = document.createElement("strong");
  title.textContent = eventLabel(event);

  const detail = document.createElement("small");

  detail.textContent =
    `Frame ${event.source_frame} · ` +
    `${event.time_seconds.toFixed(2)}s · ` +
    CONFIDENCE_NAMES[normalizedConfidence(event)];

  const notes = document.createElement("small");

  if (event.notes) {
    notes.textContent = event.notes;
  }

  jump.append(title, detail);

  if (event.notes) {
    jump.appendChild(notes);
  }

  jump.addEventListener("click", () => {
    jumpToEvent(event);
  });

  const edit = document.createElement("button");
  edit.type = "button";
  edit.className = "event-edit";
  edit.textContent = "✎";
  edit.title = "Edit annotation";

  edit.setAttribute(
    "aria-label",
    `Edit ${eventLabel(event)}`
  );

  edit.addEventListener("click", () => {
    loadEventForEditing(event);
  });

  const remove = document.createElement("button");
  remove.type = "button";
  remove.className = "event-delete";
  remove.textContent = "✕";
  remove.title = "Delete annotation";

  remove.setAttribute(
    "aria-label",
    `Delete ${eventLabel(event)}`
  );

  remove.addEventListener("click", async () => {
    if (eventsBusy) return;

    if (!window.confirm(
      "Delete this annotation permanently?"
    )) {
      return;
    }

    eventsBusy = true;
    remove.disabled = true;

    try {
      await eventRequest(
        `/api/events/${encodeURIComponent(event.id)}` +
        `?rally=${encodeURIComponent(currentRallyId)}`,
        { method: "DELETE" }
      );

      if (editingEventId === event.id) {
        resetEventForm();
      }

      await refreshEvents();

      showEventMessage("Annotation deleted.");

    } catch (error) {
      showEventMessage(error.message, true);

    } finally {
      eventsBusy = false;
      remove.disabled = false;
    }
  });

  item.append(jump, edit, remove);

  return item;
}

function renderEventList() {
  const container = $("eventList");
  container.replaceChildren();

  const events = filteredEvents();

  $("eventCount").textContent =
    `${rallyEvents.length} events`;

  $("eventsButton").textContent =
    `Events (${rallyEvents.length})`;

  $("eventFilterSummary").textContent =
    `Showing ${events.length} of ` +
    `${rallyEvents.length} annotations`;

  if (!events.length) {
    const empty = document.createElement("p");
    empty.className = "muted";

    empty.textContent = rallyEvents.length
      ? "No annotations match these filters."
      : "No annotations yet.";

    container.appendChild(empty);
    return;
  }

  for (const event of events) {
    container.appendChild(
      createEventListItem(event)
    );
  }

  updateCurrentEventHighlight();
}

function renderAllEvents() {
  if (!annotationsReady) return;

  renderEventMarkers();
  renderEventList();
}

// ============================================================
// LOAD / REFRESH
// ============================================================

async function refreshEvents() {
  const data = await eventRequest(eventsUrl());

  if (!Array.isArray(data.events)) {
    throw new Error("Invalid events response");
  }

  rallyEvents = [...data.events];

  rallyEvents.sort((a, b) =>
    a.local_frame - b.local_frame ||
    a.created_at.localeCompare(b.created_at) ||
    a.id.localeCompare(b.id)
  );

  renderAllEvents();
  updateEventFrameLabel();
}

// ============================================================
// PANEL
// ============================================================

function showEventPanel(open) {
  $("eventPanel").hidden = !open;

  $("eventsButton").classList.toggle(
    "active",
    open
  );

  $("eventsButton").setAttribute(
    "aria-expanded",
    String(open)
  );

  if (open) {
    pause();
    showSettings(false);
    updateEventFrameLabel();
  }
}

// ============================================================
// SAVE OR UPDATE
// ============================================================

async function saveCurrentEvent(event) {
  event.preventDefault();

  if (!timeline.length || eventsBusy) return;

  pause();

  const button = $("saveEvent");
  button.disabled = true;
  eventsBusy = true;

  const payload = {
    rally_id: currentRallyId,
    local_frame: currentFrame,
    kind: $("eventKind").value,
    player: $("eventPlayer").value,
    shot: $("eventShot").value,
    confidence: $("eventConfidence").value,
    notes: $("eventNotes").value.trim()
  };

  const wasEditing = Boolean(editingEventId);

  try {
    let result;

    if (wasEditing) {
      result = await eventRequest(
        `/api/events/${encodeURIComponent(editingEventId)}` +
        `?rally=${encodeURIComponent(currentRallyId)}`,
        {
          method: "PATCH",
          headers: {
            "Content-Type": "application/json"
          },
          body: JSON.stringify(payload)
        }
      );

    } else {
      result = await eventRequest(
        "/api/events",
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json"
          },
          body: JSON.stringify(payload)
        }
      );
    }

    resetEventForm();

    await refreshEvents();

    showEventMessage(
      `${wasEditing ? "Updated" : "Saved"} ` +
      `annotation at frame ` +
      `${result.event.source_frame}.`
    );

  } catch (error) {
    showEventMessage(error.message, true);

  } finally {
    button.disabled = false;
    eventsBusy = false;
  }
}

// ============================================================
// KEYBOARD SHORTCUTS
// ============================================================

function canUseEventShortcut(event) {
  const target = event.target;

  if (
    target?.isContentEditable ||
    ["INPUT", "SELECT", "TEXTAREA"].includes(
      target?.tagName
    )
  ) {
    return false;
  }

  return !(
    event.ctrlKey ||
    event.altKey ||
    event.metaKey
  );
}

function quickAnnotation(kind) {
  if (!timeline.length) return;

  showEventPanel(true);

  resetEventForm();

  $("eventKind").value = kind;
  updateShotField();

  showEventMessage(
    `Ready to annotate ${eventFrameText()}.`
  );
}

function handleAnnotationKeyboard(event) {
  if (!annotationsReady || !timeline.length) {
    return;
  }

  if (event.code === "Escape") {
    if (!$("eventPanel").hidden) {
      showEventPanel(false);
    }
    return;
  }

  if (!canUseEventShortcut(event)) {
    return;
  }

  if (event.code === "KeyB") {
    event.preventDefault();
    quickAnnotation("bookmark");
  }

  if (event.code === "KeyC") {
    event.preventDefault();
    quickAnnotation("contact");
  }

  if (event.code === "BracketLeft") {
    event.preventDefault();
    navigateEvents(-1);
  }

  if (event.code === "BracketRight") {
    event.preventDefault();
    navigateEvents(1);
  }
}

// ============================================================
// INITIALIZATION
// ============================================================

async function initAnnotations() {
  if (annotationsReady) return;

  annotationsReady = true;

  createFilterUI();
  createConfidenceUI();

  const rally = rallyCatalog.find(
    item => item.id === currentRallyId
  );

  $("eventRallyLabel").textContent =
    rally?.title || currentRallyId;

  $("eventsButton").addEventListener(
    "click",
    () => {
      showEventPanel(
        $("eventPanel").hidden
      );
    }
  );

  $("closeEvents").addEventListener(
    "click",
    () => showEventPanel(false)
  );

  $("eventKind").addEventListener(
    "change",
    updateShotField
  );

  $("eventForm").addEventListener(
    "submit",
    saveCurrentEvent
  );

  $("previousEvent").addEventListener(
    "click",
    () => navigateEvents(-1)
  );

  $("nextEvent").addEventListener(
    "click",
    () => navigateEvents(1)
  );

  document.addEventListener(
    "keydown",
    handleAnnotationKeyboard
  );

  resetEventForm();
  updateEventFrameLabel();

  try {
    await refreshEvents();

  } catch (error) {
    console.error(
      "Annotation loading failed:",
      error
    );

    showEventMessage(
      "Could not load annotations: " +
      error.message,
      true
    );
  }
}
