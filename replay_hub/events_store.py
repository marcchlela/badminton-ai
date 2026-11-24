
import json
import os
import re
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parent.parent
STORE = ROOT / "outputs" / "annotations"
LOCK = threading.RLock()

VALID_KINDS = {"contact", "bookmark", "rally_event"}
VALID_PLAYERS = {"Near", "Far", "Unknown"}

VALID_SHOTS = {
    "unknown", "serve", "clear", "drop", "smash",
    "lift", "drive", "net", "push", "other",
}

VALID_CONFIDENCE = {
    "confirmed", "approximate", "uncertain",
}

RALLY_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
EVENT_PATTERN = re.compile(r"^[a-f0-9]{32}$")


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def storage_path(rally_id):
    if not isinstance(rally_id, str):
        raise ValueError("Invalid rally ID")

    if not RALLY_PATTERN.fullmatch(rally_id):
        raise ValueError("Invalid rally ID")

    return STORE / f"{rally_id}.json"


def read_events(rally_id):
    path = storage_path(rally_id)

    if not path.exists():
        return []

    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, dict):
        raise ValueError("Invalid annotation file")

    if data.get("rally_id") != rally_id:
        raise ValueError("Annotation file rally mismatch")

    events = data.get("events")

    if not isinstance(events, list):
        raise ValueError("Invalid annotation events")

    return events


def write_events(rally_id, events):
    STORE.mkdir(parents=True, exist_ok=True)

    destination = storage_path(rally_id)

    payload = {
        "schema_version": 2,
        "rally_id": rally_id,
        "events": events,
    }

    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=STORE,
            prefix=".events_",
            suffix=".tmp",
            delete=False,
        ) as file:
            temp_path = Path(file.name)

            json.dump(
                payload,
                file,
                indent=2,
                ensure_ascii=False,
            )
            file.write("\n")

        os.replace(temp_path, destination)

    finally:
        if temp_path and temp_path.exists():
            temp_path.unlink()


def sort_events(events):
    return sorted(
        events,
        key=lambda event: (
            event["local_frame"],
            event["created_at"],
            event["id"],
        ),
    )


def normalize_existing(event):
    """
    Old V1 events remain readable.

    Missing confidence is marked uncertain, never
    automatically upgraded to confirmed.
    """
    item = dict(event)

    if item.get("confidence") not in VALID_CONFIDENCE:
        item["confidence"] = "uncertain"

    return item


def list_events(rally_id):
    with LOCK:
        events = read_events(rally_id)
        return sort_events([
            normalize_existing(event)
            for event in events
        ])


def validate_fields(rally_info, payload):
    if not isinstance(payload, dict):
        raise ValueError("Expected a JSON object")

    index = payload.get("local_frame")

    if type(index) is not int:
        raise ValueError("local_frame must be an integer")

    count = int(rally_info["frame_count"])

    if not 0 <= index < count:
        raise ValueError("Frame outside this rally")

    kind = payload.get("kind")
    player = payload.get("player", "Unknown")
    shot = payload.get("shot", "unknown")

    confidence = payload.get(
        "confidence", "uncertain"
    )

    notes = payload.get("notes", "")

    if kind not in VALID_KINDS:
        raise ValueError("Invalid event type")

    if player not in VALID_PLAYERS:
        raise ValueError("Invalid player")

    if shot not in VALID_SHOTS:
        raise ValueError("Invalid shot type")

    if confidence not in VALID_CONFIDENCE:
        raise ValueError("Invalid confidence")

    if not isinstance(notes, str):
        raise ValueError("Notes must be text")

    notes = notes.strip()

    if len(notes) > 500:
        raise ValueError("Notes cannot exceed 500 characters")

    if kind != "contact":
        shot = "unknown"

    original_frame = int(
        rally_info["start_frame"]
    ) + index

    fps = float(rally_info["fps"])

    return {
        "local_frame": index,
        "source_frame": original_frame,
        "time_seconds": round(
            original_frame / fps, 6
        ),
        "kind": kind,
        "player": player,
        "shot": shot,
        "confidence": confidence,
        "notes": notes,
    }


def ensure_not_duplicate(events, fields, exclude_id=None):
    for existing in events:
        if existing.get("id") == exclude_id:
            continue

        if all(
            existing.get(key) == fields[key]
            for key in (
                "local_frame",
                "kind",
                "player",
                "shot",
            )
        ):
            raise ValueError(
                "A matching annotation already exists "
                "at this frame"
            )


def add_event(rally_id, rally_info, payload):
    fields = validate_fields(rally_info, payload)

    event = {
        "id": uuid4().hex,
        "rally_id": rally_id,
        **fields,
        "source": "manual",
        "created_at": utc_now(),
    }

    with LOCK:
        events = read_events(rally_id)

        ensure_not_duplicate(events, fields)

        events.append(event)
        write_events(rally_id, sort_events(events))

    return event


def update_event(
    rally_id,
    rally_info,
    event_id,
    payload,
):
    if not isinstance(event_id, str):
        raise ValueError("Invalid event ID")

    if not EVENT_PATTERN.fullmatch(event_id):
        raise ValueError("Invalid event ID")

    fields = validate_fields(rally_info, payload)

    with LOCK:
        events = read_events(rally_id)

        index = next(
            (
                i for i, item in enumerate(events)
                if item.get("id") == event_id
            ),
            None,
        )

        if index is None:
            return None

        ensure_not_duplicate(
            events, fields, exclude_id=event_id
        )

        original = events[index]

        updated = {
            **original,
            **fields,
            "source": "manual",
            "updated_at": utc_now(),
        }

        # Keep event identity and creation timestamp.
        updated["id"] = original["id"]
        updated["rally_id"] = rally_id
        updated["created_at"] = original["created_at"]

        events[index] = updated

        write_events(
            rally_id,
            sort_events(events),
        )

    return updated


def delete_event(rally_id, event_id):
    if not isinstance(event_id, str):
        raise ValueError("Invalid event ID")

    if not EVENT_PATTERN.fullmatch(event_id):
        raise ValueError("Invalid event ID")

    with LOCK:
        events = read_events(rally_id)

        remaining = [
            event for event in events
            if event.get("id") != event_id
        ]

        if len(remaining) == len(events):
            return False

        write_events(rally_id, remaining)
        return True
