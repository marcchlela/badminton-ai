
import csv
import json
import math
from collections import Counter
from pathlib import Path

# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent

PLAYER_FILE = (
    ROOT / "outputs/demo_36_46/tracking_v4_hybrid.json"
)

SHUTTLE_FILE = (
    ROOT / "outputs/shuttle/tracknet/"
    "coach_demo_36_46/shuttle_positions.csv"
)

SHUTTLE_METADATA = (
    ROOT / "outputs/shuttle/tracknet/"
    "coach_demo_36_46/metadata.json"
)

OUT = ROOT / "outputs/unified/coach_demo_36_46"

START_FRAME = 900
END_FRAME = 1150  # Exclusive
FPS = 25.0

COURT_WIDTH_M = 6.10
COURT_LENGTH_M = 13.40

OUT.mkdir(parents=True, exist_ok=True)


# ============================================================
# HELPERS
# ============================================================

def read_json(path):
    if not path.exists():
        raise FileNotFoundError(path)

    with path.open(encoding="utf-8") as f:
        return json.load(f)


def valid_point(value):
    if not isinstance(value, (list, tuple)):
        return None

    if len(value) < 2:
        return None

    if value[0] is None or value[1] is None:
        return None

    x = float(value[0])
    y = float(value[1])

    if not (math.isfinite(x) and math.isfinite(y)):
        return None

    return [x, y]


def normalize_player(player):
    """
    Preserve both measured image information
    and the estimated ground-plane position.

    Important: bbox-foot pixels do not necessarily
    correspond to the pose-hybrid court position.
    """

    if not isinstance(player, dict):
        return None

    court = valid_point(player.get("court_m"))

    bbox_foot = valid_point(player.get("pixel"))

    image_center = valid_point(
        player.get("image_center")
    )

    pose_ankle = valid_point(
        player.get("ankle_midpoint_pixel")
    )

    if pose_ankle is None:
        pose_ankle = valid_point(
            player.get("pose_ankle_midpoint_pixel")
        )

    bbox = player.get("bbox")

    if bbox is not None:
        bbox = [float(v) for v in bbox]

    within_court = None

    if court is not None:
        within_court = (
            0 <= court[0] <= COURT_WIDTH_M
            and 0 <= court[1] <= COURT_LENGTH_M
        )

    confidence = player.get("confidence")

    return {
        "court_m": court,
        "court_position_valid": within_court,
        "bbox_foot_pixel": bbox_foot,
        "image_center_pixel": image_center,
        "pose_ankle_pixel": pose_ankle,
        "bbox": bbox,
        "confidence": (
            float(confidence)
            if confidence is not None
            else None
        ),
        "position_source": player.get(
            "position_source"
        ),
        "visual_detection": player.get(
            "visual_detection"
        ),
        "ground_updated": player.get(
            "ground_updated"
        ),
        "ground_age_seconds": player.get(
            "ground_age_seconds"
        ),
        "pose_accepted": player.get(
            "pose_accepted"
        ),
        "pose_reason": player.get(
            "pose_reason"
        ),
    }


# ============================================================
# LOAD PLAYER DATA
# ============================================================

def load_players():
    data = read_json(PLAYER_FILE)

    if not isinstance(data, list):
        raise ValueError(
            "Expected player tracking JSON to be a list"
        )

    by_frame = {}

    for record in data:
        frame = int(record["frame"])

        if not START_FRAME <= frame < END_FRAME:
            continue

        if frame in by_frame:
            raise ValueError(
                f"Duplicate player frame: {frame}"
            )

        players = record.get("players", {})

        by_frame[frame] = {
            "Near": normalize_player(
                players.get("Near")
            ),
            "Far": normalize_player(
                players.get("Far")
            ),
        }

        # Verify the recorded time agrees with frame ID.
        recorded_time = float(
        record.get("time", frame / FPS)
    )

    expected_time = frame / FPS
    time_difference = abs(recorded_time - expected_time)

    # Reject genuinely inconsistent timestamps.
    # Allow small differences from player sampling and rounding.
    if time_difference > 0.11:
        raise ValueError(
            f"Player timestamp mismatch at frame {frame}: "
            f"recorded={recorded_time:.4f}s, "
            f"frame_time={expected_time:.4f}s, "
            f"difference={time_difference:.4f}s"
        )

    # The original frame number remains authoritative
    # when constructing the unified timeline.

    return by_frame


# ============================================================
# LOAD SHUTTLE DATA
# ============================================================

def load_shuttle():
    metadata = read_json(SHUTTLE_METADATA)

    if int(metadata["first_frame"]) != START_FRAME:
        raise ValueError(
            "Unexpected shuttle starting frame"
        )

    if int(metadata["frame_count"]) != (
        END_FRAME - START_FRAME
    ):
        raise ValueError(
            "Unexpected shuttle frame count"
        )

    if abs(float(metadata["fps"]) - FPS) > 0.001:
        raise ValueError(
            "Shuttle FPS does not match timeline"
        )

    by_frame = {}

    with SHUTTLE_FILE.open(
        newline="",
        encoding="utf-8-sig"
    ) as f:

        reader = csv.DictReader(f)

        for row in reader:
            frame = int(row["frame"])

            if not START_FRAME <= frame < END_FRAME:
                raise ValueError(
                    f"Unexpected shuttle frame {frame}"
                )

            if frame in by_frame:
                raise ValueError(
                    f"Duplicate shuttle frame {frame}"
                )

            if int(row["local_frame"]) != (
                frame - START_FRAME
            ):
                raise ValueError(
                    f"Incorrect local frame at {frame}"
                )

            time = float(row["time"])

            if abs(time - frame / FPS) > 0.001:
                raise ValueError(
                    f"Shuttle timestamp mismatch at {frame}"
                )

            visible = int(row["visible"]) == 1

            point = None

            if visible:
                point = valid_point([
                    row["x"],
                    row["y"]
                ])

                if point is None:
                    raise ValueError(
                        f"Missing shuttle coordinates at {frame}"
                    )

            by_frame[frame] = {
                "visible": visible,
                "pixel": point,
                "model": "TrackNetV3",
                "prediction_source": "tracknet_only",
            }

    return by_frame


# ============================================================
# BUILD UNIFIED TIMELINE
# ============================================================

def build_timeline(players, shuttle):
    expected_frames = set(
        range(START_FRAME, END_FRAME)
    )

    if set(shuttle) != expected_frames:
        missing = sorted(
            expected_frames - set(shuttle)
        )

        raise ValueError(
            f"Missing shuttle frames: {missing[:10]}"
        )

    timeline = []

    for frame in range(START_FRAME, END_FRAME):
        time = frame / FPS

        player_sample = players.get(frame)

        timeline.append({
            "frame": frame,
            "time": round(time, 4),
            "local_frame": frame - START_FRAME,
            "players_sampled": (
                player_sample is not None
            ),
            "players": (
                player_sample
                if player_sample is not None
                else {
                    "Near": None,
                    "Far": None
                }
            ),
            "shuttle": shuttle[frame],
        })

    return timeline


# ============================================================
# VALIDATION AND EXPORT
# ============================================================

def main():
    print("========== UNIFIED RALLY TIMELINE ==========")

    print("\nLoading player tracking...")
    players = load_players()

    print("Player samples:", len(players))

    print("\nLoading shuttle tracking...")
    shuttle = load_shuttle()

    print("Shuttle samples:", len(shuttle))

    print("\nSynchronizing by original frame numbers...")
    timeline = build_timeline(players, shuttle)

    total = len(timeline)

    player_sampled = sum(
        row["players_sampled"]
        for row in timeline
    )

    shuttle_visible = sum(
        row["shuttle"]["visible"]
        for row in timeline
    )

    both_available = sum(
        row["players_sampled"]
        and row["shuttle"]["visible"]
        for row in timeline
    )

    missing_player_samples = total - player_sampled

    position_sources = Counter()

    for row in timeline:
        if not row["players_sampled"]:
            continue

        for side in ("Near", "Far"):
            player = row["players"].get(side)

            if player is not None:
                source = player.get(
                    "position_source"
                )

                position_sources[
                    source or "unknown"
                ] += 1

    metadata = {
        "video": "data/raw/clip.mp4",
        "start_frame": START_FRAME,
        "end_frame_exclusive": END_FRAME,
        "fps": FPS,
        "total_frames": total,
        "player_sampled_frames": player_sampled,
        "player_unsampled_frames": (
            missing_player_samples
        ),
        "shuttle_predicted_visible_frames": (
            shuttle_visible
        ),
        "frames_with_player_sample_and_shuttle": (
            both_available
        ),
        "coordinate_systems": {
            "players_court_m": (
                "Estimated ground-plane positions in metres"
            ),
            "player_pixels": (
                "Image positions in original source frame"
            ),
            "shuttle_pixel": (
                "Image-space location of airborne shuttle"
            ),
        },
        "sources": {
            "players": str(
                PLAYER_FILE.relative_to(ROOT)
            ),
            "shuttle": str(
                SHUTTLE_FILE.relative_to(ROOT)
            ),
        },
        "player_position_sources": dict(
            position_sources
        ),
        "notes": [
            "No player positions are interpolated.",
            "Missing player samples are explicit null values.",
            "TrackNet visibility is a model prediction.",
            "Airborne shuttle pixels are not court-plane metres.",
            "Player pose-hybrid court positions are experimental.",
        ],
    }

    output_timeline = OUT / "timeline.json"
    output_metadata = OUT / "metadata.json"

    with output_timeline.open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(timeline, f, indent=2)

    with output_metadata.open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(metadata, f, indent=2)

    print("\n========== TIMELINE SUMMARY ==========")
    print("Total frames:", total)
    print("First frame:", timeline[0]["frame"])
    print("Last frame:", timeline[-1]["frame"])
    print("Player sampled frames:", player_sampled)
    print(
        "Player unsampled frames:",
        missing_player_samples
    )
    print(
        "Shuttle predicted visible frames:",
        shuttle_visible
    )
    print(
        "Frames with both sources:",
        both_available
    )

    print("\nPlayer position sources:")
    for source, count in position_sources.items():
        print(f"  {source}: {count}")

    print("\n========== EXAMPLE FRAME ==========")

    example = next(
        (
            row for row in timeline
            if row["players_sampled"]
        ),
        None
    )

    if example:
        print(
            json.dumps(example, indent=2)[:2500]
        )

    print("\nSaved:")
    print(output_timeline)
    print(output_metadata)

    print("\nDone!")


if __name__ == "__main__":
    main()
