
"""
Badminton AI — Contact Candidate Reviewer V1

Review heuristic contact candidates using original footage.

Controls:
    A       Accept contact
    R       Reject candidate
    U       Uncertain
    J / L   Previous / next frame
    K       Return to proposed frame
    P / N   Previous / next candidate
    1       Near player
    2       Far player
    3       Unknown player
    T       Cycle shot type
    Space   Play/pause local preview
    H       Toggle shuttle overlay
    Q/Esc   Quit

Reviews are saved automatically.
Manual annotation files are not modified.
"""

import argparse
import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parent

CATALOG = ROOT / "replay_hub" / "rallies.json"

CONTACT_ROOT = ROOT / "outputs" / "contacts" / "v1"

WINDOW_NAME = "Badminton AI - Contact Review"

SHOTS = [
    "unknown",
    "serve",
    "clear",
    "drop",
    "smash",
    "lift",
    "drive",
    "net",
    "push",
    "other",
]

COLORS = {
    "bg": (25, 17, 11),
    "surface": (42, 29, 17),
    "white": (245, 245, 245),
    "muted": (164, 155, 145),
    "accent": (194, 207, 104),
    "near": (243, 207, 104),
    "far": (103, 188, 246),
    "accept": (150, 220, 105),
    "reject": (130, 130, 245),
    "uncertain": (103, 188, 246),
    "shuttle": (170, 246, 138),
}


def read_json(path):
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def file_hash(path):
    return hashlib.sha256(
        path.read_bytes()
    ).hexdigest()


def safe_name(value):
    return (
        bool(value)
        and len(value) <= 80
        and all(
            char.isascii()
            and (
                char.isalnum()
                or char in "-_"
            )
            for char in value
        )
    )


def project_file(value):
    path = (ROOT / value).resolve()

    if not path.is_relative_to(ROOT):
        raise ValueError("Invalid project path")

    if not path.is_file():
        raise FileNotFoundError(path)

    return path


def atomic_json(path, payload):
    temporary = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            suffix=".tmp",
            delete=False,
        ) as file:
            temporary = Path(file.name)

            json.dump(
                payload,
                file,
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
            )

            file.write("\n")

        os.replace(temporary, path)

    finally:
        if temporary and temporary.exists():
            temporary.unlink()


def text(canvas, value, x, y,
         color=None, scale=.55, thickness=1):
    cv2.putText(
        canvas,
        str(value),
        (int(x), int(y)),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color or COLORS["white"],
        thickness,
        cv2.LINE_AA,
    )


class ContactReviewer:

    def __init__(self, rally_id, experiment, radius):

        self.rally_id = rally_id
        self.experiment = experiment
        self.radius = radius

        self.directory = (
            CONTACT_ROOT /
            rally_id /
            experiment
        )

        self.candidates_path = (
            self.directory /
            "candidates.json"
        )

        self.review_path = (
            self.directory /
            "reviews.json"
        )

        self.candidate_hash = file_hash(
            self.candidates_path
        )

        candidate_data = read_json(
            self.candidates_path
        )

        if candidate_data.get("rally_id") != rally_id:
            raise ValueError(
                "Candidate rally mismatch"
            )

        self.candidates = sorted(
            candidate_data["candidates"],
            key=lambda item: item["local_frame"],
        )

        if not self.candidates:
            raise ValueError(
                "No candidates to review"
            )

        self.load_rally()

        self.reviews = {}

        self.load_reviews()

        self.candidate_index = 0

        self.current_frame = 0

        self.selected_player = "Unknown"
        self.selected_shot = "unknown"

        self.playing = False
        self.overlay = True

        self.message = ""

        self.cap = cv2.VideoCapture(
            str(self.video_path)
        )

        if not self.cap.isOpened():
            raise RuntimeError(
                "Cannot open source video"
            )

        try:
            self.validate_video()
        except Exception:
            self.cap.release()
            raise

        self.cached_index = None
        self.cached_frame = None

        # Resume at first unreviewed candidate.
        first_pending = next(
            (
                index
                for index, candidate in enumerate(
                    self.candidates
                )
                if candidate["candidate_id"]
                not in self.reviews
            ),
            0,
        )

        self.select_candidate(first_pending)

    # ========================================================
    # DATA LOADING
    # ========================================================

    def load_rally(self):

        catalog = read_json(CATALOG)

        entry = next(
            (
                item
                for item in catalog["rallies"]
                if item["id"] == self.rally_id
            ),
            None,
        )

        if entry is None:
            raise ValueError("Unknown rally")

        self.timeline = read_json(
            project_file(entry["timeline"])
        )

        self.metadata = read_json(
            project_file(entry["metadata"])
        )

        self.video_path = project_file(
            entry["video"]
        )

        self.fps = float(
            self.metadata["fps"]
        )

        self.start_frame = int(
            self.metadata["start_frame"]
        )

        self.count = len(self.timeline)

        if self.count != int(
            self.metadata["total_frames"]
        ):
            raise ValueError(
                "Timeline length mismatch"
            )

        for index, row in enumerate(self.timeline):
            if row["local_frame"] != index:
                raise ValueError(
                    "Invalid local frame alignment"
                )

            if row["frame"] != self.start_frame + index:
                raise ValueError(
                    "Invalid source frame alignment"
                )

        ids = set()

        for candidate in self.candidates:
            candidate_id = candidate["candidate_id"]
            index = candidate["local_frame"]

            if candidate_id in ids:
                raise ValueError(
                    "Duplicate candidate ID"
                )

            ids.add(candidate_id)

            if type(index) is not int:
                raise ValueError(
                    "Invalid candidate frame"
                )

            if not 0 <= index < self.count:
                raise ValueError(
                    "Candidate outside rally"
                )

            if candidate["source_frame"] != (
                self.start_frame + index
            ):
                raise ValueError(
                    "Candidate source-frame mismatch"
                )

    def validate_video(self):

        count = int(
            self.cap.get(
                cv2.CAP_PROP_FRAME_COUNT
            )
        )

        fps = float(
            self.cap.get(
                cv2.CAP_PROP_FPS
            )
        )

        if count != self.count:
            raise ValueError(
                "Video frame count mismatch"
            )

        if abs(fps - self.fps) > .001:
            raise ValueError(
                "Video FPS mismatch"
            )

    def load_reviews(self):

        if not self.review_path.exists():
            return

        data = read_json(self.review_path)

        if data.get("rally_id") != self.rally_id:
            raise ValueError("Review rally mismatch")

        if data.get("candidates_sha256") != (
            self.candidate_hash
        ):
            raise ValueError(
                "Candidate file changed since review. "
                "Use a separate experiment folder."
            )

        self.reviews = {
            item["candidate_id"]: item
            for item in data.get("reviews", [])
        }

    # ========================================================
    # CANDIDATE STATE
    # ========================================================

    @property
    def candidate(self):
        return self.candidates[
            self.candidate_index
        ]

    @property
    def candidate_frame(self):
        return self.candidate["local_frame"]

    def select_candidate(self, index):

        self.candidate_index = max(
            0,
            min(
                index,
                len(self.candidates) - 1,
            ),
        )

        candidate = self.candidate

        saved = self.reviews.get(
            candidate["candidate_id"]
        )

        self.playing = False

        if saved:
            corrected = saved.get(
                "corrected_local_frame"
            )

            self.current_frame = (
                corrected
                if corrected is not None
                else self.candidate_frame
            )

            self.selected_player = (
                saved.get("player") or "Unknown"
            )

            self.selected_shot = (
                saved.get("shot") or "unknown"
            )

            self.message = (
                "Previously reviewed: "
                + saved["decision"].upper()
            )

        else:
            self.current_frame = (
                self.candidate_frame
            )

            # Require reviewer to identify the player.
            # Do not silently accept predicted identity.
            self.selected_player = "Unknown"

            self.selected_shot = "unknown"

            self.message = (
                "Review this candidate."
            )

    def move_frame(self, amount):

        self.playing = False

        self.current_frame = max(
            0,
            min(
                self.current_frame + amount,
                self.count - 1,
            ),
        )

    def cycle_shot(self):

        index = SHOTS.index(
            self.selected_shot
        )

        self.selected_shot = SHOTS[
            (index + 1) % len(SHOTS)
        ]

    # ========================================================
    # SAVE REVIEW
    # ========================================================

    def save_review(self, decision):

        if decision not in {
            "accepted",
            "rejected",
            "uncertain",
        }:
            raise ValueError(
                "Invalid review decision"
            )

        candidate = self.candidate

        accepted = decision == "accepted"

        corrected = (
            self.current_frame
            if accepted else None
        )

        record = {
            "candidate_id": (
                candidate["candidate_id"]
            ),
            "decision": decision,
            "proposed_local_frame": (
                candidate["local_frame"]
            ),
            "proposed_source_frame": (
                candidate["source_frame"]
            ),
            "original_score": (
                candidate["score"]
            ),
            "predicted_player": (
                candidate["predicted_player"]
            ),
            "corrected_local_frame": corrected,
            "corrected_source_frame": (
                self.start_frame + corrected
                if accepted else None
            ),
            "corrected_time_seconds": (
                (self.start_frame + corrected) /
                self.fps
                if accepted else None
            ),
            "frame_correction": (
                corrected - self.candidate_frame
                if accepted else None
            ),
            "player": (
                self.selected_player
                if accepted else None
            ),
            "shot": (
                self.selected_shot
                if accepted else None
            ),
            "source": "manual_candidate_review",
            "reviewed_at": datetime.now(
                timezone.utc
            ).isoformat(),
        }

        updated = {
            **self.reviews,
            candidate["candidate_id"]: record,
        }

        payload = {
            "schema_version": 1,
            "rally_id": self.rally_id,
            "experiment": self.experiment,
            "candidates_sha256": (
                self.candidate_hash
            ),
            "reviews": sorted(
                updated.values(),
                key=lambda item: (
                    item["proposed_local_frame"]
                ),
            ),
        }

        atomic_json(
            self.review_path,
            payload,
        )

        self.reviews = updated

        print(
            f"{candidate['candidate_id']} | "
            f"Frame {candidate['source_frame']} | "
            f"{decision.upper()} | "
            f"Corrected: "
            f"{record['corrected_source_frame']} | "
            f"Player: {record['player']} | "
            f"Shot: {record['shot']}"
        )

        self.playing = False

        # Advance after each successful review.
        if self.candidate_index < (
            len(self.candidates) - 1
        ):
            self.select_candidate(
                self.candidate_index + 1
            )

        else:
            self.message = (
                "Saved. End of candidate list."
            )

    # ========================================================
    # VIDEO FRAME READING
    # ========================================================

    def read_frame(self):

        if self.cached_index == self.current_frame:
            return self.cached_frame.copy()

        self.cap.set(
            cv2.CAP_PROP_POS_FRAMES,
            self.current_frame,
        )

        ok, frame = self.cap.read()

        if not ok:
            raise RuntimeError(
                f"Cannot read frame "
                f"{self.current_frame}"
            )

        self.cached_index = self.current_frame
        self.cached_frame = frame.copy()

        return frame

    # ========================================================
    # SHUTTLE OVERLAY
    # ========================================================

    def draw_shuttle(self, frame):

        if not self.overlay:
            return

        previous = None

        first = max(
            0,
            self.current_frame - 6,
        )

        for index in range(
            first,
            self.current_frame + 1,
        ):
            shuttle = (
                self.timeline[index]
                .get("shuttle") or {}
            )

            pixel = shuttle.get("pixel")

            if (
                not shuttle.get("visible")
                or not isinstance(pixel, list)
                or len(pixel) != 2
            ):
                previous = None
                continue

            try:
                point = tuple(
                    int(round(float(value)))
                    for value in pixel
                )
            except (TypeError, ValueError, OverflowError):
                previous = None
                continue

            if previous is not None:
                cv2.line(
                    frame,
                    previous,
                    point,
                    COLORS["shuttle"],
                    2,
                    cv2.LINE_AA,
                )

            if index == self.current_frame:
                cv2.circle(
                    frame,
                    point,
                    7,
                    COLORS["shuttle"],
                    2,
                    cv2.LINE_AA,
                )

                cv2.circle(
                    frame,
                    point,
                    14,
                    COLORS["accent"],
                    1,
                    cv2.LINE_AA,
                )

            previous = point

    # ========================================================
    # INTERFACE
    # ========================================================

    def draw_interface(self):

        frame = self.read_frame()

        self.draw_shuttle(frame)

        original_h, original_w = frame.shape[:2]

        width = min(
            1100,
            original_w,
        )

        height = round(
            original_h *
            width / original_w
        )

        frame = cv2.resize(
            frame,
            (width, height),
            interpolation=cv2.INTER_AREA,
        )

        header_h = 74
        footer_h = 205

        canvas = np.full(
            (
                header_h + height + footer_h,
                width,
                3,
            ),
            COLORS["bg"],
            dtype=np.uint8,
        )

        canvas[
            header_h:header_h + height
        ] = frame

        candidate = self.candidate

        source_frame = (
            self.start_frame +
            self.current_frame
        )

        correction = (
            self.current_frame -
            self.candidate_frame
        )

        # Header.
        text(
            canvas,
            "BADMINTON AI / CONTACT REVIEW",
            18,
            24,
            COLORS["accent"],
            .52,
        )

        text(
            canvas,
            f"Candidate "
            f"{self.candidate_index + 1}/"
            f"{len(self.candidates)}",
            18,
            53,
            scale=.8,
            thickness=2,
        )

        text(
            canvas,
            f"Reviewed "
            f"{len(self.reviews)}/"
            f"{len(self.candidates)}",
            width - 220,
            25,
            COLORS["muted"],
        )

        text(
            canvas,
            f"Score: "
            f"{candidate['score']:.3f}",
            width - 220,
            53,
            COLORS["accent"],
            .65,
        )

        bottom = header_h + height

        # Current frame data.
        text(
            canvas,
            f"Proposed: "
            f"{candidate['source_frame']}   |   "
            f"Selected: {source_frame}   |   "
            f"Correction: {correction:+d} frames   |   "
            f"{source_frame / self.fps:.2f}s",
            18,
            bottom + 24,
            scale=.53,
        )

        # Local review timeline.
        track_left = 24
        track_right = width - 24

        first = max(
            0,
            self.candidate_frame - self.radius,
        )

        last = min(
            self.count - 1,
            self.candidate_frame + self.radius,
        )

        # Extend display if reviewer steps outside window.
        first = min(first, self.current_frame)
        last = max(last, self.current_frame)

        span = max(1, last - first)

        def x_at(index):
            return round(
                track_left +
                (index - first) /
                span *
                (track_right - track_left)
            )

        track_y = bottom + 47

        cv2.line(
            canvas,
            (track_left, track_y),
            (track_right, track_y),
            (75, 64, 55),
            4,
            cv2.LINE_AA,
        )

        for index in range(first, last + 1):
            x = x_at(index)

            cv2.line(
                canvas,
                (x, track_y - 3),
                (x, track_y + 3),
                COLORS["muted"],
                1,
            )

        cv2.line(
            canvas,
            (
                x_at(self.candidate_frame),
                track_y - 10,
            ),
            (
                x_at(self.candidate_frame),
                track_y + 10,
            ),
            COLORS["far"],
            2,
        )

        cv2.circle(
            canvas,
            (
                x_at(self.current_frame),
                track_y,
            ),
            7,
            COLORS["accent"],
            -1,
            cv2.LINE_AA,
        )

        # Proposed vs human-entered labels.
        text(
            canvas,
            f"Detector player: "
            f"{candidate['predicted_player']}",
            18,
            bottom + 79,
            COLORS["muted"],
        )

        player_color = {
            "Near": COLORS["near"],
            "Far": COLORS["far"],
            "Unknown": COLORS["white"],
        }[self.selected_player]

        text(
            canvas,
            f"Review player: "
            f"{self.selected_player}   |   "
            f"Shot: {self.selected_shot.upper()}",
            18,
            bottom + 103,
            player_color,
            .6,
        )

        # Controls.
        text(
            canvas,
            "[A] ACCEPT",
            18,
            bottom + 129,
            COLORS["accept"],
            .6,
            2,
        )

        text(
            canvas,
            "[R] REJECT",
            180,
            bottom + 129,
            COLORS["reject"],
            .6,
            2,
        )

        text(
            canvas,
            "[U] UNCERTAIN",
            345,
            bottom + 129,
            COLORS["uncertain"],
            .6,
            2,
        )

        text(
            canvas,
            "[J/L] Frame  [K] Reset  "
            "[P/N] Candidate  "
            "[SPACE] Preview  [Q] Quit",
            18,
            bottom + 153,
            COLORS["muted"],
            .45,
        )

        text(
            canvas,
            "[1] Near  [2] Far  [3] Unknown  "
            "[T] Cycle shot  [H] Overlay",
            18,
            bottom + 173,
            COLORS["muted"],
            .45,
        )

        text(
            canvas,
            self.message,
            18,
            bottom + 194,
            COLORS["accent"],
            .47,
        )

        return canvas

    # ========================================================
    # PLAYBACK
    # ========================================================

    def advance_preview(self):

        if not self.playing:
            return

        first = max(
            0,
            self.candidate_frame - self.radius,
        )

        last = min(
            self.count - 1,
            self.candidate_frame + self.radius,
        )

        if (
            self.current_frame < first
            or self.current_frame >= last
        ):
            self.current_frame = first
        else:
            self.current_frame += 1

    # ========================================================
    # KEYBOARD INPUT
    # ========================================================

    def handle_key(self, key):

        if key in (27, ord("q")):
            return False

        if key == ord("j"):
            self.move_frame(-1)

        elif key == ord("l"):
            self.move_frame(1)

        elif key == ord("k"):
            self.playing = False
            self.current_frame = (
                self.candidate_frame
            )

        elif key == ord("p"):
            self.select_candidate(
                self.candidate_index - 1
            )

        elif key == ord("n"):
            self.select_candidate(
                self.candidate_index + 1
            )

        elif key == ord("1"):
            self.selected_player = "Near"

        elif key == ord("2"):
            self.selected_player = "Far"

        elif key == ord("3"):
            self.selected_player = "Unknown"

        elif key == ord("t"):
            self.cycle_shot()

        elif key == ord("h"):
            self.overlay = not self.overlay

        elif key == ord(" "):
            self.playing = not self.playing

        elif key in (
            ord("a"),
            ord("r"),
            ord("u"),
        ):
            if self.playing:
                self.playing = False

                self.message = (
                    "Paused. Inspect frame, "
                    "then press decision key again."
                )

                return True

            decisions = {
                ord("a"): "accepted",
                ord("r"): "rejected",
                ord("u"): "uncertain",
            }

            self.save_review(
                decisions[key]
            )

        return True

    # ========================================================
    # REVIEW SUMMARY
    # ========================================================

    def summary(self):

        accepted = [
            item
            for item in self.reviews.values()
            if item["decision"] == "accepted"
        ]

        rejected = [
            item
            for item in self.reviews.values()
            if item["decision"] == "rejected"
        ]

        uncertain = [
            item
            for item in self.reviews.values()
            if item["decision"] == "uncertain"
        ]

        unique_contacts = {
            item["corrected_source_frame"]
            for item in accepted
        }

        corrections = [
            abs(item["frame_correction"])
            for item in accepted
        ]

        decided = (
            len(accepted) + len(rejected)
        )

        acceptance = (
            len(accepted) / decided
            if decided else None
        )

        known_player = [
            item
            for item in accepted
            if item["player"] != "Unknown"
        ]

        correct_player = [
            item
            for item in known_player
            if item["player"]
            == item["predicted_player"]
        ]

        result = {
            "rally_id": self.rally_id,
            "total_candidates": (
                len(self.candidates)
            ),
            "reviewed_candidates": (
                len(self.reviews)
            ),
            "accepted": len(accepted),
            "rejected": len(rejected),
            "uncertain": len(uncertain),
            "unreviewed": (
                len(self.candidates)
                - len(self.reviews)
            ),
            "unique_accepted_source_frames": (
                len(unique_contacts)
            ),
            "mean_absolute_correction_frames": (
                sum(corrections) /
                len(corrections)
                if corrections else None
            ),
            "acceptance_among_decided_candidates": (
                acceptance
            ),
            "known_player_review_count": (
                len(known_player)
            ),
            "correct_player_predictions": (
                len(correct_player)
            ),
            "player_agreement": (
                len(correct_player) /
                len(known_player)
                if known_player else None
            ),
            "limitations": [
                "Candidate acceptance is not overall recall.",
                "Unreviewed and uncertain candidates are excluded from acceptance denominator.",
                "Duplicate accepted contacts may occur.",
                "A complete independent rally annotation is still required to find missed contacts.",
            ],
        }

        atomic_json(
            self.directory /
            "review_summary.json",
            result,
        )

        print()
        print("========== REVIEW SUMMARY ==========")

        for key, value in result.items():
            if key != "limitations":
                print(f"{key}: {value}")

        print()
        print("Saved reviews:", self.review_path)

    # ========================================================
    # MAIN REVIEW LOOP
    # ========================================================

    def run(self):

        try:
            cv2.namedWindow(
                WINDOW_NAME,
                cv2.WINDOW_NORMAL,
            )

            cv2.resizeWindow(
                WINDOW_NAME,
                1100,
                850,
            )

            while True:

                canvas = self.draw_interface()

                cv2.imshow(
                    WINDOW_NAME,
                    canvas,
                )

                # Deliberate half-speed review playback.
                delay = (
                    max(
                        1,
                        round(2000 / self.fps),
                    )
                    if self.playing
                    else 35
                )

                key = cv2.waitKey(delay)

                try:
                    visible = cv2.getWindowProperty(
                        WINDOW_NAME,
                        cv2.WND_PROP_VISIBLE,
                    )

                    if visible < 1:
                        break

                except cv2.error:
                    break

                if key != -1:
                    if not self.handle_key(
                        key & 0xFF
                    ):
                        break

                self.advance_preview()

        finally:
            self.cap.release()
            cv2.destroyAllWindows()
            self.summary()


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Review automatic badminton "
            "racket-contact candidates"
        )
    )

    parser.add_argument(
        "--rally",
        default="coach-36-46",
    )

    parser.add_argument(
        "--experiment",
        default="baseline",
    )

    parser.add_argument(
        "--radius",
        type=int,
        default=10,
    )

    args = parser.parse_args()

    if not safe_name(args.rally):
        parser.error("Invalid rally ID")

    if not safe_name(args.experiment):
        parser.error("Invalid experiment name")

    if not 3 <= args.radius <= 100:
        parser.error(
            "--radius must be between 3 and 100"
        )

    reviewer = ContactReviewer(
        args.rally,
        args.experiment,
        args.radius,
    )

    reviewer.run()


if __name__ == "__main__":
    main()
