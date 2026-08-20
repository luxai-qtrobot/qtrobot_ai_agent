"""Natural human-attention tracking for QTrobot."""

from __future__ import annotations

import math
import random
import statistics
import time
from dataclasses import dataclass

from luxai.magpie.nodes import BaseNode
from luxai.magpie.utils import Logger


# Tracking policy. These values intentionally live here so the behavior can be
# tuned without changing the robot SDK or the kinematics plugin.
MAX_TRACKING_DISTANCE_METERS = 3.0
MIN_TRACKING_DISTANCE_METERS = 0.35
MIN_PERSON_CONFIDENCE = 0.5
MIN_KEYPOINT_CONFIDENCE = 0.4
MIN_SINGLE_LANDMARK_CONFIDENCE = 0.75

MIN_FOCUS_DURATION_SECONDS = 4.0
MAX_FOCUS_DURATION_SECONDS = 8.0
SWITCH_SCORE_MARGIN = 0.12
SIMILAR_SCORE_MARGIN = 0.08
SWITCH_CONFIRMATION_SECONDS = 0.8
LOST_PERSON_GRACE_SECONDS = 0.8

TARGET_SMOOTHING_ALPHA = 0.45
MAX_TARGET_ANGULAR_SPEED_DEGREES_PER_SECOND = 60.0
MAX_FILTER_TIMESTEP_SECONDS = 0.25
TRACK_FILTER_TTL_SECONDS = 2.0

MAX_FACE_DEPTH_DEVIATION_METERS = 0.25
FACE_TARGET_Z_OFFSET_METERS = 0.15

_FACE_LANDMARKS = (
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
)


@dataclass(frozen=True)
class _PersonCandidate:
    person_id: str
    target: tuple[float, float, float]
    distance: float
    score: float


@dataclass
class _TrackFilterState:
    filtered: tuple[float, float, float]
    last_seen: float


class _TargetFilter:
    """Smooth targets while limiting how fast their direction can change."""

    def __init__(self) -> None:
        self._tracks: dict[str, _TrackFilterState] = {}

    def update(
        self,
        person_id: str,
        target: tuple[float, float, float],
        now: float,
    ) -> tuple[float, float, float]:
        state = self._tracks.get(person_id)
        if state is None:
            self._tracks[person_id] = _TrackFilterState(target, now)
            return target

        elapsed = min(now - state.last_seen, MAX_FILTER_TIMESTEP_SECONDS)
        state.last_seen = now
        smoothed = _blend(
            state.filtered,
            target,
            TARGET_SMOOTHING_ALPHA,
        )

        angular_change = _angle_between_degrees(state.filtered, smoothed)
        maximum_change = (
            MAX_TARGET_ANGULAR_SPEED_DEGREES_PER_SECOND * max(elapsed, 0.0)
        )
        if angular_change > maximum_change > 0.0:
            smoothed = _spherical_blend(
                state.filtered,
                smoothed,
                maximum_change / angular_change,
            )

        state.filtered = smoothed
        return state.filtered

    def prune(self, now: float) -> None:
        expired = [
            person_id
            for person_id, state in self._tracks.items()
            if now - state.last_seen > TRACK_FILTER_TTL_SECONDS
        ]
        for person_id in expired:
            del self._tracks[person_id]


class _AttentionSelector:
    """Keep attention stable while allowing deliberate person changes."""

    def __init__(self) -> None:
        self.current_person_id: str | None = None
        self.focus_started_at = 0.0
        self.current_last_seen_at = 0.0
        self.challenger_id: str | None = None
        self.challenger_since = 0.0

    def select(
        self,
        candidates: dict[str, _PersonCandidate],
        now: float,
    ) -> _PersonCandidate | None:
        current = candidates.get(self.current_person_id or "")
        if current is not None:
            self.current_last_seen_at = now
        elif self.current_person_id is not None:
            if now - self.current_last_seen_at <= LOST_PERSON_GRACE_SECONDS:
                return None
            self._clear_focus()

        if self.current_person_id is None:
            if not candidates:
                return None
            return self._set_focus(max(candidates.values(), key=lambda item: item.score), now)

        current = candidates[self.current_person_id]
        others = [
            candidate
            for person_id, candidate in candidates.items()
            if person_id != self.current_person_id
        ]
        if not others:
            self._clear_challenger()
            return current

        challenger = max(others, key=lambda item: item.score)
        focused_for = now - self.focus_started_at
        clearly_better = challenger.score >= current.score + SWITCH_SCORE_MARGIN
        fair_turn = (
            focused_for >= MAX_FOCUS_DURATION_SECONDS
            and challenger.score >= current.score - SIMILAR_SCORE_MARGIN
        )

        if focused_for < MIN_FOCUS_DURATION_SECONDS or not (clearly_better or fair_turn):
            self._clear_challenger()
            return current

        if challenger.person_id != self.challenger_id:
            self.challenger_id = challenger.person_id
            self.challenger_since = now
            return current

        if now - self.challenger_since < SWITCH_CONFIRMATION_SECONDS:
            return current

        return self._set_focus(challenger, now)

    def _set_focus(
        self,
        candidate: _PersonCandidate,
        now: float,
    ) -> _PersonCandidate:
        self.current_person_id = candidate.person_id
        self.focus_started_at = now
        self.current_last_seen_at = now
        self._clear_challenger()
        return candidate

    def _clear_focus(self) -> None:
        self.current_person_id = None
        self.focus_started_at = 0.0
        self.current_last_seen_at = 0.0
        self._clear_challenger()

    def _clear_challenger(self) -> None:
        self.challenger_id = None
        self.challenger_since = 0.0


class HumanAttentionBehavior(BaseNode):
    """Track an engaged nearby person, or occasionally look around when idle."""

    def __init__(
        self,
        robot,
        *,
        detector_endpoint: str,
        idle_attention_timeout: float = 5.0,
        default_depth: float = 1.0,
        use_vad: bool = False,
        minimum_keypoint_confidence: float = MIN_KEYPOINT_CONFIDENCE,
        maximum_tracking_distance: float = MAX_TRACKING_DISTANCE_METERS,
        look_velocity: float = 300.0,
        tracking_fps: float = 10.0,
    ) -> None:
        if tracking_fps <= 0:
            raise ValueError("tracking_fps must be greater than zero")
        if maximum_tracking_distance <= MIN_TRACKING_DISTANCE_METERS:
            raise ValueError("maximum_tracking_distance is too small")

        self.robot = robot
        self.detector_endpoint = detector_endpoint
        self.idle_attention_timeout = idle_attention_timeout
        self.default_depth = default_depth
        self.use_vad = use_vad
        self.minimum_keypoint_confidence = minimum_keypoint_confidence
        self.maximum_tracking_distance = maximum_tracking_distance
        self.look_velocity = look_velocity
        self.tracking_interval = 1.0 / tracking_fps
        self.reader = None

        now = time.monotonic()
        self.last_idle_look = now
        self.last_valid_person_seen = now
        self.last_tracking_update = 0.0
        self._target_filter = _TargetFilter()
        self._selector = _AttentionSelector()
        super().__init__(name="human-attention")

    def setup(self) -> None:
        self.robot.enable_plugin_local("human-detector")
        configured = self.robot.perception.configure_human_detector(
            endpoint=self.detector_endpoint,
            default_depth=self.default_depth,
            use_vad=self.use_vad,
        )
        if not configured:
            raise RuntimeError(
                "Failed to configure the QTrobot human detector at "
                f"{self.detector_endpoint}"
            )

        self.robot.enable_plugin_local("kinematics")
        self.reader = self.robot.perception.stream.open_human_presence_reader(
            queue_size=0
        )
        Logger.info(
            "Human attention enabled: stable engaged-person tracking; "
            f"maximum distance {self.maximum_tracking_distance:g}m; "
            f"tracking at {1.0 / self.tracking_interval:g} FPS; "
            f"idle look every {self.idle_attention_timeout:g}s"
        )

    def process(self) -> None:
        try:
            frame = self.reader.read(timeout=self.idle_attention_timeout)
        except TimeoutError:
            frame = None

        now = time.monotonic()
        persons = frame.value.get("persons", {}) if frame is not None else {}
        if persons and now - self.last_tracking_update < self.tracking_interval:
            return
        if persons:
            self.last_tracking_update = now

        candidates = self._build_candidates(persons, now)
        self._target_filter.prune(now)
        if candidates:
            self.last_valid_person_seen = now

        previous_person_id = self._selector.current_person_id
        selected = self._selector.select(candidates, now)
        if selected is not None:
            if selected.person_id != previous_person_id:
                Logger.debug(
                    "Human attention focus changed: "
                    f"person={selected.person_id}, "
                    f"score={selected.score:.3f}, "
                    f"distance={selected.distance:.2f}m"
                )
            self._set_look_target(
                selected.target[0],
                selected.target[1],
                selected.target[2] - FACE_TARGET_Z_OFFSET_METERS,
            )
            return

        self._update_idle_attention(now)

    def _build_candidates(
        self,
        persons: dict,
        now: float,
    ) -> dict[str, _PersonCandidate]:
        candidates: dict[str, _PersonCandidate] = {}
        for raw_person_id, person in persons.items():
            candidate = self._build_candidate(str(raw_person_id), person, now)
            if candidate is not None:
                candidates[candidate.person_id] = candidate
        return candidates

    def _build_candidate(
        self,
        person_id: str,
        person: dict,
        now: float,
    ) -> _PersonCandidate | None:
        confidence = _finite_float(person.get("confidence"))
        if confidence is None or confidence < MIN_PERSON_CONFIDENCE:
            return None

        keypoints = person.get("keypoints") or {}
        reliable = []
        for name in _FACE_LANDMARKS:
            keypoint = keypoints.get(name) or {}
            keypoint_confidence = _finite_float(keypoint.get("conf"))
            xyz = _valid_xyz(keypoint.get("xyz"))
            if (
                keypoint_confidence is not None
                and keypoint_confidence >= self.minimum_keypoint_confidence
                and xyz is not None
            ):
                reliable.append((name, keypoint_confidence, xyz))

        if not reliable:
            return None

        median_distance = statistics.median(_length(item[2]) for item in reliable)
        reliable = [
            item
            for item in reliable
            if abs(_length(item[2]) - median_distance)
            <= MAX_FACE_DEPTH_DEVIATION_METERS
        ]
        if not reliable:
            return None

        if len(reliable) == 1:
            name, confidence, _xyz = reliable[0]
            if name != "nose" or confidence < MIN_SINGLE_LANDMARK_CONFIDENCE:
                return None

        target = tuple(
            statistics.median(item[2][axis] for item in reliable)
            for axis in range(3)
        )
        distance = statistics.median(_length(item[2]) for item in reliable)
        if not (
            MIN_TRACKING_DISTANCE_METERS
            <= distance
            <= self.maximum_tracking_distance
        ):
            return None

        target = self._target_filter.update(person_id, target, now)
        score = self._attention_score(person, distance)
        return _PersonCandidate(person_id, target, distance, score)

    def _attention_score(self, person: dict, distance: float) -> float:
        proximity = max(0.0, 1.0 - distance / self.maximum_tracking_distance)
        facing = _facing_score(person, self.minimum_keypoint_confidence)

        if not self.use_vad:
            return 0.6 * facing + 0.4 * proximity

        voice_score = _finite_float((person.get("voice") or {}).get("score")) or 0.0
        voice_score = min(1.0, max(0.0, voice_score))
        return 0.5 * facing + 0.3 * proximity + 0.2 * voice_score

    def _update_idle_attention(self, now: float) -> None:
        if now - self.last_valid_person_seen < self.idle_attention_timeout:
            return
        if now - self.last_idle_look < self.idle_attention_timeout:
            return

        self._set_look_target(
            2.0,
            random.uniform(-1.0, 1.0),
            0.65,
        )
        self.last_idle_look = now

    def _set_look_target(self, x: float, y: float, z: float) -> None:
        try:
            self.robot.kinematics.set_look_target(
                x,
                y,
                z,
                only_gaze=False,
                velocity=self.look_velocity,
            )
        except Exception as exc:
            Logger.warning(f"Human attention look target ignored: {exc}")

    def cleanup(self) -> None:
        if self.reader is not None:
            self.reader.close()
            self.reader = None


def _facing_score(person: dict, minimum_confidence: float) -> float:
    keypoints = person.get("keypoints") or {}
    nose = _valid_uv(keypoints.get("nose"), minimum_confidence)
    left_eye = _valid_uv(keypoints.get("left_eye"), minimum_confidence)
    right_eye = _valid_uv(keypoints.get("right_eye"), minimum_confidence)
    if nose is not None and left_eye is not None and right_eye is not None:
        left_distance = math.dist(nose, left_eye)
        right_distance = math.dist(nose, right_eye)
        total = left_distance + right_distance
        if total >= 1.0:
            asymmetry = abs(left_distance - right_distance) / total
            return (1.0 - asymmetry) ** 2

    yaw = _finite_float((person.get("face") or {}).get("yaw"))
    if yaw is None:
        return 0.0
    return max(0.0, 1.0 - abs(yaw) / 60.0)


def _valid_uv(keypoint: dict | None, minimum_confidence: float):
    if not keypoint:
        return None
    confidence = _finite_float(keypoint.get("conf"))
    uv = keypoint.get("uv")
    if confidence is None or confidence < minimum_confidence or not uv or len(uv) != 2:
        return None
    values = tuple(_finite_float(value) for value in uv)
    return values if all(value is not None for value in values) else None


def _valid_xyz(value) -> tuple[float, float, float] | None:
    if not value or len(value) != 3:
        return None
    xyz = tuple(_finite_float(component) for component in value)
    if any(component is None for component in xyz):
        return None
    if xyz[0] <= 0.0:
        return None
    return xyz


def _finite_float(value) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _length(vector: tuple[float, float, float]) -> float:
    return math.sqrt(sum(component * component for component in vector))


def _angle_between_degrees(
    first: tuple[float, float, float],
    second: tuple[float, float, float],
) -> float:
    denominator = _length(first) * _length(second)
    if denominator <= 0.0:
        return 180.0
    cosine = sum(a * b for a, b in zip(first, second)) / denominator
    return math.degrees(math.acos(min(1.0, max(-1.0, cosine))))


def _blend(
    previous: tuple[float, float, float],
    current: tuple[float, float, float],
    alpha: float,
) -> tuple[float, float, float]:
    return tuple(
        old + alpha * (new - old)
        for old, new in zip(previous, current)
    )


def _spherical_blend(
    previous: tuple[float, float, float],
    current: tuple[float, float, float],
    fraction: float,
) -> tuple[float, float, float]:
    """Interpolate two target directions while also blending their distance."""
    previous_length = _length(previous)
    current_length = _length(current)
    if previous_length <= 0.0 or current_length <= 0.0:
        return _blend(previous, current, fraction)

    previous_unit = tuple(value / previous_length for value in previous)
    current_unit = tuple(value / current_length for value in current)
    cosine = sum(a * b for a, b in zip(previous_unit, current_unit))
    angle = math.acos(min(1.0, max(-1.0, cosine)))
    sine = math.sin(angle)
    if angle < 1e-6 or abs(sine) < 1e-6:
        return _blend(previous, current, fraction)

    previous_weight = math.sin((1.0 - fraction) * angle) / sine
    current_weight = math.sin(fraction * angle) / sine
    distance = previous_length + fraction * (current_length - previous_length)
    return tuple(
        distance * (previous_weight * old + current_weight * new)
        for old, new in zip(previous_unit, current_unit)
    )
