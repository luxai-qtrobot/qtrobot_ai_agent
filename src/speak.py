"""
speak.py - Speaker: owns TTS playback plus (optional) barge-in/echo filtering for the
real ASR/TTS pipeline (main.py). The ActionHandle, the lock protecting it, and the
SequenceMatcher echo threshold are all private to this class - callers just await
talk() to speak, and feed ASR events into on_interim_speech()/is_echo().
"""

import asyncio
import threading
import time
from difflib import SequenceMatcher

from luxai.magpie.utils import Logger


class Speaker:

    # How similar recognized text has to be to what the robot just said to be
    # treated as self-echo (mic picking up the robot's own TTS through imperfect
    # echo cancellation) rather than genuine user speech.
    ECHO_SIMILARITY_THRESHOLD = 0.5

    # How long after TTS stops (naturally, or cancelled via barge-in) a result is
    # still checked against what was just said. Needed because a barge-in can be
    # triggered by a short echoed fragment that doesn't itself score as an echo (see
    # _is_echo_locked) - it clears the handle before the fuller, more comparable
    # final transcript ever arrives. Without this grace window, that final
    # transcript would skip is_echo() entirely, since "nothing's playing" would
    # otherwise mean "nothing to compare against".
    ECHO_GRACE_SECONDS = 2.0

    # Minimum interim-text length (characters) before treating it as a real
    # barge-in attempt rather than acting on the very first interim callback,
    # which can be a single fragment or word. "low" requires a more substantial
    # fragment before reacting - safest when the mic's echo cancellation is
    # unreliable and prone to leaking brief noise/echo into interim results;
    # "high" reacts immediately to any interim result, even one character.
    BARGE_IN_SENSITIVITY = {
        "high": 0,
        "moderate": 6,
        "low": 15,
    }

    def __init__(self, robot, enable_barge_in: bool = True, on_barge_in=None, engine: str = "azure",
                 barge_in_sensitivity: str = "moderate"):
        """
        robot:                 a connected Robot instance - talk() calls robot.tts.say_text_async().
        enable_barge_in:       if False, on_interim_speech()/is_echo() are no-ops -
                               useful for testing/tuning without barge-in interfering.
        on_barge_in:           called (no args) when a real barge-in happens - e.g.
                               llm_engine.interrupt(). Plain callback, not an LLMEngine
                               reference, so this class stays independently testable
                               and reusable outside this one app (same pattern as
                               WorkingMemory.bind(max_tokens, on_evict=...)).
        engine:                TTS engine name passed to every say_text_async() call.
        barge_in_sensitivity:  one of BARGE_IN_SENSITIVITY's keys - see there.
        """
        self._robot = robot
        self._enable_barge_in = enable_barge_in
        self._on_barge_in = on_barge_in
        self._engine = engine
        self._min_interim_chars = self.BARGE_IN_SENSITIVITY[barge_in_sensitivity]
        self._lock = threading.Lock()
        self._handle = None
        self._text = ""
        self._active_until = 0.0  # monotonic deadline - see ECHO_GRACE_SECONDS

    async def talk(self, text: str) -> None:
        """Speaks one sentence, tracking it internally so on_interim_speech()/
        is_echo() can react while it's playing.

        await asyncio.to_thread(handle.wait) - not a bare handle.wait() - matters
        here: ActionHandle.wait() is a plain blocking call, not a coroutine, so
        calling it directly would stall the whole event loop (and every other task on
        it, including ASR-driven submit()s) for as long as TTS is speaking."""
        handle = self._robot.tts.say_text_async(text, engine=self._engine)
        with self._lock:
            self._handle, self._text = handle, text
        await asyncio.to_thread(handle.wait)
        with self._lock:
            self._handle = None
            self._active_until = time.monotonic() + self.ECHO_GRACE_SECONDS

    def on_interim_speech(self, text: str = "") -> None:
        """Call from the ASR interim-speech callback. Cancels in-flight TTS unless
        nothing's playing, the fragment is too short to trust yet (see
        BARGE_IN_SENSITIVITY), or text looks like the mic picking up our own speech."""
        if not self._enable_barge_in:
            return
        if len(text.strip()) < self._min_interim_chars:
            return
        with self._lock:
            if self._handle is None or (text and self._is_echo_locked(text)):
                return
            handle, self._handle = self._handle, None
            self._active_until = time.monotonic() + self.ECHO_GRACE_SECONDS
        Logger.info("Barge-in: cancelling TTS.")
        handle.cancel()
        if self._on_barge_in:
            self._on_barge_in()

    def is_echo(self, text: str) -> bool:
        """Call from the ASR final-speech callback before submitting - True if text
        looks like the mic picking up our own recently-spoken sentence rather than
        genuine user speech. Checked both while actively speaking and for a short
        grace period afterward (see ECHO_GRACE_SECONDS) - always False once that
        window has passed, or when barge-in filtering is disabled."""
        if not self._enable_barge_in:
            return False
        with self._lock:
            if self._handle is None and time.monotonic() >= self._active_until:
                return False
            return self._is_echo_locked(text)

    def _is_echo_locked(self, text: str) -> bool:
        """Assumes the caller already holds self._lock.

        Compares text against a same-length PREFIX of the last-spoken sentence, not
        the whole thing. A short interim fragment (or a final result truncated by a
        barge-in cutting TTS off mid-sentence) naturally scores low via
        SequenceMatcher against a much longer full sentence even when it IS a
        perfect echo, since the ratio is dominated by the length difference rather
        than actual similarity - e.g. "I'm sorry." vs "I'm sorry, I didn't quite
        understand that. Could you say it again?" scores far lower than "I'm
        sorry." vs "I'm sorry,", even though both are the same echo."""
        text_norm = text.strip().lower()
        last_norm = self._text.strip().lower()
        if not text_norm or not last_norm:
            return False
        prefix = last_norm[:len(text_norm)]
        ratio = SequenceMatcher(None, text_norm, prefix).ratio()
        return ratio >= self.ECHO_SIMILARITY_THRESHOLD
