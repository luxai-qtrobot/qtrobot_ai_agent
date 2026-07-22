"""
speak.py - Speaker: owns TTS playback plus (optional) barge-in/echo filtering for the
real ASR/TTS pipeline (main.py). The ActionHandle, the lock protecting it, and the
SequenceMatcher echo threshold are all private to this class - callers just await
talk() to speak, and feed ASR events into on_interim_speech()/is_echo().
"""

import asyncio
import threading
from difflib import SequenceMatcher

from luxai.magpie.utils import Logger


class Speaker:

    # How similar recognized text has to be to what the robot just said to be
    # treated as self-echo (mic picking up the robot's own TTS through imperfect
    # echo cancellation) rather than genuine user speech.
    ECHO_SIMILARITY_THRESHOLD = 0.5

    def __init__(self, robot, enable_barge_in: bool = True, on_barge_in=None, engine: str = "azure"):
        """
        robot:            a connected Robot instance - talk() calls robot.tts.say_text_async().
        enable_barge_in:  if False, on_interim_speech()/is_echo() are no-ops - useful
                          for testing/tuning without barge-in interfering.
        on_barge_in:      called (no args) when a real barge-in happens - e.g.
                          llm_engine.interrupt(). Plain callback, not an LLMEngine
                          reference, so this class stays independently testable and
                          reusable outside this one app (same pattern as
                          WorkingMemory.bind(max_tokens, on_evict=...)).
        engine:           TTS engine name passed to every say_text_async() call.
        """
        self._robot = robot
        self._enable_barge_in = enable_barge_in
        self._on_barge_in = on_barge_in
        self._engine = engine
        self._lock = threading.Lock()
        self._handle = None
        self._text = ""

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

    def on_interim_speech(self, text: str = "") -> None:
        """Call from the ASR interim-speech callback. Cancels in-flight TTS unless
        nothing's playing or text looks like the mic picking up our own speech."""
        if not self._enable_barge_in:
            return
        with self._lock:
            if self._handle is None or (text and self._is_echo_locked(text)):
                return
            handle, self._handle = self._handle, None
        Logger.info("Barge-in: cancelling TTS.")
        handle.cancel()
        if self._on_barge_in:
            self._on_barge_in()

    def is_echo(self, text: str) -> bool:
        """Call from the ASR final-speech callback before submitting - True if text
        looks like the mic picking up our own just-spoken sentence rather than
        genuine user speech. Always False once we're no longer actively speaking, or
        when barge-in filtering is disabled."""
        if not self._enable_barge_in:
            return False
        with self._lock:
            return self._handle is not None and self._is_echo_locked(text)

    def _is_echo_locked(self, text: str) -> bool:
        """Assumes the caller already holds self._lock."""
        if not self._text:
            return False
        ratio = SequenceMatcher(None, text.lower(), self._text.lower()).ratio()
        return ratio >= self.ECHO_SIMILARITY_THRESHOLD
