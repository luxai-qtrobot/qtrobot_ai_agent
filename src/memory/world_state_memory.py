"""
world_state_memory.py - WorldStateMemory: transient "what's true right now" memory -
in-flight background actions (tool/agent calls) and live environment/sensor state.

Not accumulated like WorkingMemory, not summarized like ShortTermMemory: every entry
represents a currently-true fact, keyed by whoever owns it, rendered fresh on every
render() call, and never persisted anywhere. WSM has no expiry logic of its own -
every entry is removed by whichever component added it, the moment its own condition
ends (a tool call resolves, a sensor stops detecting something - see finish() for the
tool-call case, a perception component would call add()/remove() directly for
the sensor case). Never holds the real tool_calls/tool_result pair - that only ever
lives in WorkingMemory; WSM entries are plain descriptive lines like
"[bg action]: search_web(...) - in progress", never a parallel structured
representation, so the model's trained tool_calls/tool_result shape stays intact.

Thread-safe: add()/remove()/finish()/render() can all be called from different
concurrent threads (the main worker thread dispatching a fresh turn, a background
thread resolving an earlier parked one, a future perception loop reporting sensor
state) - every public method acquires self._lock.
"""

import threading

from .base import WorldStateMemoryBase

HEADER = "[World state - background activity and environment, not something the user said]"


class WorldStateMemory(WorldStateMemoryBase):

    USAGE_NOTE = '''
        A message beginning with "[World state ...]" is live system context, not something the user said.
        Entries marked "[bg action]" describe actions currently running.
        Entries marked "[state]" describe things currently true about the robot or its environment.
        Use this information to answer questions about current actions or surroundings and avoid contradicting the current state.
        If no world-state message is present, assume no special background activity or state is known.
    '''

    def __init__(self):
        self._lock = threading.Lock()
        self._entries: dict[str, str] = {}

    def add(self, key: str, text: str) -> None:
        with self._lock:
            self._entries[key] = text

    def remove(self, key: str) -> None:
        with self._lock:
            self._entries.pop(key, None)

    def finish(self, updates: dict[str, str]) -> str:
        """Atomically: overwrite each key in updates with its final ('just finished')
        text, render the whole current state (including any OTHER, still-running
        entries), then remove exactly these keys - used right before invoking the one
        completion that should see the finished note, so nothing needs to remember to
        clean these up later. Returns the rendered snapshot to append to that call's
        context; the caller decides where."""
        with self._lock:
            self._entries.update(updates)
            text = self._render_locked()
            for key in updates:
                self._entries.pop(key, None)
            return text

    def render(self) -> str:
        with self._lock:
            return self._render_locked()

    def _render_locked(self) -> str:
        if not self._entries:
            return HEADER
        return HEADER + "\n" + "\n".join(self._entries.values())
