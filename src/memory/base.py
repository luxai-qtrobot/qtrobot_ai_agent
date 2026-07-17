"""
base.py - Per-role abstract contracts for memory tiers.

Not one shared interface across all tiers - working and short-term memory genuinely
handle different-shaped data (single live messages vs. batches-in/summary-out), so
forcing one common signature would just erase useful type information. Instead, each
role gets its own contract, so a custom implementation (e.g. MyWorkingMemory) can be
swapped in without RobotMemory needing to change.
"""

from abc import ABC, abstractmethod


class WorkingMemoryBase(ABC):
    """Contract for the working-memory role: raw, unsummarized recent chat history."""

    @abstractmethod
    def bind(self, total_tokens: int, on_evict) -> None: ...

    @abstractmethod
    def add(self, message: dict) -> None: ...

    @abstractmethod
    def insert_after(self, anchor: dict, messages: list[dict]) -> None: ...

    @abstractmethod
    def get(self) -> list[dict]: ...

    @abstractmethod
    def turn_boundary(self) -> int: ...

    @abstractmethod
    def end_turn(self) -> None: ...

    @abstractmethod
    def flush(self, notify: bool = True) -> None: ...


class ShortTermMemoryBase(ABC):
    """Contract for the short-term-memory role: condensed, LLM-extracted facts."""

    @abstractmethod
    def bind(self, total_tokens: int, on_evict=None) -> None: ...

    @abstractmethod
    def add(self, messages: list[dict]) -> None: ...

    @abstractmethod
    def get(self) -> str: ...

    @abstractmethod
    def flush(self) -> None: ...


class WorldStateMemoryBase(ABC):
    """Contract for the world-state-memory role: transient, currently-true facts about
    in-flight background activity and live environment/sensor state - never
    accumulated, never summarized, rendered fresh from whatever's currently held."""

    @abstractmethod
    def add(self, key: str, text: str) -> None: ...

    @abstractmethod
    def remove(self, key: str) -> None: ...

    @abstractmethod
    def finish(self, updates: dict[str, str]) -> str: ...

    @abstractmethod
    def render(self) -> str: ...
