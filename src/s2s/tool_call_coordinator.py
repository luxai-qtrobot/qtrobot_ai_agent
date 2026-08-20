"""Coordinate asynchronous MCP tool calls emitted by an S2S session.

S2S emits a tool call more than once during its lifecycle: first as streamed
output metadata, then as completed arguments, and finally in ``response.done``.
This coordinator executes each call once, keeps results scoped to their origin
response, and commits a complete batch back to S2S in conversation order.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from luxai.magpie.utils import Logger


_TOOL_CREATE_ID_METADATA_KEY = "s2s_local_tool_create_id"
_IMAGE_CAPTURE_ACKNOWLEDGEMENT = "Image captured and attached."


@dataclass(frozen=True, slots=True)
class _ImagePart:
    image_url: str
    detail: str = "auto"


@dataclass(frozen=True, slots=True)
class _ToolResult:
    call_id: str
    output: str
    images: tuple[_ImagePart, ...] = ()


@dataclass(slots=True)
class _ToolBatch:
    call_ids: set[str] = field(default_factory=set)
    call_sequence: dict[str, int] = field(default_factory=dict)
    output_indices: dict[str, int] = field(default_factory=dict)
    execution_tasks: dict[str, asyncio.Task[None]] = field(default_factory=dict)
    results: dict[str, _ToolResult] = field(default_factory=dict)
    discarded_call_ids: set[str] = field(default_factory=set)
    delivered_call_ids: set[str] = field(default_factory=set)
    completed: bool = False
    cancelled: bool = False

    def ordered_call_ids(self) -> list[str]:
        def order(call_id: str) -> tuple[int, int, int]:
            if call_id in self.output_indices:
                return (0, self.output_indices[call_id], self.call_sequence[call_id])
            return (1, 0, self.call_sequence[call_id])

        return sorted(self.call_sequence, key=order)


class ToolCallCoordinator:
    """Execute and return S2S tool calls without blocking event reception.

    ``client`` must provide the synchronous ``send_event(mapping)`` method used
    by :class:`S2SClient`. ``executor`` normally is ``ToolEngine`` and must
    provide async ``execute(calls)``; ``execute_tracked(calls)`` is preferred
    automatically when available.
    """

    def __init__(self, client: Any, executor: Any) -> None:
        if not callable(getattr(client, "send_event", None)):
            raise TypeError("client must provide send_event(event)")
        if not callable(getattr(executor, "execute", None)) and not callable(
            getattr(executor, "execute_tracked", None)
        ):
            raise TypeError("executor must provide execute(calls)")

        self._client = client
        self._executor = executor
        self._active_response_id: str | None = None
        self._batches: dict[str, _ToolBatch] = {}
        self._batch_order: list[str] = []
        self._background_tasks: set[asyncio.Task[Any]] = set()
        self._delivery_lock = asyncio.Lock()
        self._follow_up_lock = asyncio.Lock()
        self._queued_follow_ups = 0
        self._next_create_sequence = 0
        self._pending_create_id: str | None = None
        self._pending_create_follow_ups = 0
        self._pending_create_saw_response = False
        self._waiting_after_collision = False
        self._pending_background_events: deque[dict[str, str]] = deque()
        self._user_input_pending = False
        self._failure: asyncio.Future[None] = (
            asyncio.get_running_loop().create_future()
        )
        self._closing = False

    def handle_event(self, event: Mapping[str, Any]) -> bool:
        """Observe one raw event from ``S2SClient.receive_events()``.

        Returns ``True`` for tool/lifecycle events understood by the
        coordinator. The caller may still log or otherwise observe the event.
        """

        if self._closing or not isinstance(event, Mapping):
            return False

        event_type = event.get("type")
        if event_type == "input_audio_buffer.speech_started":
            self._user_input_pending = True
            return False
        if event_type == "conversation.item.input_audio_transcription.completed":
            if not str(event.get("transcript") or "").strip():
                self._user_input_pending = False
                self._kick_follow_up()
            return False
        if event_type == "response.created":
            self._user_input_pending = False
            self._handle_response_created(event)
            return True
        if event_type in {"response.output_item.added", "response.output_item.done"}:
            self._handle_output_item(event, schedule=event_type.endswith(".done"))
            return True
        if event_type == "response.function_call_arguments.done":
            response_id = self._response_id(event)
            if response_id is not None:
                self._schedule_call(response_id, event)
            return True
        if event_type == "response.done":
            self._handle_response_done(event)
            return True
        if event_type == "error":
            self._handle_error(event)
            return False
        return False

    def queue_background_event(self, event: Mapping[str, Any]) -> None:
        """Queue one trusted application event for the next safe response."""

        if self._closing:
            raise RuntimeError("tool coordinator is closing")
        if not isinstance(event, Mapping):
            raise TypeError("background event must be a mapping")

        event_type = str(event.get("type") or "").strip()
        event_id = str(event.get("id") or "").strip()
        payload = str(event.get("payload") or "").strip()
        if not event_type:
            raise ValueError("background event requires type")
        if not event_id:
            raise ValueError("background event requires id")
        if not payload:
            raise ValueError("background event requires payload")

        self._pending_background_events.append(
            {"type": event_type, "id": event_id, "payload": payload}
        )
        Logger.info(f"S2S background event queued: {event_type} / {event_id}")
        self._kick_follow_up()

    async def close(self) -> None:
        """Cancel outstanding local executions and stop accepting events."""

        if self._closing:
            return
        self._closing = True
        if any(batch.execution_tasks for batch in self._batches.values()):
            self._cancel_executor_actions()
        tasks = list(self._background_tasks)
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        wait_for_cancellations = getattr(
            self._executor,
            "wait_for_cancellations",
            None,
        )
        if callable(wait_for_cancellations):
            await wait_for_cancellations()
        self._batches.clear()
        self._batch_order.clear()
        self._pending_background_events.clear()
        if not self._failure.done():
            self._failure.cancel()

    async def wait_for_failure(self) -> None:
        """Block until an internal protocol/delivery error must stop the app."""

        await asyncio.shield(self._failure)

    def _handle_response_created(self, event: Mapping[str, Any]) -> None:
        response = event.get("response")
        if not isinstance(response, Mapping):
            return
        response_id = response.get("id")
        if isinstance(response_id, str) and response_id:
            self._active_response_id = response_id

        metadata = response.get("metadata")
        create_id = (
            metadata.get(_TOOL_CREATE_ID_METADATA_KEY)
            if isinstance(metadata, Mapping)
            else None
        )
        if self._pending_create_id is None:
            return
        if create_id == self._pending_create_id:
            self._queued_follow_ups = max(
                0, self._queued_follow_ups - self._pending_create_follow_ups
            )
            self._pending_create_id = None
            self._pending_create_follow_ups = 0
            self._pending_create_saw_response = False
        else:
            self._pending_create_saw_response = True

    def _handle_output_item(
        self,
        event: Mapping[str, Any],
        *,
        schedule: bool,
    ) -> None:
        item = event.get("item")
        if not isinstance(item, Mapping) or item.get("type") != "function_call":
            return
        response_id = self._response_id(event)
        call_id = item.get("call_id")
        output_index = event.get("output_index")
        if response_id is None or not isinstance(call_id, str) or not call_id:
            return
        self._register_call_order(response_id, call_id, output_index)
        if schedule:
            self._schedule_call(response_id, item, output_index=output_index)

    def _handle_response_done(self, event: Mapping[str, Any]) -> None:
        response = event.get("response")
        if not isinstance(response, Mapping):
            return
        response_id = response.get("id")
        if not isinstance(response_id, str) or not response_id:
            return
        if response_id == self._active_response_id:
            self._active_response_id = None
        self._waiting_after_collision = False

        status = response.get("status")
        if status != "completed":
            batch = self._batches.get(response_id)
            if batch is not None:
                self._cancel_batch(response_id, batch)
            self._kick_follow_up()
            return

        output = response.get("output")
        calls: list[tuple[int, Mapping[str, Any]]] = []
        if isinstance(output, Sequence) and not isinstance(output, (str, bytes, bytearray)):
            calls = [
                (index, item)
                for index, item in enumerate(output)
                if isinstance(item, Mapping) and item.get("type") == "function_call"
            ]

        batch = self._batches.get(response_id)
        if calls and batch is None:
            batch = self._batch(response_id)
        for output_index, call in calls:
            call_id = call.get("call_id")
            if isinstance(call_id, str) and call_id:
                self._register_call_order(response_id, call_id, output_index)
            self._schedule_call(response_id, call, output_index=output_index)

        if batch is not None:
            batch.completed = True
            self._kick_delivery()
        self._kick_follow_up()

    def _handle_error(self, event: Mapping[str, Any]) -> None:
        error = event.get("error")
        if not isinstance(error, Mapping) or self._pending_create_id is None:
            return
        if error.get("event_id") != self._pending_create_id:
            return

        error_kind = {error.get("type"), error.get("code")}
        collision = "conversation_already_has_active_response" in error_kind
        rejected_id = self._pending_create_id
        saw_response = self._pending_create_saw_response
        self._pending_create_id = None
        self._pending_create_follow_ups = 0
        self._pending_create_saw_response = False

        if collision:
            # Keep queued outputs pending. A response that won the race must
            # finish before retrying, otherwise response.create can spin.
            self._waiting_after_collision = not saw_response
            if saw_response and self._active_response_id is None:
                self._kick_follow_up()
            return

        self._queued_follow_ups = 0
        message = (
            f"S2S rejected tool follow-up {rejected_id!r}: "
            f"{error.get('message') or error}"
        )
        Logger.error(message)
        if not self._failure.done():
            self._failure.set_exception(RuntimeError(message))

    def _response_id(self, event: Mapping[str, Any]) -> str | None:
        response_id = event.get("response_id") or self._active_response_id
        return response_id if isinstance(response_id, str) and response_id else None

    def _batch(self, response_id: str) -> _ToolBatch:
        batch = self._batches.get(response_id)
        if batch is None:
            batch = _ToolBatch()
            self._batches[response_id] = batch
            self._batch_order.append(response_id)
        return batch

    def _register_call_order(
        self,
        response_id: str,
        call_id: str,
        output_index: Any,
    ) -> None:
        batch = self._batch(response_id)
        batch.call_sequence.setdefault(call_id, len(batch.call_sequence))
        if isinstance(output_index, int) and not isinstance(output_index, bool):
            batch.output_indices[call_id] = output_index

    def _schedule_call(
        self,
        response_id: str,
        call: Mapping[str, Any],
        *,
        output_index: Any = None,
    ) -> None:
        call_id = call.get("call_id") or call.get("id")
        if not isinstance(call_id, str) or not call_id:
            Logger.warning("Ignoring an S2S tool call without call_id")
            return
        batch = self._batch(response_id)
        if batch.cancelled or call_id in batch.call_ids:
            return

        effective_output_index = (
            output_index if output_index is not None else call.get("output_index")
        )
        self._register_call_order(response_id, call_id, effective_output_index)
        batch.call_ids.add(call_id)
        tool_call = {
            "id": call_id,
            "name": call.get("name"),
            "arguments": call.get("arguments") or "{}",
        }
        Logger.info(
            f"S2S tool call: {tool_call['name']}({tool_call['arguments']})"
        )
        task = asyncio.create_task(
            self._execute_call(response_id, batch, tool_call),
            name=f"s2s-tool-{tool_call['name']}-{call_id}",
        )
        batch.execution_tasks[call_id] = task
        task.add_done_callback(lambda _task, cid=call_id: batch.execution_tasks.pop(cid, None))
        self._track(task)

    async def _execute_call(
        self,
        response_id: str,
        batch: _ToolBatch,
        call: Mapping[str, Any],
    ) -> None:
        call_id = str(call["id"])
        name = str(call.get("name") or "<unnamed>")
        try:
            execute = getattr(self._executor, "execute_tracked", None)
            if not callable(execute):
                execute = self._executor.execute
            raw_results = await execute([call])
            if self._is_cancelled_sentinel(raw_results):
                result = None
            else:
                if not isinstance(raw_results, Sequence) or isinstance(
                    raw_results, (str, bytes, bytearray)
                ):
                    raise TypeError("tool executor must return a result sequence")
                if len(raw_results) != 1:
                    raise ValueError(
                        f"tool executor returned {len(raw_results)} results for one call"
                    )
                result = self._normalize_result(call_id, raw_results[0])
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            Logger.error(f"S2S tool {name!r} failed: {exc}")
            result = _ToolResult(call_id, f"Error while running {name}: {exc}")

        if batch.cancelled or self._batches.get(response_id) is not batch:
            return
        if result is None:
            batch.discarded_call_ids.add(call_id)
        else:
            batch.results[call_id] = result
        self._kick_delivery()

    @staticmethod
    def _is_cancelled_sentinel(value: Any) -> bool:
        try:
            from tool.tool_engine import CANCELLED
        except (ImportError, AttributeError):
            return False
        return value is CANCELLED

    @classmethod
    def _normalize_result(cls, call_id: str, raw: Any) -> _ToolResult:
        if not isinstance(raw, Mapping):
            raise TypeError("tool result must be a mapping")
        returned_id = raw.get("tool_call_id")
        if returned_id not in {None, call_id}:
            Logger.warning(
                f"Ignoring mismatched tool result id {returned_id!r}; expected {call_id!r}"
            )

        output = raw.get("output", "")
        images: list[_ImagePart] = []
        if isinstance(output, str):
            output_text = output
        elif isinstance(output, Sequence) and not isinstance(
            output, (bytes, bytearray)
        ):
            output_text, compatible_images = cls._normalize_responses_output(output)
            images.extend(compatible_images)
        else:
            output_text = json.dumps(output, ensure_ascii=False, default=str)

        raw_images = raw.get("images") or []
        if not isinstance(raw_images, Sequence) or isinstance(
            raw_images, (str, bytes, bytearray)
        ):
            raise TypeError("tool result images must be a sequence")
        images.extend(cls._normalize_canonical_image(image) for image in raw_images)
        if images and not output_text.strip():
            output_text = _IMAGE_CAPTURE_ACKNOWLEDGEMENT
        return _ToolResult(call_id, output_text, tuple(images))

    @staticmethod
    def _normalize_canonical_image(raw: Any) -> _ImagePart:
        if not isinstance(raw, Mapping):
            raise TypeError("tool image must be a mapping")
        mime_type = raw.get("mime_type") or raw.get("mimeType")
        data = raw.get("data")
        if not isinstance(mime_type, str) or not mime_type.startswith("image/"):
            raise ValueError("tool image has an invalid MIME type")
        if not isinstance(data, str) or not data:
            raise ValueError("tool image has no base64 data")
        return _ImagePart(f"data:{mime_type};base64,{data}")

    @staticmethod
    def _normalize_responses_output(
        output: Sequence[Any],
    ) -> tuple[str, list[_ImagePart]]:
        text_parts: list[str] = []
        images: list[_ImagePart] = []
        for part in output:
            if not isinstance(part, Mapping):
                continue
            part_type = part.get("type")
            if part_type == "input_text":
                text = part.get("text")
                if isinstance(text, str) and text:
                    text_parts.append(text)
            elif part_type == "input_image":
                image_url = part.get("image_url")
                if isinstance(image_url, str) and image_url:
                    detail = part.get("detail")
                    images.append(
                        _ImagePart(
                            image_url,
                            detail if isinstance(detail, str) and detail else "auto",
                        )
                    )
        text = "\n".join(text_parts)
        if images and not text:
            text = _IMAGE_CAPTURE_ACKNOWLEDGEMENT
        return text, images

    def _kick_delivery(self) -> None:
        if self._closing:
            return
        self._track(
            asyncio.create_task(
                self._deliver_ready_batches(), name="s2s-tool-output-delivery"
            )
        )

    async def _deliver_ready_batches(self) -> None:
        async with self._delivery_lock:
            while self._batch_order:
                response_id = self._batch_order[0]
                batch = self._batches.get(response_id)
                if batch is None or batch.cancelled:
                    self._batch_order.pop(0)
                    continue

                # Deliver outputs as soon as execution finishes. S2S safely
                # defers them behind any still-streaming assistant output and
                # can prefetch the follow-up while TTS is still playing.
                pending_ids = [
                    call_id
                    for call_id in batch.ordered_call_ids()
                    if call_id not in batch.delivered_call_ids
                ]
                if pending_ids:
                    call_id = pending_ids[0]
                    if call_id in batch.discarded_call_ids:
                        batch.delivered_call_ids.add(call_id)
                        continue
                    result = batch.results.get(call_id)
                    if result is None:
                        return
                    self._send_result(result)
                    batch.results.pop(call_id, None)
                    batch.delivered_call_ids.add(call_id)
                    continue

                if not batch.completed:
                    # More calls may still be emitted by the active response.
                    return

                self._batch_order.pop(0)
                self._batches.pop(response_id, None)
                if batch.delivered_call_ids - batch.discarded_call_ids:
                    self._queued_follow_ups += 1

            self._kick_follow_up()

    def _send_result(self, result: _ToolResult) -> None:
        previous_item_id: str | None = None
        if result.images:
            previous_item_id = f"msg_{uuid.uuid4().hex}"
            self._client.send_event(
                {
                    "type": "conversation.item.create",
                    "item": {
                        "id": previous_item_id,
                        "type": "message",
                        "role": "user",
                        "content": [
                            {
                                "type": "input_image",
                                "image_url": image.image_url,
                                "detail": image.detail,
                            }
                            for image in result.images
                        ],
                    },
                }
            )

        event: dict[str, Any] = {
            "type": "conversation.item.create",
            "item": {
                "type": "function_call_output",
                "call_id": result.call_id,
                "output": result.output,
            },
        }
        if previous_item_id is not None:
            event["previous_item_id"] = previous_item_id
        self._client.send_event(event)
        Logger.info(f"S2S tool result returned: call_id={result.call_id}")

    def _kick_follow_up(self) -> None:
        if self._closing:
            return
        self._track(
            asyncio.create_task(
                self._maybe_send_follow_up(), name="s2s-tool-follow-up"
            )
        )

    async def _maybe_send_follow_up(self) -> None:
        async with self._follow_up_lock:
            if (
                self._closing
                or self._active_response_id is not None
                or self._user_input_pending
                or self._pending_create_id is not None
                or self._waiting_after_collision
                or self._batch_order
            ):
                return

            while self._pending_background_events:
                event = self._pending_background_events[0]
                self._send_background_event(event)
                self._pending_background_events.popleft()
                self._queued_follow_ups += 1

            if self._queued_follow_ups == 0:
                return
            self._next_create_sequence += 1
            create_id = f"tool_{self._next_create_sequence}"
            self._pending_create_id = create_id
            self._pending_create_follow_ups = self._queued_follow_ups
            self._pending_create_saw_response = False
            try:
                self._client.send_event(
                    {
                        "event_id": create_id,
                        "type": "response.create",
                        "response": {
                            "metadata": {_TOOL_CREATE_ID_METADATA_KEY: create_id}
                        },
                    }
                )
            except BaseException:
                self._pending_create_id = None
                self._pending_create_follow_ups = 0
                self._pending_create_saw_response = False
                raise

    def _send_background_event(self, event: Mapping[str, str]) -> None:
        text = (
            "[BACKGROUND_EVENT "
            f"type={json.dumps(event['type'], ensure_ascii=False)} "
            f"id={json.dumps(event['id'], ensure_ascii=False)} "
            f"payload={json.dumps(event['payload'], ensure_ascii=False)}]"
        )
        self._client.send_event(
            {
                "type": "conversation.item.create",
                "item": {
                    "id": f"msg_{uuid.uuid4().hex}",
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": text}],
                },
            }
        )
        Logger.info(
            "S2S background event injected: "
            f"{event['type']} / {event['id']}"
        )

    def _cancel_batch(self, response_id: str, batch: _ToolBatch) -> None:
        batch.cancelled = True
        # ToolEngine.cancel_all() can dispatch paired stop tools before local
        # execution tasks are torn down.
        self._cancel_executor_actions()
        for task in tuple(batch.execution_tasks.values()):
            task.cancel()
        self._batches.pop(response_id, None)
        if response_id in self._batch_order:
            self._batch_order.remove(response_id)
        self._kick_delivery()

    def _cancel_executor_actions(self) -> None:
        cancel_all = getattr(self._executor, "cancel_all", None)
        if not callable(cancel_all):
            return
        try:
            cancel_all()
        except Exception as exc:
            Logger.warning(f"Could not cancel active tool actions: {exc}")

    def _track(self, task: asyncio.Task[Any]) -> None:
        self._background_tasks.add(task)

        def done(completed: asyncio.Task[Any]) -> None:
            self._background_tasks.discard(completed)
            if completed.cancelled():
                return
            exception = completed.exception()
            if exception is not None:
                Logger.error(f"S2S tool coordinator task failed: {exception}")
                if not self._failure.done():
                    self._failure.set_exception(exception)

        task.add_done_callback(done)


__all__ = ["ToolCallCoordinator"]
