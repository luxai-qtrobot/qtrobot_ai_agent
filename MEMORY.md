# Memory

Five distinct memory concepts, each with a different job, orchestrated by
`RobotMemory` (`memory/robot_memory.py`) into one message list handed to the
LLM on every call. For how memory fits into the wider conversation engine,
see [ARCHITECTURE.md](ARCHITECTURE.md).

| Tier | Class | Lifetime | Mutability | Purpose |
|---|---|---|---|---|
| Static | plain string | process lifetime | never trimmed | system prompt |
| WRM (working) | `WorkingMemory` | until evicted | append/compact | raw recent conversation |
| STM (short-term) | `ShortTermMemory` | until re-summarized | folded in place | condensed narrative of older content |
| LTM (long-term) | `LongTermMemory` | persisted to disk | append-only | durable, searchable archive + documents |
| WSM (world state) | `WorldStateMemory` | this instant | add/remove | what's happening *right now* |

Everything ends up as one flat `messages` list via `RobotMemory.get()` -
`[static, doc_index, *working, stm_summary, world_state]` - trimmed to the one
absolute token ceiling (`RobotMemory.max_tokens`) as the very last step,
non-destructively.

---

## `RobotMemory` - the orchestrator

```python
RobotMemory(static=SYSTEM_PROMPT, max_tokens=31000,
            working=WorkingMemory(size_ratio=0.75, flush_ratio=0.2),
            short_term=ShortTermMemory(llm=client, model=MODEL, size_ratio=0.25),
            long_term=long_term_instance,
            world_state=WorldStateMemory())
```

Only `RobotMemory` knows the real token ceiling. Each tier only knows its own
local, ratio-derived budget (`size_ratio` of `max_tokens`) and is otherwise
self-contained - they never need runtime awareness of each other or of the
parent. `working.bind(max_tokens, on_evict=...)` and
`short_term.bind(max_tokens, on_evict=...)` wire the eviction cascade:
WRM evicts → feeds STM's running summary (and, via a second callback, an
unlossy raw copy into LTM); STM's own eviction feeds LTM directly.

### `get()` assembly order, and why

```
[0] static system prompt              - stable instruction prefix, never trimmed
[1] document index (if LTM has docs)  - stable/instructional, front-loaded
[2..] working memory's raw messages   - the actual conversation
[N-1] short-term summary (if any)     - appended last: recency
[N]   world state (if configured)     - appended after that: even fresher
```

Two placement rules, both deliberate:

- **Stable/instructional content goes right after static** (system prompt,
  document index) - front-loaded, where models attend most reliably and
  where it won't blur into fast-changing content that would defeat any
  KV-cache reuse a serving backend does on an unchanged prefix.
- **Fast-changing, recency-sensitive content goes at the very end** - models
  attend most reliably to the *end* of context too, and burying a
  fast-changing summary mid-conversation is the position least likely to
  actually influence the response. Not `role: "system"` either, for the
  same cache-blurring reason - both STM's summary and WSM are `role: "user"`,
  clearly tagged in their own text as memory/context, not literal
  conversation.

### Read-time, non-destructive budget enforcement

`_trim_to_budget()` runs only if the assembled list actually exceeds
`max_tokens`. It never deletes anything from WRM/STM/LTM's own stores - it
only decides what's *included in this one call*:

- Lead messages (static, doc index) are always kept.
- The **current, still-open turn** is never trimmed (`WorkingMemory.
  turn_boundary()` marks the protected region - same protection WRM's own
  eviction gives it, so the read path can't reintroduce the bug of cutting a
  turn's own tool result out of its own follow-up round).
- The two optional **tail messages** (STM summary, WSM) are each kept whole
  or dropped whole - never trimmed piecemeal. WSM is tried first (it's
  later = higher recency value), so it's the one that survives if budget is
  tight enough that both can't fit.
- Everything else (older working-memory messages) is trimmed oldest-first,
  stopping at the first message that doesn't fit rather than skipping over
  it to pack in smaller older ones - a contiguous recent window keeps
  tool_call/tool_result pairing and conversational order intact.

---

## Working Memory (WRM) - `memory/working_memory.py`

Raw, unsummarized recent chat history. Self-managed: image superseding and
overflow eviction happen automatically on `add()`; turn-boundary tool-result
compaction happens on `end_turn()`. Lock-protected throughout, since more
than one thread can touch it at once (the main worker processing a fresh
turn, a background thread resolving an earlier parked one).

- **`add(message)`** - appends; if the message carries an image, any *prior*
  image in memory is stripped first (only the most recent image is
  cognitively useful).
- **`insert_after(anchor, messages)`** - splices `messages` right after
  `anchor`'s *current* position (matched by identity, not index, since
  positions shift as other turns get added) - this is the mechanism behind
  chronologically-correct parked-call resolution (see ARCHITECTURE.md).
  Falls back to appending at the end if the anchor was evicted in the
  meantime.
- **`end_turn()`** - compacts the current turn's own tool_call/tool_result
  exchanges into one short summary line (`[tool actions] name(args) ->
  result`, long results truncated) now that the turn is fully resolved and
  the LLM already used the raw result to answer - only *future* turns need
  less detail. A message spliced in via `insert_after()` does **not** get
  this compaction immediately (it operates on the tail, not an arbitrary
  earlier position) - it stays full/uncompacted until the *next* turn that
  completes normally sweeps it in. Confirmed in practice: a parked call
  resolved before the next `end_turn()` shows fully compacted; one resolved
  after still shows full detail until the following turn closes.
- **Eviction** - triggered when total tokens exceed this tier's own
  ratio-derived budget; evicts oldest-first down to `flush_ratio` below
  budget, never touching the protected (still-open-turn) region, and hands
  evicted messages to STM via the `on_evict` callback wired by `RobotMemory`.

---

## Short-Term Memory (STM) - `memory/short_term_memory.py`

One plain-prose running summary, updated in place. Deliberately simple: no
fact list, no tags, no separate condense step - staying short is just part
of every update (the summarization prompt asks for bullet points, one short
sentence each, under a target token count).

- **`add(messages)`** - never blocks. Archives the raw batch (if an
  `on_evict` callback is wired to LTM) and starts a background
  summarization thread, returning immediately.
- **`get()`** - the current summary, *plus raw text for anything still being
  summarized* (the "pending" holding area) - so nothing the user said is
  ever silently missing from context, just sometimes less compact
  mid-transition.
- Summarization is an LLM call (can take seconds) and always runs on a real
  OS thread, not an asyncio task - a real thread can't be starved by a long
  blocking call elsewhere in the process (the live streaming chat call),
  it just runs.
- **`flush()`** - joins all in-flight summarization threads; called on
  shutdown so nothing pending is silently lost.

---

## Long-Term Memory (LTM) - `memory/long_term_memory.py`

Durable, embedding-searchable archive sitting below STM in the
working → short-term → long-term flow, plus a separate ingestion path for
loaded reference documents. **Both chat history and documents live in the
same vector store**, distinguished only by a `kind` tag (`CHAT_KIND` /
`DOCUMENT_KIND`) - this is what lets `search_memory()` and
`search_documents()` be two distinct LLM-facing tools backed by one index.

- **Embedding**: `fastembed` (ONNX, CPU, in-process) - no extra server, no
  torch, doesn't touch the chat LLM server at all. Off the hot path:
  `add()`/`add_document()` return immediately, embed+store happens on a
  background thread.
- **Search**: two-stage retrieve-then-rerank. Stage 1 is cheap brute-force
  cosine similarity over the whole (optionally kind/time-filtered) store,
  casting a wide net (`RERANK_POOL_SIZE` candidates). Stage 2 runs a
  cross-encoder over that shortlist, scoring query+chunk jointly for real
  relevance - too slow over the whole store, cheap over a short pool. Only
  the reranked order decides what's returned.
- **Time filtering** (`search_memory` only - documents have no meaningful
  "when"): structured buckets (`today`, `yesterday`, `this_week`,
  `last_week`, `this_month`, `earlier`) - not freeform NL parsing, and
  results carry a human-relative `"when"` (`"3 days ago"`), not a raw
  timestamp, since that's what the model can actually reason with.
- **Document index**: `document_index_summary()` gives the LLM an always-on
  list of what's loaded and searchable, deduplicated by summary text, so it
  doesn't need a wasted tool round-trip just to discover documents exist.
- **Persistence**: `save()`/`load()` cover **chat history only** - written
  as plain, human-readable JSON (text/timestamp/meta, no vectors), so the
  file doubles as a readable chat log a user can open or hand-edit.
  `load()` **re-embeds every record fresh** rather than trusting stored
  vectors - an old save file can never go silently stale if the embedding
  model is ever changed later. Documents are never persisted; they're
  cheap to reindex from source files every run (see `document_reader.py`)
  and this guarantees the index always matches what's actually on disk.

### Document ingestion - `memory/document_reader.py`

Deliberately separate from `long_term_memory.py` - that file only knows
text-in/vectors-out; this one only knows filesystem-in/text-out.

- **`meta`** (source/path/modified date) - auto-extracted, handed straight
  to `add_document(meta=...)` unchanged.
- **`summary`** - a human-written one-line "what this is about", used only
  for the LLM's document listing. Can't be reliably auto-extracted, so
  readers never guess it - callers always supply it.
- `DirectoryReader.read(dir, summary=...)` shares one `summary` across every
  file in a batch, which collapses to a single line in the LLM's document
  listing regardless of how many files it came from.

---

## World State Memory (WSM) - `memory/world_state_memory.py`

The newest tier, and the one built specifically to fix real, observed
model failures: stale answers with no sense of elapsed time, self-
contradiction after an action was superseded, and hallucinated
self-unawareness ("I'm just waiting for instructions" while actually
mid-search). Not accumulated like WRM, not summarized like STM: every entry
represents a **currently-true fact**, rendered fresh on every call, and
**never persisted anywhere**.

### The contract

```python
class WorldStateMemoryBase(ABC):
    def add(self, key: str, text: str) -> None: ...
    def remove(self, key: str) -> None: ...
    def finish(self, updates: dict[str, str]) -> str: ...
    def render(self) -> str: ...
```

A plain, thread-safe, keyed store (`dict[key, text]` under one lock) -
deliberately generic. It has **no opinion about content or tags** - the
owner adding an entry picks its own tag (`ToolEngine`/`LLMEngine` always
write `[bg action]: ...`; a future perception component would write
`[state]: ...`). WSM doesn't validate or interpret any of it.

### Why it's a reserved slot, not a conditional append

`RobotMemory.get()` appends WSM's render **unconditionally** whenever a
`world_state` tier is configured - never omitted just because it's
currently empty. `render()` with no entries returns just the header, with
no fabricated "nothing in progress" body text (a claim like that doesn't
generalize once entries can be `[state]` facts, not just `[bg action]`s -
it's not this class's place to invent content).

This mattered in practice: when WSM was implemented as a *conditional*
append (only present when non-empty), a model that had just said "I am
currently searching..." would go on to answer a later "what are you doing?"
by falling back on that stale self-statement, because an *absent* section
is too weak a signal to override a strong, recent, first-person claim
already sitting in the transcript. An always-present section - even with
just the bare header and no entries - gives the model something concrete
to reconcile against, rather than something to simply not notice.

### Lifecycle: who adds, who removes, and when

WSM has **no expiry logic of its own**. Every entry is removed by whichever
component added it, the moment its own condition ends - there's no timeout,
no generic "stale entry" sweep.

**Background tool/agent calls** (today's only real producer, driven by
`LLMEngine`, `llm/llm_engine.py`):

1. **Dispatch** (`_wsm_add`) - right before a tool call is handed to
   `ToolEngine.execute_async()`, an entry goes in:
   `[bg action]: search_web({"query": "..."}) - in progress`. This is what
   makes the action visible to *any* concurrently-processed round, not just
   the one that dispatched it - the actual fix for the "what are you
   doing?" failure mode.
2. **Resolve** (`_wsm_finish`) - the moment a result comes back, *before*
   the synthesis completion is even invoked: mark the entry `"- just
   finished"`, capture one fresh render of the *whole* current world state
   (atomically, under one lock - see `finish()` below), and delete the
   entry - all before the completion call runs. That captured render is
   what gets appended to the one completion that's reacting to this
   result; nothing needs to remember to clean it up afterward, because it's
   already gone.
3. **Discard** (`_wsm_discard`) - if a batch was cancelled rather than
   resolved, its entry is just removed, no "finished" narration (the
   cancellation reply already covers what the user sees).

This **delete-before-invoke** design (rather than delete-after-the-response)
is deliberate and fixes two real bugs found during design:

- **No orphaned entries across nested tool rounds.** If a parked round's
  own follow-up asks for *another* tool call, that batch gets its own
  add/finish/discard cycle, right at that point in the loop - not a single
  cleanup at the very end using only the first batch's ids. Every batch
  that resolves cleans up its own ids as part of handling it.
- **No stale-visibility race.** Deleting only after the (possibly
  multi-second) LLM response would leave a window where a *different*,
  concurrently-processed round could render WSM and see an entry that's
  about to disappear anyway. Deleting immediately at request-build time
  means the store is always internally consistent.

**`cancel()`** (the explicit "stop everything" case) clears WSM entries for
its cancelled ids **immediately** too, via `ToolEngine.cancel_all()`'s
return value - not waiting for the eventual resolve-triggered discard
above. This matters because a tool without a cancel-tool pairing can keep
running in the background for a while after being cancelled; without this,
its `"in progress"` entry would stay visible - and misleading - for that
whole window even though the user already cancelled it.

### `finish()` - the one atomic operation

```python
def finish(self, updates: dict[str, str]) -> str:
    with self._lock:
        self._entries.update(updates)
        text = self._render_locked()
        for key in updates:
            self._entries.pop(key, None)
        return text
```

Update → render → delete, all under one lock acquisition - no other thread
can observe the "just finished" text and then also see it lingering, and no
other thread's own `add()`/`remove()` can interleave mid-sequence.

### What deliberately does *not* go in WSM

The real `tool_calls`/`tool_result` message pair **never** appears in WSM -
that only ever lives in WRM, added exactly once when the round resolves,
via the normal splice mechanism. WSM entries are always plain descriptive
lines (`"[bg action]: search_web(...) - in progress"`), never a parallel
structured representation of a tool call - mixing those shapes would
confuse the model's trained expectation of the tool_calls/tool_result
pairing shape, and create two sources of truth for the same fact.

### Not yet built

`[state]` entries (person in camera view, force detected on an arm, etc.)
are designed for - the `add`/`remove` contract is general enough to host
them - but no perception component exists yet to drive them. When one is
built, it owns its own add/remove timing entirely (with its own debounce
logic so a one-frame flicker doesn't spam add/remove), completely
independent of the tool-call lifecycle above.
