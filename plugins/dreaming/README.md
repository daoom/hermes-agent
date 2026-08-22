# Dreaming — automatic memory consolidation

Reference implementation from [Willow 2.0](https://github.com/rudi193-cmd/willow-2.0), adapted to the Hermes plugin interface. Addresses [issue #25309](https://github.com/NousResearch/hermes-agent/issues/25309).

## What it does

Three-phase pipeline that runs automatically during idle periods:

| Phase | What happens |
|---|---|
| **Light Sleep** | Scans recent direct-user transcript records, filters scaffolding and unsafe source material, stages schema-versioned observations for review, and aggregates bounded duplicates/paraphrases with provenance. Observation is deliberately wider than promotion and carries no claim about what a sentence means |
| **REM** | Sends safe candidates as untrusted structured data to the configured Hermes auxiliary LLM, validates the JSON response, and otherwise writes a deterministic fallback to `DREAMS.md` |
| **Deep Sleep** | Re-derives promotion-critical fields from exact source text, runs the two-model preference pipeline below, re-fetches and re-validates provenance at the write boundary, and atomically promotes at most one preference per cycle to `memories/MEMORY.md` — and only when the promotion mode explicitly allows a write |

## Trust model

Semantic authority for automatic promotion is **agreement between two independently prompted models**:

Both stages are given the **entire exact authoritative source message**, character-for-character, re-read from the `state.db` row at assessment time. Light Sleep stages sentence fragments, but a fragment is only ever used to locate and bind its authoritative row — it is never what a model is asked to judge, so a second speech act elsewhere in the same message cannot disappear before the trust boundary.

1. **Extractor** (`dreaming.preference.extract.v1`) returns one typed assessment per supplied candidate: decision, subject, assertion mode, durability scope, whether a second speech act is present, a character-for-character echo of the source, and a bounded `preference_object`.
2. Only records that are `decision=preference`, `subject=user`, `assertion_mode=direct`, `durability_scope=stable`, and `additional_speech_act=false` continue.
3. **Verifier** (`dreaming.preference.verify.v1`) is routed to a different provider/model, is given the complete exact utterance and the proposed object, and is told nothing about how that object was produced — no extractor rationale, no confidence. It independently judges direct user assertion, standing durability, full entailment, whether the complete utterance is accounted for, single speech act, and absence of retraction or condition.
4. Promotion requires `verdict=accept`, every check exactly `true`, and an empty reason list.

Deterministic code owns everything that is not a semantic judgement: exact schemas, candidate/evidence identity binding, exact source echoes, enums, strict booleans, object bounds, canonical deduplication, provenance re-verification, the per-cycle cap, and the atomic write. It renders the memory sentence itself as `User prefers {preference_object}.` — models control only the bounded object, never the subject, the bullet syntax, or the sentence.

### What this does not guarantee

There is no allow grammar, separator table, or phrase list acting as a third semantic authority, and code cannot decide whether an utterance means one thing or two. If **both** independently prompted models wrongly accept the same mixed utterance, nothing downstream will catch it. Using two different model classes and two independently written prompts reduces but does not eliminate correlated error, and both routes currently point at the same vendor, which reduces the diversity further. That residual risk is accepted deliberately and is the reason automatic writes stay behind a separate gate.

Any malformed extraction or verification item — bad JSON, wrong envelope, extra or missing keys, count mismatch, unknown or duplicate ID, foreign evidence ID, altered source echo, invalid enum, non-strict boolean, unknown reason code, or a verdict contradicting its own checks — invalidates the **entire** semantic batch. No valid-looking sibling record may promote from a contract-violating response. Malformed or negative output is never retried into acceptance; only transport failure gets one bounded retry.

A **failed semantic batch does not consume the staged candidates**: a malformed extractor or verifier response, a provider failure surviving the bounded retry, and an unconfigured semantic route all leave `staging.jsonl` byte-for-byte intact so the same candidates can be reassessed in a later clean cycle. A batch is clean — and staging is consumed as usual — when both models answered within contract, including when every answer was negative, and when nothing was admitted to either model by the objective gates.

Configuring both stages to the **identical provider/model pair** removes the boundary entirely, so exact `(provider, model)` equality is refused with `SemanticRouteError` before any call is made. The same provider with two *distinct* model identifiers stays valid — the boundary is the model pair, not the vendor. Code cannot tell whether two different identifiers alias the same deployed weights; establishing genuine model diversity remains a real-provider acceptance question, not something this check can settle.

No confidence score is accepted or used as authorization.

### Bounded batches

Both stages echo every `source_text` character-for-character inside a structured response capped at 3072 tokens, so the source a batch carries dominates what the models must emit. Three explicit finite constants in `_preference_semantics.py` bound that, and all three are checked before either call:

| Constant | Value | Rationale |
|---|---|---|
| `MAX_SOURCE_CANDIDATES` | `30` | Never larger than the scheduler's own semantic batch limit, so the cap that selects candidates and the cap that guards the call cannot drift apart |
| `SOURCE_TEXT_MAX_CHARS` | `2000` | One ordinary Hermes/Discord user message — 2000 is the Discord single-message ceiling — so a complete normal utterance is admitted intact |
| `TOTAL_SOURCE_MAX_CHARS` | `6000` | Aggregate echo budget. At the ~3 characters per token compact JSON tokenizes to, this is roughly 2000 of the 3072 available output tokens, leaving the remainder for per-item keys, enums, and bounded objects |

A source is **never truncated, split, or fragmented to fit**: a trimmed source is no longer the complete utterance both stages are required to judge. `SOURCE_TEXT_MAX_CHARS` is also declared as `maxLength` on `source_text` in both JSON schemas.

In the scheduler flow these are objective gates rather than errors. An authoritative message longer than `SOURCE_TEXT_MAX_CHARS` can never fit, so it is review-only with reason `source_too_large` and is **retired** — retrying it forever would pin `staging.jsonl` on an input no cycle could ever complete. A message that fits but falls outside the cycle's committed source budget is review-only with reason `source_budget_deferred` and stays staged for the next cycle, exactly like overflow past the batch limit.

These bounds keep a batch proportionate; they do not prove that every admitted batch fits a given provider's output cap. A response the provider truncates simply fails local validation, and nothing promotes.

### Concurrent staging safety

`staging.jsonl` is appended by session hooks and light sleep while a cycle may be running. Appends and partial consumption share one stable advisory lock on the sidecar `staging.jsonl.lock` — a sidecar rather than the file itself, because the atomic `os.replace` swaps the inode out from under any open descriptor. The lock is held only across local file work, never across a provider call.

A cycle captures the exact byte snapshot of staging it read, and a clean partial consumption retires selected identities **only inside that snapshot's prefix**, preserving any concurrently appended suffix byte-for-byte. If the current file no longer begins with that snapshot — staging was rewritten or truncated mid-cycle — nothing is guessed: the file is left exactly as found. The replacement itself remains atomic. On platforms without `fcntl` the lock degrades to a no-op and only the snapshot/suffix discipline applies.

## Opt-in

Disabled by default. Configure it per profile under
`plugins.entries.dreaming.settings` in that profile's `config.yaml`:

```yaml
plugins:
  entries:
    dreaming:
      settings:
        enabled: true
        promotion_mode: shadow
        rem:
          mode: auto
          provider: mistral
          model: mistral-small-latest
        extract:
          provider: mistral
          model: dreaming-rem
        verify:
          provider: mistral
          model: hindsight-extract
```

## Configuration

All behavioral controls are profile-scoped. Secrets remain in the provider's
private environment/configuration; Dreaming does not accept process-global
behavior overrides. Unknown keys, wrong types, non-finite numbers, and values
outside the documented bounds invalidate the profile settings and disable the
entrypoint before scheduling or provider work.

| Setting under `plugins.entries.dreaming.settings` | Default | Description |
|---|---|---|
| `enabled` | `false` | Enables hooks, commands, and the unattended entrypoint for this profile only |
| `promotion_mode` | `off` | `off` \| `shadow` \| `auto`; only `auto` can write memory |
| `extract.provider` / `extract.model` | blank | Independent extractor route; either blank component closes the semantic route |
| `verify.provider` / `verify.model` | blank | Independent verifier route; either blank component closes the semantic route |
| `rem.mode` | `auto` | `auto` uses the host LLM facade when available; `off` uses deterministic fallback |
| `rem.provider` / `rem.model` | `mistral` / `mistral-small-latest` | REM narration route only |
| `llm_timeout_seconds` | `60` | Finite number greater than zero and no more than 300 |
| `min_hours` | `24` | Minimum hours between cycles, from 0 through 8760 |
| `min_sessions` | `5` | Minimum queued sessions before a cycle, from 0 through 100000 |
| `quiet_minutes` | `60` | Required inactivity window, from 0 through 10080 minutes |
| `lookback_days` | `7` | Source lookback window, from 1 through 3650 days |
| `min_score` | `0.72` | Finite promotion threshold in `[0.0, 1.0]` |
| `max_promotions` | `1` | Per-cycle automatic promotion cap, restricted to 0 or 1 |
| `memory_char_limit` | `12000` | Target memory capacity limit, from 0 through 10000000 characters |

The REM narration route is independent: it is never reused as an extractor or verifier route, and the verifier route never falls back to the extractor route. Both semantic routes must be configured explicitly or no semantic call is attempted at all.

Interactive `/dream run` and `/dream preview` calls use the host plugin LLM facade, so configured provider/model overrides must also be allowed by `plugins.entries.dreaming.llm`. Scheduled cycles use the host auxiliary client directly but enforce the same validated profile settings, semantic route, and runtime-identity checks.

### Modes

- **`off`** — no semantic calls at all and no automatic writes. Every candidate is review-only with reason `promotion_mode_off`. This is the default; any unrecognized value invalidates the profile settings before scheduling or provider work.
- **`shadow`** — extraction, verification, policy, scoring, and the write-boundary re-validation all run, and the cycle reports `would_promote`, but `_write_to_memory` is never called.
- **`auto`** — the only mode that may write, and still subject to every gate above.

`/dream preview` never writes `MEMORY.md`, never consumes staged candidates, and never records a completed cycle or resets the queued-session count. It does run Light Sleep and REM: newly scanned messages may be staged, the Light Sleep scan cursor may advance, and a `DREAMS.md` entry may be appended. A non-preview `shadow` cycle is a real cycle — it clears staging on a clean semantic batch and advances the completed-cycle state — it simply cannot write memory. A cycle whose semantic batch failed keeps its staging in every mode.

## Scoring

Candidates that pass deterministic source and policy gates are scored using these calibrated weights:

| Dimension | Weight |
|---|---|
| Relevance | 28% |
| Direct-source quality | 24% |
| Durability | 18% |
| Independent-session support | 12% |
| Recency | 8% |
| Novelty vs existing memory | 7% |
| Conceptual richness | 3% |

Promotion threshold: **0.72**. Durability is raised above its neutral baseline only after a verifier acceptance, so ranking never rewards a candidate the second model refused. Repetition inside one session does not increase independent-session support. Candidates below threshold remain review-only, and the pilot permits at most one automatic promotion per profile per cycle.

A computed score that is **non-finite** — `nan` or an infinity, however the scoring inputs came to produce one — cannot be compared against the threshold in either direction, so it can never authorize a write. Such a candidate is review-only with reason `invalid_score`, and its decision record carries `score: null` rather than a value that is not a number.

## Policy gates

Injected summaries, markdown scaffolding, tool/scheduler material, code blocks, volatile task state, assistant speculation, prompt-injection phrases, memory-management meta-talk, and recognizable secret material are rejected before either model is asked anything. These floors are re-run against the **complete** authoritative message, not the staged fragment, so blocked material anywhere in the utterance keeps the whole message away from both models. Promotion re-derives canonical text, identity, source quality, relevance, and durability rather than trusting cached staging fields. Every claimed session/message source must resolve to an active direct-user row in `state.db` whose session ID matches and whose text the candidate binds to exactly — as the complete message or as one exact sentence of it; unverifiable or mixed forged provenance remains review-only.

Candidate records are **schema v3**. Legacy records written by an earlier schema keep their cached claims and are therefore review-only — they are never upgraded into the semantic pipeline.

At the write boundary the source rows are re-fetched and provenance is re-validated, and both model records are re-bound to the freshly derived source: the entire message is compared character-for-character against the text the models were bound to. A source row that changed — including in a part of the message the staged candidate never covered — was deactivated, changed role or session, or was rewound between assessment and the write fails closed. The rendered fact carries its own `promotion_key`, separate from source and staging identity, so two distinct sources rendering the same fact write once.

## Meta-entry filter

Entries about memory management itself (e.g. "memory is full", "update SKILL.md", "memory capacity") are detected and left out of `MEMORY.md`. They still appear in `DREAMS.md` for review, but the plugin does not auto-write them to skills or long-term memory. This addresses the most common source of memory rot described in [#25309](https://github.com/NousResearch/hermes-agent/issues/25309#issuecomment-4638538182).

## Rollout status

Initial operation is **shadow-first**. Enabling `auto` on any profile requires, as a separate authorization:

- a frozen adversarial corpus evaluated against the real extractor and verifier with zero unsafe accepts;
- at least seven consecutive successful scheduled shadow cycles spanning seven days;
- at least 25 semantically assessed real candidates, with every `would_promote` reviewed by hand.

**That shadow acceptance gate has not been run.** No real-provider acceptance, alias, profile-config, deployment, or live-promotion step is part of this change, and no test in this repository can establish semantic safety: the test doubles prove plumbing, contract enforcement, and the deterministic gates only.

## Slash commands

```
/dream            show status + last diary entry
/dream run        force a consolidation cycle immediately
/dream preview    run scoring and both semantic stages without writing MEMORY.md
/dream status     show mode, all three routes, hours since last cycle, ready state
/dream diary      show the last dream diary entry
```

`/dream status` reports the promotion mode and the REM, extractor, and verifier provider/model pairs. It reports route names only; no credential is ever read or displayed.

## File layout

```
{HERMES_HOME}/
  memories/MEMORY.md      ← promoted entries appended here
  DREAMS.md               ← REM narrative diary
  dreams/
    staging.jsonl         ← candidates queued from session_end hooks
    state.json            ← last_dream_at, sessions_since_dream
    lock                  ← present while a cycle is running
```

## Without an LLM

The plugin does not require Ollama, a local server, or any local model. If `ctx.llm` is unavailable, the cron path uses Hermes' auxiliary client with the same routes, the same schemas, and the same local validators. If either semantic route is unconfigured or unavailable, Light Sleep, deterministic REM fallback, scoring, review decisions, and diary writes still work, but Deep Sleep fails closed and performs no automatic memory promotion.

## Relationship to Willow 2.0

In Willow this runs as `dream_run` (SAP MCP tool) + `dream_check` (gate) + `scripts/sleep_consolidation.py` (nightly batch) + `tension_scan` for semantic deduplication. The Hermes plugin is a standalone port — no Willow dependency.
