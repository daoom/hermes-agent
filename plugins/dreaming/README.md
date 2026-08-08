# Dreaming — automatic memory consolidation

Reference implementation from [Willow 2.0](https://github.com/rudi193-cmd/willow-2.0), adapted to the Hermes plugin interface. Addresses [issue #25309](https://github.com/NousResearch/hermes-agent/issues/25309).

## What it does

Three-phase pipeline that runs automatically during idle periods:

| Phase | What happens |
|---|---|
| **Light Sleep** | Scans recent direct-user transcript records, filters scaffolding and unsafe source material, stages schema-versioned observations for review (an observation outside the direct-assertion grammar is staged without an assertion envelope and stays review-only), and aggregates bounded duplicates/paraphrases with provenance |
| **REM** | Sends safe candidates as untrusted structured data to the configured Hermes auxiliary LLM, validates the JSON response, and otherwise writes a deterministic fallback to `DREAMS.md` |
| **Deep Sleep** | Re-derives promotion-critical fields from exact source text, requires a complete assertion plus a current semantic assessment, applies calibrated scoring, and atomically promotes at most one direct stable preference per cycle to `memories/MEMORY.md` |

## Opt-in

Disabled by default. Enable with:

```bash
export HERMES_DREAMING=1
```

## Configuration

| Variable | Default | Description |
|---|---|---|
| `HERMES_DREAMING` | _(unset)_ | Set to `1` to enable |
| `HERMES_DREAM_REM_MODE` | `auto` | `auto` uses `ctx.llm` when available; `off` always uses deterministic fallback |
| `HERMES_DREAM_PROMOTION_MODE` | value of `HERMES_DREAM_REM_MODE` | Only `auto` enables semantic promotion; `off`, empty, unknown, or malformed values fail closed and make every candidate review-only |
| `HERMES_DREAM_PROVIDER` | _(unset)_ | Optional Hermes provider override for REM; gated by `plugins.entries.dreaming.llm.allow_provider_override` |
| `HERMES_DREAM_MODEL` | _(unset)_ | Optional Hermes model override for REM; gated by `plugins.entries.dreaming.llm.allow_model_override` |
| `HERMES_DREAM_LLM_TIMEOUT` | _(unset)_ | Optional timeout in seconds for REM LLM calls |
| `HERMES_DREAM_MIN_HOURS` | `24` | Minimum hours between cycles |
| `HERMES_DREAM_MIN_SESSIONS` | `5` | Minimum sessions queued before cycle runs |
| `HERMES_DREAM_MIN_SCORE` | `0.72` | Minimum calibrated score after policy gates |
| `HERMES_DREAM_MAX_PROMOTIONS` | `1` | Requested automatic promotions per profile and cycle; the Gate A pilot enforces a hard upper bound of one |
| `HERMES_DREAM_MEMORY_CHAR_LIMIT` | `12000` | Fail-closed capacity limit for the target memory file |
| `HERMES_DREAM_POLL_SECONDS` | `300` | Background thread poll interval (seconds) |

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

Promotion threshold: **0.72**. Positive and negative fixtures keep a margin on both sides of that boundary. Repetition inside one session does not increase independent-session support. Candidates below threshold remain review-only, and the pilot permits at most one automatic promotion per profile per cycle.

## Policy gates

Injected summaries, markdown scaffolding, tool/scheduler material, code blocks, volatile task state, assistant speculation, prompt-injection phrases, and recognizable secret material are rejected before scoring. Promotion re-derives canonical text, relation-derived category, assertion structure, source quality, relevance, and durability rather than trusting cached staging fields. Every claimed session/message source must resolve to an active direct-user row in `state.db` whose session ID and exact sentence match; unverifiable or mixed forged provenance remains review-only. Legacy/unversioned records, incomplete assertions, non-preference categories, and candidates without a valid current semantic assessment also remain review-only. The semantic response is an exact candidate-ID disposition: only a complete direct assertion with stable scope is eligible. Existing-memory similarity is calculated before ranking so duplicate facts are review-only rather than rewarded as novel.

## Meta-entry filter

Entries about memory management itself (e.g. "memory is full", "update SKILL.md", "memory capacity") are detected and left out of `MEMORY.md`. They still appear in `DREAMS.md` for review, but the plugin does not auto-write them to skills or long-term memory. This addresses the most common source of memory rot described in [#25309](https://github.com/NousResearch/hermes-agent/issues/25309#issuecomment-4638538182).

## Slash commands

```
/dream            show status + last diary entry
/dream run        force a consolidation cycle immediately
/dream status     show hours since last cycle, sessions queued, ready state
/dream diary      show the last dream diary entry
```

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

The plugin does not require Ollama, a local server, or any local model. If `ctx.llm` is unavailable, Hermes can use the configured auxiliary provider. If semantic validation is disabled or unavailable, Light Sleep, deterministic REM fallback, scoring, review decisions, and diary writes still work, but Deep Sleep fails closed and performs no automatic memory promotion.

## Relationship to Willow 2.0

In Willow this runs as `dream_run` (SAP MCP tool) + `dream_check` (gate) + `scripts/sleep_consolidation.py` (nightly batch) + `tension_scan` for semantic deduplication. The Hermes plugin is a standalone port — no Willow dependency.
