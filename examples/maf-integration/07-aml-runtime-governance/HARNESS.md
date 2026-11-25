# Governance harness: instrumenting the pipeline and sizing its context window

`instrumentation.py` adds a governance harness to the existing healthcare pipeline without
rebuilding it. Everything that varies comes from the `harness` block in the customer JSON,
[`client_pipelines/healthcare_pipeline.json`](client_pipelines/healthcare_pipeline.json), which sits
next to the client's `system` and `policy`. The harness observes the model calls the pipeline
already makes, records one trace entry per agent step, and finds the smallest context window at
which the care decision is still safe.

## Quick start

Requires the local Ollama runtime with `llama3.1:8b`.

```bash
# One instrumented run at a given window (prints one trace record per agent, then the verdict)
python instrumentation.py run --window 1024

# Smallest safe window between lo and hi, to within --resolution tokens
python instrumentation.py size --lo 1024 --hi 4096 --resolution 16
```

`--definition` points at the customer JSON with the harness block. `--records` points at the case
to run; the default is [`harness_cases/healthcare_contraindication.json`](harness_cases/healthcare_contraindication.json),
a synthetic patient whose Amoxicillin order conflicts with a penicillin allergy.

From Python:

```python
import json
from instrumentation import case_json_for, load_harness, run_case, smallest_safe_window

records = json.load(open("harness_cases/healthcare_contraindication.json"))
case = case_json_for("client_pipelines/healthcare_pipeline.json", records)   # definition + harness + records

harness = load_harness(case)            # fails loudly if tokenizer or template don't match the runtime
trace, verdict = run_case(case, 1024)   # one reproducible run at num_ctx=1024
best = smallest_safe_window(case, 1024, 4096)
```

## Long-running projection

[`gaco_longrun.py`](gaco_longrun.py) is the advisory sibling to window sizing. It takes a few
chronological runs, carries the configured agent's retained state from one run to the next, fits
occupied context against run index, and then synthesizes larger retained state to locate and
validate the first trace-integrity failure without waiting for months of wall-clock operation.

```bash
python gaco_longrun.py \
  --definition client_pipelines/healthcare_pipeline.json \
  --records harness_cases/healthcare_contraindication.json \
            harness_cases/healthcare_contraindication.json \
            harness_cases/healthcare_contraindication.json \
  --window 4096 \
  --cadence-seconds 86400
```

The repeated sample above means three chronological executions of the sample workload, not three
pre-planted outcomes. In customer use, pass that customer's chronological case files instead. The
window is an explicit input (or the harness's `model.current_window` when omitted); neither the
window nor a future run horizon is embedded in the algorithm.

The report contains:

- every measured occupied-context value and retained-memory increment;
- the linear fit, R², uncertainty bounds, and an explicit poor/insufficient-fit result when needed;
- the first failing occupied-context interval and the detected onset mechanism (`overflow`,
  `degradation`, `mixed`, or `indeterminate`);
- a projected run/time range when the fit supports one; and
- fresh fast-forward runs just below and at onset, including whether the crossover reproduced.

This is a projection, not a guarantee. It is extrapolated from the supplied runs and assumes the
measured growth relationship continues. Fast-forwarding is an accelerated proxy made from
synthesized accumulated context, not a real month of operation. The module advises and warns; it
does not pin evidence, compact memory, enlarge a window, or otherwise modify client settings.

### Example result (llama3.1:8b, the contraindication case)

```text
harness: llama3.1:8b via langgraph; calibration {'tokens': 30, 'runtime_tokens': 30, 'window': 8192}
window   4096: SAFE   assembled=1384 processed=1384 truncated=False
window   1024: unsafe assembled=1384 processed=514  truncated=True
window   2560: SAFE   assembled=1384 processed=1384 truncated=False
window   1792: SAFE   assembled=1384 processed=1384 truncated=False
window   1408: SAFE   assembled=1384 processed=1384 truncated=False
window   1216: unsafe assembled=1384 processed=610  truncated=True
window   1312: unsafe assembled=1384 processed=658  truncated=True
window   1360: unsafe assembled=1384 processed=682  truncated=True
window   1384: unsafe assembled=1384 processed=694  truncated=True
window   1396: SAFE   assembled=1384 processed=1384 truncated=False
smallest safe window: 1396
```

The care-decision prompt is 1,384 tokens. The demo's `num_ctx=1024` is unsafe; 1,396 is the
smallest safe window found (to within 16 tokens), against an advertised 131,072. A window equal to
the prompt size (1,384) still truncates: Ollama needs the prompt to be strictly smaller than
`num_ctx`, so the true threshold lies between 1,385 and 1,396.

## Long-running demo: early vs. aged context

[`longrun_demo.py`](longrun_demo.py) shows the same case, the same window and the same check twice:
once with a little accumulated history (Case 1) and once with a large one (Case 2). Only the amount
of history in the care-decision agent's context changes.

```bash
python longrun_demo.py                               # window/sizes from harness.longrun_demo
python longrun_demo.py --window 4096 --early 200 --aged 2400 --json report.json
```

- **History** is synthesized prior-run notes (routing, queue, audit, handoff entries), sized with the
  harness's real tokenizer and dated one run per `cadence_days`. A guard refuses any filler that
  matches the evidence format or contains the finding's words, so the demo can't be rigged.
- **Placement** of the finding relative to that history:
  - `recent_end`: all history before the finding (Case 1).
  - `buried_mid`: half before and half after, inside the window (Case 2's first try).
  - `overflowed`: the history grows past the window after the finding (Case 2's retry, used only
    when burial alone doesn't lose it).
- **Mechanism** is what the trace measured:
  - `overflow`: the runtime processed fewer tokens than were assembled, and the finding was in the
    dropped part.
  - `degradation`: the finding was inside the kept window but the agent didn't use it.
  - `indeterminate`: the trace can't settle it.

The history is passed through the existing `accumulated_state` argument. A mapping
`{"before_finding": ..., "after_finding": ...}` places it around the finding; a plain string keeps
the original single slot, and an empty value leaves the prompt byte-identical.

### Live result (llama3.1:8b, window 4096)

```text
|                        | Case 1 (early)                                          | Case 2 (~4 months of history, simulated)                                           |
|------------------------|---------------------------------------------------------|------------------------------------------------------------------------------------|
| occupied context       | 1620 tokens (227 of history, ~7 prior runs over 6 days) | 5521 tokens (4128 of history, ~128 prior runs over 127 days)                       |
| window                 | 4096                                                    | 4096                                                                               |
| finding placement      | near end (after the history)                            | buried mid-history -> overflowed (history after it); burial alone was insufficient |
| finding inside window? | yes                                                     | no (pushed past the kept window)                                                   |
| agent used finding?    | yes                                                     | no                                                                                 |
| decision               | HOLD_FOR_CLINICAL_REVIEW                                | APPROVE_TREATMENT                                                                  |
| governance check       | PASS                                                    | CATCHES LOSS                                                                       |
| failure mechanism      | -                                                       | overflow                                                                           |
```

Burying the finding in the middle of 2,423 tokens of history (3,825 occupied, inside the window) did
not lose it: llama3.1:8b still used it and the check passed, so no degradation was observed at this
size. The loss came when the history grew past the window: 5,521 tokens assembled, 2,050 processed,
and the finding was in the dropped part.

Case 2 is an accelerated simulation, not months of real operation. Case 1 shows no loss observed in
these runs, which is not a guarantee of zero loss. These are real LLM runs, so results are
probabilistic.

## Degradation demo: inside the window, but unused

The overflow demo loses the finding because it falls out of the window. The degradation demo tests
the other long-running failure ("lost in the middle"): the finding stays **inside** the window the
whole time, buried under a large pile of newer history, and the question is whether the model stops
using it.

```bash
python longrun_demo.py --mode degradation          # settings from harness.longrun_demo.degradation
python longrun_demo.py --mode degradation --window 16384 --aged 8000 --depth 0.9 --json report.json
```

- **Large window.** llama3.1:8b handles a 4k context easily, so the demo uses a large window
  (`window`, default 16,384) to leave room to bury the finding deep while it still fits.
- **`depth`** is the fraction of the history that is newer than the finding. At the default 0.9, the
  newest 90% of the history sits on top of it.
- **No truncation, proven.** A Case 2 run only counts as degradation evidence if the runtime processed
  every assembled token (`processed_tokens == assembled_tokens`). A truncated run is marked invalid
  and retried with a larger window (up to `max_window`), so an overflow is never reported as
  degradation.
- **Escalation sweep.** If the model still uses the finding, the history grows by `growth_factor`
  (and the window with it) up to `max_aged_tokens`. The first size where the model stops using the
  finding is reported as the degradation onset.
- **If no onset is found**, the report says "no degradation observed up to N tokens; the model held
  the finding throughout". It never manufactures a loss.

### Live result (llama3.1:8b, depth 0.9): robust

| History tokens | Occupied / processed | Window | Finding used? | Check |
|---|---|---|---|---|
| 8,027 | 9,429 / 9,429 | 16,384 | yes | PASS |
| 12,030 | 13,432 / 13,432 | 16,384 | yes | PASS |
| 18,024 | 19,426 / 19,426 | 24,576 | yes | PASS |
| 27,029 | 28,431 / 28,431 | 32,768 | yes | PASS |

No degradation observed up to 27,029 tokens of accumulated history. With the finding buried under the
newest 90% of it, inside the window and never truncated, llama3.1:8b held the finding throughout. No
loss observed in these runs. The synthesized history is deliberately neutral and repetitive (so it
can't rig the result), which is also easy for a model to skip past.

## GACO: suggest the smallest safe window (advisory)

`gaco.py` suggests a context window for one case; it never sets one or edits the client's pipeline.
Run `python gaco.py --initial-window 4096` (add `--json gaco_report.json` for the full report, or
`--definition` / `--records` for another case). It first measures the **floor** with no model runs:
the exact token count of the decision agent's assembled prompt, plus the reply room the agent
requests (`num_predict`), plus an optional `floor_margin_tokens`. It then probes your initial window
with one real run of the clean case (no synthesized history). If that run passes, it binary-searches
down toward the floor; if it fails, it grows the window (`growth_factor`) up to
`model.advertised_window` or `harness.gaco.max_window`, then searches back down. A window is only
suggested after it passes a repeat run, and it must sit above every window that ever failed. The last
line is the answer: `Suggested context window: W  (N tokens below / above your initial setting of
<initial>).` Above that line, the report prints every probe, so you can audit the search. The floor
is tagged *measured*, and the suggestion *found-by-search*. The suggestion fits *this* case only: for
a workload, run several sample cases and take the largest suggestion. The suggestion means the
evidence fits and survives; it does not prove the model reasons over it well. On this Ollama runtime
the evidence check is *measured-only*. All settings are optional and live in `harness.gaco`.

### Live result (llama3.1:8b, the contraindication case)

| Initial window | Probes (each a real run) | Suggested |
|---|---|---|
| 4,096 (PASS) | 4096 PASS → 1544 PASS → 1544 PASS (repeat) | **1,544** (2,552 below) |
| 1,024 (below the floor, not probed) | 1544 PASS → 1544 PASS (repeat) | **1,544** (520 above) |

The floor is 1,544 = 1,384 prompt + 160 reply room. The runtime processed exactly 1,384 tokens on
every probe, which matches the measured prompt count. For this clean case, the floor already passes,
so the search confirms it. GACO never probes below the floor, so it can't suggest less than that.

## What `load_harness` checks

It refuses to load, with a `HarnessConfigError`, unless:

1. **Tokenizer.** The runtime's own vocabulary and merges (read from Ollama's model metadata)
   rebuild into a tokenizer of the declared `type` and `pre`. Only verified pre-tokenizers are
   supported (`llama-bpe`); anything else fails rather than guessing.
2. **Chat template.** The runtime's Go template is byte-identical to the one the config's Jinja
   template was written for (`runtime_template_sha256`). If Ollama's template changes, the harness
   stops instead of drifting.
3. **Calibration.** A calibration prompt rendered and counted by the harness matches the runtime's
   `prompt_eval_count` exactly.

## The four hooks

| Hook | What it does | Where its inputs come from |
|---|---|---|
| A: assemble + count | Captures the exact request body sent to `/api/chat`, renders it through the chat template, and counts it with the runtime's tokenizer (`assembled_tokens`). | `observability.model_call`, `context_assembly.chat_template`, `model.tokenizer` |
| B: window as parameter | Reads the window from run state (`pipeline.window_state_key` / `window_config_field`) and checks the request sent the same value under `model.window_param`. The pipeline already threads it through, so the hook verifies and never injects. | `pipeline`, `model.window_param` |
| C: truncation at the source | Reads `observability.processed_token_field`; `truncated = assembled_tokens > processed_tokens`. Matches each required finding against the prompt with the structured `evidence_formats` pattern (agent id, version, and finding, not a substring). | `observability`, `evidence_formats`, `agents[].required_evidence` |
| D: trace record | Appends one record per agent step to `state["trace"]`. | all of the above |

### Measured-only vs measured-and-verified

Ollama truncates inside the runtime and never returns the prompt the model actually saw, so this
config sets `can_capture_model_visible_prompt: false`. The harness therefore:

- sets every `evidence_visible` value to `null`, with `model_visibility_confirmed: false` and
  `confirmation: "measured-only"`;
- still records what it can measure: `evidence_present_in_prompt` (in the prompt that was sent) and
  `evidence_in_kept_window_estimate` (whether that finding's position fits in the tail the runtime
  kept, under `truncation_policy: "keep_tail"`);
- sets `token_count_verified: true` when a run is not truncated and the harness's count equals the
  runtime's processed count.

With a runtime that can show the model-visible prompt, `evidence_visible` becomes
`present AND not truncated`, and records are marked `measured-and-verified`.

If the runtime rejects an over-long prompt instead of truncating it, the step is recorded with
`truncated: true` and `overflow_error: true`, and the verdict is `runtime-overflow`. The run does
not crash.

## What "safe" means

`run_case` ingests the run through the existing client-pipeline ingestion and runs the existing
deterministic trace-integrity check. A window is **safe** when:

- trace integrity passes, **and**
- every agent carried each of its `required_evidence` findings through exactly as emitted.

The second condition matters. The existing integrity check skips an agent that records no
provenance at all, so an agent that lost *every* finding (a very small window) would pass it. The
harness catches that case from the declared `required_evidence` (`category:
"required-evidence-not-carried"`) without changing the governance check.

`smallest_safe_window` binary-searches `[lo, hi]` to within `resolution` (default from
`sizing.resolution`). It assumes safety is monotonic in the window, and every probe is a real LLM
run, so results are probabilistic and each probe costs about 15-40 seconds (Ollama reloads the
model when `num_ctx` changes).

## How the config maps to the code

| Harness field | Used by |
|---|---|
| `framework` | Reported only; the pipeline wiring is in `pipeline` |
| `model.name`, `provider`, `base_url` | Calibration, template check, and the model-name check on every captured call |
| `model.tokenizer` | `_gguf_tokenizer` (source `ollama-gguf`, `type` and `pre` checked against the runtime) |
| `model.window_param` | Hook B (the request option that carries the window) and calibration |
| `model.current_window`, `advertised_window` | Defaults and context for sizing (`run` without `--window` uses `current_window`) |
| `context_assembly.chat_template` | Hook A rendering; `runtime_template_sha256` checked at load |
| `context_assembly.includes` | Documents what each model-calling agent's prompt contains |
| `context_assembly.calibration` | `load_harness` calibration call |
| `observability.*` | Hooks A and C, visibility flags, overflow detection |
| `run_entry_point` | Names `instrumentation.run_case(case_json, window)` |
| `pipeline.*` | `run_case`: which module, nodes, state schema, builder, run function, and config factory to use |
| `pipeline.accumulated_state_argument` | Optional pass-through used by long-run projection to supply starting history/memory; ordinary runs use an empty value and are unchanged |
| `evidence_formats` | Hook C structured matching |
| `agents[]` | One `AgentSpec` per node; every pipeline node must be declared |
| `sizing.resolution` | `smallest_safe_window` default resolution |

## Tests

`test_instrumentation.py` runs offline against a fake Ollama-like runtime (same tail-keeping
truncation, reports `prompt_eval_count`, answers from the reports it can see). It covers the config
checks, every hook, the read-only guarantee (identical decisions and identical requests with and
without instrumentation), overflow, the lost-everything case, and the binary search. Two live tests
run against the real runtime when `llama3.1:8b` is available.
