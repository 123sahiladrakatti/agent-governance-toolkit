"""Config-driven governance harness for an existing agentic pipeline.

Everything that varies (framework, model, tokenizer, chat template, window parameter,
observability, pipeline entry points, per-agent evidence) comes from the ``harness`` block
of the customer JSON. This module only knows how to read that block and instrument
what is already there; it does not rebuild the pipeline.

    harness = load_harness(customer_json)          # resolve + verify tokenizer and template
    trace, verdict = run_case(case_json, window)    # one reproducible, instrumented run
    smallest_safe_window(case_json, lo, hi)         # binary search over the window

The instrumentation is read-only with respect to agent behaviour: it observes the
model calls the pipeline already makes and appends trace records; it never changes
prompts, evidence, windows, or decisions.
"""

from __future__ import annotations

import contextlib
import copy
import hashlib
import importlib
import json
import operator
import re
import threading
from dataclasses import dataclass, field
from typing import Annotated, Any, Callable, Mapping, TypedDict, get_type_hints

import httpx
import jinja2

# llama.cpp's pre-tokenizer regex for each GGUF ``tokenizer.ggml.pre`` value we have verified
# against the runtime. Unknown values fail loudly rather than guess.
PRE_TOKENIZER_PATTERNS = {
    "llama-bpe": (
        r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*"
        r"|\s*[\r\n]+|\s+(?!\S)|\s+"
    ),
}
_REQUIRED = {
    "": ("framework", "model", "context_assembly", "observability", "run_entry_point", "pipeline",
         "agents", "evidence_formats"),
    "model": ("name", "provider", "tokenizer", "window_param", "current_window", "advertised_window"),
    "context_assembly": ("chat_template", "includes"),
    "observability": ("can_capture_model_visible_prompt", "exposes_processed_token_count",
                      "processed_token_field", "can_log_trimming", "model_call"),
    "run_entry_point": ("module", "function", "window_argument"),
    "pipeline": ("module", "nodes", "state_schema", "graph_builder", "run_function", "normalize",
                 "model_config_factory", "window_state_key", "window_config_field", "records_codecs"),
}


class HarnessConfigError(RuntimeError):
    """The harness block is incomplete or cannot be resolved against the runtime."""


class HarnessOverflow(RuntimeError):
    """The runtime rejected a prompt as too long instead of truncating it."""

    def __init__(self, record: dict[str, Any]) -> None:
        super().__init__(f"{record['agent']}: runtime rejected the prompt as too long")
        self.record = record


# ---------------------------------------------------------------- config + resolution
@dataclass(frozen=True)
class AgentSpec:
    name: str
    required_evidence: tuple[str, ...]
    high_consequence_action: bool
    keeps_history: bool
    calls_model: bool
    evidence_format: str | None
    harness: "Harness" = field(repr=False, compare=False)


@dataclass(frozen=True)
class Harness:
    config: dict[str, Any]
    tokenizer: Any
    template: jinja2.Template
    add_bos: bool
    calibration: dict[str, Any] | None
    agents: dict[str, AgentSpec] = field(default_factory=dict)

    @property
    def model(self) -> dict[str, Any]:
        return self.config["model"]

    @property
    def observability(self) -> dict[str, Any]:
        return self.config["observability"]

    @property
    def pipeline(self) -> dict[str, Any]:
        return self.config["pipeline"]

    def agent(self, name: str) -> AgentSpec:
        try:
            return self.agents[name]
        except KeyError:
            raise HarnessConfigError(f"agent {name!r} is not declared in the harness") from None

    def render(self, messages: list[Mapping[str, Any]], tools: Any = None) -> str:
        """Render messages exactly as the runtime's chat template does."""
        return self.template.render(messages=list(messages), tools=tools)

    def count(self, rendered: str, with_bos: bool = True) -> int:
        """Tokens the runtime processes for an already-rendered prompt."""
        ids = self.tokenizer.encode(rendered, add_special_tokens=False).ids
        return len(ids) + (1 if with_bos and self.add_bos else 0)


def _require(config: Mapping[str, Any], path: str, keys: tuple[str, ...]) -> None:
    missing = [key for key in keys if key not in config]
    if missing:
        where = f"harness.{path}" if path else "harness"
        raise HarnessConfigError(f"{where} is missing {', '.join(missing)}")


def _resolve(reference: str) -> Any:
    """``package.module:attr`` (or ``package.module.attr``) -> the object."""
    module_name, sep, attr = reference.partition(":")
    if not sep:
        module_name, _, attr = reference.rpartition(".")
    try:
        return getattr(importlib.import_module(module_name), attr)
    except (ImportError, AttributeError) as exc:
        raise HarnessConfigError(f"cannot resolve {reference!r}: {exc}") from exc


def _runtime(config: Mapping[str, Any], path: str, payload: dict[str, Any]) -> dict[str, Any]:
    base = config["model"].get("base_url", "http://localhost:11434").rstrip("/")
    try:
        response = httpx.post(f"{base}{path}", json=payload, timeout=300)
        response.raise_for_status()
        return response.json()
    except (httpx.HTTPError, ValueError) as exc:
        raise HarnessConfigError(f"runtime call {path} failed at {base}: {exc}") from exc


_TOKENIZERS: dict[tuple[str, str, str], Any] = {}


def _gguf_tokenizer(config: Mapping[str, Any]) -> Any:
    """Rebuild the runtime's own tokenizer from the GGUF vocabulary Ollama exposes."""

    from tokenizers import AddedToken, Regex, Tokenizer, decoders, models, pre_tokenizers

    spec = config["model"]["tokenizer"]
    key = (config["model"].get("base_url", ""), spec["model"], spec.get("pre", ""))
    if key in _TOKENIZERS:
        return _TOKENIZERS[key]
    info = _runtime(config, "/api/show", {"model": spec["model"], "verbose": True}).get("model_info", {})
    kind, pre = info.get("tokenizer.ggml.model"), info.get("tokenizer.ggml.pre")
    if kind != spec.get("type") or pre != spec.get("pre"):
        raise HarnessConfigError(
            f"tokenizer mismatch for {spec['model']}: runtime has model={kind!r} pre={pre!r}, "
            f"harness declares type={spec.get('type')!r} pre={spec.get('pre')!r}"
        )
    if kind != "gpt2" or pre not in PRE_TOKENIZER_PATTERNS:
        raise HarnessConfigError(f"no verified tokenizer implementation for model={kind!r} pre={pre!r}")
    tokens, types = info.get("tokenizer.ggml.tokens"), info.get("tokenizer.ggml.token_type")
    merges = info.get("tokenizer.ggml.merges")
    if not tokens or not merges or not types:
        raise HarnessConfigError(f"runtime did not return the vocabulary for {spec['model']} (verbose /api/show)")
    tokenizer = Tokenizer(models.BPE(
        vocab={token: index for index, token in enumerate(tokens)},
        merges=[tuple(merge.split(" ", 1)) for merge in merges],
        ignore_merges=True,
    ))
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(Regex(PRE_TOKENIZER_PATTERNS[pre]), behavior="isolated", invert=False),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
    ])
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.add_special_tokens([
        AddedToken(tokens[index], special=True, normalized=False)
        for index, kind_id in enumerate(types) if kind_id == 3  # control tokens
    ])
    _TOKENIZERS[key] = tokenizer
    return tokenizer


def _unsupported(what: str) -> str:
    raise HarnessConfigError(f"chat template does not support {what!r} messages")


def _compile_template(config: Mapping[str, Any]) -> jinja2.Template:
    spec = config["context_assembly"]["chat_template"]
    if spec.get("engine") != "jinja2" or not spec.get("template"):
        raise HarnessConfigError("chat_template must provide a jinja2 'template'")
    environment = jinja2.Environment(undefined=jinja2.StrictUndefined, keep_trailing_newline=True, autoescape=False)
    environment.globals["unsupported"] = _unsupported
    try:
        return environment.from_string(spec["template"])
    except jinja2.TemplateError as exc:
        raise HarnessConfigError(f"chat_template does not compile: {exc}") from exc


_HARNESSES: dict[str, Harness] = {}


def load_harness(config: Mapping[str, Any], *, tokenizer: Any = None, runtime_checks: bool = True) -> Harness:
    """Parse the harness block (or a customer JSON containing one) and resolve it.

    With ``runtime_checks`` (the default) this fails loudly unless:
    * the runtime's tokenizer matches the declared one (rebuilt from the runtime's vocabulary),
    * the runtime's chat template is byte-identical to the one this config was written for, and
    * a calibration prompt counts to exactly the runtime's processed-token count.
    ``tokenizer`` injects a tokenizer (tests only; skips the runtime tokenizer).
    """

    config = dict(config.get("harness", config))
    injected = tokenizer is not None
    cache_key = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest() + f":{runtime_checks}"
    if not injected and cache_key in _HARNESSES:
        return _HARNESSES[cache_key]

    _require(config, "", _REQUIRED[""])
    for section in ("model", "context_assembly", "observability", "run_entry_point", "pipeline"):
        _require(config[section], section, _REQUIRED[section])
    if config["observability"]["exposes_processed_token_count"] is not True:
        raise HarnessConfigError("truncation detection needs a runtime that reports processed tokens")
    template = _compile_template(config)

    if tokenizer is None:
        if config["model"]["tokenizer"].get("source") != "ollama-gguf":
            raise HarnessConfigError(f"unsupported tokenizer source {config['model']['tokenizer'].get('source')!r}")
        tokenizer = _gguf_tokenizer(config)
    if runtime_checks:
        runtime_template = _runtime(config, "/api/show", {"model": config["model"]["name"]}).get("template", "")
        digest = hashlib.sha256(runtime_template.encode()).hexdigest()
        expected = config["context_assembly"]["chat_template"].get("runtime_template_sha256")
        if digest != expected:
            raise HarnessConfigError(
                f"runtime chat template for {config['model']['name']} changed (sha256 {digest}, harness expects "
                f"{expected}); update the harness template before trusting token counts"
            )

    harness = Harness(
        config=config, tokenizer=tokenizer, template=template,
        add_bos=bool(config["model"]["tokenizer"].get("add_bos", False)), calibration=None,
    )
    if runtime_checks:
        harness = _calibrate(harness)
    agents = {}
    for entry in config["agents"]:
        _require(entry, f"agents[{entry.get('name', '?')}]",
                 ("name", "required_evidence", "high_consequence_action", "keeps_history"))
        fmt = entry.get("evidence_format")
        if entry["required_evidence"] and fmt not in config["evidence_formats"]:
            raise HarnessConfigError(f"agent {entry['name']!r} needs an evidence_format from evidence_formats")
        agents[entry["name"]] = AgentSpec(
            entry["name"], tuple(entry["required_evidence"]), bool(entry["high_consequence_action"]),
            bool(entry["keeps_history"]), bool(entry.get("calls_model", False)), fmt, harness,
        )
    harness.agents.update(agents)
    if not injected:
        _HARNESSES[cache_key] = harness
    return harness


def _calibrate(harness: Harness) -> Harness:
    """Our template + tokenizer must count a calibration prompt exactly as the runtime does."""
    spec = harness.config["context_assembly"].get("calibration")
    if not spec:
        raise HarnessConfigError("context_assembly.calibration is required for runtime checks")
    obs, model = harness.observability, harness.model
    reply = _runtime(harness.config, harness.observability["model_call"]["endpoint"], {
        "model": model["name"], "messages": spec["messages"], "stream": False,
        "options": {model["window_param"]: spec["window"], "num_predict": 1, "temperature": 0},
    })
    runtime_count = reply.get(obs["processed_token_field"])
    ours = harness.count(harness.render(spec["messages"]))
    if runtime_count != ours:
        raise HarnessConfigError(
            f"tokenizer/template drift: harness counts {ours} tokens, runtime processed {runtime_count} "
            "for the calibration prompt"
        )
    return Harness(harness.config, harness.tokenizer, harness.template, harness.add_bos,
                   {"tokens": ours, "runtime_tokens": runtime_count, "window": spec["window"]}, harness.agents)


# ---------------------------------------------------------------- hooks
_LOCAL = threading.local()  # per-thread recorder: LangGraph runs parallel branches in worker threads


@contextlib.contextmanager
def _patched(target: Any, attribute: str, value: Any):
    original = getattr(target, attribute)
    setattr(target, attribute, value)
    try:
        yield original
    finally:
        setattr(target, attribute, original)


@contextlib.contextmanager
def _observe_model_calls(harness: Harness):
    """Record every call to the model endpoint, unchanged, for whichever node is running."""
    spec = harness.observability["model_call"]
    module_name, attribute = spec["http_function"].rsplit(".", 1)
    module = importlib.import_module(module_name)
    original = getattr(module, attribute)

    def recording(url, *args, **kwargs):
        calls = getattr(_LOCAL, "calls", None)
        if calls is None or not str(url).endswith(spec["endpoint"]):
            return original(url, *args, **kwargs)
        entry: dict[str, Any] = {"request": copy.deepcopy(kwargs.get("json"))}
        calls.append(entry)
        try:
            response = original(url, *args, **kwargs)
        except Exception as exc:
            entry["error"] = f"{type(exc).__name__}: {exc}"
            raise
        entry["status"] = response.status_code
        try:
            entry["response"] = response.json()
        except ValueError:
            entry["response_text"] = response.text[:2000]
        return response

    with _patched(module, attribute, recording):
        yield


def _state_window(state: Mapping[str, Any], harness: Harness) -> Any:
    """Hook B: the window lives in run state, never in a constant."""
    holder = state.get(harness.pipeline["window_state_key"])
    return getattr(holder, harness.pipeline["window_config_field"], None) if holder is not None else None


def _evidence(harness: Harness, agent: AgentSpec, state: Mapping[str, Any], messages: list[Mapping[str, Any]],
              rendered: str, processed: int | None, truncated: bool | None) -> dict[str, Any]:
    """Hook C evidence: structured match of each required finding in the assembled prompt."""
    fmt = harness.config["evidence_formats"][agent.evidence_format]
    pattern = re.compile(fmt["pattern"], re.M)
    source = fmt["source"]
    emitted = state.get(source["state_key"], {})
    found: dict[str, dict[str, Any]] = {}
    for message in messages:
        content = str(message.get("content", ""))
        offset = rendered.find(content)
        for match in pattern.finditer(content):
            found.setdefault(match["id"], {
                "version": int(match["version"]), "finding": match["finding"],
                "offset": offset + match.start() if offset >= 0 else None,
            })
    present, mismatch, in_tail = {}, {}, {}
    for evidence_id in agent.required_evidence:
        parent = emitted.get(evidence_id)
        seen = found.get(evidence_id)
        expected = None if parent is None else (
            int(getattr(parent, source["version_field"])), str(getattr(parent, source["finding_field"])).lower()
        )
        ok = bool(seen and expected and (seen["version"], seen["finding"].lower()) == expected)
        present[evidence_id] = ok
        if seen and not ok:
            mismatch[evidence_id] = {"in_prompt": [seen["version"], seen["finding"]],
                                     "emitted": list(expected) if expected else None}
        if not truncated:
            in_tail[evidence_id] = ok
        elif harness.observability.get("truncation_policy") == "keep_tail" and seen and seen["offset"] is not None:
            suffix = harness.count(rendered[seen["offset"]:])
            in_tail[evidence_id] = ok and processed is not None and suffix <= processed
        else:
            in_tail[evidence_id] = None
    return {"present": present, "mismatch": mismatch, "in_kept_window_estimate": in_tail}


def _overflowed(error: Exception | None, calls: list[dict[str, Any]], harness: Harness) -> bool:
    markers = [m.lower() for m in harness.observability.get("overflow_error_markers", [])]
    texts = [str(error or "")] + [str(c.get("error", "")) + str(c.get("response_text", "")) for c in calls]
    return any(marker in text.lower() for text in texts for marker in markers)


def _record(harness: Harness, agent: AgentSpec, state: Mapping[str, Any], update: Mapping[str, Any] | None,
            calls: list[dict[str, Any]], overflow: bool = False) -> dict[str, Any]:
    """Hook D: one trace record per agent step."""
    obs, model = harness.observability, harness.model
    window = _state_window(state, harness)
    step = ((update or {}).get("steps") or {}).get(agent.name)
    decision = None if step is None else (getattr(step, "disposition", None) or getattr(step, "finding", None))
    record: dict[str, Any] = {
        "agent": agent.name, "window": window, "assembled_tokens": None, "processed_tokens": None,
        "truncated": None, "required_evidence": list(agent.required_evidence),
        "evidence_visible": {evidence_id: None for evidence_id in agent.required_evidence},
        "decision": decision, "high_consequence_action": agent.high_consequence_action,
        "model_call": False, "model_visibility_confirmed": False, "confirmation": "not-applicable",
    }
    primary = calls[0] if calls else None  # observability.model_call.primary_call == "first"
    if not agent.calls_model or primary is None:
        return record
    request = primary.get("request") or {}
    if request.get("model") != model["name"]:
        raise HarnessConfigError(f"{agent.name} called model {request.get('model')!r}, harness expects {model['name']!r}")
    messages = request.get("messages") or []
    rendered = harness.render(messages, request.get("tools"))
    assembled = harness.count(rendered)                                   # Hook A
    sent_window = (request.get("options") or {}).get(model["window_param"])  # Hook B
    response = primary.get("response") or {}
    processed = response.get(obs["processed_token_field"])
    truncated = True if overflow else (None if processed is None else assembled > processed)  # Hook C
    evidence = _evidence(harness, agent, state, messages, rendered, processed, truncated)
    visible_confirmed = bool(obs["can_capture_model_visible_prompt"])
    record.update({
        "model_call": True,
        "window_sent": sent_window,
        "window_matches_state": sent_window == window,
        "assembled_tokens": assembled,
        "processed_tokens": processed,
        "truncated": truncated,
        "overflow_error": bool(overflow),
        "token_count_verified": (truncated is False and processed == assembled),
        "evidence_present_in_prompt": evidence["present"],
        "evidence_in_kept_window_estimate": evidence["in_kept_window_estimate"],
        "evidence_mismatch": evidence["mismatch"],
        "evidence_visible": {
            evidence_id: ((ok and truncated is False) if visible_confirmed else None)
            for evidence_id, ok in evidence["present"].items()
        },
        "model_visibility_confirmed": visible_confirmed,
        "confirmation": "measured-and-verified" if visible_confirmed else "measured-only",
        "auxiliary_model_calls": len(calls) - 1,
        "assembled_prompt": rendered,
    })
    return record


def instrument(node: Callable[[Mapping[str, Any]], dict], agent_config: AgentSpec) -> Callable:
    """Wrap one existing agent node with hooks A-D. The node's own output is passed through unchanged."""

    harness = agent_config.harness

    def wrapped(state):
        previous = getattr(_LOCAL, "calls", None)
        _LOCAL.calls = calls = []
        try:
            try:
                update = node(state)
            except Exception as exc:
                if agent_config.calls_model and _overflowed(exc, calls, harness):
                    raise HarnessOverflow(_record(harness, agent_config, state, None, calls, overflow=True)) from exc
                raise
        finally:
            _LOCAL.calls = previous
        return {**update, "trace": [_record(harness, agent_config, state, update, calls)]}

    wrapped.__name__ = getattr(node, "__name__", agent_config.name)
    wrapped.__wrapped__ = node
    return wrapped


# ---------------------------------------------------------------- run + size
def _instrumented_graph(harness: Harness, module: Any) -> Any:
    """Compile the pipeline's own graph builder with wrapped nodes and a ``trace`` channel."""
    pipe = harness.pipeline
    nodes = getattr(module, pipe["nodes"])
    undeclared = sorted(set(nodes) - set(harness.agents))
    if undeclared:
        raise HarnessConfigError(f"pipeline nodes missing from harness.agents: {', '.join(undeclared)}")
    schema = getattr(module, pipe["state_schema"])
    hints = get_type_hints(schema, include_extras=True)
    traced = TypedDict(f"Traced{schema.__name__}", {**hints, "trace": Annotated[list, operator.add]})
    wrapped = {name: instrument(fn, harness.agent(name)) for name, fn in nodes.items()}
    builder = getattr(module, pipe["graph_builder"])
    builder = getattr(builder, "__wrapped__", builder)  # bypass the builder's cache: compile a fresh copy
    with _patched(module, pipe["nodes"], wrapped), _patched(module, pipe["state_schema"], traced):
        return builder()


def _carried(outcome: Any, harness: Harness) -> dict[str, dict[str, bool]]:
    """Did each agent consume every required finding exactly as its parent emitted it?

    The existing trace-integrity check skips a consumer that records no provenance at all,
    so an agent that lost *every* finding would pass it. The harness closes that gap from
    the declared ``required_evidence`` without changing the governance check itself.
    """
    steps = {step.agent: step for step in outcome.steps}
    carried: dict[str, dict[str, bool]] = {}
    for agent in harness.agents.values():
        if not agent.required_evidence or agent.name not in steps:
            continue
        consumer = steps[agent.name]
        carried[agent.name] = {
            evidence_id: evidence_id in steps
            and consumer.received_findings_by_parent.get(evidence_id) == steps[evidence_id].finding
            and consumer.received_versions_by_parent.get(evidence_id) == steps[evidence_id].finding_version
            for evidence_id in agent.required_evidence
        }
    return carried


def _verdict(integrity: Any, carried: dict[str, dict[str, bool]], window: Any, harness: Harness) -> dict[str, Any]:
    verified = bool(harness.observability["can_capture_model_visible_prompt"])
    all_carried = all(ok for per_agent in carried.values() for ok in per_agent.values())
    lost = {agent: [e for e, ok in per_agent.items() if not ok] for agent, per_agent in carried.items()}
    lost = {agent: ids for agent, ids in lost.items() if ids}
    return {
        "passed": bool(integrity.passed) and all_carried,
        "integrity_passed": bool(integrity.passed),
        "required_evidence_carried": all_carried,
        "evidence_not_carried": lost,
        "category": integrity.category or (None if all_carried else "required-evidence-not-carried"),
        "reason": integrity.reason if not integrity.passed or all_carried else (
            "trace integrity passed, but required evidence was not carried through: "
            + "; ".join(f"{agent} lost {', '.join(ids)}" for agent, ids in lost.items())
        ),
        "origin_agent": integrity.origin_agent or (next(iter(lost)) if lost else None),
        "related_agents": list(integrity.related_agents) or sorted({e for ids in lost.values() for e in ids}),
        "window": window, "evidence_check": "measured-and-verified" if verified else "measured-only",
    }


def run_case(
    case_json: Mapping[str, Any],
    window: int,
    accumulated_state: Any = None,
    *,
    harness: Harness | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run one case through the instrumented pipeline at ``window``; return (trace, verdict).

    ``case_json`` is the customer JSON: the pipeline definition (system, policy), its
    ``harness`` block, and this case's encoded ``records``. ``harness`` overrides the one
    resolved from ``case_json`` (tests only). ``accumulated_state`` is an optional starting
    history/memory value. It is passed through only when the customer harness declares
    ``pipeline.accumulated_state_argument``; the harness never invents or mutates it.
    """

    from client_pipeline import compose_client_payload, ingest_client_pipeline, run_to_json
    from graph_integrity import check_graph_integrity

    harness = harness or load_harness(case_json)
    pipe = harness.pipeline
    module = importlib.import_module(pipe["module"])
    codecs = _resolve(pipe["records_codecs"])
    schema = case_json["system"]["schema"]
    if schema not in codecs:
        raise HarnessConfigError(f"no records codec for schema {schema!r}")
    records = codecs[schema][1](case_json["records"])
    model_config = _resolve(pipe["model_config_factory"])(**{
        pipe["window_config_field"]: window, pipe.get("model_config_field", "model"): harness.model["name"],
    })
    graph = _instrumented_graph(harness, module)
    run_function = getattr(module, pipe["run_function"])
    accumulated_arguments: dict[str, Any] = {}
    if accumulated_state is not None:
        argument = pipe.get("accumulated_state_argument")
        if not argument:
            raise HarnessConfigError(
                "accumulated_state was supplied, but harness.pipeline has no "
                "accumulated_state_argument"
            )
        accumulated_arguments[argument] = accumulated_state
    try:
        with _observe_model_calls(harness), _patched(module, pipe["graph_builder"], lambda: graph):
            state = run_function(
                records,
                **pipe.get("run_arguments", {}),
                **{pipe["window_state_key"]: model_config},
                **accumulated_arguments,
            )
    except HarnessOverflow as overflow:
        verdict = {"passed": False, "integrity_passed": None, "required_evidence_carried": None,
                   "evidence_not_carried": {}, "category": "runtime-overflow", "reason": str(overflow),
                   "origin_agent": overflow.record["agent"], "related_agents": [], "window": window,
                   "evidence_check": "overflow"}
        return [overflow.record], verdict

    steps = getattr(module, pipe["normalize"])(records, state)
    run_id = str(getattr(records, pipe.get("run_id_field", "case_id")))
    prefix = case_json["system"].get("resource_prefix", "pipeline-run")
    run = run_to_json(steps, run_id=run_id, resource_id=f"{prefix}:{run_id}")
    definition = {key: case_json[key] for key in ("format", "system", "policy")}
    payload = compose_client_payload(definition, run, case_json["records"])
    outcome = ingest_client_pipeline(json.dumps(payload), codecs).outcome
    carried = _carried(outcome, harness)
    verdict = _verdict(check_graph_integrity(outcome), carried, window, harness)
    for record in state.get("trace", []):
        record["evidence_used"] = carried.get(record["agent"], {})
    return list(state.get("trace", [])), verdict


def smallest_safe_window(case_json: Mapping[str, Any], lo: int, hi: int, *, resolution: int | None = None,
                         probes: list[dict[str, Any]] | None = None, harness: Harness | None = None) -> int | None:
    """Smallest window in [lo, hi] whose run is safe, to within ``resolution``.

    Safe = trace integrity passes AND every agent carried all its required evidence through.

    Assumes safety is monotonic in the window. Returns None if even ``hi`` is unsafe. Each
    probe is a real run, appended to ``probes`` (window, passed, trace) when given.
    """

    if lo > hi:
        raise ValueError("lo must be <= hi")
    harness = harness or load_harness(case_json)
    step = max(1, int(resolution or harness.config.get("sizing", {}).get("resolution", 1)))
    results: dict[int, bool] = {}

    def safe(window: int) -> bool:
        if window not in results:
            trace, verdict = run_case(case_json, window, harness=harness)
            results[window] = verdict["passed"]
            if probes is not None:
                probes.append({"window": window, "passed": verdict["passed"], "verdict": verdict, "trace": trace})
        return results[window]

    if not safe(hi):
        return None
    if safe(lo):
        return lo
    unsafe, ok = lo, hi
    while ok - unsafe > step:
        middle = (unsafe + ok) // 2
        if safe(middle):
            ok = middle
        else:
            unsafe = middle
    return ok


def case_json_for(definition_path: str, records: Mapping[str, Any]) -> dict[str, Any]:
    """Customer JSON for one case: the pipeline definition (with its harness) plus encoded records."""
    with open(definition_path, encoding="utf-8") as handle:
        definition = json.load(handle)
    if "harness" not in definition:
        raise HarnessConfigError(f"{definition_path} has no harness block")
    return {**definition, "records": dict(records)}


def _main() -> None:
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Instrumented run / window sizing for one case.")
    parser.add_argument("command", choices=("run", "size"))
    parser.add_argument("--definition", default="client_pipelines/healthcare_pipeline.json")
    parser.add_argument("--records", default="harness_cases/healthcare_contraindication.json")
    parser.add_argument("--window", type=int)
    parser.add_argument("--lo", type=int, default=512)
    parser.add_argument("--hi", type=int, default=8192)
    parser.add_argument("--resolution", type=int)
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    with open(args.records, encoding="utf-8") as handle:
        case = case_json_for(args.definition, json.load(handle))
    harness = load_harness(case)
    print(f"harness: {harness.model['name']} via {harness.config['framework']}; calibration {harness.calibration}")
    if args.command == "run":
        window = args.window or harness.model["current_window"]
        trace, verdict = run_case(case, window)
        for record in trace:
            print(json.dumps({k: v for k, v in record.items() if k != "assembled_prompt"}, default=str))
        print("verdict:", json.dumps(verdict))
    else:
        probes: list[dict[str, Any]] = []
        best = smallest_safe_window(case, args.lo, args.hi, resolution=args.resolution, probes=probes)
        for probe in probes:
            model = next((r for r in probe["trace"] if r.get("model_call")), {})
            print(f"window {probe['window']:6}: {'SAFE' if probe['passed'] else 'unsafe'} "
                  f"assembled={model.get('assembled_tokens')} processed={model.get('processed_tokens')} "
                  f"truncated={model.get('truncated')}")
        print("smallest safe window:", best)


if __name__ == "__main__":
    _main()


__all__ = [
    "AgentSpec",
    "Harness",
    "HarnessConfigError",
    "HarnessOverflow",
    "case_json_for",
    "instrument",
    "load_harness",
    "run_case",
    "smallest_safe_window",
]
