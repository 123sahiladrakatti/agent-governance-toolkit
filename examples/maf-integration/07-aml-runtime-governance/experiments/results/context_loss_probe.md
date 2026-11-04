# Real-LLM context-loss probe

**Conclusion: CONTEXT LOSS NOT VERIFIED**

## Runtime

- Ollama: `0.40.14`
- Model: `llama3.1:latest`
- Model identifier: `46e0c10c039e019119339687c3c1757cc81b9da49709a3b3924863ba87ca666e`
- Temperature: `0.1`; seed: `42`
- Requested contexts: full `2048`, constrained `512`, restored `8192`
- Finding placement: `early`; filler target: `4000` characters
- Trials per condition: `10`

## Controlled comparison

| Condition | Prompt contains screening | Recall HIT | ESCALATE | CLEAR | Abstain | Runtime/parse failures |
|---|---:|---:|---:|---:|---:|---:|
| full_context | yes | 10/10 | 10/10 | 0/10 | 0/10 | 0/10 |
| constrained_context | yes | 0/10 | 0/10 | 0/10 | 10/10 | 0/10 |
| restored_context | yes | 10/10 | 10/10 | 0/10 | 0/10 | 0/10 |

## Causal checks

- Long-prompt token reduction at constrained context: `True`
- Full-context accuracy: `10/10`
- Constrained incorrect-clear rate: `0/10`
- Restoration recovery rate: `10/10`

A failed recall does not directly expose Ollama's retained token sequence. When prompt-token processing is reduced and the separate recall probe fails, the artifact labels the specific finding's visibility as `INFERRED_NOT_RETAINED`, not directly verified truncation.

## Integration gate

Stop before pipeline integration. The required reproducible full → incorrect CLEAR → restored sequence was not established by this run.
