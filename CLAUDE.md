# Project conventions

## Commits

Conventional Commits. One line, lower case, imperative, no trailing period:

```
type(scope): what changed
```

Keep the subject under ~72 characters. It should answer "what does this add or fix" on its own,
without opening the diff.

### Types

| Type | Use for |
|---|---|
| `feat` | new capability |
| `fix` | corrected behaviour that was wrong |
| `docs` | notes, design docs, README |
| `test` | tests only |
| `refactor` | restructuring with no behaviour change |
| `perf` | speed or memory |
| `chore` | tooling, deps, config, CI |

### Scopes

The module or area touched, matching `src/llm/`:

`tokenizer` `data` `config` `attention` `norms` `ffn` `rope` `block` `model` `generate` `kv-cache`
`train` `optim` `checkpoint` `gpt2-weights` `chat`

Plus: `notes` (the learning notes), `docs` (design docs), `env` (venv, deps, CI).

Omit the scope when a change genuinely spans the project.

### Examples

```
feat(attention): add causal self-attention with GQA and SDPA paths
feat(tokenizer): add BPE and word-level tokenizers
fix(config): round SwiGLU width to nearest multiple of 64
fix(attention): mask before softmax so rows renormalize to 1
docs(notes): correct the claim that residuals alone prevent vanishing gradients
test(block): add parity check against torch.nn.LayerNorm
chore(env): pin torch to the cu128 index for Blackwell
```

### Body

Optional, and only when the subject cannot carry it. Use it for **why**, for measured numbers, and
for corrections to earlier claims. Wrap at 80 characters. Skip it for small, obvious changes.

Attribution line on every commit:

```
Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
```

### Not used here

Upstream repos show a `(#1234)` suffix — that is added automatically when GitHub squash-merges a
pull request. We commit straight to `master`, so there is no number to reference. If we move to a
PR workflow later, the suffix appears on its own.

## Code

- Every module in `src/llm/` is importable **and** runnable standalone via `__main__`, printing
  shapes or a small demo.
- Comments explain *why*, not *what*. The learning notes in `notes/` carry the teaching.
- Each note ships tests: shape, property, and parity against a PyTorch reference where one exists.
- Never hardcode dimensions in tests — derive them from the config preset, so changing a preset
  does not silently break unrelated tests.

## Things that must not be committed

`.gitignore` covers these, but they are worth stating: the Raschka book PDF (copyrighted), the
`data/` corpus, `checkpoints/`, and `.venv/`.
