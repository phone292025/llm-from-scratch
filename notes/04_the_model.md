# Note 4 — The whole model

Assembling the pieces into something that runs. Short note — the hard ideas are behind us, and this
is mostly wiring. It ends with the single most convincing test in the project.

**Produces:** `src/llm/config.py`, `src/llm/model.py`, `src/llm/gpt2_weights.py`,
`tests/test_04_model.py`

---

## 1. The stack

```python
def forward(self, idx):                      # idx: (B, T)
    B, T = idx.shape
    h = self.tok_emb(idx)                    # (B, T, d)
    h = h + self.pos_emb(torch.arange(T))    # learned positions
    h = self.drop(h)
    for block in self.blocks:                # (B, T, d) throughout
        h = block(h)
    h = self.final_norm(h)                   # pre-norm needs this
    return self.head(h)                      # (B, T, V)
```

That is the model. Two embedding tables, N identical blocks, a norm, a projection to vocabulary
size. Everything interesting happened in notes 2 and 3.

The `final_norm` exists because pre-norm blocks normalize on the way *in*, leaving the last block's
output unnormalized. Forgetting it produces a model that trains but underperforms, with no error —
exactly the kind of bug the GPT-2 oracle catches.

## 2. Weight tying

The output head maps `d → V`. The token embedding maps `V → d`. Tying means they are the same
matrix:

```python
self.head.weight = self.tok_emb.weight     # shared storage, not a copy
```

The argument for it is not just parameter count. "The vector that means *cat*" and "the direction
that predicts *cat*" describe the same thing, so they should be the same vector — and forcing that
gives the embedding table gradient signal from both ends, which helps rare tokens that appear
seldom as input but must remain predictable as output.

It also saves 38,597,376 parameters, 24% of the model. GPT-2 does this, and it is why a model with
163 M parameters is called 124 M.

## 3. Initialization

Random init is not arbitrary. Too large and activations explode across 12 layers; too small and
they vanish. GPT-2 uses normal with σ = 0.02, plus one refinement worth keeping:

```python
# residual projections scaled by 1/sqrt(2 * n_layers)
```

Every block adds twice into the residual stream (attention and MLP). Over 12 blocks the stream's
variance grows with depth. Scaling down the *output* projections keeps it stable regardless of how
deep the model is. It is one loop over named parameters and it visibly improves early training.

## 4. Configuration, and the presets

`ModelConfig` is described in [ARCHITECTURE §3](../docs/ARCHITECTURE.md). Three presets:

```python
GPT2_124M = ModelConfig()                                  # the 2019 architecture exactly
SMALL     = ModelConfig(emb_dim=384, n_layers=6, n_heads=6,
                        context_length=512)                # ~30M, trains well in 8GB
MODERN    = replace(SMALL, norm="rmsnorm", pos="rope",
                    act="swiglu", n_kv_heads=2,
                    qkv_bias=False)                        # v2
```

`GPT2_124M` is for loading pretrained weights and doing inference. `SMALL` is what we actually
train — see [ARCHITECTURE §6](../docs/ARCHITECTURE.md) for why chasing 124 M on a laptop is a trap.

## 5. Sanity checks before training anything

Three things to verify the moment the model first runs, in this order:

**Parameter count is exactly 163,009,536** untied, 124,412,160 tied. This number is unforgiving —
a missing bias, a dropped layer, or a wrong hidden dimension all change it. Matching it exactly is
strong evidence the architecture is right.

**Loss at random init is ≈ 10.82.** An untrained model should be uniformly uncertain over 50,257
tokens, giving `ln(50257) = 10.8247`. If it starts much lower, something is leaking information
(check the mask). Much higher, and initialization is wrong.

**Generation produces gibberish.** Run the untrained model. The output is noise — and that is
correct. The architecture is complete; the knowledge is not. Seeing this before training makes the
contrast afterwards much more legible.

## 6. The oracle

This is the payoff of note 4, and the strongest reason for building GPT-2's architecture first
rather than jumping to a modern one.

GPT-2's trained weights are public. Loading them into **my own class** and getting fluent English
proves that every matrix, every transpose, every normalization and every wiring decision in this
repo is correct. No unit test comes close to that level of evidence.

```python
def load_gpt2_weights(model: GPTModel, size: str = "124M") -> GPTModel
```

Weights come from HuggingFace as `.safetensors` — identical numbers to OpenAI's release, without
installing TensorFlow on Windows to read a 2019 checkpoint format.

The one real trap: **OpenAI stored the linear weights transposed** relative to PyTorch's
convention, because the original code used `Conv1D` rather than `Linear`. Every `W_qkv`, `W_o` and
MLP weight needs a `.T` on the way in. Get it wrong and the model loads without error and emits
perfect nonsense — which is itself an instructive twenty minutes.

Mapping, in outline:

```
wte                      -> tok_emb.weight
wpe                      -> pos_emb.weight
h.{i}.ln_1.{weight,bias} -> blocks[i].norm1.{scale,shift}
h.{i}.attn.c_attn.weight -> blocks[i].attn.W_qkv.weight     (transposed)
h.{i}.attn.c_proj.weight -> blocks[i].attn.W_o.weight       (transposed)
h.{i}.mlp.c_fc.weight    -> blocks[i].ffn.up.weight         (transposed)
h.{i}.mlp.c_proj.weight  -> blocks[i].ffn.down.weight       (transposed)
ln_f.{weight,bias}       -> final_norm.{scale,shift}
```

After loading, `"The capital of France is"` should continue with `" Paris"`. When it does, the
architecture is verified and training can start with confidence that failures are about data and
hyperparameters, not about a silently wrong transpose.

## 7. What gets built

```python
# config.py
@dataclass(frozen=True)
class ModelConfig: ...
    def head_dim(self) -> int
    def n_params(self) -> int          # computed, for comparison against the real count
GPT2_124M: ModelConfig
SMALL:     ModelConfig
MODERN:    ModelConfig

# model.py
class GPTModel(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None
    def forward(self, idx: Tensor, targets: Tensor | None = None) -> Tensor | tuple[Tensor, Tensor]
    def num_parameters(self, non_embedding: bool = False) -> int

# gpt2_weights.py
def load_gpt2_weights(model: GPTModel, size: str = "124M") -> GPTModel
```

`forward` optionally takes targets and returns the loss alongside logits — it keeps the training
loop short and avoids duplicating the flattening logic.

## Tests

| Test | Catches |
|---|---|
| logits shape `(B, T, V)` | wiring |
| **parameter count is exactly 163,009,536 untied** | any structural mistake at all |
| tied count is 124,412,160, and the two weights share storage | tying implemented as a copy instead of a reference |
| loss at init is 10.82 ± 0.5 | initialization scale, mask leaks |
| forward is deterministic in eval mode | dropout left active |
| every `SMALL` config variant builds and runs | that the config switches are wired |
| **GPT-2 weights load and produce " Paris"** | the transposes, and the entire architecture end to end |
| a save → load → forward round trip gives identical logits | checkpoint code |

## Demo

`scripts/demo_04.py` prints the parameter breakdown by component against the table in
ARCHITECTURE §5, generates from the untrained model (noise), then loads GPT-2 weights and generates
again from the same prompt. The before/after in one output is the moment the project becomes real.

---

Next: **[note 5 — training](05_training.md)**.
