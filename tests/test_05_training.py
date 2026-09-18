"""Tests for note 5 - loss, optimizer, schedule, checkpointing and the loop.

The one that matters most is test_loss_decreases_on_a_repeated_batch: if a model cannot drive the
loss down on a single batch shown over and over, nothing else in this file is worth running.
"""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest
import torch
from torch import nn

from llm import checkpoint
from llm.config import SMALL, ModelConfig
from llm.dataset import make_dataloader
from llm.losses import compute_loss, evaluate
from llm.model import GPTModel
from llm.optim import lr_at_step, make_optimizer, make_schedule
from llm.train import TrainConfig, check_vocabulary, load_text_corpus, plot_history, train

VOCAB, CTX = 512, 32


@pytest.fixture
def cfg() -> ModelConfig:
    # Tiny and CPU-friendly: these tests are about the training machinery, not the architecture.
    return replace(
        SMALL,
        vocab_size=VOCAB,
        context_length=CTX,
        emb_dim=64,
        n_layers=2,
        n_heads=4,
        attn="manual",
        dropout=0.0,
    )


@pytest.fixture
def model(cfg: ModelConfig) -> GPTModel:
    torch.manual_seed(0)
    return GPTModel(cfg)


@pytest.fixture
def ids() -> np.ndarray:
    return np.random.default_rng(0).integers(0, VOCAB, 4000, dtype=np.uint16)


@pytest.fixture
def loader(ids: np.ndarray):
    return make_dataloader(ids, CTX, CTX, batch_size=4, shuffle=False)


# --- loss -----------------------------------------------------------------------------------


def test_compute_loss_matches_cross_entropy(cfg: ModelConfig) -> None:
    logits = torch.randn(2, 8, VOCAB)
    targets = torch.randint(0, VOCAB, (2, 8))
    expected = nn.functional.cross_entropy(logits.reshape(-1, VOCAB), targets.reshape(-1))
    assert torch.allclose(compute_loss(logits, targets), expected)


def test_perfect_prediction_gives_zero_loss() -> None:
    targets = torch.randint(0, VOCAB, (2, 8))
    logits = torch.full((2, 8, VOCAB), -1e4)
    logits.scatter_(-1, targets.unsqueeze(-1), 1e4)
    assert compute_loss(logits, targets).item() < 1e-3


def test_untrained_loss_is_ln_vocab(model: GPTModel, cfg: ModelConfig) -> None:
    x = torch.randint(0, VOCAB, (4, CTX))
    y = torch.randint(0, VOCAB, (4, CTX))
    with torch.no_grad():
        _, loss = model(x, targets=y)
    assert abs(loss.item() - math.log(VOCAB)) < 0.5


def test_evaluate_reports_loss_and_perplexity(model: GPTModel, loader) -> None:
    stats = evaluate(model, loader, max_batches=3)
    assert stats["batches"] == 3
    assert math.isclose(stats["perplexity"], math.exp(stats["loss"]), rel_tol=1e-6)


def test_evaluate_restores_training_mode(model: GPTModel, loader) -> None:
    # Leaving the model in eval mode mid-run silently disables dropout for the rest of training.
    model.train()
    evaluate(model, loader, max_batches=1)
    assert model.training


def test_evaluate_does_not_change_weights(model: GPTModel, loader) -> None:
    before = model.tok_emb.weight.clone()
    evaluate(model, loader, max_batches=2)
    assert torch.equal(before, model.tok_emb.weight)


# --- optimizer ------------------------------------------------------------------------------


def test_weight_decay_skips_biases_and_norms(model: GPTModel) -> None:
    """Pulling a norm's scale toward zero fights the normalization it exists to perform."""
    decay, plain = make_optimizer(model, lr=1e-3).param_groups
    assert decay["weight_decay"] > 0
    assert plain["weight_decay"] == 0.0
    assert all(p.ndim >= 2 for p in decay["params"])
    assert all(p.ndim < 2 for p in plain["params"])


def test_every_parameter_lands_in_exactly_one_group(model: GPTModel) -> None:
    groups = make_optimizer(model, lr=1e-3).param_groups
    grouped = {id(p) for g in groups for p in g["params"]}
    assert grouped == {id(p) for p in model.parameters()}
    assert sum(len(g["params"]) for g in groups) == len(list(model.parameters()))


def test_betas_use_the_transformer_convention(model: GPTModel) -> None:
    assert make_optimizer(model, lr=1e-3).param_groups[0]["betas"] == (0.9, 0.95)


# --- schedule -------------------------------------------------------------------------------


def test_warmup_rises_to_peak_exactly_at_warmup_steps() -> None:
    warmup, total = 100, 1000
    assert lr_at_step(0, warmup_steps=warmup, total_steps=total) == pytest.approx(0.01)
    assert lr_at_step(warmup - 1, warmup_steps=warmup, total_steps=total) == pytest.approx(1.0)


def test_first_step_is_not_a_dead_step() -> None:
    # Off-by-one here makes step 0 a no-op at lr exactly 0.
    assert lr_at_step(0, warmup_steps=10, total_steps=100) > 0


def test_decay_reaches_the_floor_at_total_steps() -> None:
    warmup, total, floor = 10, 100, 0.1
    end = lr_at_step(total - 1, warmup_steps=warmup, total_steps=total, min_lr_ratio=floor)
    assert end == pytest.approx(floor, abs=1e-3)
    # Flat past the end rather than negative or wrapping.
    assert lr_at_step(total + 50, warmup_steps=warmup, total_steps=total, min_lr_ratio=floor) == floor


def test_schedule_decreases_monotonically_after_warmup() -> None:
    values = [lr_at_step(s, warmup_steps=10, total_steps=200) for s in range(10, 200)]
    assert all(a >= b - 1e-9 for a, b in zip(values, values[1:]))


def test_scheduler_drives_the_optimizer(model: GPTModel) -> None:
    opt = make_optimizer(model, lr=1e-3)
    sched = make_schedule(opt, warmup_steps=5, total_steps=50)
    assert opt.param_groups[0]["lr"] == pytest.approx(1e-3 * 0.2)
    for _ in range(5):
        opt.step()
        sched.step()
    assert opt.param_groups[0]["lr"] == pytest.approx(1e-3, rel=1e-3)


# --- gradient accumulation --------------------------------------------------------------------


def test_accumulation_matches_one_large_batch(cfg: ModelConfig) -> None:
    """Four micro-batches of 2 must give the same gradient as one batch of 8.

    This is what the `loss / accum_steps` division buys. Without it the effective learning rate
    is multiplied by accum_steps, which looks like a badly tuned LR rather than a bug.
    """
    torch.manual_seed(0)
    x = torch.randint(0, VOCAB, (8, CTX))
    y = torch.randint(0, VOCAB, (8, CTX))

    torch.manual_seed(0)
    big = GPTModel(cfg)
    _, loss = big(x, targets=y)
    loss.backward()

    torch.manual_seed(0)
    accum = GPTModel(cfg)
    for i in range(4):
        sl = slice(i * 2, (i + 1) * 2)
        _, micro = accum(x[sl], targets=y[sl])
        (micro / 4).backward()

    for (name, a), (_, b) in zip(big.named_parameters(), accum.named_parameters()):
        assert torch.allclose(a.grad, b.grad, atol=1e-5), name


def test_gradient_clipping_bounds_the_norm(model: GPTModel, cfg: ModelConfig) -> None:
    x = torch.randint(0, VOCAB, (2, CTX))
    _, loss = model(x, targets=x)
    (loss * 1000).backward()  # deliberately enormous
    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    total = torch.norm(torch.stack([p.grad.norm() for p in model.parameters()]))
    assert total.item() <= 1.0 + 1e-4


# --- the loop --------------------------------------------------------------------------------


def test_loss_decreases_on_a_repeated_batch(cfg: ModelConfig) -> None:
    """The 'can it learn at all' test. If this fails, nothing else matters."""
    torch.manual_seed(0)
    model = GPTModel(cfg)
    opt = make_optimizer(model, lr=3e-3, weight_decay=0.0)
    x = torch.randint(0, VOCAB, (2, CTX))
    y = torch.randint(0, VOCAB, (2, CTX))

    losses = []
    for _ in range(25):
        opt.zero_grad(set_to_none=True)
        _, loss = model(x, targets=y)
        loss.backward()
        opt.step()
        losses.append(loss.item())

    assert losses[-1] < losses[0] * 0.5, f"{losses[0]:.3f} -> {losses[-1]:.3f}"


def test_train_runs_and_records_history(cfg: ModelConfig, ids: np.ndarray) -> None:
    loader = make_dataloader(ids, CTX, CTX, batch_size=2)
    torch.manual_seed(0)
    history = train(
        GPTModel(cfg),
        loader,
        loader,
        TrainConfig(max_steps=12, warmup_steps=2, eval_every=6, log_every=0, bf16=False),
        device=torch.device("cpu"),
        verbose=False,
    )
    assert len(history["train_loss"]) == 12
    # Evaluates at step 0, at every multiple of eval_every, and always on the final step:
    # 0, 6, and 11.
    assert history["val_step"] == [0, 6, 11]
    assert history["tokens_seen"] == 12 * 2 * CTX


def test_forgetting_zero_grad_would_be_caught(cfg: ModelConfig) -> None:
    """backward() accumulates into .grad; it does not replace it."""
    torch.manual_seed(0)
    model = GPTModel(cfg)
    x = torch.randint(0, VOCAB, (2, CTX))
    _, loss = model(x, targets=x)
    loss.backward()
    first = model.tok_emb.weight.grad.clone()
    _, loss = model(x, targets=x)
    loss.backward()
    assert torch.allclose(model.tok_emb.weight.grad, 2 * first, atol=1e-5)


# --- refusing to start a doomed run -----------------------------------------------------------


def test_warmup_longer_than_the_run_is_refused(cfg: ModelConfig, loader) -> None:
    """Warmup that outlasts the run means the cosine never decays.

    The model finishes at peak learning rate, and the last stretch of decay is where a surprising
    share of the final loss improvement comes from. Nothing in the log says this happened: the
    loss falls, the run completes, the result is quietly worse.
    """
    with pytest.raises(ValueError, match="never decays"):
        train(
            GPTModel(cfg),
            loader,
            None,
            TrainConfig(max_steps=20, warmup_steps=20, bf16=False),
            device=torch.device("cpu"),
            verbose=False,
        )


@pytest.mark.parametrize("field,value", [("max_steps", 0), ("accum_steps", 0)])
def test_nonsensical_step_counts_are_refused(cfg: ModelConfig, loader, field, value) -> None:
    kwargs = {"max_steps": 20, "warmup_steps": 2, "bf16": False, field: value}
    tcfg = TrainConfig(**kwargs)
    with pytest.raises(ValueError, match=field):
        train(GPTModel(cfg), loader, None, tcfg, device=torch.device("cpu"), verbose=False)


def test_vocabulary_mismatch_is_caught_before_the_run_starts(cfg: ModelConfig) -> None:
    """The failure this replaces is a CUDA device-side assert: asynchronous, so the traceback
    points somewhere unrelated, and fatal to the context, so nothing after it is recoverable."""
    corpus = np.array([0, 1, VOCAB], dtype=np.uint16)  # one id past the end of the table
    with pytest.raises(ValueError, match="vocab_size"):
        check_vocabulary(cfg, corpus, "test corpus")

    check_vocabulary(cfg, np.array([0, VOCAB - 1], dtype=np.uint16), "test corpus")


def test_empty_corpus_is_caught(cfg: ModelConfig) -> None:
    with pytest.raises(ValueError, match="no tokens"):
        check_vocabulary(cfg, np.array([], dtype=np.uint16), "test corpus")


# --- reporting --------------------------------------------------------------------------------


def test_history_carries_the_vocabulary_for_the_plot(cfg: ModelConfig, loader) -> None:
    """The plot's reference line is ln(vocab_size). Hardcoded at ln(50257) it sits 1.6 nats too
    high on a 10k-vocabulary run, and every curve under it looks better than it is."""
    history = train(
        GPTModel(cfg),
        loader,
        None,
        TrainConfig(max_steps=2, warmup_steps=1, log_every=0, eval_every=0, bf16=False),
        device=torch.device("cpu"),
        verbose=False,
    )
    assert history["vocab_size"] == cfg.vocab_size


def test_plot_history_survives_without_matplotlib(tmp_path) -> None:
    history = {"step": [0], "train_loss": [1.0], "val_step": [], "val_loss": [], "vocab_size": 64}
    out = plot_history(history, tmp_path / "loss.png")
    assert out is None or out.exists()


# --- checkpointing ---------------------------------------------------------------------------


def test_save_load_round_trip_gives_identical_logits(cfg: ModelConfig, tmp_path) -> None:
    torch.manual_seed(0)
    model = GPTModel(cfg).eval()
    x = torch.randint(0, VOCAB, (2, CTX))
    with torch.no_grad():
        before = model(x)

    checkpoint.save(tmp_path / "c.pt", model, step=7, cfg=cfg)
    restored = GPTModel(cfg).eval()
    assert checkpoint.load(tmp_path / "c.pt", restored) == 7
    with torch.no_grad():
        assert torch.equal(before, restored(x))


def test_checkpoint_restores_optimizer_and_scheduler(cfg: ModelConfig, tmp_path) -> None:
    """Weights alone are not resumable: Adam's moments and the LR position matter too."""
    torch.manual_seed(0)
    model = GPTModel(cfg)
    opt = make_optimizer(model, lr=1e-3)
    sched = make_schedule(opt, warmup_steps=2, total_steps=20)
    x = torch.randint(0, VOCAB, (2, CTX))
    for _ in range(5):
        opt.zero_grad(set_to_none=True)
        _, loss = model(x, targets=x)
        loss.backward()
        opt.step()
        sched.step()

    checkpoint.save(tmp_path / "c.pt", model, opt, sched, step=5, cfg=cfg)

    model2 = GPTModel(cfg)
    opt2 = make_optimizer(model2, lr=1e-3)
    sched2 = make_schedule(opt2, warmup_steps=2, total_steps=20)
    checkpoint.load(tmp_path / "c.pt", model2, opt2, sched2)

    assert sched2.get_last_lr() == sched.get_last_lr()
    state, state2 = opt.state_dict()["state"], opt2.state_dict()["state"]
    assert torch.allclose(state[0]["exp_avg"], state2[0]["exp_avg"])


def test_checkpoint_carries_its_architecture(cfg: ModelConfig, tmp_path) -> None:
    checkpoint.save(tmp_path / "c.pt", GPTModel(cfg), cfg=cfg)
    assert checkpoint.load_config(tmp_path / "c.pt") == cfg


# --- corpus ---------------------------------------------------------------------------------


def test_load_text_corpus_splits_without_overlap(tmp_path) -> None:
    from llm.tokenizer import BPETokenizer

    path = tmp_path / "t.txt"
    path.write_text("the cat sat on the mat. " * 200, encoding="utf-8")
    train_ids, val_ids = load_text_corpus(path, BPETokenizer(), val_fraction=0.1)
    assert len(val_ids) > 0
    assert abs(len(val_ids) / (len(train_ids) + len(val_ids)) - 0.1) < 0.02


# --- crash safety ---------------------------------------------------------------------------------


def _tiny_run(cfg, ids, out, steps, resume=False, save_every=4):
    from llm.train import TrainConfig, train

    loader = make_dataloader(ids, CTX, CTX, batch_size=2)
    torch.manual_seed(0)
    model = GPTModel(cfg)
    tcfg = TrainConfig(max_steps=steps, warmup_steps=2, eval_every=0, log_every=0, bf16=False,
                       save_every=save_every, out_dir=out)
    history = train(model, loader, None, tcfg, device=torch.device("cpu"), verbose=False,
                    resume_from=out / "latest.pt" if resume else None,
                    metrics_path=out / "metrics.jsonl")
    return model, history


def test_resume_continues_from_the_saved_step(cfg, ids, tmp_path) -> None:
    """Interrupt a run partway, resume it, and check it picks up where it stopped.

    Nothing short of doing this proves the checkpoint carries everything a restart needs.
    """
    _tiny_run(cfg, ids, tmp_path, steps=9, save_every=4)   # saves at 4, 8 and the last step
    first = torch.load(tmp_path / "latest.pt", weights_only=True)
    assert first["step"] == 8
    assert first["tokens_seen"] == 9 * 2 * CTX

    # "Crash" is simulated by running a longer schedule from that checkpoint.
    _, history = _tiny_run(cfg, ids, tmp_path, steps=14, resume=True, save_every=4)
    assert history["step"][0] == 9, "resumed run should start at the step after the checkpoint"
    assert history["tokens_seen"] == 14 * 2 * CTX


def test_resume_restores_the_learning_rate_schedule(cfg, ids, tmp_path) -> None:
    """A fresh scheduler would restart warmup: the LR would drop to near zero mid-run."""
    from llm.optim import lr_at_step

    _tiny_run(cfg, ids, tmp_path, steps=8, save_every=4)
    _, history = _tiny_run(cfg, ids, tmp_path, steps=8, resume=True, save_every=4)
    # The run already finished at step 7, so resuming does no more work.
    assert history["step"] == []

    _tiny_run(cfg, ids, tmp_path / "b", steps=5, save_every=4)  # checkpoint at 4
    _, history = _tiny_run(cfg, ids, tmp_path / "b", steps=20, resume=True, save_every=4)
    assert history["step"][0] == 5
    # The first resumed step uses step 5's learning rate, not step 0's.
    expected = 3e-4 * lr_at_step(6, warmup_steps=2, total_steps=20)
    assert history["lr"][0] == pytest.approx(expected, rel=1e-6)


def test_metrics_log_survives_and_prefers_the_latest_record(cfg, ids, tmp_path) -> None:
    """After a crash the steps since the last checkpoint run twice; the rerun must win."""
    from llm.train import read_metrics

    _tiny_run(cfg, ids, tmp_path, steps=9, save_every=4)
    lines_before = (tmp_path / "metrics.jsonl").read_text().count("\n")
    assert lines_before == 9

    # Pretend the process died after step 8 and a rerun covered 6..8 again.
    with open(tmp_path / "metrics.jsonl", "a") as f:
        f.write('{"kind": "train", "step": 7, "loss": 1.234, "lr": 0.0001, "tokens": 0}\n')
    history = read_metrics(tmp_path / "metrics.jsonl")
    assert history["step"] == list(range(9))
    assert history["train_loss"][7] == 1.234


def test_checkpoint_extras_round_trip(cfg, tmp_path) -> None:
    extras = {}
    checkpoint.save(tmp_path / "c.pt", GPTModel(cfg), step=3, cfg=cfg, tokens_seen=12345)
    checkpoint.load(tmp_path / "c.pt", GPTModel(cfg), extras=extras)
    assert extras["tokens_seen"] == 12345 and extras["step"] == 3
