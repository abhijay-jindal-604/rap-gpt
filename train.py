"""Train one model, generate from it, and score its output.

    python train.py --block-size 128 --seed 0
"""

import argparse
import json
import os
import time

import torch

from rapgpt.data import load_corpus, reverse_lines
from rapgpt.model import GPT, GPTConfig
from rapgpt.rhyme import line_pairs, score_text, stanzas


def autocast(device):
    return torch.autocast(device, dtype=torch.bfloat16)


def get_batch(data, block_size, batch_size, device, artist_ids=None):
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    artist = artist_ids[ix].to(device) if artist_ids is not None else None
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    return x.to(device), y.to(device), artist


@torch.no_grad()
def estimate_loss(model, data, args, device, artist_ids=None):
    """Mean loss over all positions, and loss at the last position alone.

    Early positions see little context under any block size, so the last-position
    loss is the fairer way to compare models trained with different context lengths.
    """
    model.eval()
    total, last = 0.0, 0.0
    for _ in range(args.eval_iters):
        x, y, artist = get_batch(data, args.block_size, args.batch_size, device, artist_ids)
        with autocast(device):
            logits, loss = model(x, y, artist)
        total += loss.item()
        last += torch.nn.functional.cross_entropy(logits[:, -1, :].float(), y[:, -1]).item()
    model.train()
    return total / args.eval_iters, last / args.eval_iters


def generate_text(model, corpus, args, device, warmup=256):
    """Sample batches, discarding a fixed warm-up so every model starts from full context."""
    newline = corpus.encode("\n")[0]
    samples, by_artist = [], {name: [] for name in corpus.artists}
    conditioned = model.cfg.n_artists > 0
    artist = torch.arange(args.sample_seqs, device=device) % len(corpus.artists) if conditioned else None
    model.eval()
    for _ in range(args.sample_batches):
        start = torch.full((args.sample_seqs, 1), newline, dtype=torch.long, device=device)
        with autocast(device):
            out = model.generate(start, args.sample_chars + warmup, artist=artist)
        for i, seq in enumerate(out[:, warmup:].tolist()):
            text = corpus.decode(seq)
            # the first and last lines are cut off by the sampling window
            text = "\n".join(text.split("\n")[1:-1])
            text = reverse_lines(text) if corpus.reversed else text
            samples.append(text)
            if conditioned:
                by_artist[corpus.artists[i % len(corpus.artists)]].append(text)
    return "\n\n".join(samples), by_artist


def evaluate_rhyme(text, corpus):
    train_lines = {ln for lines in stanzas(corpus.train_text) for ln in lines}
    lines = [ln for group in stanzas(text) for ln in group]
    long_lines = [ln for ln in lines if len(ln) >= 20]
    report = {
        "lines": len(lines),
        "copied_line_rate": sum(ln in train_lines for ln in long_lines) / max(1, len(long_lines)),
    }
    for distance in (1, 2):
        train_pairs = line_pairs(corpus.train_text, distance)
        report[f"d{distance}"] = score_text(text, distance)
        report[f"d{distance}_novel"] = score_text(text, distance, exclude=train_pairs)
    return report


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/lyrics_raw.csv")
    p.add_argument("--out", default="runs")
    p.add_argument("--block-size", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--reversed", action="store_true", help="generate each line backwards, rhyme word first")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--artist-embedding", action="store_true", help="condition the model on the artist")
    p.add_argument("--max-iters", type=int, default=5000)
    p.add_argument("--eval-interval", type=int, default=250)
    p.add_argument("--eval-iters", type=int, default=50)
    p.add_argument("--patience", type=int, default=4, help="evals without improvement before stopping")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--n-layer", type=int, default=6)
    p.add_argument("--n-head", type=int, default=6)
    p.add_argument("--n-embd", type=int, default=384)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--sample-batches", type=int, default=2)
    p.add_argument("--sample-seqs", type=int, default=20)
    p.add_argument("--sample-chars", type=int, default=2000)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    corpus = load_corpus(args.data, reversed_lines=args.reversed)
    cfg = GPTConfig(corpus.vocab_size, args.block_size, args.n_layer, args.n_head, args.n_embd, args.dropout,
                    len(corpus.artists) if args.artist_embedding else 0)
    model = GPT(cfg).to(device)
    train_artist = corpus.train_artist if args.artist_embedding else None
    val_artist = corpus.val_artist if args.artist_embedding else None
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    params = sum(p.numel() for p in model.parameters())
    print(f"{params / 1e6:.2f}M parameters, vocab {corpus.vocab_size}, "
          f"{len(corpus.train):,} train / {len(corpus.val):,} val chars, {device}", flush=True)

    direction = "reversed" if args.reversed else "forward"
    run_dir = os.path.join(args.out, f"{direction}_block{args.block_size}_seed{args.seed}")
    os.makedirs(run_dir, exist_ok=True)
    ckpt_path = os.path.join(run_dir, "best.pt")
    history, best = [], None
    started = time.time()
    for step in range(args.max_iters + 1):
        if step % args.eval_interval == 0:
            train_loss, _ = estimate_loss(model, corpus.train, args, device, train_artist)
            val_loss, val_last = estimate_loss(model, corpus.val, args, device, val_artist)
            history.append({"step": step, "train": train_loss, "val": val_loss, "val_last": val_last})
            print(f"step {step}: train {train_loss:.4f}, val {val_loss:.4f}, "
                  f"val@last {val_last:.4f} ({time.time() - started:.0f}s)", flush=True)
            if best is None or val_loss < best["val"]:
                best = history[-1]
                torch.save({"model": model.state_dict(), "cfg": cfg, "itos": corpus.itos}, ckpt_path)
            elif step - best["step"] >= args.patience * args.eval_interval:
                print(f"early stop at step {step}, best was {best['step']}", flush=True)
                break
        if step == args.max_iters:
            break
        xb, yb, artist = get_batch(corpus.train, args.block_size, args.batch_size, device, train_artist)
        with autocast(device):
            _, loss = model(xb, yb, artist)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    train_seconds = time.time() - started
    model.load_state_dict(torch.load(ckpt_path, weights_only=False)["model"])
    text, by_artist = generate_text(model, corpus, args, device)
    if args.artist_embedding:
        with open(os.path.join(run_dir, "samples_by_artist.json"), "w", encoding="utf-8") as f:
            json.dump(by_artist, f)
    with open(os.path.join(run_dir, "samples.txt"), "w", encoding="utf-8") as f:
        f.write(text)
    result = {
        "block_size": args.block_size,
        "direction": direction,
        "seed": args.seed,
        "params": params,
        "train_seconds": train_seconds,
        "best": best,
        "history": history,
        "rhyme": evaluate_rhyme(text, corpus),
    }
    with open(os.path.join(run_dir, "result.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps({k: result["rhyme"][k] for k in ("copied_line_rate", "d1", "d1_novel")}, indent=2), flush=True)
    return result


if __name__ == "__main__":
    main()
