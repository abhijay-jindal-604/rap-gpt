"""Does the model use the rhyme word, or only sound like rap?

Take a held-out context, replace the last line's end word with a target word that
does not rhyme with it, and sample the next line. If the model tracks rhyme, the next
line should rhyme with the target more often than it does without the swap.

    python counterfactual.py runs/reversed_block256_seed0 runs/forward_block256_seed0
"""

import argparse
import json
import os
import random
import re
from collections import Counter

import torch

from rapgpt.data import load_corpus, reverse_lines
from rapgpt.model import GPT
from rapgpt.rhyme import end_word, rhymes, stanzas


def target_pool(text, min_count=10):
    counts = Counter(w for lines in stanzas(text) for w in map(end_word, lines) if w)
    return [w for w, n in counts.items() if n >= min_count and rhymes(w, "zebra") is not None]


def build_contexts(text, pool, block_size, count, seed):
    """(context, original context's end word, target) triples ending right after a newline."""
    rng = random.Random(seed)
    contexts = []
    for lines in stanzas(text):
        for i in range(5, len(lines)):
            history = "\n".join(lines[:i]) + "\n"
            word = end_word(lines[i - 1])
            if len(history) < block_size or not word or not re.search(rf"\b{re.escape(word)}$", lines[i - 1].lower()):
                continue
            candidates = [t for t in pool if (r := rhymes(word, t)) and not r[0]]
            if candidates:
                contexts.append((lines[:i], word, rng.choice(candidates)))
    rng.shuffle(contexts)
    return contexts[:count]


def swap_end_word(lines, word, target):
    return lines[:-1] + [lines[-1][: -len(word)] + target]


@torch.no_grad()
def next_end_words(model, corpus, batch_lines, block_size, samples, device):
    """Sample `samples` next lines per context, returning their end words."""
    prompts = []
    for lines in batch_lines:
        text = "\n".join(lines) + "\n"
        text = reverse_lines(text) if corpus.reversed else text
        prompts.append(corpus.encode(text)[-block_size:])
    idx = torch.tensor(prompts, device=device).repeat_interleave(samples, dim=0)
    with torch.autocast(device, dtype=torch.bfloat16):
        out = model.generate(idx, 70)[:, block_size:].tolist()
    words = []
    for seq in out:
        line = corpus.decode(seq).split("\n")[0]
        line = line[::-1] if corpus.reversed else line
        words.append(end_word(line))
    return [words[i : i + samples] for i in range(0, len(words), samples)]


def rate(groups, targets):
    hits, scored = 0, 0
    for group, target in zip(groups, targets):
        for word in group:
            result = rhymes(word, target) if word else None
            if result is not None:
                scored += 1
                hits += result[0]
    return hits / scored if scored else 0.0


def evaluate(run_dir, data, contexts_n, samples, batch, device):
    checkpoint = torch.load(os.path.join(run_dir, "best.pt"), weights_only=False)
    cfg = checkpoint["cfg"]
    corpus = load_corpus(data, reversed_lines=run_dir.split("/")[-1].startswith("reversed"))
    model = GPT(cfg).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    pool = target_pool(corpus.train_text)
    contexts = build_contexts(corpus.val_text, pool, cfg.block_size, contexts_n, seed=0)
    torch.manual_seed(0)
    base, swapped = [], []
    for i in range(0, len(contexts), batch):
        chunk = contexts[i : i + batch]
        base += next_end_words(model, corpus, [c[0] for c in chunk], cfg.block_size, samples, device)
        swapped += next_end_words(model, corpus, [swap_end_word(*c) for c in chunk], cfg.block_size, samples, device)
    originals = [c[1] for c in contexts]
    targets = [c[2] for c in contexts]
    result = {
        "run": os.path.basename(run_dir),
        "contexts": len(contexts),
        "samples_per_context": samples,
        "target_rate_without_swap": rate(base, targets),
        "target_rate_with_swap": rate(swapped, targets),
        "original_rate_without_swap": rate(base, originals),
        "original_rate_with_swap": rate(swapped, originals),
    }
    result["target_effect"] = result["target_rate_with_swap"] - result["target_rate_without_swap"]
    result["original_effect"] = result["original_rate_with_swap"] - result["original_rate_without_swap"]
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="+")
    p.add_argument("--data", default="data/lyrics_raw.csv")
    p.add_argument("--contexts", type=int, default=500)
    p.add_argument("--samples", type=int, default=8)
    p.add_argument("--batch", type=int, default=25)
    p.add_argument("--out", default="results/counterfactual.json")
    args = p.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    results = []
    for run_dir in args.runs:
        results.append(evaluate(run_dir.rstrip("/"), args.data, args.contexts, args.samples, args.batch, device))
        print(json.dumps(results[-1]), flush=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
