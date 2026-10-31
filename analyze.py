"""Aggregate finished runs into results/summary.json and results/context_vs_rhyme.png."""

import glob
import json
import os
import statistics as st

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rapgpt.data import load_corpus
from rapgpt.rhyme import end_word_pairs, line_pairs, score_text


def mean_sd(values):
    values = list(values)
    return {"mean": st.mean(values), "sd": st.stdev(values) if len(values) > 1 else 0.0}


def rescore(run_path, corpus, copied, seen):
    """Score the saved samples with the current metric, so changing it needs no retraining."""
    with open(run_path) as f:
        run = json.load(f)
    with open(os.path.join(os.path.dirname(run_path), "samples.txt"), encoding="utf-8") as f:
        text = f.read()
    for distance in (1, 2):
        run["rhyme"][f"d{distance}"] = score_text(text, distance, seen_word_pairs=seen[distance])
        run["rhyme"][f"d{distance}_novel"] = score_text(
            text, distance, exclude=copied[distance], seen_word_pairs=seen[distance])
    return run


def main():
    corpus = load_corpus("data/lyrics_raw.csv")
    copied = {d: line_pairs(corpus.train_text, d) for d in (1, 2)}
    seen = {d: end_word_pairs(corpus.train_text, d) for d in (1, 2)}
    runs = [rescore(p, corpus, copied, seen) for p in glob.glob("runs/*_block*_seed*/result.json")]
    real = {
        "train_d1": score_text(corpus.train_text, 1),
        "train_d2": score_text(corpus.train_text, 2),
        "val_d1": score_text(corpus.val_text, 1, seen_word_pairs=seen[1]),
        "val_d2": score_text(corpus.val_text, 2, seen_word_pairs=seen[2]),
    }
    table = []
    for direction, block in sorted({(r["direction"], r["block_size"]) for r in runs}):
        group = [r for r in runs if r["direction"] == direction and r["block_size"] == block]
        row = {
            "direction": direction,
            "block_size": block,
            "seeds": len(group),
            "val_loss": mean_sd(r["best"]["val"] for r in group),
            "val_last": mean_sd(r["best"]["val_last"] for r in group),
            "best_step": mean_sd(r["best"]["step"] for r in group),
            "copied_line_rate": mean_sd(r["rhyme"]["copied_line_rate"] for r in group),
        }
        for key in ("d1", "d2", "d1_novel", "d2_novel"):
            for metric in ("perfect", "slant", "lift_perfect", "lift_slant", "lift_per_pair",
                           "coverage", "same_spelling_share", "seen_pair_share"):
                row[f"{key}_{metric}"] = mean_sd(r["rhyme"][key][metric] for r in group)
        table.append(row)

    os.makedirs("results", exist_ok=True)
    with open("results/summary.json", "w") as f:
        json.dump({"real": real, "table": table}, f, indent=2)

    colors = {"forward": "#2a6f97", "reversed": "#d1495b"}
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for direction, color in colors.items():
        rows = [r for r in table if r["direction"] == direction]
        blocks = [r["block_size"] for r in rows]
        axes[0].errorbar(blocks, [100 * r["d1_novel_lift_perfect"]["mean"] for r in rows],
                         yerr=[100 * r["d1_novel_lift_perfect"]["sd"] for r in rows],
                         marker="o", capsize=3, label=direction, color=color)
        axes[1].errorbar(blocks, [100 * r["d1_novel_coverage"]["mean"] for r in rows],
                         yerr=[100 * r["d1_novel_coverage"]["sd"] for r in rows],
                         marker="o", capsize=3, label=direction, color=color)
    axes[0].axhline(0, color="#888", lw=1, ls=":")
    axes[0].axhline(100 * real["val_d1"]["lift_perfect"], color="#555", lw=1, ls="--", label="real held-out songs")
    axes[0].set_ylabel("Perfect-rhyme lift over chance (pct. points)")
    axes[0].set_title("Rhyme between adjacent lines, novel pairs only", fontsize=10)
    axes[1].set_ylabel("Line pairs with dictionary words (%)")
    axes[1].set_title("Share of line pairs that could be scored", fontsize=10)
    for ax in axes:
        ax.set_xscale("log", base=2)
        ticks = sorted({r["block_size"] for r in table})
        ax.set_xticks(ticks, [str(t) for t in ticks])
        ax.minorticks_off()
        ax.set_xlabel("Context length (characters)")
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig("results/context_vs_rhyme.png", dpi=160)

    def pct(cell):
        return f"{100 * cell['mean']:+.1f} ± {100 * cell['sd']:.1f}"

    print("| direction | context | val loss | loss@last | d1 lift | d2 lift | d1 lift per pair | d1 slant lift "
          "| coverage | same spelling | seen word pair | copied lines |")
    print("|" + "---|" * 12)
    for r in table:
        print(f"| {r['direction']} | {r['block_size']} | {r['val_loss']['mean']:.3f} | {r['val_last']['mean']:.3f} "
              f"| {pct(r['d1_novel_lift_perfect'])} | {pct(r['d2_novel_lift_perfect'])} "
              f"| {pct(r['d1_novel_lift_per_pair'])} | {pct(r['d1_novel_lift_slant'])} "
              f"| {100 * r['d1_novel_coverage']['mean']:.0f}% | {100 * r['d1_novel_same_spelling_share']['mean']:.0f}% "
              f"| {100 * r['d1_novel_seen_pair_share']['mean']:.0f}% | {100 * r['copied_line_rate']['mean']:.1f}% |")
    for name, res in real.items():
        print(name, {k: round(v, 3) for k, v in res.items() if isinstance(v, float)})


if __name__ == "__main__":
    main()
