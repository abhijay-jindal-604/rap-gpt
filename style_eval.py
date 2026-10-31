"""Do artist-conditioned samples read like the requested artist?

Fit a character n-gram classifier on real training songs, then score it on held-out
songs (the ceiling) and on generated text (the model). Chance is 1 / number of artists.

    python style_eval.py runs_artist/reversed_block256_seed0 ...
"""

import argparse
import json
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from rapgpt.data import load_songs, split_songs


def chunks(text, size=1000):
    return [text[i : i + size] for i in range(0, len(text) - size + 1, size)]


def labelled(pairs):
    texts, labels = [], []
    for text, label in pairs:
        pieces = chunks(text)
        texts += pieces
        labels += [label] * len(pieces)
    return texts, labels


def main():
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="+")
    p.add_argument("--data", default="data/lyrics_raw.csv")
    p.add_argument("--out", default="results/style.json")
    args = p.parse_args()
    train, val = split_songs(load_songs(args.data))
    x_train, y_train = labelled((s["lyrics"], s["artist"]) for s in train)
    x_val, y_val = labelled((s["lyrics"], s["artist"]) for s in val)
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=2, sublinear_tf=True)
    classifier = LogisticRegression(max_iter=2000, C=10.0)
    classifier.fit(vectorizer.fit_transform(x_train), y_train)
    results = {
        "chance": 1 / len(set(y_train)),
        "real_held_out_accuracy": classifier.score(vectorizer.transform(x_val), y_val),
        "real_held_out_chunks": len(y_val),
        "runs": [],
    }
    for run_dir in args.runs:
        with open(os.path.join(run_dir, "samples_by_artist.json"), encoding="utf-8") as f:
            by_artist = json.load(f)
        x, y = labelled((text, artist) for artist, texts in by_artist.items() for text in texts)
        results["runs"].append({"run": os.path.basename(run_dir.rstrip("/")), "chunks": len(y),
                                "accuracy": classifier.score(vectorizer.transform(x), y)})
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
