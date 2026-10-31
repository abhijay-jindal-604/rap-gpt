"""Score rhyme between line endings using the CMU Pronouncing Dictionary.

perfect: identical sounds from the last stressed vowel onward (plan / man)
slant:   same last stressed vowel, whatever follows (time / line). Closer to how rap
         rhymes, but really assonance: it also accepts weak matches.

Repeating a word is not a rhyme, so those pairs are skipped. Otherwise a hook
repeated over and over would score as perfect rhyming. Words missing from the
dictionary are skipped too, and `coverage` records how many pairs survived.
"""

import random
import re

import pronouncing

PAREN_RE = re.compile(r"\([^)]*\)?")  # ad-libs such as "(Ayy, don't)"
WORD_RE = re.compile(r"[a-z]+(?:'[a-z]+)*'?")


def end_word(line: str) -> str | None:
    words = WORD_RE.findall(PAREN_RE.sub(" ", line.lower()))
    if not words:
        return None
    word = words[-1]
    if word.endswith("in'"):  # wishin' -> wishing
        return word[:-1] + "g"
    return word.rstrip("'")


def _endings(word: str) -> tuple[set[tuple[str, ...]], set[str]]:
    perfect, slant = set(), set()
    for phones in pronouncing.phones_for_word(word):
        sounds = phones.split()
        stressed = [i for i, p in enumerate(sounds) if p[-1] in "12"]
        if stressed:  # readings with no stress, like the/a, cannot rhyme
            tail = sounds[stressed[-1]:]
            perfect.add(tuple(p.rstrip("012") for p in tail))
            slant.add(tail[0].rstrip("012"))
    return perfect, slant


_cache: dict[str, tuple[set, set]] = {}


def endings(word: str) -> tuple[set, set]:
    if word not in _cache:
        _cache[word] = _endings(word)
    return _cache[word]


def rhymes(a: str, b: str) -> tuple[bool, bool] | None:
    """(perfect, slant) for two distinct known words, else None."""
    if a == b:
        return None
    perfect_a, slant_a = endings(a)
    perfect_b, slant_b = endings(b)
    if not perfect_a or not perfect_b:
        return None
    return bool(perfect_a & perfect_b), bool(slant_a & slant_b)


def stanzas(text: str) -> list[list[str]]:
    """Non-empty lines grouped by blank-line breaks; rhymes are not counted across a break."""
    groups = re.split(r"\n\s*\n", text)
    return [[ln for ln in g.split("\n") if ln.strip()] for g in groups if g.strip()]


def _rates(pairs: list[tuple[str, str]]) -> dict:
    scored = [r for r in (rhymes(a, b) for a, b in pairs) if r is not None]
    n = len(scored)
    return {
        "pairs": len(pairs),
        "scored": n,
        "coverage": n / len(pairs) if pairs else 0.0,
        "perfect": sum(r[0] for r in scored) / n if n else 0.0,
        "slant": sum(r[1] for r in scored) / n if n else 0.0,
        # Counts unscored pairs as misses, so low coverage cannot inflate it.
        "perfect_per_pair": sum(r[0] for r in scored) / len(pairs) if pairs else 0.0,
    }


def end_word_pairs(text: str, distance: int = 1) -> set[frozenset]:
    pairs = set()
    for lines in stanzas(text):
        ends = [end_word(ln) for ln in lines]
        pairs.update(frozenset((a, b)) for a, b in zip(ends, ends[distance:]) if a and b)
    return pairs


def score_text(
    text: str,
    distance: int = 1,
    exclude: set | None = None,
    shuffles: int = 20,
    seed: int = 0,
    seen_word_pairs: set | None = None,
) -> dict:
    """Rhyme rate for lines `distance` apart, plus a chance baseline.

    The baseline re-pairs the end words of the kept pairs at random. Subtracting it
    gives the lift: rhyme the text has beyond what its word distribution yields by
    accident. `exclude` holds (line, line) pairs to skip, such as ones copied from
    training data. `seen_word_pairs` adds the share of perfect rhymes reusing a
    word pair that appears in training.
    """
    pairs = []
    for lines in stanzas(text):
        ends = [end_word(ln) for ln in lines]
        for i in range(len(lines) - distance):
            a, b = ends[i], ends[i + distance]
            if a and b and not (exclude and (lines[i], lines[i + distance]) in exclude):
                pairs.append((a, b))
    out = _rates(pairs)
    words = [w for pair in pairs for w in pair]
    rng = random.Random(seed)
    chance_perfect, chance_slant, chance_per_pair = [], [], []
    for _ in range(shuffles):
        rng.shuffle(words)
        rates = _rates(list(zip(words[::2], words[1::2])))
        chance_perfect.append(rates["perfect"])
        chance_slant.append(rates["slant"])
        chance_per_pair.append(rates["perfect_per_pair"])
    out["chance_perfect"] = sum(chance_perfect) / shuffles if shuffles else 0.0
    out["chance_slant"] = sum(chance_slant) / shuffles if shuffles else 0.0
    out["lift_perfect"] = out["perfect"] - out["chance_perfect"]
    out["lift_slant"] = out["slant"] - out["chance_slant"]
    out["lift_per_pair"] = out["perfect_per_pair"] - (sum(chance_per_pair) / shuffles if shuffles else 0.0)
    perfect = [(a, b) for a, b in pairs if (r := rhymes(a, b)) and r[0]]
    # Rhymes that share their last three letters can come from copying spelling alone.
    out["same_spelling_share"] = sum(a[-3:] == b[-3:] for a, b in perfect) / len(perfect) if perfect else 0.0
    if seen_word_pairs is not None:
        out["seen_pair_share"] = sum(frozenset((a, b)) in seen_word_pairs for a, b in perfect) / len(perfect) if perfect else 0.0
    return out


def line_pairs(text: str, distance: int = 1) -> set[tuple[str, str]]:
    pairs = set()
    for lines in stanzas(text):
        pairs.update(zip(lines, lines[distance:]))
    return pairs
