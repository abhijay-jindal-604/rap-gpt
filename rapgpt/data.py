"""Load the lyrics, split them by song, and encode them as characters."""

import csv
import random
import re
from dataclasses import dataclass

import torch

csv.field_size_limit(10**8)

# Genius sprinkles in typographic variants that would only inflate the vocabulary.
REPLACEMENTS = {"’": "'", "‘": "'", "“": '"', "”": '"', "—": "-", "–": "-", "―": "-", "…": "..."}
INVISIBLE_RE = re.compile("[​⁠﻿]")
ODD_SPACE_RE = re.compile("[   ]")
# Genius also swaps in Cyrillic look-alikes: "comеdian" has a Cyrillic "е".
HOMOGLYPHS = str.maketrans("аеорсухіАЕОРСХ", "aeopcyxiAEOPCX")
RELEASE_LINE_RE = re.compile(r' - ".*"|\(\)$')
MAX_RELEASE_LINE_SHARE = 0.2
MAX_SONG_CHARS = 15000


def normalize(text: str) -> str:
    for src, dst in REPLACEMENTS.items():
        text = text.replace(src, dst)
    text = INVISIBLE_RE.sub("", text)
    return ODD_SPACE_RE.sub(" ", text).translate(HOMOGLYPHS)


def is_latin(line: str) -> bool:
    return all(ord(c) <= 0x24F for c in line)


def is_release_list(lyrics: str) -> bool:
    """Genius release calendars ('Artist - "Track"') sometimes get scraped as songs."""
    lines = lyrics.split("\n")
    matches = sum(bool(RELEASE_LINE_RE.search(ln)) for ln in lines)
    return len(lyrics) > MAX_SONG_CHARS or matches / len(lines) > MAX_RELEASE_LINE_SHARE


def load_songs(path: str, min_chars: int = 200) -> list[dict]:
    """Read a lyrics CSV, using `artist_verses` (the named artist only) when present."""
    songs = []
    with open(path, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            lyrics = normalize(row.get("artist_verses") or row.get("lyrics") or "")
            lyrics = "\n".join(ln.strip() for ln in lyrics.split("\n") if ln.strip() and is_latin(ln))
            if len(lyrics) >= min_chars and not is_release_list(lyrics):
                songs.append({"track_name": row["track_name"], "artist": row["artist"], "lyrics": lyrics})
    return songs


def split_songs(songs: list[dict], val_per_artist: int = 5, seed: int = 0):
    """Hold out whole songs per artist.

    A plain 90/10 cut of the concatenated corpus would leak repeated hooks across
    the boundary and leave only the last artist in validation.
    """
    rng = random.Random(seed)
    by_artist: dict[str, list[dict]] = {}
    for song in songs:
        by_artist.setdefault(song["artist"], []).append(song)
    train, val = [], []
    for artist in sorted(by_artist):
        shuffled = sorted(by_artist[artist], key=lambda s: s["track_name"])
        rng.shuffle(shuffled)
        val += shuffled[:val_per_artist]
        train += shuffled[val_per_artist:]
    return train, val


def join_songs(songs: list[dict]) -> str:
    return "".join(song["lyrics"] + "\n\n" for song in songs)


def reverse_lines(text: str) -> str:
    """Reverse the characters of each line and keep the line order, so a line starts with its rhyme."""
    return "\n".join(line[::-1] for line in text.split("\n"))


@dataclass
class Corpus:
    train_text: str
    val_text: str
    train: torch.Tensor
    val: torch.Tensor
    itos: list[str]
    reversed: bool = False
    artists: tuple[str, ...] = ()
    train_artist: torch.Tensor | None = None  # artist id of every character
    val_artist: torch.Tensor | None = None

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    def encode(self, text: str) -> list[int]:
        stoi = {ch: i for i, ch in enumerate(self.itos)}
        return [stoi[c] for c in text]

    def decode(self, ids) -> str:
        return "".join(self.itos[i] for i in ids)


def load_corpus(path: str, val_per_artist: int = 5, split_seed: int = 0, reversed_lines: bool = False) -> Corpus:
    """`train_text` and `val_text` stay forward for scoring; the tensors hold what the model sees."""
    train_songs, val_songs = split_songs(load_songs(path), val_per_artist, split_seed)
    train_text, val_text = join_songs(train_songs), join_songs(val_songs)
    itos = sorted(set(train_text) | set(val_text))
    stoi = {ch: i for i, ch in enumerate(itos)}
    to_tensor = lambda s: torch.tensor([stoi[c] for c in s], dtype=torch.long)
    view = reverse_lines if reversed_lines else (lambda s: s)
    artists = tuple(sorted({s["artist"] for s in train_songs}))
    ids = lambda songs: torch.cat(
        [torch.full((len(s["lyrics"]) + 2,), artists.index(s["artist"]), dtype=torch.long) for s in songs])
    return Corpus(train_text, val_text, to_tensor(view(train_text)), to_tensor(view(val_text)), itos,
                  reversed_lines, artists, ids(train_songs), ids(val_songs))
