"""
Scrape rap lyrics from Genius, preserving line breaks (needed for the
context-length-vs-rhyme experiment). Personal/educational use only.

Usage:
    export GENIUS_ACCESS_TOKEN="..."
    python scrape_lyrics.py

Writes:
    data/lyrics_lines.csv   (track_name, artist, lyrics)  -- lyrics has real \n
    data/lyrics_corpus.txt  -- all lyrics concatenated, one song per blank-line block
"""

import os
import re
import csv
import time

import lyricsgenius

ARTISTS = [
    "Drake", "Nicki Minaj", "Dave", "Nas", "Rapsody", "Skepta",
    "Eminem", "2Pac", "Kendrick Lamar", "Future", "J. Cole",
]
SONGS_PER_ARTIST = 50
OUT_DIR = os.path.join(os.path.dirname(__file__), "data")

SECTION_TAG_RE = re.compile(r"\[.*?\]")          # [Verse 1], [Chorus], etc.
EMBED_JUNK_RE = re.compile(r"\d*Embed$", re.MULTILINE)
LEADING_HEADER_RE = re.compile(r"^.*?Lyrics\n", re.DOTALL)


def clean_lyrics(raw: str) -> str:
    text = LEADING_HEADER_RE.sub("", raw, count=1)   # strip "Song Title Lyrics" header
    text = SECTION_TAG_RE.sub("", text)              # strip [Verse]/[Chorus] tags
    text = EMBED_JUNK_RE.sub("", text)                # strip trailing "123Embed"
    lines = [ln.strip() for ln in text.split("\n")]
    lines = [ln for ln in lines if ln]                # drop blank lines
    return "\n".join(lines)


def main():
    token = os.environ.get("GENIUS_ACCESS_TOKEN")
    if not token:
        raise SystemExit("Set GENIUS_ACCESS_TOKEN before running.")

    genius = lyricsgenius.Genius(
        token,
        skip_non_songs=True,
        excluded_terms=["(Remix)", "(Live)"],
        remove_section_headers=False,  # we strip them ourselves in clean_lyrics
        timeout=15,
        retries=3,
    )
    genius.verbose = True

    os.makedirs(OUT_DIR, exist_ok=True)
    csv_path = os.path.join(OUT_DIR, "lyrics_lines.csv")
    txt_path = os.path.join(OUT_DIR, "lyrics_corpus.txt")

    csv_exists = os.path.exists(csv_path) and os.path.getsize(csv_path) > 0
    csv_f = open(csv_path, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(csv_f, fieldnames=["track_name", "artist", "lyrics"])
    if not csv_exists:
        writer.writeheader()
    txt_f = open(txt_path, "a", encoding="utf-8")

    total_songs = 0
    total_chars = 0

    for artist_name in ARTISTS:
        print(f"\n=== {artist_name} ===", flush=True)
        try:
            artist = genius.search_artist(
                artist_name, max_songs=SONGS_PER_ARTIST, sort="popularity"
            )
        except Exception as e:
            print(f"  failed to fetch {artist_name}: {e}", flush=True)
            continue

        if artist is None:
            continue

        for song in artist.songs:
            lyrics = clean_lyrics(song.lyrics)
            if len(lyrics) < 200:
                continue
            writer.writerow({"track_name": song.title, "artist": artist_name, "lyrics": lyrics})
            txt_f.write(lyrics)
            txt_f.write("\n\n")
            csv_f.flush()
            txt_f.flush()
            total_songs += 1
            total_chars += len(lyrics)

        print(f"  running total: {total_songs} songs, {total_chars} chars", flush=True)

    csv_f.close()
    txt_f.close()
    print(f"\nSaved {total_songs} songs, {total_chars} chars -> {csv_path}, {txt_path}")


if __name__ == "__main__":
    main()
