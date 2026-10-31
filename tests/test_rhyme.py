from rapgpt.data import is_release_list, normalize, reverse_lines, split_songs
from rapgpt.rhyme import end_word, line_pairs, rhymes, score_text


def test_end_word_ignores_adlibs_and_restores_dropped_g():
    assert end_word("we walked along the river road (hey, yeah)") == "road"
    assert end_word("my friends are singin'") == "singing"
    assert end_word("(yeah)") is None


def test_perfect_and_slant():
    assert rhymes("plan", "man") == (True, True)
    assert rhymes("time", "line") == (False, True)
    assert rhymes("cat", "moon") == (False, False)


def test_repeats_and_unknown_words_are_not_scored():
    assert rhymes("plan", "plan") is None
    assert rhymes("plan", "zzxqj") is None


def test_unstressed_function_words_do_not_rhyme():
    assert rhymes("the", "a")[0] is False


def test_score_text_counts_within_stanzas_only():
    result = score_text("my plan\nyour man\n\nthe cat\nthe moon", shuffles=0)
    assert result["pairs"] == 2
    assert result["perfect"] == 0.5


def test_per_pair_rate_counts_unknown_words_as_misses():
    result = score_text("my plan\nyour man\nzzxqj\nthe cat", shuffles=0)
    assert result["perfect"] == 1.0
    assert result["perfect_per_pair"] == 1 / 3


def test_score_text_can_exclude_copied_pairs():
    text = "my plan\nyour man\nthe cat\nthe moon"
    copied = line_pairs("my plan\nyour man")
    assert score_text(text, exclude=copied, shuffles=0)["pairs"] == 2


def test_split_holds_out_whole_songs_per_artist():
    songs = [{"track_name": f"{a}{i}", "artist": a, "lyrics": "x"} for a in "ab" for i in range(10)]
    train, val = split_songs(songs, val_per_artist=2)
    assert len(val) == 4 and len(train) == 16
    assert not {s["track_name"] for s in train} & {s["track_name"] for s in val}
    assert sum(s["artist"] == "a" for s in val) == 2


def test_normalize_maps_typography():
    assert normalize("God’s​ plan—") == "God's plan-"


def test_reverse_lines_keeps_line_order_and_round_trips():
    text = "my plan\nyour man\n\nnext"
    assert reverse_lines(text) == "nalp ym\nnam ruoy\n\ntxen"
    assert reverse_lines(reverse_lines(text)) == text


def test_release_calendars_are_not_songs():
    assert is_release_list('Klem - "Mort vivant"\nLIM - "Gamberge"\nAfro S - "Mode demo"')
    assert is_release_list("Blanca - Shattered ()\nJess Kent - Girl ()")
    assert is_release_list("a line\n" * 3000)
    assert not is_release_list('She texts me "How is your life?"\nI said - "I am good"\n' + "a line\n" * 20)


def test_normalize_replaces_cyrillic_lookalikes():
    assert normalize("com\u0435dian") == "comedian"
