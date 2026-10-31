# rap-gpt

How much context does a character-level transformer need before it can rhyme, and does
generating each line backwards, rhyme word first, help?

Work in progress: results and a full write-up are coming once the experiment grid finishes.

The model is the GPT from Andrej Karpathy's [Let's build GPT](https://github.com/karpathy/ng-video-lecture)
lecture, lightly restructured. This project's own work is the experiment around it: data
cleaning, the song-level split, line reversal, the rhyme metric and its controls.

Lyrics are copyrighted and are not in this repository. `scrape_lyrics.py` collects a similar
corpus through the Genius API.
