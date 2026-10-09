# manga2anime_predictor

Predict whether a manga/manhwa gets an anime adaptation, using models written
from scratch in NumPy (logistic regression, then a one-hidden-layer MLP).

## Data

No public MyAnimeList dataset (Kaggle or Hugging Face) includes manga→anime
adaptation links, so `scripts/fetch_jikan.py` pulls them from the
[Jikan API](https://docs.api.jikan.moe/):

```bash
pip install -r requirements.txt
python scripts/fetch_jikan.py --n 3000 --anime-dates   # ~1.5 h, resumable
```

- Top manga by MAL score, filtered to Manga/Manhwa/Manhua.
- Label source: each title's `/manga/{id}/relations`. A relation `"Adaptation"`
  whose entry has `type == "anime"` means the manga was adapted.
- `--anime-dates` also fetches the air date of each linked anime (to help
  choose the cutoff year; never use it as a feature).
- Every response is cached in `data/cache/`. Ctrl-C and re-run anytime.
- Output: `data/raw/manga_raw.jsonl`, one Jikan manga object per line plus
  `relations` (and `adaptation_anime` with `--anime-dates`).

## Roadmap

1. Dataset: clean features, genre 0/1 columns, label, train/test split,
   normalization fit on train only.
2. Logistic regression from scratch + confusion matrix, precision/recall/F1.
3. One-hidden-layer ReLU MLP with hand-derived backprop; compare to step 2.
