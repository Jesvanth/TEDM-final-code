# Data
The datasets are not redistributed. Prepare an utterance-level CSV with:
`dialogue_id,turn_index,text,audio_path,video_path,label`.
Then run `features/extract_features.py`. Preserve the generated split ID files in `data/splits/` for reproducibility.
