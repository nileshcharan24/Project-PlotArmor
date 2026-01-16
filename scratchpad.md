# Project Scratchpad

## Progress Log

- 2026-01-14: Initial commit - Established Training Pipeline for BDH vs GPT-2
- 2026-01-14: Fixed model errors (unused L, gelu callable, RoPE shapes, matmul ops)
- 2026-01-14: Created dataset.py, train.py, test_train.py, config updates
- 2026-01-14: Created dummy.txt, verified training pipeline
- 2026-01-14: Initialized Git repo
- 2026-01-14: Created .gitignore, bundle_kaggle.py, verified bundle
- 2026-01-14: Refactored codebase into professional structure (config/, model/, utils/, metrics/, inference/)
- 2026-01-14: Updated imports, created comparison script with perplexity and generation
- 2026-01-14: Updated project_context.md directory structure
- 2026-01-14: Downloaded TinyStories dataset (1.9GB train, 19MB val)
- 2026-01-14: Tested real training on Kaggle: BDH PPL=27k, GPT2 PPL=22k after 3 epochs
- 2026-01-14: Replaced WandB with timestamped CSV logging in results/ folder
- 2026-01-14: Added automatic file copying to Kaggle working dir for download

## Current Status

Modified dataset.py to tokenize text in 10MB chunks to prevent memory issues during encoding. Changes pushed for user to test on Kaggle.

## Latest Work (2026-01-16)

- Updated tools/pre_tokenize.py to streaming, two-pass memmap writer with tqdm progress (line-chunked; safe for 16GB RAM; Kaggle runnable).
- Updated research/utils/dataset.py to prefer loading pretokenized .bin via np.memmap (read-only); falls back to slow on-the-fly tokenization with warning; added optional pretokenized_path plumbed into get_dataloaders.
- Added pretokenization flag to research/utils/train.py (args.pretokenized_path) to skip tokenization when cache exists.
- Inserted Kaggle notebook cell in project-plotarmour.ipynb to run tools/pre_tokenize.py (writes /kaggle/working/research/data/tinystories_train.bin) before training.

## Next Steps

- Local quick test on small sample to validate memmap read path and truncation logic.
- If needed, add notebook cell in project-plotarmour.ipynb to run tools/pre_tokenize.py on Kaggle before training.

## Plan (approved — 2026-01-16, user-modified for safety/visibility)

1) Update [tools/pre_tokenize.py](tools/pre_tokenize.py) to stream-read the input text (no full file in RAM), tokenize per chunk with `GPT2TokenizerFast`, and write token IDs directly to a `.bin`/memmap-compatible file; wrap the loop in `tqdm` to show progress (bytes processed or lines processed). Include chunked reading (e.g., 10k lines) and avoid unbounded lists.
2) Update [research/utils/dataset.py](research/utils/dataset.py) so `TextDataset` loads a `.bin` pretokenized file via `np.memmap` in read-only mode when available; otherwise fall back to existing text tokenization with a clear warning. Keep training loop unchanged but faster when cache exists.
3) Ensure Kaggle compatibility: the script must be runnable as a separate step (e.g., `!python tools/pre_tokenize.py`) without breaking `train.py` usage.
4) Verify locally with a small test run to ensure RAM safety (16GB laptop) and that memmap loading works; report results here.
