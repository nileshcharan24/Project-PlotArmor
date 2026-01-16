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

Modified dataset.py to tokenize text in 10MB chunks to prevent memory issues during encoding. Changes pushed for user to test on Kaggle. Fixed Vite 404 by adding root-level index.html in app/client. Dev server now runs on port 5174 due to port conflict. Verified HTTP 200 response from dev server root.
Updated project-plotarmour.ipynb to remove pretokenization cell and point training to Kaggle dataset /kaggle/input/tinystories-pretokenized/tinystories_train.bin using kaggle_long_train.py.

## Plan (2026-01-17) — Kaggle notebook pretokenized path & training UX

1) Edit [project-plotarmour.ipynb](project-plotarmour.ipynb) to remove the pretokenization cell that ran `tools/pre_tokenize.py`.
2) Update the training cell to call `!python research/utils/train.py --config research/config/kaggle_long_train.py --pretokenized_path /kaggle/input/tinystories-pretokenized/tinystories_train.bin`.
3) Ensure config still uses kaggle_long_train.py (batch size 16, grad accumulation 4). Keep progress logging (tqdm/progress bar) and add early stopping hook if absent.
4) Roughly estimate Kaggle P100 runtime for 900MB bin; propose utilization tweaks if GPU not saturated; ensure progress bar shows percent and loss.

## Latest Work (2026-01-16)

- Updated tools/pre_tokenize.py to streaming, two-pass memmap writer with tqdm progress (line-chunked; safe for 16GB RAM; Kaggle runnable).
- Updated research/utils/dataset.py to prefer loading pretokenized .bin via np.memmap (read-only); falls back to slow on-the-fly tokenization with warning; added optional pretokenized_path plumbed into get_dataloaders.
- Added pretokenization flag to research/utils/train.py (args.pretokenized_path) to skip tokenization when cache exists.
- Inserted Kaggle notebook cell in project-plotarmour.ipynb to run tools/pre_tokenize.py (writes /kaggle/working/research/data/tinystories_train.bin) before training.
- OOM/tokenizer fixes: set batch_size=16 and gradient_accumulation_steps=4 in research/config/kaggle_long_train.py; added gradient accumulation + lr from config + torch.cuda.empty_cache() in research/utils/train.py; set truncation=True, max_length=1024 in tools/pre_tokenize.py.

## Next Steps

- Local quick test on small sample to validate memmap read path and truncation logic.
- If needed, add notebook cell in project-plotarmour.ipynb to run tools/pre_tokenize.py on Kaggle before training.

## New Task Plan (2026-01-17) — Fix Vite 404 at app/client

1) Inspect Vite entry points and dev server config: verify [app/client/src/index.html](app/client/src/index.html) references `/src/main.jsx` and `#root`, and check [app/client/src/main.jsx](app/client/src/main.jsx) mounts React to `root`.
2) Check Vite config and package scripts: ensure `npm run dev` uses Vite defaults (no custom base misconfiguration) and that dependencies are installed; review [app/client/package.json](app/client/package.json) for scripts.
3) Run `npm run dev` in app/client to confirm server binds to 5173 and that the console shows no missing entry errors; if 404 persists, inspect [app/client/tailwind.config.js](app/client/tailwind.config.js) and public paths for misaligned base.
4) Verify routing: ensure [app/client/src/App.jsx](app/client/src/App.jsx) renders content without relying on client-side routing that could 404; add minimal smoke test rendering.
5) After fixes, rerun dev server, open http://localhost:5173/, and confirm page renders; document changes and update directory map if any files change.

## New Task Plan (2026-01-16) — Kaggle OOM + tokenizer warning

1) config fix: set batch_size=16 and add gradient_accumulation_steps=4 in [research/config/kaggle_long_train.py](research/config/kaggle_long_train.py) to simulate effective 64 without OOM.
2) training fix: implement gradient accumulation and torch.cuda.empty_cache() at start of main() in [research/utils/train.py](research/utils/train.py); ensure optimizer.step/zero_grad only after accumulation.
3) tokenizer fix: in [tools/pre_tokenize.py](tools/pre_tokenize.py) set truncation=True, max_length=1024 (aligned with context_len) in tokenizer calls to silence warnings and cap length.
4) notebook check: verify pretokenize cell still present in [project-plotarmour.ipynb](project-plotarmour.ipynb); update if needed and flag if changed.
5) sanity: skim project_context.md to ensure goal alignment; note no structure change so directory map unchanged.

## Plan (approved — 2026-01-16, user-modified for safety/visibility)

1) Update [tools/pre_tokenize.py](tools/pre_tokenize.py) to stream-read the input text (no full file in RAM), tokenize per chunk with `GPT2TokenizerFast`, and write token IDs directly to a `.bin`/memmap-compatible file; wrap the loop in `tqdm` to show progress (bytes processed or lines processed). Include chunked reading (e.g., 10k lines) and avoid unbounded lists.
2) Update [research/utils/dataset.py](research/utils/dataset.py) so `TextDataset` loads a `.bin` pretokenized file via `np.memmap` in read-only mode when available; otherwise fall back to existing text tokenization with a clear warning. Keep training loop unchanged but faster when cache exists.
3) Ensure Kaggle compatibility: the script must be runnable as a separate step (e.g., `!python tools/pre_tokenize.py`) without breaking `train.py` usage.
4) Verify locally with a small test run to ensure RAM safety (16GB laptop) and that memmap loading works; report results here.
