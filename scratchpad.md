File: scratchpad.md
Lines 1-75:
 1 | # Project Scratchpad
 2 | 
 3 | ## Progress Log
 4 | 
 5 | - 2026-01-14: Initial commit - Established Training Pipeline for BDH vs GPT-2
 6 | - 2026-01-14: Fixed model errors (unused L, gelu callable, RoPE shapes, matmul ops)
 7 | - 2026-01-14: Created dataset.py, train.py, test_train.py, config updates
 8 | - 2026-01-14: Created dummy.txt, verified training pipeline
 9 | - 2026-01-14: Initialized Git repo
10 | - 2026-01-14: Created .gitignore, bundle_kaggle.py, verified bundle
11 | - 2026-01-14: Refactored codebase into professional structure (config/, model/, utils/, metrics/, inference/)
12 | - 2026-01-14: Updated imports, created comparison script with perplexity and generation
13 | - 2026-01-14: Updated project_context.md directory structure
14 | - 2026-01-14: Downloaded TinyStories dataset (1.9GB train, 19MB val)
15 | - 2026-01-14: Tested real training on Kaggle: BDH PPL=27k, GPT2 PPL=22k after 3 epochs
16 | - 2026-01-14: Replaced WandB with timestamped CSV logging in results/ folder
17 | - 2026-01-14: Added automatic file copying to Kaggle working dir for download
18 | - 2026-01-17: Fixed backend slider logic to be continuous (blended probabilities) instead of discrete switch.
19 | - 2026-01-17: Updated Frontend Slider labels (Creativity=0%, Logic=100%) and output label (Logic Influence %).
20 | 
21 | ## Current Status
22 | 
23 | Backend `inference.py` now uses linear interpolation of probabilities between BDH and GPT-2 based on slider value. Frontend updated to match.
24 | 
25 | ## Plan (2026-01-17) — Fix backend validate & generation (COMPLETED)
26 | 
27 | 1) Refactor `inference.py` to use `_generate_blended`.
28 | 2) Update Frontend labels in `HemisphereSlider.jsx` and `SplitView.jsx`.
29 | 
30 | ## Plan (2026-01-17) — Kaggle notebook pretokenized path & training UX
31 | 
32 | 1) Edit [project-plotarmour.ipynb](project-plotarmour.ipynb) to remove the pretokenization cell that ran `tools/pre_tokenize.py`.
33 | 2) Update the training cell to call `!python research/utils/train.py --config research/config/kaggle_long_train.py --pretokenized_path /kaggle/input/tinystories-pretokenized/tinystories_train.bin`.
34 | 3) Ensure config still uses kaggle_long_train.py (batch size 16, grad accumulation 4). Keep progress logging (tqdm/progress bar) and add early stopping hook if absent.
35 | 4) Roughly estimate Kaggle P100 runtime for 900MB bin; propose utilization tweaks if GPU not saturated; ensure progress bar shows percent and loss.
36 | 
37 | ## Latest Work (2026-01-16)
38 | 
39 | - Updated tools/pre_tokenize.py to streaming, two-pass memmap writer with tqdm progress (line-chunked; safe for 16GB RAM; Kaggle runnable).
40 | - Updated research/utils/dataset.py to prefer loading pretokenized .bin via np.memmap (read-only); falls back to slow on-the-fly tokenization with warning; added optional pretokenized_path plumbed into get_dataloaders.
41 | - Added pretokenization flag to research/utils/train.py (args.pretokenized_path) to skip tokenization when cache exists.
42 | - Inserted Kaggle notebook cell in project-plotarmour.ipynb to run tools/pre_tokenize.py (writes /kaggle/working/research/data/tinystories_train.bin) before training.
43 | - OOM/tokenizer fixes: set batch_size=16 and gradient_accumulation_steps=4 in research/config/kaggle_long_train.py; added gradient accumulation + lr from config + torch.cuda.empty_cache() in research/utils/train.py; set truncation=True, max_length=1024 in tools/pre_tokenize.py.
44 | 
45 | ## Next Steps
46 | 
47 | - Local quick test on small sample to validate memmap read path and truncation logic.
48 | - If needed, add notebook cell in project-plotarmour.ipynb to run tools/pre_tokenize.py on Kaggle before training.
49 | 
50 | ## New Task Plan (2026-01-17) — Fix Vite 404 at app/client
51 | 
52 | 1) Inspect Vite entry points and dev server config: verify [app/client/src/index.html](app/client/src/index.html) references `/src/main.jsx` and `#root`, and check [app/client/src/main.jsx](app/client/src/main.jsx) mounts React to `root`.
53 | 2) Check Vite config and package scripts: ensure `npm run dev` uses Vite defaults (no custom base misconfiguration) and that dependencies are installed; review [app/client/package.json](app/client/package.json) for scripts.
54 | 3) Run `npm run dev` in app/client to confirm server binds to 5173 and that the console shows no missing entry errors; if 404 persists, inspect [app/client/tailwind.config.js](app/client/tailwind.config.js) and public paths for misaligned base.
55 | 4) Verify routing: ensure [app/client/src/App.jsx](app/client/src/App.jsx) renders content without relying on client-side routing that could 404; add minimal smoke test rendering.
56 | 5) After fixes, rerun dev server, open http://localhost:5173/, and confirm page renders; document changes and update directory map if any files change.
57 | 
58 | ## New Task Plan (2026-01-16) — Kaggle OOM + tokenizer warning
59 | 
60 | 1) config fix: set batch_size=16 and add gradient_accumulation_steps=4 in [research/config/kaggle_long_train.py](research/config/kaggle_long_train.py) to simulate effective 64 without OOM.
61 | 2) training fix: implement gradient accumulation and torch.cuda.empty_cache() at start of main() in [research/utils/train.py](research/utils/train.py); ensure optimizer.step/zero_grad only after accumulation.
62 | 3) tokenizer fix: in [tools/pre_tokenize.py](tools/pre_tokenize.py) set truncation=True, max_length=1024 (aligned with context_len) in tokenizer calls to silence warnings and cap length.
63 | 4) notebook check: verify pretokenize cell still present in [project-plotarmour.ipynb](project-plotarmour.ipynb); update if needed and flag if changed.
64 | 5) sanity: skim project_context.md to ensure goal alignment; note no structure change so directory map unchanged.
65 | 
66 | ## Plan (approved — 2026-01-16, user-modified for safety/visibility)
67 | 
68 | 1) Update [tools/pre_tokenize.py](tools/pre_tokenize.py) to stream-read the input text (no full file in RAM), tokenize per chunk with `GPT2TokenizerFast`, and write token IDs directly to a `.bin`/memmap-compatible file; wrap the loop in `tqdm` to show progress (bytes processed or lines processed). Include chunked reading (e.g., 10k lines) and avoid unbounded lists.
69 | 2) Update [research/utils/dataset.py](research/utils/dataset.py) so `TextDataset` loads a `.bin` pretokenized file via `np.memmap` in read-only mode when available; otherwise fall back to existing text tokenization with a clear warning. Keep training loop unchanged but faster when cache exists.
70 | 3) Ensure Kaggle compatibility: the script must be runnable as a separate step (e.g., `!python tools/pre_tokenize.py`) without breaking `train.py` usage.
71 | 4) Verify locally with a small test run to ensure RAM safety (16GB laptop) and that memmap loading works; report results here.
