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

Local training pipeline fully functional with CSV logging and progress prints. Kaggle setup detects GPU but execution hangs - may need further debugging. Ready to resume tomorrow.