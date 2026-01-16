---
# Bugs and Fixes Registry

This file records major errors/bugs with symptoms, root causes, fixes, and prevention notes.

---

## Bug: Vite 404 on dev server
- **Symptoms:** Dev server ran but browser showed 404 on localhost:5173.
- **Root Cause:** Missing root-level index.html for Vite dev server.
- **Fix:** Added [app/client/index.html](app/client/index.html:1) as root entry point.
- **Prevention:** Ensure Vite root index.html exists for dev.

---

## Bug: Pre-tokenization step redundant
- **Symptoms:** Repeated tokenization slowed training startup.
- **Root Cause:** Dataset already pretokenized and published as Kaggle dataset.
- **Fix:** Removed pre-tokenization cell in [project-plotarmour.ipynb](project-plotarmour.ipynb:39); training now uses `/kaggle/input/tinystories-pretokenized/tinystories_train.bin`.
- **Prevention:** Prefer pretokenized datasets when available; avoid duplicate tokenization steps.

