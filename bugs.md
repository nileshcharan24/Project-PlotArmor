# Bug Tracking

## Resolved

### [Fixed] Continuous Slider Logic
- **Date:** 2026-01-17
- **Issue:** The "Story Generation" slider was acting as a discrete switch (0-40% BDH, 60-100% GPT-2, 41-59% Hybrid) instead of a continuous control.
- **Fix:** Refactored `inference.py` to use linear interpolation of probabilities between BDH and GPT-2 models based on slider value. Updated Frontend labels to correctly reflect "Creativity" (0%) vs "Logic" (100%).
