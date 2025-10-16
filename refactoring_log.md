# Project Refactoring and Cleanup Log

## Stage 1: Initial Cleanup and Archiving
- Merged `NLP_basic` and `NLP_advance` into `archived_nlp_materials`.
- Archived `進階工具`.

## Stage 2: "Mckinsey-style" Restructuring
- Reorganized `nlp-course` into a `part-X/module-Y/chapter-Z` structure.
- Cleaned up legacy files from `nlp-course`.
- Refined chapter contents into `sub-chapters`.
- Normalized all notebook filenames to a systematic `C-S-U` convention.

## Stage 3: Image Path Correction
- Created central `nlp-course/assets/img` directory.
- **File:** `part-2_applications/chapter-7_language-modeling/C7-S1-U1_language-modeling-slides.ipynb`
    - Corrected path for `drumpf.png`.
    - Corrected path for `quiz_time.png`.
- **File:** `part-2_applications/chapter-10_sequence-to-sequence/C10-S1-U1_nmt-slides.ipynb`
    - Corrected path for `quiz_time.png`.