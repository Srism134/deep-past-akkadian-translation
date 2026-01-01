# Deep Past Challenge — Akkadian → English Machine Translation (Baseline)

A Kaggle notebook solution for the **Deep Past Challenge: Translate Akkadian to English**, focused on building a reproducible **neural machine translation (NMT)** baseline for **Old Assyrian (Akkadian) transliterations → English** using **SentencePiece** tokenization and a **Transformer** model.

## Project Overview
Old Assyrian texts are **morphologically dense** and contain **scribal artifacts** (determinatives, subscripts, logograms, line marks, etc.). The target English translations can be **noisy** (OCR-derived) and proper nouns are inconsistent.

This baseline aims to:
- Normalize transliteration/translation text for stable training
- Learn a joint subword vocabulary with SentencePiece
- Train a compact Transformer sequence-to-sequence model
- Decode with beam search + length penalty
- Generate a valid `submission.csv` for Kaggle notebook submission

## Competition
- **Name:** Deep Past Challenge — Translate Akkadian to English  
- **Task:** Transliteration (Old Assyrian/Akkadian) → English translation  
- **Metric:** `sqrt(BLEU * chrF++)` (micro-averaged corpus-level)

## Repository Contents
- `akkadian_mt_sentencepiece_transformer_smriti.ipynb` — end-to-end Kaggle notebook:
  - Data loading
  - Cleaning & normalization
  - SentencePiece training + loading
  - Encoding + DataLoaders
  - Transformer training
  - Beam decoding + quick validation evaluation
  - `submission.csv` generation

## Data
The notebook expects Kaggle competition data mounted at:

`/kaggle/input/deep-past-initiative-machine-translation/`

Key files used:
- `train.csv`
- `test.csv`
- `sample_submission.csv`
- (optional) `OA_Lexicon_eBL.csv` for future post-processing experiments

## Method
### 1) Text Cleaning
- Transliteration:
  - Normalize Unicode subscripts (₀–₉ → 0–9)
  - Canonicalize common markers (e.g., `(d)` → `D_`)
  - Collapse extra whitespace
- English translation:
  - Normalize quotes/apostrophes
  - Collapse whitespace

### 2) Tokenization (SentencePiece)
- Train a **joint** SentencePiece Unigram model over concatenated:
  - cleaned transliteration + cleaned English
- Default vocab size: **5000** (kept small to fit data scale)

### 3) Model (Transformer Seq2Seq)
- Compact Transformer using PyTorch `nn.Transformer` with:
  - Embeddings + positional encoding
  - Encoder/decoder layers
  - Output linear projection to vocab

### 4) Training
- Teacher forcing with shifted target inputs
- AdamW optimizer
- Label smoothing loss
- Gradient clipping

### 5) Decoding
- Beam search decoding
- Length penalty to control verbosity
- Generates one-sentence translations for each test sample

## How to Run (Kaggle)
1. Open the notebook in Kaggle Code
2. Attach competition dataset: **Deep Past Initiative — Machine Translation**
3. Ensure:
   - Internet access **disabled**
   - Runtime within Kaggle limits
4. Click **Save & Run All**
5. Confirm `submission.csv` is created
6. Use **Submit** button in notebook after commit

## Results
This repository provides a **working baseline** that:
- Trains successfully
- Produces valid submissions
- Provides a foundation for leaderboard improvements

> Note: Scores vary by training time, model size, and decoding parameters.

## Next Improvements (Roadmap)
- Longer training with checkpointing (best val loss)
- Hyperparameter sweeps: `beam`, `length_penalty`, `max_len`
- Better normalization of determinatives, gaps, and logograms
- Proper noun normalization using lexicon + constrained decoding
- Back-translation / synthetic augmentation (if allowed)
- Fine-tune a pretrained seq2seq model (e.g., T5/BART variants) offline

## Author
**Smriti**  
Kaggle: `smritismrit`  
GitHub: `Srism134`

## License
For competition code and notebook content: add a license if you plan to reuse publicly.  
Competition data belongs to the competition organizers and Kaggle rules apply.
