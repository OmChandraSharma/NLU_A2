# NLU Assignment 2 — IIT Jodhpur

> **Word Embeddings from IIT Jodhpur Data** · **Character-Level Name Generation with RNN Variants**

---

## Repository Structure

```
.
├── README.md
│
├── problem1/                       # Word Embeddings (Word2Vec)
│   ├── t1.py                       # Main script — all four tasks
│   ├── corpus.txt                  # Cleaned IIT Jodhpur corpus
│   ├── documents.json              # Parsed document metadata
│   ├── stats_report.txt            # Dataset statistics
│   ├── vocab.pkl                   # Vocabulary pickle
│   ├── cbow_embeddings.npy         # Scratch CBOW vectors (numpy)
│   ├── sg_embeddings.npy           # Scratch Skip-gram vectors (numpy)
│   ├── gensim_cbow.model           # Trained Gensim CBOW model
│   ├── gensim_sg.model             # Trained Gensim Skip-gram model
│   ├── plot_neighbors.png          # Top-5 nearest neighbour bar charts
│   ├── plot_analogies.png          # Analogy experiment result table
│   ├── plot_pca.png                # PCA 2-D embedding projection
│   ├── plot_tsne.png               # t-SNE 2-D embedding projection
│   └── run_log.txt                 # Full timestamped execution log
│
├── problem2/                       # Character-level Name Generation
│   ├── t2.py                       # Main script — all tasks
│   ├── TrainingNames.txt           # 1000 Indian names (one per line)
│   ├── plot_loss_curves.png        # Training loss per epoch (all models)
│   └── plot_comparison.png         # Novelty / diversity / param comparison
│
└── report/
    ├── NLU_Assignment2_Report.pdf  # Compiled report
    └── NLU_Assignment2_Report.tex  # LaTeX source
```

---

## Problem 1 — Word Embeddings from IIT Jodhpur Data

### Overview

Trains and evaluates four Word2Vec variants on a domain-specific corpus scraped from
IIT Jodhpur sources (website pages, academic regulations, faculty profiles, syllabi).

| Model | Backend | Embed Dim | Window | Neg. Samples |
|---|---|---|---|---|
| CBOW | Scratch (NumPy/PyTorch) | 100 | 5 | — |
| Skip-gram | Scratch | 100 | 5 | 10 |
| CBOW | Gensim | 100 | 5 | 5 |
| Skip-gram | Gensim | 100 | 5 | 5 |

### Dataset Statistics

| Stat | Value |
|---|---|
| Total documents | 321 (125 HTML + 196 PDF) |
| Total sentences | 72,054 |
| Total tokens | 1,102,073 |
| Vocabulary size | 50,771 |

### Setup

```bash
pip install gensim numpy scipy scikit-learn matplotlib wordcloud requests beautifulsoup4
```

### Running

```bash
cd problem1
python t1.py
```

The script is idempotent — if `corpus.txt` or trained models already exist on disk,
it skips those steps and loads from cache. To force a full retrain, delete the relevant
files before running.

**Outputs produced:**

| File | Description |
|---|---|
| `corpus.txt` | Cleaned, lowercased corpus (one sentence per line) |
| `stats_report.txt` | Token/vocab counts and top-50 word frequency table |
| `cbow_embeddings.npy` / `sg_embeddings.npy` | Scratch model weight matrices |
| `gensim_cbow.model` / `gensim_sg.model` | Serialised Gensim models |
| `plot_neighbors.png` | Cosine-similarity bar charts for 5 probe words |
| `plot_analogies.png` | Colour-coded analogy result table |
| `plot_pca.png` | PCA projection of selected embeddings |
| `plot_tsne.png` | t-SNE projection of selected embeddings |
| `run_log.txt` | Timestamped log of all task outputs |

### Task Summary

**Task 1 — Dataset Preparation**
Scrapes and preprocesses text from IIT Jodhpur sources. Applies tokenisation,
lowercasing, and removal of boilerplate/non-English content.

**Task 2 — Model Training**
Trains all four Word2Vec variants. Scratch implementations use negative sampling
(Skip-gram) and softmax (CBOW). Gensim models use optimised C extensions.

**Task 3 — Semantic Analysis**
Computes top-5 nearest neighbours by cosine similarity for: `research`, `student`,
`phd`, `exam`, `laboratory`. Runs six analogy experiments (e.g., `ug:btech::pg:?`).

**Analogy Accuracy Summary:**

| Model | Correct / 6 | Accuracy |
|---|---|---|
| CBOW (Scratch) | 0 | 0% |
| Skip-gram (Scratch) | 1 | 17% |
| Gensim CBOW | 2 | 33% |
| Gensim Skip-gram | 0 | 0% |

**Task 4 — Visualisation**
Projects embeddings to 2-D using PCA and t-SNE. Words are colour/shape-coded by
semantic category: Academic Roles, Programs, Departments, Activities.

---

## Problem 2 — Character-Level Name Generation

### Overview

Implements and compares three character-level sequence models for generating Indian names,
trained on a dataset of 1,000 unique names.

### Models

#### 1. Vanilla RNN
```
Embedding(27→64) → RNN(64→128, 2 layers, dropout=0.2) → Linear(128→27)
Trainable parameters: 63,067
```
A two-layer causal RNN. Chosen for its direct alignment with the auto-regressive
generation task. **Best-performing model.**

#### 2. Bidirectional LSTM (BLSTM)
```
Embedding(27→64) → BiLSTM(64→128×2) → Linear(256→27)
Trainable parameters: 207,323
```
Processes sequences in both directions. **Fails at inference** because the backward pass
requires future tokens unavailable during generation, leading to degenerate repetitive
output (e.g., `fafafafafafafaf`).

#### 3. Attention RNN (GRU + Bahdanau Attention)
```
Embedding(27→64) → GRU(64→128) → Additive Attention → Linear(256→27)
Trainable parameters: 99,804
```
GRU encoder with soft attention over previous outputs. **Partially fails** due to a
training-inference mismatch in attention scope and overfitting to rare character bigrams
(e.g., `fzfzzfz` loops).

### Setup

```bash
pip install torch numpy matplotlib
```

GPU is used automatically if available (`cuda`). CPU fallback is supported.

### Dataset

Generate or provide `TrainingNames.txt` in the working directory — one name per line,
alphabetic only. Example:

```
Arjun
Priya
Kavitha
Rohan
...
```

### Running

```bash
cd problem2
python t2.py
```

Training runs for 60 epochs per model. Progress is printed every 10 epochs with
5 sample generated names.

**Outputs produced:**

| File | Description |
|---|---|
| `plot_loss_curves.png` | Average cross-entropy loss per epoch for all three models |
| `plot_comparison.png` | Bar charts: novelty rate, diversity, parameter count |

### Results

| Model | Generated | Unique | Novelty | Diversity | Avg Length |
|---|---|---|---|---|---|
| Vanilla RNN | 500 | 406 | 90.2% | **81.2%** | 7.3 chars |
| BLSTM | 500 | 18 | 100.0% | 3.6% | 15.0 chars |
| Attention RNN | 500 | 109 | 100.0% | 21.8% | 12.6 chars |

### Why Models Succeed or Fail

| Model | Status | Root Cause |
|---|---|---|
| Vanilla RNN | ✅ Works | Causal architecture matches auto-regressive generation; mild dropout prevents overfitting |
| BLSTM | ❌ Fails | Bidirectional design requires future tokens unavailable at inference; 207K params overfit 1K names (loss → 0.000 by epoch 10) |
| Attention RNN | ❌ Fails | Attention spans full sequence during training but only partial sequence at inference; overfits rare bigrams (`fz`) |

### Hyperparameters

| Parameter | Value |
|---|---|
| Embedding dimension | 64 |
| Hidden size | 128 |
| Learning rate | 0.002 |
| Epochs | 60 |
| Gradient clip | 2.0 |
| Loss | Cross-entropy |
| Optimiser | Adam |
| Temperature (sampling) | 0.6 (RNN) / 0.7 (BLSTM, Attention) / 1.0 (evaluation) |
| Top-k sampling | k = 5 |

---

## Requirements

```
torch>=2.0
numpy
matplotlib
gensim>=4.0
scikit-learn
scipy
requests
beautifulsoup4
wordcloud
```

Install all at once:

```bash
pip install torch numpy matplotlib gensim scikit-learn scipy requests beautifulsoup4 wordcloud
```

---

## Key Takeaways

1. **Architectural suitability matters more than model size.** The smallest model (Vanilla
   RNN, 63K params) outperforms the largest (BLSTM, 207K params) because its causal design
   matches the generation task.

2. **Gensim CBOW produces the most semantically coherent embeddings** on a small
   domain-specific corpus, benefiting from sub-sampling and optimised training routines.

3. **Analogy arithmetic is hard on domain corpora.** Low accuracy (0–33%) is expected when
   vocabulary is small and concepts appear in narrow, formulaic contexts.

4. **Bidirectional models should not be used for auto-regressive generation** without
   masking future positions, as the inference-time mismatch corrupts the hidden state.

---

*NLU Assignment 2 — IIT Jodhpur*
