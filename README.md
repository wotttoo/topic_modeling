# Topic Modeling — arXiv Abstract Classification

A machine learning project that classifies scientific paper abstracts from arXiv into their respective research domains. The project benchmarks **3 text encoding strategies** against **4 classification models** to identify which combination yields the best performance on multi-class topic classification.

---

## Problem Statement

Given the abstract text of an arXiv paper, predict which scientific domain it belongs to — `astro-ph`, `cond-mat`, `cs`, `math`, or `physics`.

This is a **5-class text classification** task that evaluates both traditional sparse representations (BoW, TF-IDF) and modern dense semantic embeddings.

---

## Dataset

| Property | Value |
|---|---|
| Source | [`UniverseTBD/arxiv-abstracts-large`](https://huggingface.co/datasets/UniverseTBD/arxiv-abstracts-large) (HuggingFace) |
| Total samples used | 1,000 |
| Categories | `astro-ph`, `cond-mat`, `cs`, `math`, `physics` |
| Split | 80% train / 20% test (stratified) |
| Filter rule | Single-category papers only |

---

## Project Structure

```
topic_modeling/
├── main.py                  # End-to-end pipeline entry point
├── requirements.txt
├── src/
│   ├── dataset.py           # Data loading, filtering, label mapping, splitting
│   ├── preprocessing.py     # Text cleaning pipeline
│   ├── encoders.py          # BoW, TF-IDF, and Sentence Embedding vectorizers
│   ├── models.py            # KMeans, KNN, Decision Tree, Naive Bayes training & evaluation
│   └── visualization.py     # Confusion matrix plotting
├── notebooks/
│   ├── explorarion.ipynb    # EDA — label distribution, abstract length analysis
│   ├── experiments.ipynb    # Full experiment runs with result output
│   └── [Code]-Project-3.1-Topic-Modeling.ipynb  # Project assignment notebook
└── outputs/
    └── figures/             # Saved confusion matrix plots (12 total)
```

---

## Pipeline

```
Raw Dataset (HuggingFace)
        │
        ▼
  Filter & Sample
  (1000 single-category papers from 5 domains)
        │
        ▼
  Text Preprocessing
  (strip, remove special chars/digits, lowercase)
        │
        ▼
  Feature Encoding  ──────────────────────────────┐
  ├── Bag of Words (CountVectorizer)               │
  ├── TF-IDF (TfidfVectorizer)                     │
  └── Sentence Embeddings (multilingual-e5-base)   │
                                                   │
        ▼                                          │
  Classification Models  ◄───────────────────────-┘
  ├── KMeans (unsupervised)
  ├── K-Nearest Neighbors (k=5)
  ├── Decision Tree
  └── Naive Bayes (Gaussian)
        │
        ▼
  Evaluation & Visualization
  (Accuracy, Classification Report, Confusion Matrices)
```

---

## Text Preprocessing

Each abstract goes through the following cleaning steps (`src/preprocessing.py`):

1. Strip leading/trailing whitespace and replace `\n` with spaces
2. Remove special characters (`[^\w\s]`)
3. Remove all digits
4. Collapse multiple spaces
5. Convert to lowercase

---

## Encoders

| Encoder | Type | Dimensionality | Library |
|---|---|---|---|
| **Bag of Words (BoW)** | Sparse, frequency-based | Vocab size (~10k+) | `scikit-learn` |
| **TF-IDF** | Sparse, frequency-weighted | Vocab size (~10k+) | `scikit-learn` |
| **Sentence Embedding** | Dense, semantic | 768 | `sentence-transformers` |

The embedding model used is [`intfloat/multilingual-e5-base`](https://huggingface.co/intfloat/multilingual-e5-base), which supports query/passage formatting for asymmetric retrieval tasks.

---

## Models

| Model | Type | Key Hyperparameters |
|---|---|---|
| **KMeans** | Unsupervised clustering | `n_clusters=5`, `random_state=42` |
| **KNN** | Supervised, instance-based | `n_neighbors=5` |
| **Decision Tree** | Supervised, tree-based | `random_state=42` |
| **Naive Bayes** | Supervised, probabilistic | `GaussianNB` |

> **KMeans note:** After clustering, each cluster is assigned the majority label from training samples. Prediction is then done by mapping test samples to their nearest cluster centroid.

---

## Results

Accuracy on the 20% held-out test set (200 samples):

| Model | BoW | TF-IDF | Embedding |
|---|---|---|---|
| KMeans | 56.00% | 61.50% | 84.00% |
| KNN | 53.00% | 81.50% | **89.00%** |
| Decision Tree | 63.50% | 59.50% | 71.50% |
| Naive Bayes | **85.00%** | 83.00% | **89.00%** |

### Key Observations

- **Sentence Embeddings consistently outperform** sparse methods (BoW, TF-IDF) across all models — confirming that semantic representations capture topic-relevant features more effectively than raw word frequencies.
- **KNN + Embedding** and **Naive Bayes + Embedding** both achieve the highest accuracy of **89.00%**.
- **Naive Bayes + BoW** achieves 85.00% — the best result for sparse encoders, showing that probabilistic models handle high-dimensional sparse data well.
- **KMeans** benefits the most from embedding features (56% → 84%), reflecting how dense representations create geometrically meaningful cluster structures.
- **Decision Tree** lags behind other models regardless of encoder, likely due to overfitting on high-dimensional sparse features.

### Confusion Matrix Samples

Confusion matrices for all 12 model-encoder combinations are saved in `outputs/figures/`. Filenames follow the pattern `{model}_{encoder}.png`.

---

## Installation

```bash
# Clone the repository
git clone <repo-url>
cd topic_modeling

# Create and activate virtual environment
python -m venv topic_modeling_venv
source topic_modeling_venv/bin/activate   # Windows: topic_modeling_venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Requirements

```
datasets
sentence-transformers
scikit-learn
numpy
matplotlib
seaborn
```

---

## Usage

### Run the full pipeline

```bash
python main.py
```

This will:
1. Download and cache the arXiv dataset from HuggingFace
2. Filter and preprocess 1,000 samples
3. Build BoW, TF-IDF, and Embedding feature matrices
4. Train and evaluate all 4 models × 3 encoders = **12 experiments**
5. Print accuracy scores to console
6. Save confusion matrix figures to `outputs/figures/`

### Configuration

Edit the constants at the top of `main.py` to adjust the run:

```python
CACHE_DIR   = './cache'                                        # HuggingFace dataset cache location
CATEGORIES  = ['astro-ph', 'cond-mat', 'cs', 'math', 'physics']  # Target classes
MAX_SAMPLES = 1000                                             # Total samples to use
```

### Use individual modules

```python
from src.dataset import load_raw_dataset, filter_sample, build_label_mappings, split_dataset
from src.preprocessing import preprocess_samples
from src.encoders import build_feature_matrices, EmbeddingVectorizer
from src.models import train_and_test_knn
from src.visualization import plot_confusion_matrix

# Load and preprocess
ds = load_raw_dataset('./cache')
samples = filter_sample(ds, ['cs', 'math'], max_samples=500)
preprocessed = preprocess_samples(samples)

# Encode and classify
sorted_labels, label_to_id, id_to_label = build_label_mappings(preprocessed)
X_train, X_test, y_train, y_test = split_dataset(preprocessed, label_to_id)
features = build_feature_matrices(X_train, X_test)

y_pred, accuracy, report = train_and_test_knn(*features['embedding'], y_train, y_test, sorted_labels)
print(f"Accuracy: {accuracy:.4f}")
```

---

## Notebooks

| Notebook | Purpose |
|---|---|
| `explorarion.ipynb` | EDA — label distribution, abstract length statistics and plots |
| `experiments.ipynb` | Full experiment run with printed accuracy results and confusion matrices |
| `[Code]-Project-3.1-Topic-Modeling.ipynb` | Project assignment notebook (solution) |
| `[Code-Hint]-Project-3.1-Topic-Modeling.ipynb` | Project assignment notebook (with hints) |
