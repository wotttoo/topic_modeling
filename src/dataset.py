from datasets import load_dataset
from sklearn.model_selection import train_test_split

# CATEGORIES_TO_SELECT = ['astro-ph', 'cond-mat', 'cs', 'math', 'physics']

def load_raw_dataset(cache_dir: str):
    return load_dataset('UniverseTBD/arxiv-abstracts-large', cache_dir=cache_dir)

def filter_sample(ds, categories, max_samples)->list[dict]:
    samples = []
    for s in ds['train']:
        if len(s['categories'].split(' ')) != 1:
            continue

        cur_category = s['categories'].strip().split('.')[0]
        if cur_category not in categories:
            continue

        samples.append(s)
        if len(samples) >= max_samples:
            break

    return samples

def build_label_mappings(preprocessed_samples: list)->tuple[dict, dict]:
    labels = set([s['label'] for s in preprocessed_samples])
    sorted_labels = sorted(labels)

    label_to_id = {label: i for i, label in enumerate(sorted_labels)}
    id_to_label = {i: label for i, label in enumerate(sorted_labels)}

    return sorted_labels, label_to_id, id_to_label

def split_dataset(preprocessed_samples, label_to_id, test_size=0.2, random_state=42)->tuple:
    X_full = [sample['text'] for sample in preprocessed_samples]
    y_full = [label_to_id[sample['label']] for sample in preprocessed_samples]

    X_train, X_test, y_train, y_test = train_test_split(X_full, y_full, test_size=test_size, random_state=random_state, stratify=y_full)
    return X_train, X_test, y_train, y_test