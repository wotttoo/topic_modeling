from src.dataset import load_raw_dataset, filter_sample, build_label_mappings, split_dataset
from src.preprocessing import preprocess_samples
from src.encoders import build_feature_matrices
from src.models import train_and_test_kmeans, train_and_test_knn, train_and_test_decision_tree, train_and_test_naive_bayes
from src.visualization import plot_confusion_matrix

CACHE_DIR = './cache'
CATEGORIES = ['astro-ph', 'cond-mat', 'cs', 'math', 'physics']
MAX_SAMPLES = 1000

if __name__=="__main__":
    # load data from HuggingFace
    ds = load_raw_dataset(CACHE_DIR)

    # filter 1000 samples
    samples = filter_sample(ds, CATEGORIES, MAX_SAMPLES)

    # preprocessing data
    preprocessed_samples = preprocess_samples(samples)

    # mapping
    sorted_labels, label_to_id, id_to_label = build_label_mappings(preprocessed_samples)

    # split to train set and test set
    X_train, X_test, y_train, y_test = split_dataset(preprocessed_samples, label_to_id, test_size=0.2, random_state=42)

    # encoders:
    encoder_names = ['bow', 'tfidf', 'embedding']
    features = build_feature_matrices(X_train, X_test)

    # models:
    for encoder_name in encoder_names:
        X_train_enc, X_test_enc = features[encoder_name]
        
        # KMeans
        y_pred, accuracy, report = train_and_test_kmeans(X_train_enc, X_test_enc, y_train, y_test, 
                                                         n_clusters=len(label_to_id), id_to_label=id_to_label)
        print(f"KMeans ({encoder_name}): {accuracy:.4f}")
        plot_confusion_matrix(y_test, y_pred, sorted_labels, f"KMeans ({encoder_name})", 
                              save_path=f'outputs/figures/kmeans_{encoder_name}.png')
        
        # knn
        y_pred, accuracy, report = train_and_test_knn(X_train_enc, X_test_enc, y_train, y_test, sorted_labels, 
                                                      n_neighbors=5)
        print(f"KNN ({encoder_name}): {accuracy:.4f}")
        plot_confusion_matrix(y_test, y_pred, sorted_labels, f"KNN ({encoder_name})",
                              save_path=f'outputs/figures/knn_{encoder_name}.png')
        
        # decision tree
        y_pred, accuracy, report = train_and_test_decision_tree(X_train_enc, X_test_enc, y_train, y_test, sorted_labels)
        print(f'Decision Tree ({encoder_name}): {accuracy:.4f}')
        plot_confusion_matrix(y_test, y_pred, sorted_labels, f"Decision tree ({encoder_name})",
                              save_path=f'outputs/figures/dt_{encoder_name}.png')
        
        # naive bayes
        y_pred, accuracy, report = train_and_test_naive_bayes(X_train_enc, X_test_enc, y_train, y_test, sorted_labels)
        print(f'Naive Bayes ({encoder_name}): {accuracy:.4f}')
        plot_confusion_matrix(y_test, y_pred, sorted_labels, f"Naive bayes ({encoder_name})",
                              save_path=f'outputs/figures/nb_{encoder_name}.png')