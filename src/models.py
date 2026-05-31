from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.naive_bayes import GaussianNB
from collections import Counter


def train_and_test_kmeans(X_train, X_test, y_train, y_test, n_clusters: int, id_to_label)->tuple:
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_ids = kmeans.fit_predict(X_train)

    # Assign label to clusters
    cluster_to_label = {}
    for cluster_id in set(cluster_ids):
        labels_in_cluster = [y_train[i] for i in range(len(y_train)) if cluster_ids[i]==cluster_id]
        most_common_label = Counter(labels_in_cluster).most_common(1)[0][0]
        cluster_to_label[cluster_id]= most_common_label

    # Predict labels for test set
    test_cluster_ids = kmeans.predict(X_test)
    y_pred = [cluster_to_label[i] for i in test_cluster_ids]

    # metrics
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(
        y_test, y_pred,
        target_names=[id_to_label[i] for i in range(len(id_to_label))],
        output_dict=True
    )

    return y_pred, accuracy, report

def train_and_test_knn(X_train, X_test, y_train, y_test, sorted_labels, n_neighbors: int=5)->tuple:
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_train, y_train)

    y_pred = knn.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=sorted_labels, output_dict=True)

    return y_pred, accuracy, report

def train_and_test_decision_tree(X_train, X_test, y_train, y_test, sorted_labels)->tuple:
    dt = DecisionTreeClassifier(random_state=42)
    dt.fit(X_train, y_train)


    y_pred = dt.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=sorted_labels, output_dict=True)

    return y_pred, accuracy, report

def train_and_test_naive_bayes(X_train, X_test, y_train, y_test, sorted_labels)->tuple:
    nb = GaussianNB()

    # Naive Bayes requires input to be in dense format
    X_train_dense = X_train.toarray() if hasattr(X_train, 'toarray') else X_train
    X_test_dense = X_test.toarray() if hasattr(X_test, 'toarray') else X_test

    nb.fit(X_train_dense, y_train)

    y_pred = nb.predict(X_test_dense)

    # metrics
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=sorted_labels, output_dict=True)

    return y_pred, accuracy, report