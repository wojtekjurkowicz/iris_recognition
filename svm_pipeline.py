from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import f1_score
from utils.visualization import plot_tsne
import matplotlib

matplotlib.use("Agg")  # lub "TkAgg" jeśli masz GUI

import matplotlib.pyplot as plt


def run_svm(X, y):
    print("Running SVM classifier...")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=0.95)
    X_pca = pca.fit_transform(X_scaled)

    subset_classes = np.random.choice(np.unique(y), 10, replace=False)
    mask = np.isin(y, subset_classes)
    X_sub = X_pca[mask]
    y_sub = y[mask]

    X_tsne = TSNE(n_components=2, perplexity=30).fit_transform(X_sub)
    plot_tsne(X_tsne, y_sub, title="t-SNE po PCA + HOG", filename="tsne_plot_svm.png")


    """
    idx = random.randint(0, len(X) - 1)
    plt.imshow(X[idx].astype(np.uint8))
    plt.title(f"Label: {y[idx]}")
    plt.savefig("plot.png")
    """

    X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, stratify=y, random_state=42)
    parameters = {
        'C': [0.1, 1, 10, 50, 100],
        'kernel': ['linear', 'rbf'],
        'gamma': ['scale', 'auto']
    }
    clf = GridSearchCV(SVC(class_weight='balanced'), parameters, cv=3, verbose=1, n_jobs=-1)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred))
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("F1 macro:", f1_score(y_test, y_pred, average="macro"))
