import sys
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import symnmfmodule # type: ignore 

SEED = 1234

def assign_clusters(H):
    return np.argmax(H, axis=1)


def main():
    if len(sys.argv) != 3:
        print("An Error Has Occurred")
        sys.exit(1)
    k = int(sys.argv[1])
    filename = sys.argv[2]

    X = np.loadtxt(filename, delimiter=",", dtype=np.float32)

    W = symnmfmodule.norm(X)
    N = W.shape[0]
    # initialize H per spec
    np.random.seed(SEED)
    m = float(np.mean(W))
    init_upper = 2.0 * np.sqrt(m / k) if k > 0 else 0.0
    H_init = np.random.uniform(0.0, init_upper, size=(N, k)).astype(np.float32)
    H = symnmfmodule.symnmf(W, H_init)
    labels_nmf = assign_clusters(H)
    nmf_score = silhouette_score(X, labels_nmf)
    print(f"nmf: {nmf_score:.4f}")

    kmeans = KMeans(n_clusters=k, random_state=0, n_init=10).fit(X)
    kmeans_score = silhouette_score(X, kmeans.labels_)
    print(f"kmeans: {kmeans_score:.4f}")

if __name__ == "__main__":
    main()