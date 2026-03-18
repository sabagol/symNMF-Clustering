import sys
import numpy as np
import symnmfmodule # type: ignore

SEED = 1234

def load_data(path):
    return np.loadtxt(path, dtype=np.float32)


def print_matrix(mat):
    for row in mat:
        print(",".join(f"{float(v):.4f}" for v in row))


def main():
    if len(sys.argv) != 4:
        print("An Error Has Occurred")
        sys.exit(1)
    k = int(sys.argv[1])
    goal = sys.argv[2]
    filename = sys.argv[3]

    X = load_data(filename)
    try:
        if goal == "sym":
            out = symnmfmodule.sym(X)
            print_matrix(out)
        elif goal == "ddg":
            out = symnmfmodule.ddg(X)
            print_matrix(out)
        elif goal == "norm":
            out = symnmfmodule.norm(X)
            print_matrix(out)
        elif goal == "symnmf":
            # compute W
            W = symnmfmodule.norm(X)
            N = W.shape[0]
            # initialization per spec: seed and uniform in [0, 2*sqrt(m/k)] where m is mean of W entries
            np.random.seed(SEED)
            m = float(np.mean(W))
            init_upper = 2.0 * np.sqrt(m / k) if k > 0 else 0.0
            H_init = np.random.uniform(0.0, init_upper, size=(N, k)).astype(np.float32)
            H = symnmfmodule.symnmf(W, H_init)
            print_matrix(H)
        else:
            print("An Error Has Occurred")
            sys.exit(1)
    except Exception:
        print("An Error Has Occurred")
        sys.exit(1)


if __name__ == "__main__":
    main()