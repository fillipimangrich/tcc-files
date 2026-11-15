import argparse
import itertools

import numpy as np


def dct_matrix(n):
    rows = [[1.0] * n]
    for i in range(1, n):
        rows.append([np.sqrt(2) * np.cos(((np.pi * i) / n) * (j + 0.5)) for j in range(n)])
    return np.array(rows)


def reordered_columns(c, padding):
    valid = [i for i in range(c.shape[1]) if i not in padding]
    return c[:, valid + list(padding)]


def composite_matrix(c, k):
    c00, c01, c10, c11 = c[:k, :k], c[:k, k:], c[k:, :k], c[k:, k:]
    return c00 - c01 @ np.linalg.inv(c11) @ c10


def padding_matrix(c, k):
    return -np.linalg.inv(c[k:, k:]) @ c[k:, :k]


def exhaustive_order(n, k):
    c = dct_matrix(n)
    best = (float("inf"), None)
    norms = []
    for padding in itertools.combinations(range(n), n - k):
        norm = np.linalg.norm(composite_matrix(reordered_columns(c, padding), k), 2)
        norms.append((norm, list(padding)))
        if norm <= best[0]:
            best = (norm, list(padding))
    for norm, padding in sorted(norms)[:2]:
        print(f"norm: {norm} positions: {' '.join(map(str, padding))}")
    return best


def heuristic_order(n, k):
    m = n - k
    padding = [int((j + 1) * n / (m + 1)) for j in range(m)]
    norm = np.linalg.norm(composite_matrix(reordered_columns(dct_matrix(n), padding), k), 2)
    print(f"norm: {norm} positions: {' '.join(map(str, padding))}")
    return norm, padding


def cpp_table(n, k, padding):
    p = padding_matrix(reordered_columns(dct_matrix(n), padding), k)
    rows = ["{ " + ", ".join(str(x)[:7] for x in row) + " }" for row in p]
    return f"const double PaddingManager::n{n}_v{k}[{n - k}][{k}] = {{" + ",\n    ".join(rows) + "};"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sizes", nargs="*", type=int, default=[4, 8, 16])
    ap.add_argument("--heuristic", action="store_true")
    args = ap.parse_args()
    for n in args.sizes:
        for k in range(n // 2 + 1, n):
            print(f"N = {n}, K = {k}")
            norm, padding = heuristic_order(n, k) if args.heuristic else exhaustive_order(n, k)
            print(f"const unsigned int PaddingManager::n{n}_v{k}_order[{n - k}] = {{{', '.join(map(str, padding))}}};")
            print(cpp_table(n, k, padding))
            print()


if __name__ == "__main__":
    main()
