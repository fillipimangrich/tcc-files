import os
os.environ.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
import argparse
import itertools
import json
import math
import time
from multiprocessing import Pool

import numpy as np

N = 32
_k = np.arange(N)[:, None]
_n = np.arange(N)[None, :]
C = np.sqrt(2 / N) * np.cos(np.pi * (2 * _n + 1) * _k / (2 * N))
C[0] /= np.sqrt(2)
CHUNK = 20000


def task(args):
    k, prefix = args
    m = N - k
    rows = C[:k]
    last_max = N - 1 - prefix[0]
    rest = itertools.combinations(range(prefix[-1] + 1, last_max + 1), m - len(prefix))
    best_s, best_p, count = -1.0, None, 0
    while True:
        chunk = list(itertools.islice(rest, CHUNK))
        if not chunk:
            break
        p = np.empty((len(chunk), m), dtype=np.int64)
        p[:, :len(prefix)] = prefix
        if m > len(prefix):
            p[:, len(prefix):] = np.array(chunk, dtype=np.int64).reshape(len(chunk), -1)
        mask = np.ones((len(p), N), bool)
        mask[np.arange(len(p))[:, None], p] = False
        v = np.nonzero(mask)[1].reshape(len(p), k)
        a = np.transpose(rows[:, v], (1, 0, 2))
        lam = np.linalg.eigvalsh(a @ np.transpose(a, (0, 2, 1)))[:, 0]
        i = int(np.argmax(lam))
        count += len(p)
        if lam[i] > best_s:
            best_s, best_p = float(lam[i]), p[i].tolist()
    return best_s, best_p, count


def prefixes(k):
    m = N - k
    length = min(m, 4)
    return [p for p in itertools.combinations(range(N), length) if p[-1] <= N - 1 - p[0]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("k", help="comma-separated values of K")
    ap.add_argument("--processes", type=int, default=os.cpu_count())
    ap.add_argument("--output", default=None)
    args = ap.parse_args()
    ks = [int(x) for x in args.k.split(",")]
    output = args.output or f"n32_exhaustive_{args.k.replace(',', '_')}.json"
    out = {}
    with Pool(args.processes) as pool:
        for k in ks:
            t0 = time.time()
            best = (-1.0, None)
            total = 0
            for s, p, c in pool.imap_unordered(task, [(k, pr) for pr in prefixes(k)], chunksize=1):
                total += c
                if s > best[0]:
                    best = (s, p)
            out[k] = dict(norm_T=1 / math.sqrt(best[0]), positions=best[1], evaluated=total,
                          candidates=math.comb(N, N - k), seconds=round(time.time() - t0, 1))
            print(k, out[k], flush=True)
            json.dump(out, open(output, "w"), indent=1)


if __name__ == "__main__":
    main()
