import argparse
import re

import numpy as np

SIZES = [(4, [3]), (8, [5, 6, 7]), (16, list(range(9, 16))), (32, list(range(17, 32)))]


def dct(n):
    i = np.arange(n)
    c = np.sqrt(2.0 / n) * np.cos(np.pi * np.outer(i, 2 * i + 1) / (2 * n))
    c[0] /= np.sqrt(2.0)
    return c


def idct_jplm(n):
    j = np.arange(n)
    m = np.sqrt(2.0) * np.cos(np.pi * np.outer(2 * j + 1, j) / (2 * n)) / n
    m[:, 0] = 1.0 / n
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("padding_cpp", help="Padding.cpp of the patched reference software (source of the orders)")
    ap.add_argument("output")
    ap.add_argument("--decimals", type=int, default=5)
    args = ap.parse_args()
    src = open(args.padding_cpp).read()
    fmt = lambda v: (lambda s: s[1:] if s.startswith("-") and float(s) == 0.0 else s)(f"{v:.{args.decimals}f}")

    out = []
    names = {}
    worst_exact = worst_rounded = 0.0
    for n, ks in SIZES:
        c = dct(n)
        for k in ks:
            m = re.search(rf"PaddingManager::n{n}_v{k}_order\[\d+\]\s*=\s*\{{([^}}]*)\}}", src)
            pad = [int(v) for v in m.group(1).split(",")]
            valid = [i for i in range(n) if i not in pad]
            t = np.linalg.inv(c[:k, valid].T)
            d = idct_jplm(n)[valid][:, :k]
            worst_exact = max(worst_exact, np.abs(np.sqrt(n) * t - np.linalg.inv(d)).max() / np.sqrt(n))
            rounded = np.array([[float(fmt(v)) for v in row] for row in t])
            worst_rounded = max(worst_rounded, np.abs(d @ (np.sqrt(n) * rounded) - np.eye(k)).max())
            name = f"n{n}_v{k}_composite"
            names[k] = name
            out.append(f"static const double {name}[{k}][{k}] = {{")
            out.append(",\n".join("    { " + ", ".join(fmt(v) for v in row) + " }" for row in t))
            out.append("};")
            out.append("")
    out.append("const double* PaddingManager::composite_look_up_table[] = {")
    for k in range(1, 33):
        out.append(f"    &{names[k]}[0][0]," if k in names else "    nullptr,")
    out.append("};")
    open(args.output, "w").write("\n".join(out) + "\n")
    print(f"{len(names)} matrices, {sum(k * k for k in names)} values -> {args.output}")
    print(f"max |T - inverse of the JPLM IDCT block / sqrt(N)| = {worst_exact:.2e}")
    print(f"max |D sqrt(N) T_rounded - I| = {worst_rounded:.2e}")


if __name__ == "__main__":
    main()
