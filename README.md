# Padding with power-of-two transforms in the JPEG Pleno 4D Transform Mode

Code and data of the undergraduate thesis by Fillipi Mangrich Costa de Souza (Computer Science, UFSC, 2026),
advised by Prof. Ismael Seidel.

## Contents

| Folder | Contents |
|---|---|
| `codec/` | Patch over the JPEG Pleno reference software (JPLM) with the proposed technique |
| `orders/` | Search of the interlacing orders and computation of the padding matrices |
| `composite/` | Tables of the composite matrices used by the encoder |
| `figures/` | Scripts of the figures of the thesis |
| `data/` | Results of the experiments of Chapter 5 |

## Codec

The patch applies to the tag `v3.0.0-m112` (commit `d8523a53`) of the
[JPEG Pleno reference software](https://gitlab.com/wg1/jpeg-pleno-refsw):

```
git clone https://gitlab.com/wg1/jpeg-pleno-refsw.git
cd jpeg-pleno-refsw
git checkout v3.0.0-m112
git apply ../codec/jplm_tcc.patch
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBIT_STUFFING_WG1M109068=ON
make -C build jpl-encoder-bin jpl-decoder-bin
```

`codec/LICENSE-JPLM.md` is the license of the reference software.

Encoder options of each configuration:

| Configuration | Blocks | Options |
|---|---|---|
| `anchor` | CTC | `--border_policy truncate` |
| `truncation` | powers of two | `--border_policy truncate` |
| `repetition` | powers of two | `--border_policy padding` |
| `proposed` | powers of two | `--border_policy power_of_2_ordered --padding_computation composite` |
| `proposed_three_step` | powers of two | `--border_policy power_of_2_ordered --padding_computation steps` |

The decoder needs no option: the border policy is read from the codestream.

## Interlacing orders and padding matrices

```
python3 orders/search_orders.py                  # N = 4, 8, 16: exhaustive search
python3 orders/search_orders.py --heuristic 32   # N = 32: uniform positions
python3 orders/search_orders_n32.py 25,26,27,28,29,30,31
```

Each command prints the order and the padding matrix of every K in the format of `Padding.cpp`.
The tables of the codec use the exhaustive search for N ≤ 16 and, for N = 32, the exhaustive search for
K ≥ 25 and the uniform positions for K ≤ 24. `orders/n32_exhaustive.json` has the result of the
exhaustive search for every K of N = 32. An order and its mirror image (p → N − 1 − p) have the same norm;
for some sizes the tables use the mirror image of the order printed by the search.

## Composite matrices

```
python3 composite/composite_tables.py <jplm>/source/Lib/Part2/Common/TransformMode/Padding.cpp composite.inc
```

Reads the orders of `Padding.cpp` and writes the tables of T = (C00^T)^-1 with five decimal places.

## Figures

| Figure | Script | Input |
|---|---|---|
| `fig_lf_views` | `figures/lf_views.py` | light fields in PGX |
| `fig_dataset_views` | `figures/dataset_views.py` | light fields in PPM |
| `fig_dct_basis`, `fig_padding_curve`, `fig_boundary_blocks`, `fig_border_policies`, `fig_padded_views`, `fig_rate_gap` | `figures/didactic.py` | light fields in PGX, `data/metrics.csv` |
| `fig_matrix_blocks`, `fig_two_orders`, `fig_padding_growth` | `figures/chapter3.py` | none |
| `fig_conditioning` | `figures/conditioning.py` | `Padding.cpp` of the patched codec |
| `fig_block_policies` | `figures/block_policies.py` | light fields in PGX |
| `rd_<light field>_<metric>` | `figures/rd_curves.py` | `data/metrics.csv`, jpeg-pleno-ctc-tools |

The light fields are those of the JPEG Pleno Common Test Conditions, available from the JPEG Pleno database.
The RD curves use the plotting classes of jpeg-pleno-ctc-tools (commit `22fe9ba`).
Tested with Python 3.12, NumPy 2.2, SciPy 1.15 and Matplotlib 3.10.

## Data

| File | Contents |
|---|---|
| `data/encodings.json` | λ, rate, PSNR and encoder command of each of the 175 encodings |
| `data/metrics.csv` | Rate and all the metrics of the CTC for the four configurations of Chapter 5 |
| `data/bd_rate.csv` | BD-Rate of each configuration with respect to the others, per light field and metric |
| `data/vmaf/` | Reports of the VMAF tool, with the values of each view |
| `data/codestreams.md5` | MD5 of each codestream |

The rates and qualities of the anchor are those published in the CTC. PSNR, SSIM and VMAF were computed
with libvmaf 2.2.1 on the views converted to YCbCr 4:4:4 with 10 bits and ordered as in the CTC tools,
and the BD-Rate with the `bjontegaard` package 1.3.0 (Akima interpolation).
