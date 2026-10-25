# FDNet — A Lightweight Feature-Dominated U-Net for Hippocampus Segmentation in Brain MRI

Official implementation of the paper:

> **FDNet: A Lightweight Feature-Dominated U-Net for Hippocampus Segmentation in Brain MRI**
> Mehmet Halis Korkmaz, Özkan Bingöl, for the Alzheimer's Disease Neuroimaging Initiative
> *Diagnostics* (MDPI), 2026. DOI: `<add DOI once published>`

---

## 📌 Repository status — code not yet uploaded

This repository has been created to accompany the manuscript listed above, which is currently under
review at *Diagnostics* (MDPI). **The source code, trained model weights and split definitions will be
uploaded once the article has been accepted and its DOI has been issued.**

Until then, this README documents the intended contents of the release: the architecture, the datasets
and preprocessing, the training and evaluation protocols, and the results reported in the paper. The
directory layout, file names and commands shown below describe the planned release and are not yet
present in the repository.

Readers who need access to the code before publication are welcome to contact the corresponding author
(see [Contact](#contact)).

---

## Overview

Hippocampus segmentation is difficult because the structure is small, irregularly shaped and poorly
contrasted against surrounding tissue. Most U-Net variants address this by *adding* machinery —
attention gates, learnable skip transforms, discriminators — at a computational cost.

FDNet takes the opposite route. It changes the **balance of information that already crosses the skip
connection**. In a standard U-Net, each encoder feature map is concatenated once with the upsampled
decoder feature map, so fine detail and coarse context enter the following convolution in a ratio of
1:2. FDNet concatenates *n* identical copies of the encoder feature map instead:

```
[E₁, E₂, …, Eₙ, D]        (unweighted, wᵢ = 1 for all i)
```

This introduces **no new learnable module** — no gate, no transform, no sub-network. The only
parameter change is that the convolution after each concatenation accepts more input channels, which
costs exactly **48,384 parameters per additional copy**:

```
P(n) = 433,378 + 48,384·n
```

Three degrees of domination are released here: **FDNetX2** (n = 2), **FDNetX3** (n = 3) and
**FDNetX4** (n = 4).

Because repetition does not expand the family of representable functions (the kernel slices seeing
identical copies can be summed after training), the gain is interpreted as a change in the
**parameterisation and optimisation dynamics** of the encoder branch rather than as added capacity.

## Key results

Slice-level five-fold protocol, Dice similarity coefficient (DSC):

| Model | Params | ADNI | Decathlon | IBSR |
|---|---|---|---|---|
| FDNetX2 | 530,146 | 0.9321 | 0.9453 | **0.9477** |
| FDNetX3 | 578,530 | **0.9339** | **0.9470** | 0.9471 |
| FDNetX4 | 626,914 | 0.9327 | 0.9446 | 0.9453 |
| U-Net | 34,512,258 | 0.9173 | 0.9415 | 0.9236 |
| Attention U-Net | 37,306,190 | 0.8910 | 0.9175 | 0.8459 |
| SegNet | 88,092,546 | 0.9240 | 0.9394 | 0.9464 |

Subject-grouped protocol on ADNI (fixed held-out test set, 19 subjects / 34 volumes), subject-level
test Dice:

| Model | Test DSC | Params |
|---|---|---|
| FDNetX3 | **0.9050** | 578,530 |
| FDNetX2 | 0.9047 | 530,146 |
| FDNetX4 | 0.9044 | 626,914 |
| nnU-Net v2 (2D, 5-fold ensemble) | 0.9019 | 7,766,344 |
| U-Net | 0.8866 | 34,512,258 |
| Attention U-Net | 0.8849 | 37,306,190 |
| SegNet | 0.8243 | 88,092,546 |

All three FDNet variants significantly outperform the three re-implemented baselines
(Wilcoxon signed-rank, Holm-adjusted *p* < 0.0001). Their advantage over nnU-Net is **not**
significant after Holm correction — FDNet is therefore comparable to, or slightly better than,
nnU-Net, obtained with one to two orders of magnitude fewer parameters.

> **Scope.** All results come from curated research datasets, on pre-cropped patches, without external
> or cross-dataset validation. They indicate competitive performance under the evaluated conditions,
> not clinical readiness.

## Repository structure

```
fdnet/
├── models/            # FDNetX2/X3/X4 and the re-implemented U-Net, Attention U-Net, SegNet
├── data/              # download helpers, cropping, sagittal slicing, augmentation
├── training/          # training loops, hybrid Dice+BCE loss, 5-fold and GroupKFold protocols
├── evaluation/        # DSC, mIoU, recall, PPV, accuracy, FNPr/FPPr; statistical tests
├── nnunet/            # dataset conversion and custom fold definitions for the nnU-Net v2 baseline
├── splits/            # subject-level split definitions and fold assignments
├── weights/           # trained FDNet weights (~2.1-2.5 MB each)
├── figures/           # scripts reproducing Figures 1, 3-10
└── configs/           # experiment configuration files
```

## Installation

```bash
git clone https://github.com/<owner>/FDNet-hippocampus-segmentation.git
cd FDNet-hippocampus-segmentation
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Core dependency: **TensorFlow 2**. The nnU-Net baseline requires a separate environment with
**nnU-Net v2 (2.8.1)** — see `nnunet/README.md`.

## Data

Three public datasets are used. **No imaging data is redistributed in this repository**; the scripts
download and preprocess data that you obtain yourself under the relevant terms.

| Dataset | Access | Volumes used | Patch | Slices (before → after augmentation) |
|---|---|---|---|---|
| ADNI | [LONI IDA](https://adni.loni.usc.edu) — requires a Data Use Agreement | 161 (slice-level) / 322 (subject-grouped) | 96×96×96 centre crop | 15,456 → 20,751 |
| Medical Segmentation Decathlon | [medicaldecathlon.com](http://medicaldecathlon.com) | 260 labelled | 32×32×32 resampled | 2,080 → 4,160 |
| IBSR | [NITRC](https://www.nitrc.org/projects/ibsr) | 18 | 96×96×120 centre crop | 2,160 → 3,241 |

All volumes are sliced **sagittally**. For the Decathlon task the anterior and posterior labels are
merged into a single hippocampal class.

> **ADNI users:** the ADNI Data Use Agreement prohibits redistribution of ADNI images and labels.
> Obtain the data directly from LONI; this repository provides only the preprocessing and split
> definitions needed to reproduce the experiments.

### Augmentation

The default scheme generates one slice from each pair of consecutive labelled slices: the new label is
the **union** of the two source labels, the new image their **voxel-wise mean**. Test slices are never
augmented. Two alternatives evaluated in the paper (a light geometric transform, and geometric plus
intensity perturbation) are also implemented and selectable from the configuration.

## Training

```bash
python -m training.train --model fdnetx3 --dataset adni --protocol slice   --config configs/adni_slice.yaml
python -m training.train --model fdnetx3 --dataset adni --protocol subject --config configs/adni_subject.yaml
```

Default configuration used in the paper:

| Setting | Value |
|---|---|
| Framework | TensorFlow 2 |
| Optimiser | Adam, initial learning rate 1×10⁻⁴ |
| Batch size | 16 |
| Dropout | 0.2, between the convolutions of every encoder/decoder block |
| Loss | L = α·L_Dice + (1−α)·L_BCE, α = 0.5 |
| Slice-level protocol | 10% held-out test split, then five-fold CV on the remainder |
| Subject-grouped protocol | 19 test subjects (34 volumes) held out; GroupKFold over 162 development subjects (288 volumes); 120 epochs |

The subject-grouped configuration additionally uses geometric augmentation and a 2.5D input of three
consecutive sagittal slices.

## Evaluation

```bash
python -m evaluation.evaluate --weights weights/fdnetx3_adni_fold1.h5 --dataset adni --split test
python -m evaluation.statistics --results results/adni_subject/        # Wilcoxon + Holm + bootstrap CIs
```

Reported metrics: DSC, mean IoU, recall, positive predictive value, accuracy, and the mean per-slice
false-negative and false-positive pixel counts (**FNPr**, **FPPr**). FNPr and FPPr are *counts per
slice*, not conventional normalised rates; their sum is the mean total number of misclassified pixels
per slice. Slices with an empty reference mask are excluded, since DSC is undefined there.

Statistical comparison follows the paper: two-sided Wilcoxon signed-rank tests on subject-paired Dice,
Holm–Bonferroni correction across the nine pairwise comparisons, and 95% bootstrap confidence
intervals from 10,000 subject-level resamples.

## Pretrained weights

Trained FDNet weights are in `weights/` (2,140 KB, 2,329 KB and 2,518 KB for X2, X3 and X4). Baseline
weights are far larger (U-Net 134 MB, Attention U-Net 146 MB, SegNet 344 MB) and are published under
[Releases](../../releases) rather than tracked in the repository.

## Hardware and runtime

Measured on an Intel Core i9-11900K, 32 GB RAM and an NVIDIA GeForce RTX 3060, on ADNI:

| Model | conv-GFLOPs / slice | Training (s/epoch) | Inference (ms/slice) |
|---|---|---|---|
| FDNetX2 | 0.776 | 8.90 | 0.70 |
| FDNetX3 | 0.904 | 9.73 | 0.75 |
| FDNetX4 | 1.031 | 10.83 | 0.99 |
| U-Net | 0.85 | 50.98 | 2.11 |
| Attention U-Net | 0.89 | 66.59 | 2.50 |
| SegNet | 11.93 | 79.28 | 3.03 |

A single GPU with ~4 GB of memory is sufficient for FDNet; peak training memory is 1.76–2.06 GiB at
matched width.

## Reproducing the paper

| Paper item | Script |
|---|---|
| Table 2 (slice-level accuracy and error balance) | `evaluation/run_table2.py` |
| Tables 3-4 (subject-grouped results, statistics) | `evaluation/run_table3_4.py` |
| Table 5 (width-matched efficiency) | `evaluation/run_table5.py` |
| Table 10 (parameters, timings) | `evaluation/run_table10.py` |
| Figures 1, 3-10 | `figures/` |

## Citation

```bibtex
@article{korkmaz2026fdnet,
  title   = {FDNet: A Lightweight Feature-Dominated U-Net for Hippocampus Segmentation in Brain MRI},
  author  = {Korkmaz, Mehmet Halis and Bing{\"o}l, {\"O}zkan},
  journal = {Diagnostics},
  year    = {2026},
  doi     = {<add DOI once published>}
}
```

If you use the code, please also cite the datasets you train on (ADNI, Medical Segmentation
Decathlon, IBSR).

## License

Code released under the MIT License — see [LICENSE](LICENSE). The accompanying article is published
by MDPI under CC BY 4.0. Dataset licences and use agreements are those of the respective providers.

## Acknowledgements

Data used in the preparation of this work were obtained from the Alzheimer's Disease Neuroimaging
Initiative (ADNI) database (adni.loni.usc.edu). The ADNI investigators contributed to the design and
implementation of ADNI and/or provided data but did not participate in the analysis or writing of this
report. A complete listing of ADNI investigators is available at
[ADNI_Acknowledgement_List.pdf](http://adni.loni.usc.edu/wp-content/uploads/how_to_apply/ADNI_Acknowledgement_List.pdf).

ADNI is funded by the National Institute on Aging (NIH Grant U19 AG024904); the grantee organisation
is the Northern California Institute for Research and Education.

## Contact

Mehmet Halis Korkmaz — korkmaz@gumushane.edu.tr
Department of Mathematics, Institute of Science, Gümüşhane University, Gümüşhane, Türkiye

