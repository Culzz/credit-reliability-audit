# Reading note — Ovadia et al. (2019), arXiv:1906.02530

**Full bibliographic record (verified on arXiv + via API, 3 Nov 2026):**
Yaniv Ovadia, Emily Fertig, Jie Ren, Zachary Nado, D Sculley, Sebastian
Nowozin, Joshua V. Dillon, Balaji Lakshminarayanan, Jasper Snoek, *Can You
Trust Your Model's Uncertainty? Evaluating Predictive Uncertainty Under
Dataset Shift*, NeurIPS 2019 (v2 dated 17 Dec 2019), stat.ML. Peer-reviewed
(oral-track benchmark paper; reviewer feedback acknowledged).

**Read level:** FULL TEXT, 3 Nov 2026 (arXiv HTML v2). No remaining UNKNOWNs
for the axes we track.

## What it does
- Large-scale benchmark of predictive-uncertainty methods **under dataset
  shift**, across image (MNIST rotation/translation; CIFAR-10 and ImageNet
  with 16 corruption types × 5 intensities = 80 shifted test sets from
  Hendrycks & Dietterich 2019), text (20 Newsgroups in-dist vs shifted/OOD),
  purely categorical ads (Criteo, 37M rows; shift = re-assigning categorical
  tokens to fresh random tokens with probability = shift intensity), plus a
  genomics appendix.
- Methods compared: vanilla maximum-softmax, **post-hoc temperature scaling
  fitted on the validation set**, MC-Dropout, deep ensembles (M=10), full
  SVI, last-layer SVI/dropout (+ variational GP in appendix).
- Metrics: accuracy, NLL, **Brier score with its reliability/resolution/
  uncertainty decomposition** (citing Brier 1950, DeGroot & Fienberg 1983,
  Bröcker 2009), ECE (idiosyncratic, non-proper caveat stated), predictive
  entropy, accuracy-vs-confidence.

## Findings that matter to RQ1
1. **Accuracy and uncertainty quality degrade with increasing shift for
   every method examined.** Brier score separates methods that accuracy
   cannot (the paper's headline demonstration).
2. **Calibration achieved on an i.i.d. validation set does NOT carry over
   to shifted data.** This is exactly the precedent for our RQ1a/b under
   temporal (not corruption) shift.
3. **Temperature scaling is the most fragile:** below small shift it is
   fine; as shift grows it is significantly outperformed by ensembles/
   dropout, and on Criteo its Brier score is *worse than vanilla* —
   "post-hoc calibration on the validation set actually harms calibration
   under dataset shift." Directly relevant to our recalibration arms: a
   validation-window-fitted recalibrator must be expected to decay, and our
   lag-feasible re-fitting schedule is the test of whether scheduled
   re-fitting recovers it.
4. **Deep ensembles dominate across most metrics/shift** (M=5 already close
   to M=50); SVI wins only on MNIST and fails to scale.
5. Under full OOD, most methods are "confidently wrong" (very low entropy).

## What it does NOT do (our gap is intact)
- No explanations of any kind, no fidelity/stability audit.
- No loan-level decision-flip or threshold-policy analysis.
- No label-delay mechanism (shift is synthetic corruption/token-sharing,
  applied at constant intensity; there is no calibration-sample-size or
  information-delay axis).
- No test of whether AUC (separability) *predicts* reliability — they
  observe accuracy/Brier co-degradation, never regress one on the other.
- No credit lending dataset (Criteo ads are the tabular/categorical
  analogue; no per-loan or per-vintage cohort design).

## Positioning
This is the strongest precedent for "post-hoc calibration is not durable
under shift" — our novelty claim must stay narrowed to the joint audit
(calibration + explanation + decisions) with label-delay-restricted
recalibration and the AUC-vs-reliability regression on real credit data,
as pre-agreed in CONTEXT.md (do not claim the general fact itself).

## Limitations stated in the paper
Uncertainty-quality focus only; compute/memory cost of the good methods
(appendix A.9); relative method ordering is dataset-dependent (MNIST is the
outlier); SVI scaling problems; synthetic shift constructions only.
