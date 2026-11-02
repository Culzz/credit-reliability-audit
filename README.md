# credit-reliability-audit

**Working title:** *Does Accuracy Predict Reliability? Calibration, Explanation Fidelity and Decision Stability of Credit-Risk Models Under Temporal Distribution Shift.*

> **Status: pre-protocol.** Phase 0 (infrastructure). The research protocol
> (`protocol/PROTOCOL.md`) is not written yet. Test cohorts are locked in code
> (`src/data/guard.py`): nothing may score or tune on them until
> `protocol/PROTOCOL_FROZEN` exists, created only after the protocol is tagged
> `protocol-v1`. No numbers reported anywhere in the repository are final.

An empirical audit of machine-learning credit-risk models on real LendingClub
loans (2007–2018Q4, 36-month loans, strictly time-ordered evaluation). The
question: does ROC-AUC, the standard headline metric, tell us anything about
three reliability dimensions on future borrower populations?

1. **Calibration (RQ1)** — how predicted probabilities degrade across future
   cohorts, and how much recalibration recovers when outcome labels arrive late.
2. **Explanations (RQ2)** — population dependence, reliance drift after
   retraining, estimation noise floor, and fidelity against independent
   reliance measures.
3. **Decisions (RQ3)** — approve/refer/decline flips between model versions and
   their association with realised outcomes.
4. **AUC vs reliability (RQ4)** — whether discrimination predicts any of the
   above.

## Working documents

| File | Purpose |
|---|---|
| `CONTEXT.md` | Conventions every change must satisfy: data rules, metric formulas, parameters, review obligations |
| `PLAN.md` | Phases, gates, tasks, cut lines; work advances gate by gate |
| `docs/blueprint.md` | Specifications carried over from the original blueprint |
| `notes/STATUS.md` | Where the project stands right now |
| `notes/DECISIONS.md` | Decision log (append-only) |
| `notes/REVIEWS.md` | Review log; substantive changes are reviewed before they count |

## Research disciplines

No invented numbers, citations or results; every manuscript number comes from a
script output; test cohorts stay locked until the protocol freezes; time-ordered
evaluation only; no resampling in the main run; every headline number gets a
confidence interval; synthetic data only inside unit tests; negative results are
results. Details in `CONTEXT.md` §4.

## Reproduction (target end state)

```powershell
git clone https://github.com/Culzz/credit-reliability-audit.git
cd credit-reliability-audit
uv sync
# follow docs/blueprint.md §B-9 sequence (data download is manual, never committed)
```
