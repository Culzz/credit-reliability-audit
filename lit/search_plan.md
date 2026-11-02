# Literature search plan (T1.1)

Databases: Google Scholar (broad discovery), Semantic Scholar (citations +
related work), OpenAlex (bibliographic metadata), Crossref (DOI verification),
arXiv (preprints), publisher/proceedings pages (authoritative records).
Reference management: Zotero, collection `credit-shift-audit`.

Every candidate found by a query below must be verified on the publisher /
arXiv / DOI page before it enters `lit/matrix.csv` (Blueprint B-1 rules; the
matrix carries its own `doi_or_url` and `read_status` columns).

## Primary query strings

```
1.  "credit scoring" AND "temporal distribution shift"
2.  "credit risk" AND "calibration under distribution shift"
3.  "probability calibration" AND "temporal validation"
4.  "explanation stability" AND "temporal drift"
5.  "SHAP" AND "feature attribution stability"
6.  "SHAP" AND "model reliance" AND "permutation importance"
7.  "credit scoring" AND "decision instability"
8.  "credit risk" AND "model recalibration" AND "label delay"
9.  "AUC" AND "calibration" AND "distribution shift"
10. "predictive performance" AND "model reliability"
11. "credit scoring" AND "out-of-time validation"
```

## Citation tracing steps

- Backward: for each relevant hit, pull its key references (who it builds on).
- Forward: for each relevant hit, pull what cites it afterwards
  (Semantic Scholar / OpenAlex citation lists).
- Anchors to trace both directions: Ovadia 2019; Guo 2017; Marx 2020;
  Gardner 2023 (TableShift); Miller 2021 ("Accuracy on the Line");
  Krishna 2024 (disagreement problem); Fisher 2019; Van Calster 2019;
  Lundberg 2017/2020; Hooker, Mentch & Zhou 2021; Gama 2014; D'Amour 2022;
  Lessmann 2015.

## Priority reading queue (before Gate 1)

1. John 2025, arXiv:2511.03807 — closest competitor; full text.
2. Nayak 2026, arXiv:2603.06733 — closest competitor; full text.
3. Krishna 2024 (TMLR, disagreement problem).
4. Gardner 2023 (TableShift).
5. Miller 2021 ("Accuracy on the Line").
6. Marx 2020 (predictive multiplicity).
7. Ovadia 2019 (calibration under dataset shift).

Target: matrix of 40–60 verified papers; closest papers read in full
(methods + limitations, not abstracts).
