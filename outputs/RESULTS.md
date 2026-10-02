# Experimental results

Selected using validation macro-F1: **Linear SVM C=1**.

Training / validation / test rows: 3529 / 756 / 757.

Final test accuracy: **88.90%**; macro-F1: **0.6834**; weighted-F1: **0.8842**.

Accuracy bootstrap 95% interval: 86.66%–91.02%.

## Audit

- long: n=263, accuracy=88.97%, macro-F1=0.6844. 
- medium: n=376, accuracy=90.43%, macro-F1=0.6702. 
- short: n=118, accuracy=83.90%, macro-F1=0.5987. 
- formal_detailed_proxy: n=113, accuracy=85.84%, macro-F1=0.6995. 
- informal_short_proxy: n=5, accuracy=80.00%, macro-F1=0.5556. Small group: interpret cautiously.
- unassigned: n=639, accuracy=89.51%, macro-F1=0.6678. 

Proxy groups are heuristic. A difference is an association, not evidence that formality causes errors. Complete the human audit before making a validated formality claim. See controlled_audit.json for equal-class comparisons and audit_per_class.csv for coverage.
