# Blocking Recall Report

## Block A - Exact Normalized Name

| Target | True Matches | Retrieved | Recall |
|---|---:|---:|---:|
| S2 | 3,693,619 | 1,554,418 | 42.0839% |
| S3 | 3,944,746 | 1,586,891 | 40.2280% |
| Overall | 7,638,365 | 3,141,309 | 41.1254% |

Block A uses exact equality on 
ame_alnum_norm.

The result confirms that exact normalized-name matching provides substantial candidate recall, but cannot recover noisy/variant business names by itself.

## Block B - Name Token Overlap

| Target | True Matches | Retrieved | Recall |
|---|---:|---:|---:|
| S2 | 8,415 | 5,183 | 61.5924% |
| S3 | 8,947 | 5,383 | 60.1654% |
| Overall | 17,362 | 10,566 | 60.8570% |

Block B uses token-overlap candidate generation on 
ame_tokens.

The implementation uses a reusable target-side inverted token index, ignores excessively common tokens according to configuration, scores candidates using token overlap/Jaccard similarity, and limits the number of candidates scored per S1 entity.

The Block B result shows that token overlap recovers substantially different matches from exact normalized-name blocking and provides a stronger baseline for noisy business-name matching.

## Notes

- These are blocking-stage recall measurements, not final matching precision or F0.5.
- Country is not used as a hard blocking filter.
- Block B was evaluated on a 5,000-S1 diagnostic sample.
- Further blocking strategies will be evaluated and combined in later M2 stages.
