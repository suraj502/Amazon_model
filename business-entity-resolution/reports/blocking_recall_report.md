# Blocking Recall Report

## Block A - Exact Normalized Name

| Target  | True Matches | Retrieved |   Recall |
| ------- | -----------: | --------: | -------: |
| S2      |    3,693,619 | 1,554,418 | 42.0839% |
| S3      |    3,944,746 | 1,586,891 | 40.2280% |
| Overall |    7,638,365 | 3,141,309 | 41.1254% |

Block A uses exact equality on `name_alnum_norm`.

The result confirms that exact normalized-name matching provides substantial candidate recall, but cannot recover noisy or variant business names by itself.

---

## Block B - Name Token Overlap

| Target  | True Matches | Retrieved |   Recall |
| ------- | -----------: | --------: | -------: |
| S2      |        8,415 |     5,183 | 61.5924% |
| S3      |        8,947 |     5,383 | 60.1654% |
| Overall |       17,362 |    10,566 | 60.8570% |

Block B uses token-overlap candidate generation on `name_tokens`.

The implementation uses a reusable target-side inverted token index, ignores excessively common tokens according to configuration, scores candidates using token overlap/Jaccard similarity, and limits the number of candidates scored per S1 entity.

Block B recovers substantially different matches from exact normalized-name blocking and provides a stronger baseline for noisy business-name matching.

---

## Block C - Character N-gram Name Blocking

Block C uses character 3-gram overlap on `name_basic_norm`.

Development subset:

* Source 1: 5,000 rows
* Source 2: 50,000 rows
* Source 3: 50,000 rows
* Ground truth: 5,000 rows
* N-gram size: 3
* Top-K: 20

### Development Recall

| Target  | True Matches | Retrieved |   Recall | Candidate Pairs |
| ------- | -----------: | --------: | -------: | --------------: |
| S2      |        8,328 |     7,007 | 84.1378% |          99,986 |
| S3      |        9,055 |     7,826 | 86.4274% |          99,990 |
| Overall |       17,383 |    14,833 | 85.3349% |         199,976 |

Block C substantially improves recall over the individual A and B development baselines by recovering matches affected by character-level noise such as typos, punctuation variation, spacing differences, and minor name changes.

---

## A+B+C Union Blocking

Blocks A, B, and C are combined using union blocking. Duplicate candidate pairs are removed before applying the configured maximum-candidate cap.

The development evaluation uses the same recall-valid subset for all three blockers:

* Source 1: 5,000 rows
* Source 2: 50,000 rows
* Source 3: 50,000 rows
* All ground-truth target records for the sampled S1 entities are included.

### Union Recall

| Target  | True Matches | Retrieved |   Recall | Candidate Pairs After Dedup | Candidate Cap |
| ------- | -----------: | --------: | -------: | --------------------------: | ------------: |
| S2      |        8,328 |     7,126 | 85.5668% |                     150,838 |            50 |
| S3      |        9,055 |     8,005 | 88.4042% |                     151,413 |            50 |
| Overall |       17,383 |    15,131 | 87.0448% |                     302,251 |            50 |

The configured candidate cap of 50 candidates per S1 entity was not binding on this development subset. Therefore, the reported union recall reflects the A+B+C union rather than recall lost due to candidate capping.

The A+B+C union improves overall development recall to **87.0448%**, compared with the individual blocking strategies.

---

## Candidate Generation Output

The integrated A+B+C blocker produces the final M3-facing candidate-pair schema:

| Column                | Description                                   |
| --------------------- | --------------------------------------------- |
| `source1_entity_id`   | Entity ID from Source 1                       |
| `candidate_entity_id` | Candidate entity ID from Source 2 or Source 3 |

Candidate pairs are deduplicated across blockers. Source identity remains available through the candidate ID prefix (`S2-` or `S3-`).

Country is not used as a hard blocking filter.

---

## Notes

* These are blocking-stage recall measurements, not final matching precision or F0.5.
* The development subset is recall-valid: all ground-truth S2/S3 target records for the sampled S1 entities are included.
* Block A and Block B full-dataset results should not be directly compared numerically with the development-subset C and A+B+C results.
* The A+B+C union was evaluated on the same development subset for a valid comparison between the combined blockers.
* The current M2 implementation stops at Blocks A, B, and C.
* Additional blocking strategies are not being added at this stage; the current candidate output will be passed to the downstream matching/modeling stage for evaluation.
