# Normalization EDA Report

Streaming EDA over the raw train/test source files. Chunk size: 50,000 rows.

**Raw data was not modified. Case-folding and other temporary representations are used only for analysis.**

## 1. Source overview

| Source | Rows | Columns | Estimated non-Latin rate |
|---|---:|---:|---:|
| `train_source1` | 2,206,821 | 4 | 0.03% |
| `train_source2` | 5,034,616 | 4 | 21.98% |
| `train_source3` | 5,285,603 | 4 | 18.72% |
| `test_source1` | 1,732,544 | 4 | 5.86% |
| `test_source2` | 4,887,273 | 4 | 29.68% |
| `test_source3` | 5,082,316 | 4 | 25.91% |

## 2. Missing/null rate

| Source | Column | Missing records | Missing rate |
|---|---|---:|---:|
| `train_source1` | `entity_id` | 0 | 0.00% |
| `train_source1` | `business_name` | 0 | 0.00% |
| `train_source1` | `business_address` | 0 | 0.00% |
| `train_source1` | `country` | 0 | 0.00% |
| `train_source2` | `entity_id` | 0 | 0.00% |
| `train_source2` | `business_name` | 6 | 0.00% |
| `train_source2` | `business_address` | 168,967 | 3.36% |
| `train_source2` | `country` | 0 | 0.00% |
| `train_source3` | `entity_id` | 0 | 0.00% |
| `train_source3` | `business_name` | 18 | 0.00% |
| `train_source3` | `business_address` | 175,916 | 3.33% |
| `train_source3` | `country` | 0 | 0.00% |
| `test_source1` | `entity_id` | 0 | 0.00% |
| `test_source1` | `business_name` | 0 | 0.00% |
| `test_source1` | `business_address` | 0 | 0.00% |
| `test_source1` | `country` | 0 | 0.00% |
| `test_source2` | `entity_id` | 0 | 0.00% |
| `test_source2` | `business_name` | 49 | 0.00% |
| `test_source2` | `business_address` | 129,408 | 2.65% |
| `test_source2` | `country` | 0 | 0.00% |
| `test_source3` | `entity_id` | 0 | 0.00% |
| `test_source3` | `business_name` | 61 | 0.00% |
| `test_source3` | `business_address` | 136,098 | 2.68% |
| `test_source3` | `country` | 0 | 0.00% |

## 3. Country distribution

### Train

**train_source1**

| Country | Count |
|---|---:|
| `US` | 1,323,633 |
| `India` | 883,188 |

**train_source2**

| Country | Count |
|---|---:|
| `US` | 3,016,817 |
| `India` | 2,017,799 |

**train_source3**

| Country | Count |
|---|---:|
| `US` | 3,170,056 |
| `India` | 2,115,547 |

### Test

**test_source1**

| Country | Count |
|---|---:|
| `India` | 809,986 |
| `US` | 663,106 |
| `France` | 259,452 |

**test_source2**

| Country | Count |
|---|---:|
| `India` | 2,312,565 |
| `US` | 1,871,330 |
| `France` | 703,378 |

**test_source3**

| Country | Count |
|---|---:|
| `India` | 2,405,000 |
| `US` | 1,945,701 |
| `France` | 731,615 |

### Country shift

Countries present in test but not training: `France`

Country is treated as a soft signal, not a hard filter.

## 4. Ground-truth singleton estimate

- Ground-truth rows: **2,206,821**
- Singleton rows: **119,157**
- Estimated singleton rate: **5.40%**

## 5. Name/address length distributions

| Source | Field | Min | Median | Mean | P95 | Max |
|---|---|---:|---:|---:|---:|---:|
| `train_source1` | `business_name` | 3 | 24.0 | 24.0 | 37.0 | 105 |
| `train_source1` | `business_address` | 11 | 41.0 | 52.1 | 103.0 | 256 |
| `train_source2` | `business_name` | 2 | 25.0 | 25.1 | 40.0 | 104 |
| `train_source2` | `business_address` | 8 | 37.0 | 47.8 | 97.0 | 249 |
| `train_source3` | `business_name` | 2 | 25.0 | 25.2 | 42.0 | 123 |
| `train_source3` | `business_address` | 2 | 42.0 | 48.3 | 92.0 | 240 |
| `test_source1` | `business_name` | 3 | 24.0 | 23.8 | 36.0 | 92 |
| `test_source1` | `business_address` | 11 | 50.0 | 57.2 | 105.0 | 268 |
| `test_source2` | `business_name` | 2 | 25.0 | 25.7 | 42.0 | 102 |
| `test_source2` | `business_address` | 5 | 43.0 | 51.8 | 99.0 | 269 |
| `test_source3` | `business_name` | 2 | 25.0 | 25.7 | 42.0 | 103 |
| `test_source3` | `business_address` | 5 | 44.0 | 50.1 | 95.0 | 267 |

## 6. Within-source duplicate detection

| Source | Field | Duplicate-record rate |
|---|---|---:|
| `train_source1` | `business_name` | 10.38% |
| `train_source1` | `business_address` | 0.30% |
| `train_source2` | `business_name` | 1.66% |
| `train_source2` | `business_address` | 0.29% |
| `train_source3` | `business_name` | 1.48% |
| `train_source3` | `business_address` | 0.23% |
| `test_source1` | `business_name` | 10.21% |
| `test_source1` | `business_address` | 0.37% |
| `test_source2` | `business_name` | 1.43% |
| `test_source2` | `business_address` | 0.31% |
| `test_source3` | `business_name` | 1.28% |
| `test_source3` | `business_address` | 0.26% |

## 7. Transliteration / non-Latin estimate

A record is flagged when its name or address contains at least one non-ASCII character. This is an approximation, not a linguistic classifier.

| Source | Estimated non-Latin rate |
|---|---:|
| `train_source1` | 0.03% |
| `train_source2` | 21.98% |
| `train_source3` | 18.72% |
| `test_source1` | 5.86% |
| `test_source2` | 29.68% |
| `test_source3` | 25.91% |

## 8. Normalization decisions and information-loss risks

Original name and address values must always remain recoverable.

- Case/punctuation/whitespace normalization can collapse distinct spellings.
- Legal-suffix removal can create collisions between different businesses.
- Token normalization can remove ordering information.
- Alphanumeric normalization can remove meaningful separators or symbols.
- Transliteration can map different scripts or spellings to the same Latin representation.
- Address parsing can produce ambiguous or incomplete components.
- Missing address components must not generate artificial tokens.
- Landmark references should be flagged but never resolved through external identity lookup.

Member 2 should check these information-loss points against blocking recall.

## 9. Gate status

EDA completed. Blocking top_k and candidate caps should be tuned from the distributions in this report rather than guessed.
