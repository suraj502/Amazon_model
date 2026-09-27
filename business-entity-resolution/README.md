# Machine Learning Business Entity Resolution Pipeline

Machine-learning-based entity resolution pipeline for the Amazon ML Challenge 2026.

The system matches Source 1 business entities against corresponding records from Source 2 and Source 3 using normalization, candidate blocking, pairwise similarity features, XGBoost classification, source-specific thresholds, and target consistency handling.

---

## 1. System Overview

### Task

For every Source 1 reference entity, identify all corresponding Source 2 and Source 3 records.

An entity may have:

- no matching records
- one matching record
- multiple matching records

### Core Pipeline

1. Text normalization
2. Country-aware candidate generation
3. Pairwise feature extraction
4. XGBoost binary classification
5. Source-specific probability thresholding
6. Source-aware target consistency
7. Singleton handling
8. Submission generation

### Final Model

The model used to generate the submitted test predictions is an XGBoost binary classifier.

Main configuration:

- `n_estimators = 500`
- `learning_rate = 0.035`
- `max_depth = 7`
- `subsample = 0.85`
- `colsample_bytree = 0.85`
- `tree_method = hist`
- random seed = `42`

The model uses 36 engineered pairwise features.

---

## 2. Directory Structure

```text
code/business_entity_resolution/
├── src/
│   ├── normalization.py      # Text normalization
│   ├── blocking.py           # Candidate generation
│   ├── features.py           # Pairwise feature extraction
│   ├── model.py              # XGBoost model wrapper
│   ├── thresholding.py       # Thresholding and consistency logic
│   ├── evaluation.py         # Validation/evaluation utilities
│   ├── data_loader.py        # Competition data loading
│   ├── output.py             # Submission output generation
│   ├── inference.py          # Test inference engine
│   └── main.py               # CLI entry point
├── requirements.txt
└── README.md

---

3. Installation

Use a compatible Python environment and install the dependencies:

pip install -r requirements.txt

Required libraries include:

NumPy
SciPy
pandas
scikit-learn
LightGBM
RapidFuzz
text-unidecode
joblib
XGBoost

XGBoost is required because the final saved model is an XGBoost classifier.

4. Input Data

The pipeline expects the competition data in the prescribed directory structure.

The relevant source files contain:

entity_id
business_name
business_address
country

The implementation does not depend on a fixed list of countries.

Country information is used during candidate generation while retaining support for countries appearing only in test data.

5. Normalization

Business names and addresses are normalized before candidate generation and feature extraction.

The normalization stage handles variations such as:

case differences
punctuation
whitespace
business-name formatting
common textual variations
Unicode/transliteration differences
address formatting differences
numeric address information

The same normalized representations are used consistently throughout the matching pipeline.

6. Candidate Generation

A complete Cartesian comparison between Source 1 and Sources 2/3 would be computationally impractical.

The pipeline therefore generates candidate pairs using country-aware inverted indexes and multiple normalized blocking signals.

Candidate generation uses combinations of normalized name and address information.

The final candidate set is written to:

output/candidate_pairs.tsv

The final test inference produced:

Source 1 entities: 1,732,544
Candidate pairs: 34,461,015

The locally measured candidate recall on validation was:

0.9527918049343529

Candidate recall represents the proportion of true validation matches that survived the blocking stage.

7. Feature Engineering

Each candidate pair is represented using 36 engineered features.

The features capture information from:

Name similarity

Examples include:

normalized equality
edit-distance similarity
Jaro-Winkler similarity
token similarity
token overlap
character-level similarity
name-length relationships
Address similarity

Examples include:

normalized equality
edit-distance similarity
token similarity
address token overlap
numeric agreement
numeric conflict
address-length relationships
Combined information

Additional features capture:

name/address interactions
combined similarity signals
source-specific information
blocking-related information

These features allow the classifier to distinguish strong matches from approximate or ambiguous candidates.

8. XGBoost Matching Model

The final matcher is an XGBoost binary classifier.

The model predicts the probability that a candidate Source 1 / Source 2 or Source 3 pair represents the same business entity.

The production configuration uses histogram-based tree construction.

The model was configured for CUDA acceleration where available. During the final local test inference, CUDA was unavailable on the execution machine, so XGBoost automatically fell back to CPU execution.

This fallback did not change the generated output format.

9. Thresholding

Different target sources use different decision thresholds.

Final thresholds:

Source 2: 0.70
Source 3: 0.75

A candidate is retained only when its predicted probability satisfies the appropriate source-specific threshold.

The thresholds were selected using the validation procedure used during model development.

10. Target Consistency

After probability scoring, the pipeline applies source-aware target assignment logic.

The final inference uses:

Greedy highest-probability target assignment with source-aware 1-to-1 target consistency.

This prevents the same target entity from being assigned to multiple Source 1 entities when the consistency constraint applies.

11. Singleton Handling

The system does not force every Source 1 entity to have a match.

If no candidate survives candidate generation, scoring, thresholding, and consistency handling, the Source 1 entity is represented as a singleton with an empty match list.

This is necessary because the competition explicitly allows unmatched Source 1 entities.

12. Validation Results

The validation results associated with the final model configuration are:

Metric	Result
Macro F0.5	0.9407646876901236
Precision	0.9913619728990142
Recall	0.8676670033184245
Macro F1	0.9040051616213294
Candidate Recall	0.9527918049343529
Singleton Accuracy	0.9796672828096118
False Positives	262
False Negatives	4586

These are local validation measurements.

They are not the final competition leaderboard score.

13. Final Test Inference

The complete competition test set was processed.

Final inference statistics:

Quantity	Result
Source 1 entities	1,732,544
Predicted singletons	121,859
Entities with matches	1,610,685
Total predicted matches	5,545,532
Source 2 matches	2,739,660
Source 3 matches	2,805,872
Candidate pairs	34,461,015
Runtime	16,788.4 seconds / 279.81 minutes

Country-level Source 1 counts:

Country	Source 1 entities
France	259,452
US	663,106
India	809,986
14. Output Files

The final pipeline generates:

output/matching_results.tsv

The final entity-resolution submission.

It contains exactly one row for every Source 1 test entity.

output/candidate_pairs.tsv

The final candidate set immediately before model scoring.

Every final predicted match is required to be present in this candidate set.

15. Validation

The official competition validator was executed with ID checking enabled.

Command:

python utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test \
    --check-ids

Validation result:

required S1 entities: 1732544
valid S2/S3 match IDs: 9969589
matching_results.tsv: 1732544 rows
candidate_pairs.tsv: 1732544 rows

PASS — no blocking issues found. Safe to submit.
16. Reproduction

Test inference can be launched using:

python code/business_entity_resolution/src/main.py \
    --test-dir dataset/test \
    --model-path models/final_entity_matcher.joblib \
    --meta-path models/model_metadata.json \
    --output-dir output

The command requires the trained model artifact and metadata referenced by the command.

The already-generated competition output is included separately under output/.

17. Compliance

The solution uses the supplied competition data and does not perform external entity lookups.

No external:

business databases
geocoding services
Google Maps lookups
Wikidata queries
OpenStreetMap lookups
external business registries
external entity-resolution APIs

were used.

The solution supports countries dynamically rather than hard-coding the test country list.

18. Final Notes

The validation metrics reported in this README are development/validation measurements only.

The final test data has no public ground truth, so the actual test performance can only be determined by the competition evaluator.

The final generated files are:

output/
├── matching_results.tsv
└── candidate_pairs.tsv

The submitted code is intended to document and reproduce the entity-resolution methodology used to generate those outputs.

