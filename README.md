# Automated Financial Customer Complaint Classification
**Author:** Shorya Jaiswal  
**Task:** NLP multiclass classification of public CFPB complaint narratives.

## Final deliverables
- [Bias and performance audit](docs/Bias_Performance_Audit.pdf)
- [Explainability analysis](docs/Explainability_Analysis.pdf)
- [Updated one-page model card](docs/Model_Card.pdf)
- [Standalone chart generation code](visuals/model_card_visuals.py)

Held-out accuracy: **88.90%**; macro-F1: **0.6834**; weighted-F1: **0.8842**.
The informal-short style subgroup has only five examples, so that comparison is inconclusive.
Triage thresholds in the model card are proposals, not implemented or validated features.

![Model card](docs/Model_Card.png)

## Start here
1. Install Python 3.11 or 3.12.
2. Extract this folder and open a terminal inside it.
3. Run `python -m pip install -r requirements.txt -r environment.txt` to match the saved model's dependencies.
4. Open `Financial_Complaints.ipynb` in Jupyter (`jupyter lab`) or follow the commands below.

```bash
python src/project.py train --csv data/complaints.csv
python src/project.py predict --text "My mortgage servicer applied my payment incorrectly and charged a late fee."
```
Training replaces files in `outputs/`. The included model can be used immediately after installing requirements. Only load model files you trust: joblib uses Python pickle. `environment.txt` records the versions used for the delivered run.

For your own CFPB or Kaggle CSV, pass its path with `--csv`. Required columns: `Consumer complaint narrative`, `Product`; capitalization/spacing variants are accepted. Optional: `Complaint ID`, `Date received`. ZIP files containing a single CSV are accepted by pandas. Increase `--max-rows 100000` for a larger experiment; default 30,000 eligible records are sampled uniformly from the supplied file using bounded-memory random priorities. Downloading the full CSV is separate from sampling it. A supplied API snapshot is not a uniform sample of the full database.

## Files
- `Financial_Complaints.ipynb`: teaching notebook, commands, metrics and explanations.
- `REPORT.md`: complete methodology, audit design, limitations and references.
- `outputs/RESULTS.md`: measured results from the executed experiment.
- `src/project.py`: preprocessing, model selection, evaluation, explanations, prediction, human audit.
- `src/download_data.py`: repeatable official API download with provenance.
- `data/complaints.csv`: actual public narrative snapshot used for the delivered run.
- `data/provenance.json`: retrieval URL, timestamp, selection caveat.
- `outputs/model.joblib`: fitted winning TF-IDF pipeline.
- `outputs/`: metrics, charts, audit, errors, split hashes and annotation sheet.
- `VIVA.md`: explanation and likely viva questions.

## Evaluation rules
Vocabulary/IDF and models are fitted on training data only. Hyperparameters and winner are selected by validation macro-F1. Only the selected model receives final test evaluation. Exact normalized duplicate narratives are removed before splitting; contradictory duplicates are excluded. Near-duplicate templates may remain. Classes with fewer than 40 eligible unique narratives in the supplied sample are excluded and recorded. No company, issue, product text, or response metadata is included as an input feature.

## Complete the validated style audit
The automated output includes length groups and explicitly heuristic style proxies. A short text is not necessarily informal. For a defensible final claim about formality:
1. Two readers independently label `outputs/human_audit_template.csv`; avoid viewing predictions.
2. Use `formal`, `neutral`, or `informal` in both rater columns. Adjudicate disagreements in `adjudicated_formality`.
3. Use `detailed` when the narrative includes at least three concrete facts (e.g. timeline, action, consequence, attempted remedy), otherwise `limited` in `detail_level`.
4. Run `python src/project.py human-audit` to measure agreement and classification performance on annotated groups.
Do not invent annotations or present heuristic labels as ground truth. More short informal examples may need purposeful collection and separate evaluation.

## Reproducibility
Random seed 42; 70/15/15 stratified split; input SHA-256 and split hashes retained. Re-running an API request later can return different rows. Re-running against the included CSV is reproducible with the recorded environment, subject to numerical platform differences. Runtime varies by machine; Random Forest is typically the slowest candidate.
