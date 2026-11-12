# Supermarket Sales SLM

This project fine-tunes a small language model (SmolLM2-135M-Instruct) to answer retail analytics questions
(revenue by branch, product line, payment method and so on, plus tax/total calculations) using a
15,000-row supermarket sales table.

Dataset on Hugging Face: https://github.com/pasupuletirohith442007-ui/supermarket_slm

## Repository layout
```
supermarket_sales_15000.csv   source sales table
data/train.csv                5,180 Q&A rows  (training)
data/validation.csv             790 Q&A rows  (checkpoint selection)
data/test.csv                   790 Q&A rows  (final evaluation)
data/DATASET_CARD.md          dataset documentation
scripts/build_dataset.py      CSV -> train/validation/test splits
scripts/train.py              fine-tunes on train, keeps the best checkpoint by validation loss
scripts/evaluate.py           test-set accuracy (fine-tuned vs base model)
scripts/upload_dataset.py     pushes the splits to the Hugging Face Hub
```
Columns in each split: `id, question, answer, category, fact_id`.

## Quick start
```bash
python -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu   # or the CUDA build
pip install -r requirements.txt

python scripts/build_dataset.py        # regenerates data/*.csv (deterministic, seed 42)
python scripts/train.py --epochs 3     # GPU recommended
python scripts/evaluate.py --base      # numeric accuracy on the test split
```
`train.py` can also mix in the research-paper text (`design_thinking_merged.jsonl`). Those papers are copyrighted
and are not in this repository. Without them, training uses only the Q&A data.

## How the splits are made
- **Aggregate facts** (totals, rankings, breakdowns, cross-tabs, concepts): each fact has 4 paraphrased questions,
  assigned 2 to train, 1 to validation and 1 to test. Validation and test check whether the model recalls the fact
  from unseen wording.
- **Calculations** (quantity × unit price → COGS, 5% tax, total): 6,000 invoices are split 80/10/10 by invoice ID,
  so no transaction appears in two splits.

## Notes
- Rows 1–1000 of the source CSV match the public Kaggle "Supermarket Sales" dataset; the other 14,000 rows are synthetic.
- `gross income` always equals the 5% tax, so the gross margin is a constant 4.76%.
