# LLaMA 3.1 8B Fine-tuning on Spider Text-to-SQL (QLoRA)

fine-tuned LLaMA 3.1 8B on the Spider Text-to-SQL dataset using QLoRA. the goal was to get the model to generate SQL queries from natural language questions.

## what's in here

- 4-bit quantization via bitsandbytes (QLoRA)
- LoRA adapters with r=16, alpha=32
- trained on Spider dataset (7000 train / 1034 val)
- MLflow integration for experiment tracking
- adapter pushed to HuggingFace after training

## stack

- `transformers` + `peft` + `trl` (SFTTrainer)
- `bitsandbytes==0.46.1`
- `mlflow` for tracking loss/hyperparams
- Kaggle T4 GPU

## prompt format
```
Task : Generate a SQL query based on the question and schema below.
Schema
{db_id}
Question
{question}
SQL
{query}
```
## training config

| param | value |
|---|---|
| base model | meta-llama/Meta-Llama-3.1-8B |
| epochs | 3 | changed it to 1 cuz it was causing overfitting
| batch size | 4 |
| grad accum steps | 4 |
| learning rate | 1e-4 |
| lora r | 16 |
| lora alpha | 32 |
| lora dropout | 0.1 |  increase the droput rate from 0.05 to 0.1
| max seq length | 512 |

## results

training loss went from ~0.70 → ~0.43 over 800+ steps. val loss started diverging a bit after step 300 (expected with small LoRA on a complex task).

## Mlflow metrics
- Loss
  <img width="1365" height="781" alt="image" src="https://github.com/user-attachments/assets/d752cad7-9ace-448f-85b8-0b5e7c82211f" />
- Mean token Accuracy
  <img width="1365" height="781" alt="image" src="https://github.com/user-attachments/assets/6a7f10d8-22a3-4f8e-9775-0f7fb1d05c1a" />

## adapter

pushed to `Awan8754/llama3.1-text2sql-adapter` on HuggingFace along with MLflow run artifacts.
