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
| epochs | 1 |
| batch size | 4 |
| grad accum steps | 4 |
| learning rate | 1e-4 |
| lora r | 16 |
| lora alpha | 32 |
| lora dropout | 0.1 |
| max seq length | 512 |

> epochs reduced from 3 → 1 and dropout increased from 0.05 → 0.1 to address overfitting.
## results

training loss went from ~0.70 → ~0.43 over 800+ steps. val loss started diverging a bit after step 300 (expected with small LoRA on a complex task).

## Mlflow metrics
 ### Loss
  <img width="566" height="507" alt="image" src="https://github.com/user-attachments/assets/e6ff227d-ea83-410a-9591-e4eb013a8b96" />

 ### Mean token Accuracy 
  <img width="1365" height="781" alt="image" src="https://github.com/user-attachments/assets/6a7f10d8-22a3-4f8e-9775-0f7fb1d05c1a" />

## results
initially overfitted hard — val loss diverged early. fixed by:
- increased dropout from 0.05 → 0.1
- reduced learning rate from 2e-4 → 1e-4

got better results after that but val loss still crept up after step 300 — expected with a small LoRA rank on a complex task like Text-to-SQL.

training loss: 1.5 → 0.5
mean token accuracy: 0.68 → 0.87

## adapter

pushed to HuggingFace along with MLflow run artifacts.
See it on : https://huggingface.co/Awan8754/llama_nl_to_sql/tree/main
