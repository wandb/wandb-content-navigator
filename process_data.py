import wandb
import pandas as pd
import os
import random
import openai
from pathlib import Path
from pprint import pprint
from getpass import getpass
from rich.markdown import Markdown
from pydantic import BaseModel, Field, validator
from typing_extensions import Literal
from enum import Enum
from typing import Optional, List
from tenacity import retry, stop_after_attempt, wait_random_exponential
import instructor
from openai import OpenAI
from tqdm.auto import tqdm

MAX_ROWS = 20 #-1

class Article(BaseModel):
    use_case: str = Field(description="Use case covered in the report")
    value_props: List[str] = Field(description="W&B value propositions covered in the report")
    application: List[str] = Field(description="ML application covered in the report")
    integrations: List[str] = Field(description="Integrations covered in the report")
    summary: str = Field(description="Concise summary of the report")
    audience: List[str] = Field(description="Target audience for the report")

run = wandb.init(project='content-suggestor')
artifact = run.use_artifact('wandbot/wandbot_public/fc-markdown-reports:latest', type='fully-connected-dataset')
artifact_dir = artifact.download()
# create a dataframe from jsonl file
df = pd.read_json(f'{artifact_dir}/reports_final.jsonl', lines=True)


print(df.head())
print("*"*100)

system_prompt = \
"""Your are an expert assistant trained on W&B (Weights & Biases) and you are helping us annotate W&B reports.
You are given a report and you need to perform the following analysis.
1. Identify the use case reports covers from the below options:
```
class UseCase(Enum):
    AUTOMOTIVE = "Autonomous Driving"
    ROBOTICS = "Robotics"
    PHARMA_LIFE_SCIENCES = "Pharma and Life Sciences"
    HEALTHCARE = "Healthcare Payer and Provider"
    FINANCE = "Finance and Banking"
    RETAIL = "Retail and E-commerce"
    ENTERTAINMENT = "Entertainment and Media"
    HIGH_TECH = "High Tech and Semis"
    OTHER = "Other"
```
2. List W&B value propositions covered in the report using below options:
```
class ValueProposition(Enum):
    Quick_Setup: "Set up W&B in 5 minutes for machine learning pipelines with tracked and versioned models and data."
    Experiment_Tracking: "Track and visualize all machine learning experiments."
    Model_Registry: "Maintain a centralized hub of all models for seamless handoff to devops and deployment."
    Launch: "Package and run ML workloads to access powerful compute resources easily."
    Sweeps: "Hyperparameter tuning and model optimization."
    LLM_Debugging_Monitoring: "Tools for LLM debugging and monitoring including usage of OpenAI's GPT API."
    Artifacts: "Version datasets and models and track lineage."
    Visualization: "Visualize and query all kinds of data including rich media like images and videos."
    Reproducibility: "Capture metrics, hyperparameters, code version and save model checkpoints for reproducibility."
    Collaboration: "Facilitate project collaboration."
    Custom_Automations: "Configure custom automations for model CI/CD workflows."
    Auditability: "A System of Record for all your ML workstreams enabling enterprise-wide visibility and governance."
    Security: "Enterprise-grade security, RBAC and SOC2 certification."
    Ease_Of_Use: "Easy to use and integrate with existing workflows."
    Productivity_and_Speed: "Productivity and faster iterations for ML teams."
```
3. Determine ML application covered in the report based on below options:
```
Literal["Computer Vision", "Natural Language Processing", "Tabular Data", "Reinforcement Learning", "Generative Models", "Anomaly Detection", "Recommendation Systems", "Other", "None"]

```
4. Determine integration covered in the report based on below options:
```
Literal["Catalyst", "Dagster", "Databricks", "DeepChecks", "DeepChem", "Docker", "Farama Gymnasium", "Fastai", "Hugging Face Transformers", "Hugging Face Diffusers", "Hugging Face Autotrain", "Hugging Face Accelerate", "Hydra", "Keras", "Kubeflow Pipelines (kfp)", "LangChain", "LightGBM", "Metaflow", "MMDetection", "MMF", "MosaicML Composer", "OpenAI API", "OpenAI Fine-Tuning", "OpenAI Gym", "PaddleDetection", "PaddleOCR", "Prodigy", "PyTorch", "PyTorch Geometric", "PyTorch Ignite", "PyTorch Lightning", "Ray Tune", "SageMaker", "Scikit-Learn", "Simple Transformers", "Skorch", "spaCy", "Stable Baselines 3", "TensorBoard", "TensorFlow", "Julia", "XGBoost", "YOLOv5", "Ultralytics", "YOLOX", "Other", "None"]
```
5. Provide a concise summary of the report.
6. Determine target audience for the report based on below options:
```
Literal["Experienced ML Engineers", "ML Managers", "ML Ops", "Beginner"]
```
Report: \n\n
"""

wandb.config.update({"system_prompt": system_prompt})

MODEL_NAME = 'gpt-4-0613'
API_KEY = os.environ.get("OPENAI_API_KEY")
client = OpenAI(api_key = API_KEY)
client = instructor.patch(client)

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def extract_with_backoff(user_prompt, **kwargs):
    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
        response_model=Article,
    )
    return resp

if MAX_ROWS == -1: MAX_ROWS = len(df)

extracted = []
rows = []
for i in tqdm(range(MAX_ROWS)):
    try:
        user_prompt = df.loc[i, 'content'] + "\n\n"
        extracted.append(extract_with_backoff(user_prompt))
        rows.append(i)
    except Exception as e:
        pass

resp_df = pd.DataFrame([x.dict() for x in extracted])
resp_df.to_csv('resp_df.csv', index=False)
run.log({"resp_df": wandb.Table(dataframe=resp_df)})

# merge with df on extracted rows
prev_df = df.loc[rows, :].reset_index(drop=True)
merged_df = pd.merge(prev_df, resp_df, left_index=True, right_index=True)
merged_df.to_csv('merged_df.csv', index=False)
run.log({"reports_metadata": wandb.Table(dataframe=merged_df)})

artifact = wandb.Artifact('processed_fc_articles', type='processed_data')
artifact.add_file('resp_df.csv')
artifact.add_file('merged_df.csv')
run.log_artifact(artifact)

wandb.finish()