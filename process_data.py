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

class UseCase(Enum):
    AUTOMOTIVE = "Automotive Industry: Self-Driving Cars"
    LIFE_SCIENCES = "Life Sciences and Healthcare"
    FINANCE = "Finance and Banking"
    RETAIL = "Retail and E-commerce"
    MANUFACTURING = "Manufacturing and Industrial Automation"
    ENTERTAINMENT = "Entertainment and Media"
    ENERGY = "Energy and Utilities"
    AGRICULTURE = "Agriculture and Food Production"
    EDUCATION = "Education and E-Learning"
    CYBERSECURITY = "Cybersecurity and Network Management"
    TRAVEL = "Travel and Hospitality"
    HR = "Human Resources and Recruitment"
    OTHER = "Other"

class ValueProposition(Enum):
    Quick_Setup = "Set up W&B in 5 minutes for machine learning pipelines with tracked and versioned models and data."
    Experiment_Tracking = "Track and visualize all machine learning experiments."
    Model_Registry = "Maintain a centralized hub of all models for seamless handoff to devops and deployment."
    Launch = "Package and run ML workloads to access powerful compute resources easily."
    Sweeps = "Hyperparameter tuning and model optimization."
    LLM_Debugging_Monitoring = "Tools for LLM debugging and monitoring including usage of OpenAI's GPT API."
    Artifacts = "Version datasets and models and track lineage."
    Visualization = "Visualize and query all kinds of data including rich media like images and videos."
    Reproducibility = "Capture metrics, hyperparameters, code version and save model checkpoints for reproducibility."
    Collaboration = "Facilitate project collaboration."
    Custom_Automations = "Configure custom automations for model CI/CD workflows."
    Auditability = "A System of Record for all your ML workstreams"
    Enterprise_Grade = "Enterprise-grade security and governance."
    Ease_Of_Use = "Easy to use and integrate with existing workflows."
    Productivity_and_Speed = "Productivity and faster iterations for ML teams."

class Article(BaseModel):
    use_case: str = Field(description="Use case covered in the report")
    value_props: str = Field(description="Value propositions covered in the report")
    application: str = Field(description="ML application covered in the report")
    integrations: str = Field(description="Integrations covered in the report")
    summary: str = Field(description="Concise summary of the report")
    quality: bool = Field(description="Boolean indicating if the report is comprehensive enough and of sufficient quality to share with potential W&B customer (True/False)")

run = wandb.init(project='content-suggestor')
artifact = run.use_artifact('wandbot/wandbot_public/fc-markdown-reports:v0', type='fully-connected-dataset')
artifact_dir = artifact.download()
df = pd.read_csv(f'{artifact_dir}/reports_final.csv')

system_prompt = \
"""Your are an expert assistant trained on W&B (Weights & Biases) and you are helping us annotate W&B reports.
You are given a report and you need to perform the following analysis.
1. Identify the use case reports covers from the below options:
```
class UseCase(Enum):
    AUTOMOTIVE = "Automotive Industry: Self-Driving Cars"
    LIFE_SCIENCES = "Life Sciences and Healthcare"
    FINANCE = "Finance and Banking"
    RETAIL = "Retail and E-commerce"
    MANUFACTURING = "Manufacturing and Industrial Automation"
    ENTERTAINMENT = "Entertainment and Media"
    ENERGY = "Energy and Utilities"
    AGRICULTURE = "Agriculture and Food Production"
    EDUCATION = "Education and E-Learning"
    CYBERSECURITY = "Cybersecurity and Network Management"
    TRAVEL = "Travel and Hospitality"
    HR = "Human Resources and Recruitment"
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
    Auditability: "A System of Record for all your ML workstreams"
    Enterprise_Grade: "Enterprise-grade security and governance."
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
6. Decide if the report is comprehensive enough and of sufficient quality to share with potential W&B customer (Yes/No).
Report: \n\n
"""

MODEL_NAME = 'gpt-4-1106-preview'
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

extracted = []
for i in tqdm(range(10)):
    user_prompt = df.loc[i, 'content'] + "\n\n"
    try:
        extracted.append(extract_with_backoff(user_prompt))
    except Exception as e:
        pass

resp_df = pd.DataFrame([x.dict() for x in extracted])
resp_df.to_csv('resp_df.csv', index=False)

run.log({"resp_df": wandb.Table(dataframe=resp_df)})

artifact = wandb.Artifact('processed_fc_articles', type='processed_data')
artifact.add_file('resp_df.csv')
run.log_artifact(artifact)

wandb.finish()