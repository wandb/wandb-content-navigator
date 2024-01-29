import wandb
import pandas as pd
import os

from datetime import datetime
from pathlib import Path
from pprint import pprint
from getpass import getpass
from rich.markdown import Markdown
from pydantic import BaseModel, Field, validator
from typing_extensions import Literal
from enum import Enum
from typing import Optional, List, Dict, Tuple
from tqdm.auto import tqdm

import asyncio

from tenacity import retry, stop_after_attempt, wait_random_exponential
import instructor
from openai import OpenAI, AsyncOpenAI
import tiktoken
import nltk
# import spacy
from pydantic import BaseModel, Field, field_validator

from dotenv import load_dotenv  
load_dotenv()

MAX_ROWS = 20 #-1

class Article(BaseModel):
    use_case: str = Field(description="Use case covered in the report")
    value_props: List[str] = Field(description="W&B value propositions covered in the report")
    application: List[str] = Field(description="ML application covered in the report")
    integrations: List[str] = Field(description="Integrations covered in the report")
    summary: str = Field(description="Concise summary of the report")
    audience: List[str] = Field(description="Target audience for the report")


def download_reports_data(output_dir: str = "reports_data", artifact_version: str="latest"):
    api = wandb.Api()
    artifact_path = f"wandbot/wandbot_public/raw_dataset:{artifact_version}"
    artifact = api.artifact(artifact_path).download(output_dir)
    print(f"Artifact content downloaded to: {artifact}")
    return output_dir

enc = tiktoken.get_encoding("cl100k_base")

reports_data_dir = "reports_data"
# reports_data_dir = download_reports_data()
df = pd.read_json(f'{reports_data_dir}/docstore_fc_reports/documents.jsonl', lines=True)

df["content_length"] = df["page_content"].apply(len)

c  = []
for d in df["page_content"]:
    c.append(len(enc.encode(d, allowed_special={'<|endoftext|>'})))
df["token_count"] = c

# extract "source" from metadata in df
df["source"] = df["metadata"].apply(lambda x: x["source"])
print(df.nlargest(10, 'token_count')['token_count'])

# run = wandb.init(project='content-suggestor')
# artifact = run.use_artifact('wandbot/wandbot_public/fc-markdown-reports:latest', type='fully-connected-dataset')
# artifact_dir = artifact.download()
# # create a dataframe from jsonl file
# df = pd.read_json(f'{artifact_dir}/reports_final.jsonl', lines=True)

print(df.columns)
print(df.head())
print("*"*100)



# system_prompt = \
# """Your are an expert assistant trained on W&B (Weights & Biases) and you are helping us annotate W&B reports.
# You are given a report and you need to perform the following analysis.
# 1. Identify the use case reports covers from the below options:
# ```
# class UseCase(Enum):
#     AUTOMOTIVE = "Autonomous Driving"
#     ROBOTICS = "Robotics"
#     PHARMA_LIFE_SCIENCES = "Pharma and Life Sciences"
#     HEALTHCARE = "Healthcare Payer and Provider"
#     FINANCE = "Finance and Banking"
#     RETAIL = "Retail and E-commerce"
#     ENTERTAINMENT = "Entertainment and Media"
#     HIGH_TECH = "High Tech and Semis"
#     OTHER = "Other"
# ```
# 2. List W&B value propositions covered in the report using below options:
# ```
# class ValueProposition(Enum):
#     Quick_Setup: "Set up W&B in 5 minutes for machine learning pipelines with tracked and versioned models and data."
#     Experiment_Tracking: "Track and visualize all machine learning experiments."
#     Model_Registry: "Maintain a centralized hub of all models for seamless handoff to devops and deployment."
#     Launch: "Package and run ML workloads to access powerful compute resources easily."
#     Sweeps: "Hyperparameter tuning and model optimization."
#     LLM_Debugging_Monitoring: "Tools for LLM debugging and monitoring including usage of OpenAI's GPT API."
#     Artifacts: "Version datasets and models and track lineage."
#     Visualization: "Visualize and query all kinds of data including rich media like images and videos."
#     Reproducibility: "Capture metrics, hyperparameters, code version and save model checkpoints for reproducibility."
#     Collaboration: "Facilitate project collaboration."
#     Custom_Automations: "Configure custom automations for model CI/CD workflows."
#     Auditability: "A System of Record for all your ML workstreams enabling enterprise-wide visibility and governance."
#     Security: "Enterprise-grade security, RBAC and SOC2 certification."
#     Ease_Of_Use: "Easy to use and integrate with existing workflows."
#     Productivity_and_Speed: "Productivity and faster iterations for ML teams."
# ```
# 3. Determine ML application covered in the report based on below options:
# ```
# Literal["Computer Vision", "Natural Language Processing", "Tabular Data", "Reinforcement Learning", "Generative Models", "Anomaly Detection", "Recommendation Systems", "Other", "None"]

# ```
# 4. Determine integration covered in the report based on below options:
# ```
# Literal["Catalyst", "Dagster", "Databricks", "DeepChecks", "DeepChem", "Docker", "Farama Gymnasium", "Fastai", "Hugging Face Transformers", "Hugging Face Diffusers", "Hugging Face Autotrain", "Hugging Face Accelerate", "Hydra", "Keras", "Kubeflow Pipelines (kfp)", "LangChain", "LightGBM", "Metaflow", "MMDetection", "MMF", "MosaicML Composer", "OpenAI API", "OpenAI Fine-Tuning", "OpenAI Gym", "PaddleDetection", "PaddleOCR", "Prodigy", "PyTorch", "PyTorch Geometric", "PyTorch Ignite", "PyTorch Lightning", "Ray Tune", "SageMaker", "Scikit-Learn", "Simple Transformers", "Skorch", "spaCy", "Stable Baselines 3", "TensorBoard", "TensorFlow", "Julia", "XGBoost", "YOLOv5", "Ultralytics", "YOLOX", "Other", "None"]
# ```
# 5. Provide a concise summary of the report.
# 6. Determine target audience for the report based on below options:
# ```
# Literal["Experienced ML Engineers", "ML Managers", "ML Ops", "Beginner"]
# ```
# Report: \n\n
# """

# wandb.config.update({"system_prompt": system_prompt})

class InitialSummary(BaseModel):
    """
    This is an initial summary which should be long ( 4-5 sentences, ~80 words)
    yet highly non-specific, containing little information beyond the entities marked as missing.
    Use overly verbose languages and fillers (Eg. This article discusses) to reach ~80 words.
    """

    summary: str = Field(
        ...,
        description="This is a summary of the article provided which is overly verbose and uses \
fillers. It should be roughly 80 words in length",
    )

    @field_validator("summary")
    def min_length(cls, v: str):
        tokens = nltk.word_tokenize(v)
        num_tokens = len(tokens)
        if num_tokens < 60:
            raise ValueError(
                "The current summary is too short. Please make sure that you generate a new summary \
that is around 80 words long."
            )
        return v


ENTITY_DEFINITION = """An Entity is a real-world object that's assigned a name - for example, a person, country \
a product, book title, a machine learning or AI topic, a Weight & Biases feature, an industry, a business use-case etc."""


class Entity(BaseModel):
    f"""{ENTITY_DEFINITION}"""
    index: str = Field(..., description="Monotonically increasing ID for the entity")
    entity: str = Field(description=f"{ENTITY_DEFINITION}")


class CountEntities(BaseModel):
    """Count the number of entities in a given `text`
    """
    chain_of_thought: str = Field(
        ...,
        description=f"Think step by step about how many entities are in the given `text`. {ENTITY_DEFINITION}",
    )
    entities: List[Entity] = Field(
        ...,
        description="This is a list of entities found in the given `text`.",
    )
    num_entities: int = Field(
        ...,
        description="This is the integer count of `entities` in the `text`, return `0` \
if no entities are found in the `text`.",
    )


class RewrittenSummary(BaseModel):
    f"""
    This is a new, denser summary called `new_summary` of identical length to `previous_summary` which covers every Entity
    and detail from the `previous_summary` plus the Missing Entities given.

    Guidelines:
    - Make every word count : Rewrite the `previous summary` to improve flow and make space for additional entities
    - Never drop entities from the `previous summary`. If space cannot be made, add fewer new entities.
    - The `new_summary` should be highly dense and concise yet self-contained, eg., easily understood without the Article.
    - Make space with fusion, compression, and removal of uninformative phrases like "the article discusses"

    {ENTITY_DEFINITION}
    """

    new_summary: str = Field(
        ...,
        description="This is a new, denser summary of identical length which covers every Entity \
and detail from the `previous_summary` plus the Missing Entities. It should have the same length ( ~ 80 words ) \
as the `previous_summary` and should be easily understood without the Article",
    )
    entities: List[Entity] = Field(
        ...,
        description="This is a list of entities found in the `new_summary`.",
    )
    absent_entities: List[str] = Field(
        default_factory=list,
        description="this is a list of Entities found absent from the `new_summary` that were present in the `previous_summary`",
    )
    missing_entities: List[str] = Field(
        default_factory=list,
        description="This is a list of 1-3 informative Entities from the Article that are missing \
from this `new_summary`.",
    )

    @field_validator("new_summary")
    def min_length(cls, v: str):
        tokens = nltk.word_tokenize(v) 
        num_tokens = len(tokens)
        if num_tokens < 60:
            raise ValueError(
                "The current summary is too short. Please make sure that you generate a new summary \
    that is around 80 words long."
            )
        return v
    
    @field_validator("new_summary")
    def min_entity_density(cls, v: str):
        tokens = nltk.word_tokenize(v)
        num_tokens = len(tokens)

        # Extract Entities
        entities: CountEntities = get_entity_count(v)
        num_entities = entities.num_entities
        density = num_entities / num_tokens
        print(f"Entity chain of thought: {entities.chain_of_thought}")
        print(f"Entities: {entities.entities}")
        print(f"{num_entities} entities found in summary, entity density: {density}")

        if density < 0.08:
            raise ValueError(
                f"The `new_summary`: {v}, has too few entities. Please regenerate a `new_summary` with more \
new entities added to it."
            )
        return v

    @field_validator("missing_entities")
    def has_missing_entities(cls, missing_entities: List[str]):
        if len(missing_entities) == 0:
            raise ValueError(
                "You must identify 1-3 informative Entities from the Article which are missing from \
the `previous_summary` to be used in the `new_summary`"
            )
        return missing_entities
    
    @field_validator("absent_entities")
    def has_no_absent_entities(cls, absent_entities: List[str]):
        # absent_entity_string = ",".join(absent_entities)
        if len(absent_entities) > 0:
            print(f"Detected absent entities of {absent_entities}")
            raise ValueError(
                f"Do not omit the following Entities {absent_entities} from the `new_summary`"
            )
        return absent_entities


def get_entity_count(text: str = ""):
    """
    Get the count of entities from a given text
    """
    print("\nchecking entity count...")
    entities: CountEntities = client.chat.completions.create(
        model=FAST_MODEL_NAME,
        temperature=0.0,
        response_model=CountEntities,
        max_retries=3,
        max_tokens=1000,
        messages=[
            {
                "role": "system",
                "content": f"""Get the count of entities in the text given. {ENTITY_DEFINITION}
            """,
            },
            {
                "role": "user", 
                "content": f"Here is the `text`: {text} \n\n please count the number of entities in the `text`",
             }
        ],
    )
    return entities


async def summarize_article(
    article: str,
    *,
    source: str,
    summary_steps: int = 3, 
    model_name: str = "gpt-4-1106-preview"
    ):
    summary_chain = []

    try:
        # We first generate an initial summary
        summary: InitialSummary = await aclient.chat.completions.create(  
            model=model_name,
            temperature=0.5,
            response_model=InitialSummary,
            max_retries=3,
            messages=[
                {
                    "role": "system",
                    "content": "Write a summary about the article that is long (4-5 sentences) yet highly \
    non-specific. Use overly, verbose language and fillers(eg.,'this article discusses') to reach ~80 words",
                },
                {
                    "role": "user", 
                    "content": f"Here is the Article: {article}"},
                {
                    "role": "user",
                    "content": "The generated summary should be about 80 words.",
                },
            ],
        )
        summary_chain.append((summary.summary, None))
        print(f"LEN SUMMARY CHAIN: {len(summary_chain)}")
        prev_summary = None
        for _ in range(summary_steps):
            missing_entity_message = (
                []
                if prev_summary is None
                else [
                    {
                        "role": "user",
                        "content": f"Please include these Entities that were missing \
    from the `previous_summary`: {','.join(prev_summary.missing_entities)}",
                    },
                ]
            )
            new_summary: RewrittenSummary = await aclient.chat.completions.create(
                model=model_name,
                temperature=0.5,
                response_model=RewrittenSummary,
                max_retries=5,
                max_tokens=1300,
                messages=[
                    {
                        "role": "system",
                        "content": """You are going to generate an increasingly concise,entity-dense summary of the following article.

                    Perform the following two tasks
                    - Given a `previous_summary`, write a new denser summary of identical length which covers every entity and detail from the previous_summary \
    plus the Missing Entities
                    - Identify 1-3 informative entities from the given article which are missing from the `previous_summary`

                    Guidelines
                    - Make every word count: re-write the `previous_summary` to improve flow and make space for additional entities
                    - Increase information density using fusion, compression, and removal of uninformative phrases like "the article discusses".
                    - The `new_summary` should become highly dense and concise yet self-contained, i.e. easily understood without the Article.
                    - Never drop entities from the `previous_summary`. If space cannot be made, add fewer new entities.
                    """,
                    },
                    {
                        "role": "user", 
                        "content": f"Here is the Article: {article}"},
                    {
                        "role": "user",
                        "content": f"Here is the `previous_summary`: {summary_chain[-1]}",
                    },
                    *missing_entity_message,
                ],
            )
            summary_chain.append((new_summary.new_summary, new_summary.entities))
            print(f"LEN SUMMARY CHAIN: {len(summary_chain)}")
            prev_summary = new_summary
    
    except Exception as e:
        print(f"SUMMARISATION ERROR DESPITE RETRIES:\nException: {e}")
        summary_chain = [(f"error - {e}", ["error"])] * summary_steps

    return summary_chain, source


async def generate_summaries(
    articles: List[str],
    sources: List[str],
    ) -> List[Tuple[List[Tuple[str, List[str]]], str]]:
    '''
    Generate summaries for a list of articles
    '''
    tasks = [summarize_article(
        article=article,
        source=source,
        summary_steps=N_SUMMARY_STEPS, 
        model_name=MODEL_NAME
        ) for article, source in zip(articles, sources)
    ]
    summaries: List[Tuple[List[Tuple[str, List[str]]], str]] = await asyncio.gather(*tasks)
    return summaries


############

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
MODEL_NAME = 'gpt-4-0125-preview' #'gpt-4-1106-preview'
FAST_MODEL_NAME = 'gpt-3.5-turbo-1106'  # upgrade to gpt-3.5-turbo-0125 when available
N_SUMMARY_STEPS = 2

aclient = AsyncOpenAI(api_key = OPENAI_API_KEY)
aclient = instructor.patch(aclient)

client = OpenAI(api_key = OPENAI_API_KEY)
client = instructor.patch(client)

articles = [df.page_content[0], df.page_content[1], df.page_content[2]]  # replace with your list of articles
sources = [df.source[0], df.source[1], df.source[2]]  # replace with your list of sources

summaries = asyncio.run(generate_summaries(
    articles = articles,
    sources = sources
    )
)

print(f"{len(summaries)} summaries in summary chain.")
# print(f"SUMMARIES: {summaries}\n\n")
print("printing summaries:\n")
# for s in summaries[0]:
#     print(s)
#     print(f"{'*'*100}\n\n")

# create variables from the summaries
final_summaries = [sum[0][-1][0] for sum in summaries]
final_entities = [entity.entity for entity in sum[0][-1][1] for sum in summaries]
sources = [sum[1] for sum in summaries]

print(f"{'*'*100}\n\n")
print(f"FINAL SUMMARY: {final_summaries}")
print(f"{'*'*100}\n\n")
print(f"FINAL ENTITIES: {final_entities}")
print(f"{'*'*100}\n\n")
print(f"FINAL SOURCES: {sources}")

# final_entities = summaries[0][-1][1]
# source_of_article = summaries[0][1]


# Convert the extracted final summary, entities, and source into a DataFrame
summary_data = {
    "final_summary": final_summaries,
    "final_entities": final_entities,
    "source": sources
}
summary_df = pd.DataFrame(summary_data)
timestamp = datetime.now().strftime("%Y%m%d-%H%M")
summary_df.to_csv(f'summary_df_{timestamp}.csv', index=False)

# @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
# def extract_with_backoff(user_prompt, **kwargs):
#     resp = client.chat.completions.create(
#         model=MODEL_NAME,
#         messages=[
#             {
#                 "role": "system",
#                 "content": system_prompt,
#             },
#             {
#                 "role": "user",
#                 "content": user_prompt,
#             },
#         ],
#         response_model=Article,
#     )
#     return resp

# if MAX_ROWS == -1: MAX_ROWS = len(df)

# # extracted = []
# # rows = []
# # for i in tqdm(range(MAX_ROWS)):
# #     try:
# #         user_prompt = df.loc[i, 'content'] + "\n\n"
# #         extracted.append(extract_with_backoff(user_prompt))
# #         rows.append(i)
# #     except Exception as e:
# #         pass



# # resp_df = pd.DataFrame([x.dict() for x in extracted])
# # resp_df.to_csv('resp_df.csv', index=False)
# # run.log({"resp_df": wandb.Table(dataframe=resp_df)})

# # # merge with df on extracted rows
# # prev_df = df.loc[rows, :].reset_index(drop=True)
# # merged_df = pd.merge(prev_df, resp_df, left_index=True, right_index=True)
# # merged_df.to_csv('merged_df.csv', index=False)
# # run.log({"reports_metadata": wandb.Table(dataframe=merged_df)})

# # artifact = wandb.Artifact('processed_fc_articles', type='processed_data')
# # artifact.add_file('resp_df.csv')
# # artifact.add_file('merged_df.csv')
# # run.log_artifact(artifact)

# # wandb.finish()