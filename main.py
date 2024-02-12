import os
import logging

from typing import List, Tuple, Dict
from collections import Counter
import numpy as np
import asyncio
import re

import uvicorn
from fastapi import FastAPI
from contextlib import asynccontextmanager

from openai import AsyncOpenAI
import instructor
from dotenv import load_dotenv

from llm_utils import ExplainedChunk, ExpandedQuery, APIRetrievalRequest, Query
from llm_utils import (
    SYSTEM_PROMPT,
    USER_PROMPT,
    EXPAND_SYSTEM_PROMPT,
    EXPAND_USER_PROMPT
)
from retriever import setup_langchain_retriever

load_dotenv('.env')

logging.basicConfig(level=logging.INFO)

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
aclient = instructor.apatch(AsyncOpenAI(api_key = OPENAI_API_KEY))


OPENAI_EXPLANATION_MODEL = "gpt-4-0125-preview"
# OPENAI_EXPLANATION_MODEL = "gpt-3.5-turbo-1106"

QUERY = 'do we have any reports I could send to a finance company?'
TOP_K = 10
INITIAL_K = 100
LANGUAGE = 'en'
INCLUDE_TAGS = ['contains-wandb-code']
EXCLUDE_TAGS = ['ml-news', 'gradient-dissent']
NON_ENGLISH_REGEX_SEARCH = r'[^\x00-\x7F]'


async def explain_usefulness(query, text, source, score):
    '''
    Given a user query and a retrieved chunk, explain why the chunk is useful
    '''
    logging.debug('Calling OpenAI to explain usefulness of retrieved chunk')
    user_prompt = USER_PROMPT.format(query=query, chunk=text)

    explanation: ExplainedChunk = await aclient.chat.completions.create(
        model = OPENAI_EXPLANATION_MODEL,
        response_model = ExplainedChunk,
        temperature = 0.0,
        messages = [{"role": "system", "content": SYSTEM_PROMPT}, 
                    {"role": "user", "content": user_prompt}]
    )
    logging.debug('Received explanation for chunk from OpenAI')
    
    return explanation, source, score


async def expand_query(query):
    '''
    Exapnd the user query to make a semantic search match more likely
    '''
    logging.debug('Calling OpenAI to expand the user query')
    user_prompt = EXPAND_USER_PROMPT.format(query=query)

    expanded_query: ExplainedChunk = await aclient.chat.completions.create(
        model = OPENAI_EXPLANATION_MODEL,
        response_model = ExpandedQuery,
        temperature = 0.1,
        messages = [{"role": "system", "content": EXPAND_SYSTEM_PROMPT}, 
                    {"role": "user", "content": user_prompt}]
    )
    logging.debug('Received explanation for chunk from OpenAI')
    
    return expanded_query


def filter_chunks(tok_k_responses: List[Dict]) -> List[Dict]:
    '''
    Filter out chunks based on sources that are not relevant to this task
    '''

    n_retrieved_responses = len(tok_k_responses)

    # Remove dummy chunnks returned when there are no results
    cleaned_chunks = [chunk for chunk in tok_k_responses if "no-result" not in chunk["metadata"]["source"].lower()]
    n_cleaned_chunks = len(cleaned_chunks)
    logging.info(f"{n_retrieved_responses - len(cleaned_chunks)} sources were filtered out due to dummy, no-result chunk returned")

    # Remove chunks than contain non-english characters
    cleaned_chunks = [chunk for chunk in cleaned_chunks if not re.search(NON_ENGLISH_REGEX_SEARCH, chunk["text"])]
    logging.info(f"{n_cleaned_chunks - len(cleaned_chunks)} sources were filtered out due to language.")
    n_cleaned_chunks = len(cleaned_chunks)

    # Remove chunks from ML-News
    cleaned_chunks = [chunk for chunk in cleaned_chunks if "ml-news" not in chunk["metadata"]["source"].lower()]
    logging.info(f"{n_cleaned_chunks - len(cleaned_chunks)} sources were filtered out due to 'ml-news'")
    n_cleaned_chunks = len(cleaned_chunks)

    # Remove chunks from Gradient Dissent podcast
    cleaned_chunks = [chunk for chunk in cleaned_chunks if "wandb_fc/gradient-dissent" not in chunk["metadata"]["source"].lower()]
    logging.info(f"{n_cleaned_chunks - len(cleaned_chunks)} sources were filtered out due to 'gradient-dissent'")
    n_cleaned_chunks = len(cleaned_chunks)

    # Temporary, remove this dodgy source:
    cleaned_chunks = [chunk for chunk in cleaned_chunks if "stacey/estuary/reports/--Vmlldzo1MjEw" not in chunk["metadata"]["source"]]
    logging.info(f"{n_cleaned_chunks - len(cleaned_chunks)} sources were filtered out due to bad data source.")
    n_cleaned_chunks = len(cleaned_chunks)

    return cleaned_chunks


### RUN THE APP

# app = FastAPI()
# # Define a global variable for the retriever
# retriever = None
# @app.on_event("startup")
# async def startup_event():
#     global retriever
#     logging.info('Setting up Retriever...')
#     # Initialize the retriever and assign it to the global variable
#     retriever = setup_langchain_retriever()
#     logging.info('Retriever setup complete.')



retriever = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global retriever
    # Load the ML model
    # ml_models["answer_to_everything"] = fake_answer_to_everything_ml_model
    logging.info('Setting up Retriever...')
    retriever = setup_langchain_retriever()
    logging.info('Retriever setup complete.')
    yield
    # Clean up and release the resources
    # del retriever

app = FastAPI(lifespan=lifespan)

# this .get works
# @app.get("/predict")
# async def predict(x: float):
#     # result = ml_models["answer_to_everything"](x)
#     results = retriever.invoke("finance example")
#     print(results)
#     return {"result": results}


# ####################
# # DEBUGGING .post, but it works ok
# from fastapi import FastAPI
# from pydantic import BaseModel

# class PredictRequest(BaseModel):
#     x: float

# @app.post("/predict")
# async def predict(request: PredictRequest):
#     results = retriever.invoke("finance example")
#     print(results)
#     return {"result": results}
# ####################


# # retriever = None

# # # Define your async context manager for lifespan events
# # @asynccontextmanager
# # async def app_lifespan(app: FastAPI):
# #     # Place your startup logic here
# #     global retriever
# #     retriever = setup_langchain_retriever()
# #     yield  # The application is now running

# # # Create your FastAPI app instance with the lifespan context manager
# # app = FastAPI(lifespan=app_lifespan)



@app.post("/get_content")
async def process_query(query: Query) -> List[Tuple[ExplainedChunk, str, List]]:
    logging.info('\n\nReceived query: %s', query.query)
    # Strip any leading or trailing whitespace
    query.query = query.query.strip()
    # Remove any user tags like <@U06A6M92DM5> 
    query.query = re.sub(r"\<@\w+\>", "", query.query).strip()

    # results = retriever.invoke("finance example")
    # print(f"RESULTS!!! {results}")

    ### RETRIEVAL ###

    # Expand the user query using OpenAI to make a retriever match more likely
    expanded_query = await expand_query(query.query)
    retriever_query = Query(query=expanded_query.expanded_query)
    logging.info(f"Expanded retriever query: {expanded_query.expanded_query}")
    if "--debug" in query.query:
        logging.info(f"Expanded query CoT: {expanded_query.chain_of_thought}")
    
    # Create API retrieval request
    formatted_request = APIRetrievalRequest(query=retriever_query.query,
                                language=LANGUAGE,
                                initial_k=INITIAL_K,
                                top_k=TOP_K,
                                include_tags=INCLUDE_TAGS,
                                exclude_tags=EXCLUDE_TAGS,
                                )

    # Retrieve response
    # retriever_response: Dict = retrieve_from_wandbot(formatted_request)
    retriever_response: Dict = retriever.invoke(formatted_request.query)
    # retriever_response = retriever_response.
    logging.info(f'{len(retriever_response["top_k"])} chunks retrieved from retrieval endpoint.')

    ### FILTERING ###
    cleaned_chunks = filter_chunks(retriever_response["top_k"])

    logging.info(f"After initial filer, there are {len(cleaned_chunks)} chunks in cleaned_chunks.")


    ### MERGING REPEATEDLY CITED SOURCES ###
    # Check if a source if retrieved more than once from the TOP_K retrieved sources, merge them if so
    source_counts = Counter(chunk["metadata"]["source"] for chunk in cleaned_chunks)
    sorted_source_counts = sorted(source_counts.items(), key=lambda item: item[1], reverse=True)
    logging.info(f"Source counts:")
    for s,c in sorted_source_counts:
        logging.info(f"{s}: {c}")
    multiple_source_citations = list(set(source for source, count in sorted_source_counts if count > 1))
    logging.info(f"After filtering, {len(multiple_source_citations)} sources were retrieved more than once.")

    # Merge chunks from the same source
    if len(multiple_source_citations) > 0:
        merged_chunks = []
        for source in multiple_source_citations:
            logging.info(f"Source: {source}")
            chunks_to_merge = [chunk for chunk in cleaned_chunks if chunk["metadata"]["source"] == source]
            for c in chunks_to_merge:
                logging.info(f"Source: {c['metadata']['source']}, chunk length: {len(c['text'])}, chunk content:")
                logging.info(c['text'])
                logging.info("END END END/n/n/n")
            logging.info(f"Number of chunks_to_merge: {len(chunks_to_merge)}")
            merged_text = "\n...\n".join([chunk["text"] for chunk in chunks_to_merge])
            merged_scores = [chunk["score"] for chunk in chunks_to_merge]
            logging.info(f"Merged chunk: {merged_text}")
            merged_chunks.append({"text": merged_text, "score": merged_scores, "metadata": {"source": source}})  
        cleaned_chunks = [chunk for chunk in cleaned_chunks if chunk["metadata"]["source"] not in multiple_source_citations]
        cleaned_chunks.extend(merged_chunks)
    
    # Convert all scores to list to handle case when multiple scores are returned due to chunks being merged
    cleaned_chunks = [dict(chunk, score=[chunk["score"]]) if not isinstance(chunk["score"], list) else chunk for chunk in cleaned_chunks]        
    logging.info(f"{len(cleaned_chunks)} chunks in cleaned_chunks after checking for multiple sources.")

    ### EXPLAIN WHY EACH CHUNK IS RELEVANT TO THE QUERY ###
    # Explain usefulness
    tasks = [explain_usefulness(query.query, chunk["text"], chunk["metadata"]["source"], chunk["score"]) for chunk in cleaned_chunks]
    result: List[Tuple[ExplainedChunk, str, List]] = await asyncio.gather(*tasks)
    logging.info(f'{len(result)} explanations from OpenAI generated')

    # Sort explanations by score
    result.sort(key=lambda x: np.max(x[2]), reverse=True)

    # for explanation, source in res:
    #     logging.info('Source: %s', source)
    #     logging.info('CoT: %s', explanation.chain_of_thought)
    #     logging.info('Explanation: %s', explanation.reason_why_helpful)
    #     logging.info('Relevant: %s', explanation.content_is_relevant)
    #     logging.info('Description: %s', explanation.content_description)
    #     print()

    return result


if __name__ == "__main__":
    logging.info('Running App')
    uvicorn.run("main:app", host="localhost", port=8008, reload=True)