import asyncio
import logging
import os
import re
from collections import Counter
from contextlib import asynccontextmanager
from typing import Dict, List, Tuple

import instructor
import numpy as np
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI
from openai import AsyncOpenAI

from llm_utils import (
    EXPAND_SYSTEM_PROMPT,
    EXPAND_USER_PROMPT,
    SYSTEM_PROMPT,
    USER_PROMPT,
    APIRetrievalRequest,
    ExpandedQuery,
    ExplainedChunk,
    Query,
    ContentNavigatorResponse,
)
from retriever import setup_langchain_retriever
from config import NavigatorConfig

load_dotenv('.env')
logging.basicConfig(level=logging.INFO)
aclient = AsyncOpenAI(api_key=os.getenv('OPENAI_API_KEY'))
aclient = instructor.patch(aclient)

config = NavigatorConfig()


async def explain_usefulness(query, text, source, score):
    '''
    Given a user query and a retrieved chunk, explain why the chunk is useful
    '''
    logging.debug('Calling OpenAI to explain usefulness of retrieved chunk')
    user_prompt = USER_PROMPT.format(query=query, chunk=text)

    explanation: ExplainedChunk = await aclient.chat.completions.create(
        model = config.EXPLANATION_MODEL,
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
        model = config.EXPLANATION_MODEL,
        response_model = ExpandedQuery,
        temperature = 0.2,
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
    cleaned_chunks = [chunk for chunk in tok_k_responses if  
                      "no-result" not in chunk["metadata"]["source"].lower()]
    n_cleaned_chunks = len(cleaned_chunks)
    logging.info(f"{n_retrieved_responses - len(cleaned_chunks)} \
sources were filtered out due to dummy, no-result chunk returned")

    # Remove chunks than contain non-english characters
    cleaned_chunks = [chunk for chunk in cleaned_chunks if \
                      not re.search(config.NON_ENGLISH_REGEX_SEARCH, chunk["text"])]
    logging.info(f"{n_cleaned_chunks - len(cleaned_chunks)} \
sources were filtered out due to language.")
    n_cleaned_chunks = len(cleaned_chunks)

    # # Temporary, remove this dodgy source:
    # cleaned_chunks = [chunk for chunk in cleaned_chunks if "stacey/estuary/reports/--Vmlldzo1MjEw" not in chunk["metadata"]["source"]]
    # logging.info(f"{n_cleaned_chunks - len(cleaned_chunks)} sources were filtered out due to bad data source.")
    # n_cleaned_chunks = len(cleaned_chunks)

    return cleaned_chunks


# def process_response(response) -> List[Tuple[ExplainedChunk, str]]:
#     '''
#     Convert httpx response to a list of tuples of ExplainedChunk and source as per
#     the api response format
#     '''
#     response = json.loads(response.text)
#     new_response = []
#     for item in response:
#         explained_chunk_dict, source, score = item  # Ignore the scores
#         explained_chunk = ExplainedChunk(**explained_chunk_dict)
#         new_response.append((explained_chunk, source, score))
#     return new_response
    

def postprocess_retriever_response(
        response: List[Tuple[ExplainedChunk, str, List]],
        username: str,
        debug_mode: bool
        ) -> Tuple[str, str]:
    ### FILTER OUT SOURCES THAT THE MODEL THINKS ARE IRRELEVANT ###
    # Remove any sources that weren't considered useful
    len_explanations = len(response)
    cleaned_response = [(explanation, source, score) for explanation, source, score in 
    response if explanation.content_is_relevant is True]
    logging.info(f"{len_explanations - len(cleaned_response)} pieces of content found to \
be irrelevant and removed")

    ### SEND CONTENT SUUGESTIONS BACK TO SLACK USER ###
    # Response if no content suggestions found
    if len(cleaned_response) == 0:
        slack_response = f"Hey <@{username}>, no content suggestions found. Try rephrasing \
your query."
    # Response if content suggestions found
    else:
        slack_response = f"Hey <@{username}>, content suggestions below:\n\n"
        for explanation, source, score in cleaned_response[:config.N_SOURCES_TO_SEND]:
            # fix links that have spaces
            source = source.replace(' ', '%20')

            # Show more info in debug mode
            if not debug_mode:
                slack_response += f"• {explanation.content_description} - <{source}|Link>\n\n"
            else:
                slack_response += f"*Score*: {score}, *Source*: {source}\n\
*reason_why_helpful*: {explanation.reason_why_helpful}\n\
*chain_of_thought*: {explanation.chain_of_thought}\n\
*content_is_relevant*: {explanation.content_is_relevant}\n\
*content_description*: {explanation.content_description}\n\n"
    
    # await say(slack_text, channel=SLACK_CHANNEL_ID, thread_ts=ts)
    # logger.info(f"Sent message: {slack_text}")
    
    ### PRINT REJECTED SOURCES IN SLACK IN DEBUG MODE ###
    rejected_slack_response = ""
    if debug_mode:
        rejected_responses = [(explanation, source, score) for 
        explanation, source, score in response if
        explanation.content_is_relevant is not True]
        for explanation, source, score in rejected_responses:
            source = source.replace(' ', '%20')
            rejected_slack_response += f"REJECTED, *Score*: {score}, *Source*: {source}\n\
*reason_why_helpful*: {explanation.reason_why_helpful}\n\
*chain_of_thought*: {explanation.chain_of_thought}\n\
*content_is_relevant*: {explanation.content_is_relevant}\n\
*content_description*: {explanation.content_description}\n\n"
            # await say(slack_response, channel=SLACK_CHANNEL_ID, thread_ts=ts)

    return slack_response, rejected_slack_response


### RUN THE APP

retriever = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load retriever
    global retriever
    logging.info('Setting up Retriever...')
    retriever = setup_langchain_retriever()
    logging.info('Retriever setup complete.')
    yield
    # Clean up and release the resources
    del retriever

# CREATE A FASTAPI APP
app = FastAPI(lifespan=lifespan)

@app.post("/get_content")
async def get_content(query: Query) -> ContentNavigatorResponse:
    logging.info('\n\nReceived query: %s from user: %s', query.query, query.username)
    # Strip any leading or trailing whitespace
    query.query = query.query.strip()
    # Remove any user tags like <@U06A6M92DM5> 
    query.query = re.sub(r"\<@\w+\>", "", query.query).strip()

    if '--debug' in query.query:  # Remove --debug from query if present
        query.query = query.query.replace('--debug', '')
        debug_mode = True
    else:
        debug_mode = False

    ### RETRIEVAL ###
    # Expand the user query using OpenAI to make a retriever match more likely
    expanded_query = await expand_query(query.query)
    retriever_query = Query(query=expanded_query.expanded_query)
    logging.info(f"Expanded retriever query: {expanded_query.expanded_query}")
    if debug_mode:
        logging.info(f"Expanded query CoT: {expanded_query.chain_of_thought}")
    
    # Create API retrieval request
    formatted_request = APIRetrievalRequest(query=retriever_query.query,
                                language=config.LANGUAGE,
                                initial_k=config.INITIAL_K,
                                top_k=config.TOP_K,
                                include_tags=config.INCLUDE_TAGS,
                                exclude_tags=config.EXCLUDE_TAGS,
                                )

    # Retrieve response
    # retriever_response: Dict = retrieve_from_wandbot(formatted_request)
    retriever_response: Dict = retriever.invoke(formatted_request.query)
    # retriever_response = retriever_response.
    logging.info(f'{len(retriever_response["top_k"])} chunks \
retrieved from retrieval endpoint.')

    ### FILTERING ###
    cleaned_chunks = filter_chunks(retriever_response["top_k"])

    logging.info(f"After initial filer, there are {len(cleaned_chunks)} \
 chunks in cleaned_chunks.")

    ### MERGING REPEATEDLY CITED SOURCES ###
    # Check if a source if retrieved more than once from the TOP_K \
    # retrieved sources, merge them if so
    source_counts = Counter(chunk["metadata"]["source"] for chunk in cleaned_chunks)
    sorted_source_counts = sorted(source_counts.items(), key=lambda item: item[1], reverse=True)
    logging.info(f"Source counts:")
    for s,c in sorted_source_counts:
        logging.info(f"{s}: {c}")
    multiple_source_citations = list(set(source for source, count in
                                          sorted_source_counts if count > 1))
    logging.info(f"After filtering, {len(multiple_source_citations)} \
sources were retrieved more than once.")

    # Merge chunks from the same source
    if len(multiple_source_citations) > 0:
        merged_chunks = []
        for source in multiple_source_citations:
            logging.info(f"Source: {source}")
            chunks_to_merge = [chunk for chunk in cleaned_chunks if 
                               chunk["metadata"]["source"] == source]
            for c in chunks_to_merge:
                logging.info(f"Source: {c['metadata']['source']}, \
chunk length: {len(c['text'])}, chunk content:")
                logging.info(c['text'])
                logging.info("END END END/n/n/n")
            logging.info(f"Number of chunks_to_merge: {len(chunks_to_merge)}")
            merged_text = "\n...\n".join([chunk["text"] for chunk in chunks_to_merge])
            merged_scores = [chunk["score"] for chunk in chunks_to_merge]
            logging.info(f"Merged chunk: {merged_text}")
            merged_chunks.append({"text": merged_text, "score": merged_scores,
                                   "metadata": {"source": source}})  
        cleaned_chunks = [chunk for chunk in cleaned_chunks if
                           chunk["metadata"]["source"] not in multiple_source_citations]
        cleaned_chunks.extend(merged_chunks)
    
    # Convert all scores to list to handle case when multiple scores are returned due to chunks being merged
    cleaned_chunks = [dict(chunk, score=[chunk["score"]]) if not
                       isinstance(chunk["score"], list) else chunk for chunk in cleaned_chunks]        
    logging.info(f"{len(cleaned_chunks)} chunks in cleaned_chunks after \
checking for multiple sources.")

    ### EXPLAIN WHY EACH CHUNK IS RELEVANT TO THE QUERY ###
    # Explain usefulness
    tasks = [explain_usefulness(query.query, chunk["text"], 
                                chunk["metadata"]["source"], chunk["score"]) for
                                chunk in cleaned_chunks]
    result: List[Tuple[ExplainedChunk, str, List]] = await asyncio.gather(*tasks)
    logging.info(f'{len(result)} explanations from OpenAI generated')

    # Sort explanations by score
    result.sort(key=lambda x: np.max(x[2]), reverse=True)

    # result = process_response(response)

    slack_response, rejected_slack_response = postprocess_retriever_response(
        result,
        query.username,
        debug_mode
    )

    response = ContentNavigatorResponse(
        slack_response=slack_response,
        rejected_slack_response=rejected_slack_response)
    return response


if __name__ == "__main__":
    logging.info('Starting App...')
    uvicorn.run("main:app", host="localhost", port=8008, reload=True)