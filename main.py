import asyncio
import logging
import os
import re
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Dict, List, Tuple

import instructor
import numpy as np
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI
from openai import AsyncOpenAI

from config import NavigatorConfig
from llm_utils import (
    EXPAND_SYSTEM_PROMPT,
    EXPAND_USER_PROMPT,
    SYSTEM_PROMPT,
    USER_PROMPT,
    APIRetrievalRequest,
    ContentNavigatorResponse,
    ExpandedQuery,
    ExplainedChunk,
    Query,
)
from retriever import setup_langchain_retriever

load_dotenv('.env')
logging.basicConfig(level=logging.INFO)
aclient = AsyncOpenAI(api_key=os.getenv('OPENAI_API_KEY'))
aclient = instructor.patch(aclient)

config = NavigatorConfig()


async def explain_usefulness(query: str, text: str, source: str, score: List[float]) -> Tuple[ExplainedChunk, str, List[float]]:
    """
    Explain the usefulness of a retrieved chunk given a user query.

    Args:
        query (str): The user's query.
        text (str): The text of the retrieved chunk.
        source (str): The source of the retrieved chunk.
        score (List[float]): The score(s) associated with the retrieved chunk.

    Returns:
        Tuple[ExplainedChunk, str, List[float]]: The explanation, source, and score.
    """
    logging.debug('Calling OpenAI to explain usefulness of retrieved chunk')
    user_prompt = USER_PROMPT.format(query=query, chunk=text)

    explanation: ExplainedChunk = await aclient.chat.completions.create(
        model=config.EXPLANATION_MODEL,
        response_model=ExplainedChunk,
        temperature=0.0,
        messages=[{"role": "system", "content": SYSTEM_PROMPT},
                  {"role": "user", "content": user_prompt}]
    )
    logging.debug('Received explanation for chunk from OpenAI')
    
    return explanation, source, score


async def expand_query(query: str) -> ExpandedQuery:
    """
    Expand the user query to improve semantic search matching.

    Args:
        query (str): The user's query.

    Returns:
        ExpandedQuery: The expanded query.
    """
    logging.debug('Calling OpenAI to expand the user query')
    user_prompt = EXPAND_USER_PROMPT.format(query=query)

    expanded_query: ExpandedQuery = await aclient.chat.completions.create(
        model=config.EXPLANATION_MODEL,
        response_model=ExpandedQuery,
        temperature=0.2,
        messages=[{"role": "system", "content": EXPAND_SYSTEM_PROMPT},
                  {"role": "user", "content": user_prompt}]
    )
    logging.debug('Received explanation for chunk from OpenAI')
    
    return expanded_query


def filter_chunks(tok_k_responses: List[Dict]) -> List[Dict]:
    """
    Filter out irrelevant chunks from the retrieved responses.

    Args:
        tok_k_responses (List[Dict]): The retrieved responses.

    Returns:
        List[Dict]: The filtered responses.
    """
    n_retrieved_responses = len(tok_k_responses)

    cleaned_chunks = [chunk for chunk in tok_k_responses if "no-result" not in chunk["metadata"]["source"].lower()]
    logging.info(f"{n_retrieved_responses - len(cleaned_chunks)} sources were filtered out due to dummy, no-result chunk returned")

    return cleaned_chunks


def postprocess_retriever_response(
        response: List[Tuple[ExplainedChunk, str, List[float]]],
        username: str,
        debug_mode: bool
        ) -> Tuple[str, str]:
    """
    Process the retriever response to generate user-friendly messages.

    Args:
        response (List[Tuple[ExplainedChunk, str, List[float]]]): The explanations, sources, and scores.
        username (str): The username of the requester.
        debug_mode (bool): Whether the request is in debug mode.

    Returns:
        Tuple[str, str]: The slack response and rejected slack response.
    """
    len_explanations = len(response)
    cleaned_response = [(explanation, source, score) for explanation, source, score in response if explanation.content_is_relevant is True]
    logging.info(f"{len_explanations - len(cleaned_response)} pieces of content found to be irrelevant and removed")

    if len(cleaned_response) == 0:
        slack_response = f"Hey <@{username}>, no content suggestions found. Try rephrasing your query."
    else:
        slack_response = f"Hey <@{username}>, content suggestions below:\n\n"
        for explanation, source, score in cleaned_response[:config.N_SOURCES_TO_SEND]:
            source = source.replace(' ', '%20')
            if not debug_mode:
                slack_response += f"• {explanation.content_description} - <{source}|Link>\n\n"
            else:
                slack_response += f"*Score*: {score}, *Source*: {source}\n*reason_why_helpful*: {explanation.reason_why_helpful}\n*chain_of_thought*: {explanation.chain_of_thought}\n*content_is_relevant*: {explanation.content_is_relevant}\n*content_description*: {explanation.content_description}\n\n"
    
    rejected_slack_response = ""
    if debug_mode:
        rejected_responses = [(explanation, source, score) for explanation, source, score in response if explanation.content_is_relevant is not True]
        for explanation, source, score in rejected_responses:
            source = source.replace(' ', '%20')
            rejected_slack_response += f"REJECTED, *Score*: {score}, *Source*: {source}\n*reason_why_helpful*: {explanation.reason_why_helpful}\n*chain_of_thought*: {explanation.chain_of_thought}\n*content_is_relevant*: {explanation.content_is_relevant}\n*content_description*: {explanation.content_description}\n\n"

    return slack_response, rejected_slack_response


retriever = None

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Context manager for the lifespan of the FastAPI app.

    Args:
        app (FastAPI): The FastAPI application instance.

    Yields:
        AsyncGenerator[None, None]: An async generator yielding None.
    """
    global retriever
    logging.info('Setting up Retriever...')
    retriever = setup_langchain_retriever()
    logging.info('Retriever setup complete.')
    yield
    del retriever

app = FastAPI(lifespan=lifespan)

@app.post("/get_content")
async def get_content(query: Query) -> ContentNavigatorResponse:
    """
    Endpoint to get content suggestions based on a user query.

    Args:
        query (Query): The user's query.

    Returns:
        ContentNavigatorResponse: The response containing content suggestions.
    """
    logging.info('\n\nReceived query: %s from user: %s', query.query, query.username)
    query.query = query.query.strip()
    query.query = re.sub(r"\<@\w+\>", "", query.query).strip()

    debug_mode = '--debug' in query.query
    if debug_mode:
        query.query = query.query.replace('--debug', '')

    expanded_query = await expand_query(query.query)
    retriever_query = Query(query=expanded_query.expanded_query)
    logging.info(f"Expanded retriever query: {expanded_query.expanded_query}")
    if debug_mode:
        logging.info(f"Expanded query CoT: {expanded_query.chain_of_thought}")
    
    formatted_request = APIRetrievalRequest(query=retriever_query.query,
                                             language=config.LANGUAGE,
                                             initial_k=config.INITIAL_K,
                                             top_k=config.TOP_K,
                                             include_tags=config.INCLUDE_TAGS,
                                             exclude_tags=config.EXCLUDE_TAGS)

    retriever_response: Dict = retriever.invoke(formatted_request.query)
    logging.info(f'{len(retriever_response["top_k"])} chunks retrieved from retrieval endpoint.')

    cleaned_chunks = filter_chunks(retriever_response["top_k"])

    logging.info(f"After initial filter, there are {len(cleaned_chunks)} chunks in cleaned_chunks.")

    tasks = [explain_usefulness(query.query, chunk["text"], chunk["metadata"]["source"], chunk["score"]) for chunk in cleaned_chunks]
    result: List[Tuple[ExplainedChunk, str, List[float]]] = await asyncio.gather(*tasks)
    logging.info(f'{len(result)} explanations from OpenAI generated')

    result.sort(key=lambda x: np.max(x[2]), reverse=True)

    slack_response, rejected_slack_response = postprocess_retriever_response(result, query.username, debug_mode)

    response = ContentNavigatorResponse(slack_response=slack_response, rejected_slack_response=rejected_slack_response)
    return response


if __name__ == "__main__":
    logging.info('Starting App...')
    uvicorn.run("main:app", host="localhost", port=8008, reload=True)
