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
from load_dotenv import load_dotenv
from openai import AsyncOpenAI

from config import NavigatorConfig
from llm_utils import (
    APIRetrievalRequest,
    ContentNavigatorResponse,
    ExplainedChunk,
    Query,
)
from retriever import setup_langchain_retriever
from utils import (
    expand_query,
    explain_usefulness,
    filter_chunks,
    postprocess_retriever_response,
)

load_dotenv(".env")
logging.basicConfig(level=logging.INFO)
aclient = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
aclient = instructor.patch(aclient)

config = NavigatorConfig()

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
    logging.info("Setting up Retriever...")
    retriever = setup_langchain_retriever(config.FULLY_CONNECTED_SUMMARIES_FILEPATH)
    logging.info("Retriever setup complete.")
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
    logging.info("\n\nReceived query: %s from user: %s", query.query, query.username)

    # Clean up the query
    query.query = query.query.strip()
    query.query = re.sub(r"\<@\w+\>", "", query.query).strip()

    debug_mode = "--debug" in query.query
    if debug_mode:
        query.query = query.query.replace("--debug", "")

    # Augment the user query to improve the changes of good retrieval
    expanded_query = await expand_query(query.query)
    logging.info(f"Expanded retriever query: {expanded_query.expanded_query}")
    if debug_mode:
        logging.info(f"Expanded retriever query CoT: {expanded_query.chain_of_thought}")

    # Retrieve relevant chunks from the retrieval chain
    formatted_request = APIRetrievalRequest(
        query=expanded_query.expanded_query,
        language=config.LANGUAGE,
        initial_k=config.INITIAL_K,
        top_k=config.TOP_K,
        include_tags=config.INCLUDE_TAGS,
        exclude_tags=config.EXCLUDE_TAGS,
    )

    retriever_response: Dict = retriever.invoke(formatted_request.query)
    logging.info(
        f'{len(retriever_response["top_k"])} chunks retrieved from retrieval endpoint.'
    )

    # Filter out irrelevant chunks from the retrieved responses
    cleaned_chunks = filter_chunks(retriever_response["top_k"])
    logging.info(
        f"After initial filter, there are {len(cleaned_chunks)} chunks in cleaned_chunks."
    )

    # Explain the relevance/usefulness of the retrieved chunks using an LLM
    tasks = [
        explain_usefulness(
            query.query, chunk["text"], chunk["metadata"]["source"], chunk["score"]
        )
        for chunk in cleaned_chunks
    ]
    result: List[Tuple[ExplainedChunk, str, List[float]]] = await asyncio.gather(*tasks)
    logging.info(f"{len(result)} explanations from OpenAI generated")

    result.sort(key=lambda x: np.max(x[2]), reverse=True)

    # Process the retriever response to generate user-friendly messages
    slack_response, rejected_slack_response = postprocess_retriever_response(
        result, query.username, debug_mode
    )

    return ContentNavigatorResponse(
        slack_response=slack_response, rejected_slack_response=rejected_slack_response
    )


if __name__ == "__main__":
    logging.info("Starting App...")
    uvicorn.run("main:app", host="localhost", port=8008, reload=True)
