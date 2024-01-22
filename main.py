import os
import logging

# from pprint import pprint
from typing import List, Tuple, Dict
from collections import Counter
# from pprint import pprint
import numpy as np
import asyncio
import requests
import re

from fastapi import FastAPI
import uvicorn

from pydantic import BaseModel, Field
from openai import AsyncOpenAI
import instructor
from dotenv import load_dotenv

load_dotenv('.env')

logging.basicConfig(level=logging.INFO)

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
aclient = instructor.apatch(AsyncOpenAI(api_key = OPENAI_API_KEY))

# from wandbot.api.schemas import APIRetrievalRequest # uncomment once wandbot v1.1 is released
class APIRetrievalRequest(BaseModel):
    query: str
    language: str = "en"
    initial_k: int = 10
    top_k: int = 5
    include_tags: List[str] = []
    exclude_tags: List[str] = []


ENDPOINT = "https://wandbot-dev.replit.app/retrieve"
HEADERS = {
    'Content-Type': 'application/json',
    'Accept': 'application/json',
}

OPENAI_EXPLANATION_MODEL = "gpt-4-1106-preview"
# OPENAI_EXPLANATION_MODEL = "gpt-3.5-turbo-1106"

QUERY = 'do we have any reports I could send to a finance company?'
TOP_K = 15
INITIAL_K = 50
LANGUAGE = 'en'
INCLUDE_TAGS = ['fc-reports']
REGEX_SEARCH = r'[^\x00-\x7F]'

SYSTEM_PROMPT = '''Our Sales team want to send recommeneded links to propects and customers. \

Given a `query` from an internal employee as well as a `chunk` of text retrieved from a blog post, \
generate an explanation of why the `chunk` is useful in answering the `query`.

Finally, frame the final response in `reason_rephrased` to speak directly to a prospect or customer, as if you were \
on our sales team.
'''

USER_PROMPT = '''Please explain why the following chunk is useful in answering the query:

>>>> query: {query}

>>>> chunk: {chunk}

Take a deep breath, if you do a good job I will tip you $200.'''


req = APIRetrievalRequest(query=QUERY,
                         language=LANGUAGE,
                         initial_k=INITIAL_K,
                         top_k=TOP_K,
                         include_tags=INCLUDE_TAGS)

def retrieve(user_request: APIRetrievalRequest) -> Dict:
    logging.debug('Sending request to retrieval endpoint')
    response = requests.post(ENDPOINT,
                             headers=HEADERS,
                             data = user_request.model_dump_json(),
                             timeout=600
                             )
    logging.debug('Received response from retrieval endpoint')
    return response.json()


class ExplainedChunk(BaseModel):
    '''Given a `query` from a user and a retrieved `chunk` of text, provide a brief description of the content of the `chunk` and \
whether it is relevant to the `query`.'''

    # query: str = Field(..., description="An internal employee's query looking for a content such as a blog post to send to a customer.")
    # chunk: str = Field(..., description="The retrieved text chunk from a database of chunks taken from blog posts.")
    chain_of_thought: str = Field(..., description="Think step by step about whether the `chunk` is useful and helpful in answering the `query`.")
    reason_why_helpful: str = Field(..., description="A brief, 1 sentence explanation on what this blog post contains and a reason whether or not it \
is relevant based on the `query`.")
    content_description: str = Field(..., description="A brief, 1 sentence description of the content of the blog post. \
DO NOT say 'this blog post' or 'this chunk'. Instead, start wtih 'This covers...'.")
    content_is_relevant: bool = Field(..., description="A boolean value indicating whether the content of the blog post is relevant to the `query`.")


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

class Query(BaseModel):
    query: str

app = FastAPI()

@app.post("/get_content")
async def process_query(query: Query) -> List[Tuple[ExplainedChunk, str, List]]:
    logging.info('Received query: %s', query.query)
    # print(query.query)

    # Create API retrieval request
    formatted_request = APIRetrievalRequest(query=query.query,
                                language=LANGUAGE,
                                initial_k=INITIAL_K,
                                top_k=TOP_K,
                                include_tags=INCLUDE_TAGS)

    # Retrieve response
    retriever_response: Dict = retrieve(formatted_request)
    logging.info(f'{len(retriever_response["top_k"])} chunks retrieved from retrieval endpoint.')

    ### FILTERING ###
    n_retrieved_responses = len(retriever_response["top_k"])
    # Remove chunks than contain non-english characters
    cleaned_chunks = [d for d in retriever_response["top_k"] if not re.search(REGEX_SEARCH, d["text"])]
    # Remove chunks from ML-News
    cleaned_chunks = [chunk for chunk in cleaned_chunks if "ml-news" not in chunk["metadata"]["source"].lower()]
    logging.info(f"{n_retrieved_responses - len(cleaned_chunks)} sources were filtered out due to language or'ml-news' source.")

    ### MERGING REPEATEDLY CITED SOURCES ###
    # Check if a source if retrieved more than once from the TOP_K retrieved sources, merge them if so
    source_counts = Counter(chunk["metadata"]["source"] for chunk in cleaned_chunks)
    sorted_source_counts = sorted(source_counts.items(), key=lambda item: item[1], reverse=True)
    logging.info(f"Source counts:")
    for s,c in sorted_source_counts:
        logging.info(f"{s}: {c}")
    multiple_source_citations = list(set(source for source, count in sorted_source_counts if count > 1))
    logging.info(f"{len(multiple_source_citations)} sources were retrieved more than once.")

    # Merge chunks from the same source
    if len(multiple_source_citations) > 0:
        merged_chunks = []
        for source in multiple_source_citations:
            logging.info(f"Source: {source}")
            chunks_to_merge = [chunk for chunk in cleaned_chunks if chunk["metadata"]["source"] == source]
            for c in chunks_to_merge:
                logging.info(f"Source: {c['metadata']['source']}, chunk length: {len(c['text'])}, chunk content:")
                for i in range(0, len(c['text']), 60):  # Adjust the chunk size as needed
                    logging.info(c['text'][i:i+60])
                logging.info("END END END/n/n/n")
            logging.info(f"Number of chunks_to_merge: {len(chunks_to_merge)}")
            merged_text = "\n...\n".join([chunk["text"] for chunk in chunks_to_merge])
            merged_scores = [chunk["score"] for chunk in chunks_to_merge]
            logging.info(f"Merged chunk: {merged_text}")
            merged_chunks.append({"text": merged_text, "score": merged_scores, "metadata": {"source": source}})  
        cleaned_chunks = [chunk for chunk in cleaned_chunks if chunk["metadata"]["source"] not in multiple_source_citations]
        cleaned_chunks.extend(merged_chunks)
    
    # Convert all scores to lit to handle case when multiple scores are returned due to chunks being merged
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
    uvicorn.run("main:app", host="localhost", port=8000, reload=True)
