import re
import logging
import requests
from typing import Dict
import pandas as pd

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnableLambda
from langchain_core.output_parsers import BaseOutputParser

from llm_utils import APIRetrievalResult,APIRetrievalRequest, APIRetrievalResponse

# ENDPOINT = "https://wandbot-dev.replit.app/retrieve"
ENDPOINT = "https://wandbot.replit.app/retrieve"
FULLY_CONNECTED_SUMMARIES_FILEPATH = "fully_connected_summaries_final.csv"

HEADERS = {
    'Content-Type': 'application/json',
    'Accept': 'application/json',
}

def retrieve_from_wandbot(user_request: APIRetrievalRequest) -> Dict:
    logging.debug('Sending request to retrieval endpoint')
    response = requests.post(ENDPOINT,
                             headers=HEADERS,
                             data = user_request.model_dump_json(),
                             timeout=600
                             )
    logging.debug('Received response from retrieval endpoint')
    return response.json()

def process_langchain_retriever_output(langchain_retriever_response: Dict) -> APIRetrievalResponse:
    '''
    Output from langchain is a dict of {"context": [list of langchain Documents]}
    Documents contains "page_content" and a "metadata" dictionary
    We need to convert this to a APIRetrievalResult format
    '''
    results = []
    for doc in langchain_retriever_response["context"]:
        text = doc.page_content
        score = doc.metadata.get("score", 0.0)  # Assuming score is stored in metadata, defaulting to 0.0 if not present
        metadata = doc.metadata
        result = APIRetrievalResult(text=text, score=score, metadata=metadata)
        results.append(result)
    return APIRetrievalResponse(
        query=langchain_retriever_response["question"], 
        top_k=results
    )

# # Custom OutputParser for LangChain retriever output
# class LangChainRetrieverOutputParser(BaseOutputParser):
#     def parse(self, output: Dict) -> Dict:
#         result: APIRetrievalResponse = process_langchain_retriever_output(output)
#         return result.model_dump()
    
#     async def aparse(self, output: Dict) -> Dict:
#         return await run_in_executor(None, self.parse, output)


def setup_langchain_retriever(fc_summaries_filepath: str = "data/fully_connected_summaries_final.csv"):
    # Load the summaries data
    reports_summaries = pd.read_csv(fc_summaries_filepath)
    raw_len_summaries = len(reports_summaries)
    print(f"{raw_len_summaries} summaries loaded, starting filtering...")
    # Filter out summaries with errors
    reports_summaries = reports_summaries.query("error_count == 0")
    # Filter out sources with "russian" in the URL
    reports_summaries = reports_summaries[~reports_summaries['sources'].str.contains("/russian/", case=False)]
    reports_summaries = reports_summaries[~reports_summaries['sources'].str.contains("/japanese/", case=False)]
    reports_summaries = reports_summaries[~reports_summaries['sources'].str.contains("/korean/", case=False)]
    reports_summaries = reports_summaries[~reports_summaries['sources'].str.contains("/chinese/", case=False)]
    reports_summaries = reports_summaries[~reports_summaries['sources'].str.contains("/german/", case=False)]
    reports_summaries = reports_summaries[~reports_summaries['sources'].str.contains("/french/", case=False)]
    reports_summaries = reports_summaries[~reports_summaries['sources'].str.contains("/seo/", case=False)]
    reports_summaries = reports_summaries[~reports_summaries['sources'].str.contains("/wb-tutorials/", case=False)]
    print(f"{raw_len_summaries-len(reports_summaries)} summaries removed due to errors or russian sources.\
{len(reports_summaries)} summaries remaining.")

    summaries = reports_summaries.summaries.values
    sources = reports_summaries.sources.values
    entities = reports_summaries.entities.values

    metadata_ls = []
    for s,e in zip(sources,entities):
        metadata_ls.append({"source":s,"entity":e})

    vectorstore = FAISS.from_texts(
        summaries, 
        embedding=OpenAIEmbeddings(
            model="text-embedding-3-large",
            tiktoken_model_name="cl100k_base"
            ),
        metadatas=metadata_ls
    )

    retriever = vectorstore.as_retriever()

    setup_and_retrieval = RunnableParallel(
        {"context": retriever, "question": RunnablePassthrough()}
    )
    langchain_output_parser = RunnableLambda(process_langchain_retriever_output)
    api_retrieval_response_to_dict = RunnableLambda(lambda x: x.model_dump())

    retrieval_chain = (
        setup_and_retrieval
        | langchain_output_parser
        | api_retrieval_response_to_dict
    )
    return retrieval_chain