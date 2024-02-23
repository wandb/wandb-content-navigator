import logging
from typing import Any, Dict, List

import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_core.callbacks.manager import (
    AsyncCallbackManagerForRetrieverRun,
    CallbackManagerForRetrieverRun,
)
from langchain_core.documents import Document
from langchain_core.runnables import (
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_openai import OpenAIEmbeddings

from config import NavigatorConfig
from llm_utils import APIRetrievalResponse, APIRetrievalResult

logging.basicConfig(level=logging.INFO)


config = NavigatorConfig()

# Custom Retriever to returns the scores along with the documents
class CustomRetriever(VectorStoreRetriever):
    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        if self.search_type == "similarity":
            docs = self.vectorstore.similarity_search(query, **self.search_kwargs)
        elif self.search_type == "similarity_score_threshold":
            docs_and_similarities = (
                self.vectorstore.similarity_search_with_relevance_scores(
                    query, **self.search_kwargs
                )
            )

            for doc, score in docs_and_similarities[:]:
                doc.metadata["score"] = score

            docs = [doc for doc, _ in docs_and_similarities]
        elif self.search_type == "mmr":
            docs = self.vectorstore.max_marginal_relevance_search(
                query, **self.search_kwargs
            )
        else:
            raise ValueError(f"search_type of {self.search_type} not allowed.")
        return docs

    async def _aget_relevant_documents(
        self, query: str, *, run_manager: AsyncCallbackManagerForRetrieverRun
    ) -> List[Document]:
        if self.search_type == "similarity":
            docs = await self.vectorstore.asimilarity_search(
                query, **self.search_kwargs
            )
        elif self.search_type == "similarity_score_threshold":
            docs_and_similarities = (
                await self.vectorstore.asimilarity_search_with_relevance_scores(
                    query, **self.search_kwargs
                )
            )
            for doc, score in docs_and_similarities[:]:
                doc.metadata["score"] = score

            docs = [doc for doc, _ in docs_and_similarities]
        elif self.search_type == "mmr":
            docs = await self.vectorstore.amax_marginal_relevance_search(
                query, **self.search_kwargs
            )
        else:
            raise ValueError(f"search_type of {self.search_type} not allowed.")
        return docs


class MyFAISS(FAISS):
    def as_retriever(self, **kwargs: Any) -> CustomRetriever:
        tags = kwargs.pop("tags", None) or []
        tags.extend(self._get_retriever_tags())
        return CustomRetriever(vectorstore=self, **kwargs, tags=tags)


def process_langchain_retriever_output(
    langchain_retriever_response: Dict,
) -> APIRetrievalResponse:
    """
    Output from langchain is a dict of {"context": [list of langchain Documents]}
    Documents contains "page_content" and a "metadata" dictionary
    We need to convert this to a APIRetrievalResult format
    """
    results = []
    for doc in langchain_retriever_response["context"]:
        text = doc.page_content
        score = doc.metadata.get("score", 0.0)  # Assuming score is stored in metadata

        metadata = doc.metadata
        result = APIRetrievalResult(text=text, score=score, metadata=metadata)
        results.append(result)
    return APIRetrievalResponse(
        query=langchain_retriever_response["question"], top_k=results
    )


def setup_langchain_retriever(
    fc_summaries_filepath: str = "data/fully_connected_summaries_final.csv",
):
    # Load the summaries data
    reports_summaries = pd.read_csv(fc_summaries_filepath)
    raw_len_summaries = len(reports_summaries)
    print(f"{raw_len_summaries} summaries loaded, starting filtering...")
    # Filter out summaries with errors
    reports_summaries = reports_summaries.query("error_count == 0")
    # Filter out sources with "russian" in the URL
    reports_summaries = reports_summaries[
        ~reports_summaries["sources"].str.contains("/russian/", case=False)
    ]
    reports_summaries = reports_summaries[
        ~reports_summaries["sources"].str.contains("/japanese/", case=False)
    ]
    reports_summaries = reports_summaries[
        ~reports_summaries["sources"].str.contains("/korean/", case=False)
    ]
    reports_summaries = reports_summaries[
        ~reports_summaries["sources"].str.contains("/chinese/", case=False)
    ]
    reports_summaries = reports_summaries[
        ~reports_summaries["sources"].str.contains("/german/", case=False)
    ]
    reports_summaries = reports_summaries[
        ~reports_summaries["sources"].str.contains("/french/", case=False)
    ]
    reports_summaries = reports_summaries[
        ~reports_summaries["sources"].str.contains("/seo/", case=False)
    ]
    reports_summaries = reports_summaries[
        ~reports_summaries["sources"].str.contains("/wb-tutorials/", case=False)
    ]
    print(
        f"{raw_len_summaries-len(reports_summaries)} summaries removed due to errors \
or non-English sources. {len(reports_summaries)} summaries remaining."
    )

    summaries = reports_summaries.summaries.values
    sources = reports_summaries.sources.values
    entities = reports_summaries.entities.values

    metadata_ls = []
    for s, e in zip(sources, entities):
        metadata_ls.append({"source": s, "entity": e})

    vectorstore = MyFAISS.from_texts(
        summaries,
        embedding=OpenAIEmbeddings(
            model="text-embedding-3-large", tiktoken_model_name="cl100k_base"
        ),
        metadatas=metadata_ls,
    )

    retriever = vectorstore.as_retriever(search_type="similarity_score_threshold",
                                     search_kwargs={"score_threshold": config.SCORE_THRESHOLD})

    setup_and_retrieval = RunnableParallel(
        {"context": retriever, "question": RunnablePassthrough()}
    )
    langchain_output_parser = RunnableLambda(process_langchain_retriever_output)
    api_retrieval_response_to_dict = RunnableLambda(lambda x: x.model_dump())

    retrieval_chain = (
        setup_and_retrieval | langchain_output_parser | api_retrieval_response_to_dict
    )

    return retrieval_chain
