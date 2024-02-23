'''
Instructor models and prompt templates
'''

from typing import List, Any
from pydantic import BaseModel, Field


class Query(BaseModel):
    '''A user query'''
    username: str = None
    query: str


class ContentNavigatorResponse(BaseModel):
    '''Response from the content navigator app'''
    slack_response: str
    rejected_slack_response: str = ""


class ExplainedChunk(BaseModel):
    '''Given a `query` from a user and a retrieved `chunk` of text, provide a brief description of the content of the `chunk` and \
whether it is relevant to the `query`.'''

    chain_of_thought: str = Field(...,
                                  description="Think step by step about whether the `chunk` is useful and helpful in answering the `query`."
                                  )
    reason_why_helpful: str = Field(...,
                                    description="A brief, 1 sentence explanation on what this blog post contains and a reason whether or not it \
is relevant based on the `query`. It is not considered relevant if the `chunk` does not directly answer the `query`."
                                    )
    content_description: str = Field(...,
                                     description="A brief, 1 sentence description of the content of the blog post. \
DO NOT say 'this blog post' or 'this chunk'. Instead, start wtih 'This covers...'. Do not use phrase like '...making it relevant to the query.'"
                                    )
    content_is_relevant: bool = Field(...,
                                      description="A boolean value indicating whether the content of the blog post is relevant to the `query`.\
If the content looks too much like a news piece covering a broad variety of topics, it is likely not relevant to the `query`.\
If the content doesn't directly address the `query` then its probably not relevant."
                                    )


class ExpandedQuery(BaseModel):
    '''Given a `query` from a user, expand on what the user may be looking for in order to make \
a semantic search match more likely.'''

    chain_of_thought: str = Field(..., 
                                  description="Think step by step about the given `query` and \
associated machine learning and artificial intelligence topics, including industry applications, and \
technological uses of ML and AI. Avoid concentrating excessively on any particular companies \
or entities mentioned in the request. Avoid topic like news or politics.")
    expanded_query: str = Field(..., 
                                description="An expanded query of machine learning topics that is more likely \
to match a semantic search based on the users `query`. Keep the topics focused on machine learning and artificial intelligence.")


# from wandbot.api.schemas import APIRetrievalRequest # uncomment once wandbot v1.1 is released
class APIRetrievalRequest(BaseModel):
    query: str
    language: str = "en"
    initial_k: int = 10
    top_k: int = 5
    include_tags: List[str] = []
    exclude_tags: List[str] = []

class APIRetrievalResult(BaseModel):
    text: str
    score: float
    metadata: dict[str, Any]

class APIRetrievalResponse(BaseModel):
    query: str
    top_k: List[APIRetrievalResult]


SYSTEM_PROMPT = '''Our Sales team want to send recommeneded links to propects and customers. \

Given a `query` from an internal employee as well as a `chunk` of text retrieved from a blog post, \
generate an explanation of why the `chunk` is useful in answering the `query`.

Finally, frame the final response in `reason_rephrased` to speak directly to a prospect or customer, as if you were \
on our sales team.
'''


EXPAND_SYSTEM_PROMPT = '''Given a `query` from a user, expand on what the user may be looking for in order to make\
a semantic search match more likely.'''


USER_PROMPT = '''Please explain why the following chunk is useful in answering the query:

>>>> query: {query}

>>>> chunk: {chunk}

Take a deep breath, if you do a good job I will tip you $200.'''


EXPAND_USER_PROMPT = '''Please expand on the following query to make a semantic search match more likely:

>>>> query: {query}

Take a deep breath, if you do a good job I will tip you $200.'''