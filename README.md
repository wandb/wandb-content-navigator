# Content Navigator

The Content Navigator is designed to suggest relevant Weights & Biases content such as articles, case studies, white papers, courses, event recordings etc based on user queries.

## Workflow

1. **User Query**: A user mentions the Slack app in a channel with a query or hits the `/get_content` endpoint with a query of type `ContentNavigatorRequest`.
2. **Query Expansion**: The query is expanded in `expand_query` by asking an LLM to add relevant AI/ML topics 
3. **Content Retrieval**: Content is retrieved over a FAISS vectorstore using the expaned query
4. **Content Explanation**: The relevance of each piece of content is explained and assessed using an LLM in `explain_usefulness`. This includes a chain of thought as well as a boolen filter for whether or not the content is relevant to the original user query.
5. **Response Generation**: The app formats the content suggestions and explanations into a Slack message in a response of type `ContentNavigatorResponse`.
6. **Debug Messaging**: If "--debug" is included in the query, the app will also generate a debug message with more detailed reasoning for each content suggestion and the content that was considered not relevant in step 4.


## Usage

Run `run.sh` to run the app, this will start the FastAPI endpoint and the Slack app. 

**Environment Variables**
You will need to set the following environment variables in a `.env` file:

```
OPENAI_API_KEY
SLACK_APP_TOKEN
SLACK_BOT_TOKEN
SLACK_CHANNEL
```
 
**Config**

You can also set the following config variables in a `config.py` file:

```
EXPLANATION_MODEL  # OpenAI model to use for content explanation
TOP_K
SCORE_THRESHOLD  # Similarity threshold for content relevance (higher = more similar)
N_SOURCES_TO_SEND  # Number of sources to send in the response
FULLY_CONNECTED_SUMMARIES_FILEPATH  # Path to the fully connected article summaries csv
```

## Run Locally

To run the app locally, you will need to install the dependencies and run the FastAPI endpoint and the Slack app.

**Slack App**

To run just the slack app, run:

```bash
python slack_app.py
```

**FastAPI Endpoint**

To run just the FastAPI endpoint, run:

```bash
python main.py
```
