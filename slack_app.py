import os
import re
import asyncio
import logging
from typing import List, Tuple
from urllib.parse import quote, unquote

from slack_bolt.async_app import AsyncApp
from slack_bolt.adapter.socket_mode.async_handler import AsyncSocketModeHandler
from dotenv import load_dotenv

from main import ExplainedChunk, Query, process_query

load_dotenv('.env')
logging.basicConfig(level=logging.INFO)

SLACK_CHANNEL_ID = os.getenv('SLACK_CHANNEL_ID')
SLACK_APP_TOKEN = os.getenv('SLACK_APP_TOKEN')
SLACK_BOT_TOKEN = os.getenv('SLACK_BOT_TOKEN')

N_SOURCES_TO_SEND = 10

# CONTENT_SUGGESTIONS = f"Source: {source}\nreason_why_helpful: {explanation.reason_why_helpful}\n \
# content_is_relevant: {explanation.content_is_relevant}\nDescription: {explanation.content_description}"

app = AsyncApp(token=SLACK_BOT_TOKEN)

@app.event("message")
async def handle_message_events(body, logger):
    logger.info("Message received")
    # logger.info(body)


@app.event("app_mention")
async def handle_app_mentions(event, say, logger):
    logger.info("App mention received:")
    logger.info(event)

    # Get user query and remove the bot's @ handle 
    user_query = event.get('text')
    logger.info(f"Received user query: {user_query}")
    user_query = re.sub(r"\<@\w+\>", "", user_query).strip()

    ts = event.get("ts")
    user = event.get('user')
    # event_type = event.get("type")
    # event_ts = event.get("event_ts")
    # channel = event.get("channel")
    
    # Get content suggestions
    logger.info("Retrieving content suggestions...")
    # Remove --debug from query if present
    if '--debug' in user_query:
        query = user_query.replace('--debug', '')
    else:
        query = user_query
    response: List[Tuple[ExplainedChunk, str]] = await process_query(Query(query=query))
    len_explanations = len(response)
    logger.info(f"{len_explanations} content explanations created.")

    ### FILTER OUT SOURCES THAT THE MODEL THINKS ARE IRRELEVANT ###
    # Remove any sources that weren't considered useful
    cleaned_response = [(explanation, source, score) for explanation, source, score in response if explanation.content_is_relevant is True]
    logger.info(f"{len_explanations - len(cleaned_response)} pieces of content found to be irrelevant and removed")

    logger.info(f"RESPONSE: LEN: {len(cleaned_response)}, {cleaned_response}")

    ### SEND CONTENT SUUGESTIONS BACK TO SLACK USER ###
    # Response if no content suggestions found
    if len(cleaned_response) == 0:
        slack_text = f"Hey <@{user}>, no content suggestions found. Try rephrasing your query."
    # Response if content suggestions found
    else:
        slack_text = f"Hey <@{user}>, content suggestions below:\n\n"
        for explanation, source, score in cleaned_response[:N_SOURCES_TO_SEND]:
            # source = unquote(quote(str(source)))
            # fix links that have spaces
            source = source.replace(' ', '%20')

            # Show more info in debug mode
            if '--debug' not in user_query:
                slack_text += f"• {explanation.content_description} - <{source}|Link>\n\n" #*{source}*\n\n" # *<{source}|Link>*\n\n"
            else:
                slack_text += f"*Score*: {score}, *Source*: {source}\n*reason_why_helpful*:\
{explanation.reason_why_helpful}\n*chain_of_thought*: {explanation.chain_of_thought}\n\
*content_is_relevant*: {explanation.content_is_relevant}\n*content_description*: {explanation.content_description}\n\n"
    
    await say(slack_text, channel=SLACK_CHANNEL_ID, thread_ts=ts)
    logger.info(f"Sent message: {slack_text}")
    
    ### PRINT REJECTED SOURCES IN SLACK IN DEBUG MODE ###
    if '--debug' in user_query:
        rejected_responses = [(explanation, source, score) for explanation, source, score in response if explanation.content_is_relevant is not True]
        for explanation, source, score in rejected_responses:
            source = source.replace(' ', '%20')
            slack_response = f"REJECTED, *Score*: {score}, *Source*: {source}\n*reason_why_helpful*:\
{explanation.reason_why_helpful}\n*chain_of_thought*: {explanation.chain_of_thought}\n\
*content_is_relevant*: {explanation.content_is_relevant}\n*content_description*: {explanation.content_description}\n\n"
            await say(slack_response, channel=SLACK_CHANNEL_ID, thread_ts=ts)
        # logger.info(f"Sent message: {slack_response}")


    # else:
    #     await say(f"There was an issue with processing source: {source}",
    #         channel=SLACK_CHANNEL_ID,
    #         thread_ts=ts)
    #     logger.error(f"There was an issue with processing source: {source}")

async def main():
    handler = AsyncSocketModeHandler(app, SLACK_APP_TOKEN)
    await handler.start_async()

if __name__ == "__main__":
    asyncio.run(main())

