import os
import re
import json
import asyncio
import logging
import httpx
from typing import List, Tuple

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
    
    # Get user query and remove the bot's @ handle 
    user_query = event.get('text')
    if "--debug" in user_query:
        logger.info(event)
    logger.info(f"Received user query: {user_query}")

    ts = event.get("ts")
    username = event.get('user')

    # Send initial response to the user
    await say(f"Working on it :D", channel=SLACK_CHANNEL_ID, thread_ts=ts)

    # Get content suggestions
    logger.info("Retrieving content suggestions...")
    # response: List[Tuple[ExplainedChunk, str]] = await process_query(Query(query=query))
    async with httpx.AsyncClient(timeout=1200.0) as content_client:
        slack_response, debug_slack_response: Tuple[str, str] = await content_client.post(
            "http://localhost:8008/get_content", 
            json={"query": user_query, "username": username}
        )
        
    await say(slack_response, channel=SLACK_CHANNEL_ID, thread_ts=ts)
    logger.info(f"Sent message: {slack_response}\n")
    if len(debug_slack_response) > 1:
        await say(debug_slack_response, channel=SLACK_CHANNEL_ID, thread_ts=ts)
        logger.info(f"Sent debug message: {debug_slack_response}\n")
    logger.info(f"Sent message: {slack_response}")

async def main():
    handler = AsyncSocketModeHandler(app, SLACK_APP_TOKEN)
    await handler.start_async()

if __name__ == "__main__":
    asyncio.run(main())

