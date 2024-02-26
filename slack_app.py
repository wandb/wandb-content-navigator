import asyncio
import logging
import os

import httpx
from dotenv import load_dotenv
from slack_bolt.adapter.socket_mode.async_handler import AsyncSocketModeHandler
from slack_bolt.async_app import AsyncApp

load_dotenv(".env")
logging.basicConfig(level=logging.INFO)

SLACK_CHANNEL_ID = os.getenv("SLACK_CHANNEL_ID")
SLACK_APP_TOKEN = os.getenv("SLACK_APP_TOKEN")
SLACK_BOT_TOKEN = os.getenv("SLACK_BOT_TOKEN")

app = AsyncApp(token=SLACK_BOT_TOKEN)


@app.event("message")
async def handle_message_events(body, logger):
    logger.info("Message received")


@app.event("app_mention")
async def handle_app_mentions(event, say, logger):
    logger.info("App mention received:")

    # Get user query and remove the bot's @ handle
    user_query = event.get("text")
    if "--debug" in user_query:
        logger.info(event)
    logger.info(f"Received user query: {user_query}")

    ts = event.get("ts")
    user_id = event.get("user")

    # Send initial response to the user
    await say("Working on it :D", channel=SLACK_CHANNEL_ID, thread_ts=ts)

    # Get content suggestions
    logger.info("Retrieving content suggestions...")
    async with httpx.AsyncClient(timeout=1200.0) as content_client:
        response = await content_client.post(
            "http://localhost:8008/get_content",
            json={"query": user_query, "user_id": user_id},
        )
    if response.status_code == 200:
        data = response.json()  # Parse the JSON response body
        logger.info(f"Received content suggestions:\n{data}")
        slack_response = data.get("slack_response")
        rejected_slack_response = data.get("rejected_slack_response")
        response_items_count = data.get("response_items_count")
    else:
        error_msg = f"Failed to get content suggestions. Status code: \
{response.status_code}\n\nresponse: {response}"
        logger.error(error_msg)
        await say(error_msg, channel=SLACK_CHANNEL_ID, thread_ts=ts)
        return None

    if response_items_count > 0:
        await say(slack_response, channel=SLACK_CHANNEL_ID, thread_ts=ts)
        logger.info(f"Sent message: {slack_response}\n")
    else:
        await say("No content suggestions found. Try rephrasing your query, but note \
there may also not be any relevant pieces of content for this query. Add '--debug' to \
your query and try again to see a detailed resoning for each suggestion.", 
                  channel=SLACK_CHANNEL_ID, thread_ts=ts)
    if len(rejected_slack_response) > 1:
        await say(rejected_slack_response, channel=SLACK_CHANNEL_ID, thread_ts=ts)
        logger.info(f"Sent debug message: {rejected_slack_response}\n")

    logger.info("Finished answering query.")
    return None


async def main():
    handler = AsyncSocketModeHandler(app, SLACK_APP_TOKEN)
    await handler.start_async()


if __name__ == "__main__":
    asyncio.run(main())
