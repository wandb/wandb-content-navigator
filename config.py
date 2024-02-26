import os

from dotenv import load_dotenv
from pydantic_settings import BaseSettings

load_dotenv()

class NavigatorConfig(BaseSettings):
    API_KEY: str = os.getenv('OPENAI_API_KEY')
    EXPLANATION_MODEL: str = "gpt-4-0125-preview"
    QUERY: str = 'do we have any reports I could send to a finance company?'
    TOP_K: int = 4
    SCORE_THRESHOLD: float = 0.15
    INITIAL_K: int = 100
    LANGUAGE: str = ''  # 'en'
    INCLUDE_TAGS: list[str] = ['']  # ['contains-wandb-code']
    EXCLUDE_TAGS: list[str] = ['']  # ['ml-news', 'gradient-dissent']
    NON_ENGLISH_REGEX_SEARCH: str = r'[^\x00-\x7F]'
    N_SOURCES_TO_SEND: int = 5  # Number of suggestions to send to the user
    FULLY_CONNECTED_SUMMARIES_FILEPATH: str = "data/fully_connected_summaries_final.csv"