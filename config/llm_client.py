# Set OpenAI API key from environment variable
import os
from dotenv import load_dotenv
load_dotenv(override=True)

# OpenAI
from openai import OpenAI


DEFAULT_LLM_MODEL = "gpt-4o-mini"


def get_model() -> tuple[OpenAI, str]:
    """OpenAI 클라이언트와 모델명을 함께 반환. 호출부에서 messages만 지정하면 됨."""
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return client, DEFAULT_LLM_MODEL