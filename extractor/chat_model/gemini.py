import os
from typing import Optional

from langchain_community.chat_models import ChatOpenAI


class ChatGemini(ChatOpenAI):
    openai_api_base: str
    openai_api_key: str
    model_name: str

    def __init__(
        self,
        model_name: str,
        openai_api_key: Optional[str] = None,
        openai_api_base: str = "https://generativelanguage.googleapis.com/v1beta/openai/",
        **kwargs,
    ):
        openai_api_key = openai_api_key or os.getenv("GOOGLE_API_KEY")
        super().__init__(
            openai_api_base=openai_api_base,
            openai_api_key=openai_api_key,
            model_name=model_name,
            **kwargs,
        )
