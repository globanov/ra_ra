import os
import requests
import logging

from dotenv import load_dotenv

from raft_utils import auto_timeit

logger = logging.getLogger(__name__)

load_dotenv()

class YandexGPT:
    def __init__(self, api_key=None, folder_id=None):
        self.api_key = api_key or os.getenv("YANDEX_API_KEY")
        self.folder_id = folder_id or os.getenv("YANDEX_FOLDER_ID")
        self.url = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"

        if not self.api_key or not self.folder_id:
            logger.warning("YandexGPT API credentials not configured")

    @auto_timeit("yandexgpt_generate")
    def generate(self, prompt, system_prompt=None, temperature=0.1):
        headers = {
            "Authorization": f"Api-Key {self.api_key}",
            "Content-Type": "application/json"
        }

        messages = []
        if system_prompt:
            messages.append({"role": "system", "text": system_prompt})
        messages.append({"role": "user", "text": prompt})

        data = {
            "modelUri": f"gpt://{self.folder_id}/yandexgpt",
            "completionOptions": {
                "stream": False,
                "temperature": temperature,
                "maxTokens": 2000
            },
            "messages": messages
        }

        try:
            response = requests.post(self.url, headers=headers, json=data, timeout=30)
            response.raise_for_status()
            result = response.json()
            return result["result"]["alternatives"][0]["message"]["text"]
        except requests.exceptions.RequestException as e:
            return f"YandexGPT request error: {str(e)}"
        except (KeyError, IndexError) as e:
            return f"YandexGPT response parsing error: {str(e)}"
