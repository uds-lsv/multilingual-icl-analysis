import openai
import asyncio
import aiolimiter
from tqdm.asyncio import tqdm_asyncio
from aiohttp import ClientSession
from typing import List
import os

from openicl.utils.logging import get_logger

logger = get_logger(__name__)

OPENICL_API_NAME_LIST = ['gpt3.5', 'gpt4']

OPENICL_API_REQUEST_CONFIG = {
    'gpt3.5': {
        'engine': "gpt-35-turbo-16k",
        'temperature': 0,
        'max_tokens': 50,
        'top_p': 1.0
    },
    'gpt4': {
        'engine': "gpt-4-32k",
        'temperature': 0,
        'max_tokens': 50,
        'top_p': 1.0
    }
}

# OpenAI APT setup
openai.api_type = "azure"
openai.api_base = os.getenv("AZURE_OPENAI_ENDPOINT")
openai.api_version = "2023-09-01-preview"
openai.api_key = os.getenv("AZURE_OPENAI_KEY")


def is_api_available(api_name):
    if api_name is None:
        return False
    return True if api_name in OPENICL_API_NAME_LIST else False


async def _throttled_openai_chat_completion_acreate(
        model: str,
        messages: List,
        temperature: float,
        max_tokens: int,
        top_p: float,
        limiter: aiolimiter.AsyncLimiter):
    async with limiter:
        for _ in range(100):
            try:
                return await openai.ChatCompletion.acreate(
                    engine=model,  # Azure uses "engine" or "deployment_id"
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                )
            except openai.error.RateLimitError:
                logger.warning(
                    "OpenAI API rate limit exceeded. Sleeping for 2 seconds."
                )
                await asyncio.sleep(2)
            except openai.error.APIConnectionError as e:
                logger.warning(f"OpenAI API connection error: {e}")
                await asyncio.sleep(2)
            except openai.error.Timeout: #asyncio.exceptions.TimeoutError:
                logger.warning(f"Request time out. Sleeping for 2 seconds.")
                await asyncio.sleep(2)
            except openai.error.APIError as e:
                logger.warning(f"OpenAI API error: {e}")
                break
            except openai.error.InvalidRequestError as e:  # e.g., content filter
                logger.warning(f"OpenAI invalid request error: {e}")
                break
        return {"choices": [{"message": {"content": ""}}]}


async def generate_from_openai_chat_completion(
        full_contexts: list,
        model: str,
        temperature: float,
        max_tokens: int,
        top_p: float,
        requests_per_minute: int = 50):
    """Generate from OpenAI Chat Completion API.

    Args:
        full_contexts: List of full contexts to generate from.
        model: The OpenAI model to use.
        temperature: Temperature to use.
        max_tokens: Maximum number of tokens to generate.
        top_p: Top p to use.
        requests_per_minute: Number of requests per minute to allow.

    Returns:
        List of generated responses.
    """
    session = ClientSession()
    openai.aiosession.set(session)
    limiter = aiolimiter.AsyncLimiter(requests_per_minute)
    async_responses = [
        _throttled_openai_chat_completion_acreate(
            model=model,
            messages=full_context,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            limiter=limiter,
        )
        for full_context in full_contexts
    ]
    responses = await tqdm_asyncio.gather(*async_responses)
    await session.close()

    outputs = []

    for response in responses:
        if 'content' in response["choices"][0]["message"]:
            outputs.append(response["choices"][0]["message"]["content"])
        else:
            outputs.append("")

    return outputs


def api_get_tokens(api_name, input_texts, max_tokens):
    if max_tokens is None:
        max_tokens = OPENICL_API_REQUEST_CONFIG[api_name]['max_tokens']

    responses = asyncio.run(generate_from_openai_chat_completion(full_contexts=input_texts,
                                                                 model=OPENICL_API_REQUEST_CONFIG[api_name]['engine'],
                                                                 temperature=OPENICL_API_REQUEST_CONFIG[api_name][
                                                                     'temperature'],
                                                                 max_tokens=max_tokens,
                                                                 top_p=OPENICL_API_REQUEST_CONFIG[api_name]['top_p']))
    return responses
