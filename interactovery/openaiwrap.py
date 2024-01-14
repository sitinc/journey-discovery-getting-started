# MIT License
#
# Copyright (c) 2023, Justin Randall, Smart Interactive Transformations Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from abc import ABC, abstractmethod

import openai
from openai import OpenAI, OpenAIError
from requests import RequestException

from interactovery import Utils

import random
import time

import logging

log = logging.getLogger('openAiLogger')


# Implement exponential backoff as per:
# https://platform.openai.com/docs/guides/rate-limits/error-mitigation?context=tier-two

def retry_with_exponential_backoff(
        func,
        initial_delay: float = 1,
        exponential_base: float = 2,
        jitter: bool = True,
        max_retries: int = 10,
        errors: tuple = (openai.RateLimitError,),
):
    """Retry a function with exponential backoff."""

    def wrapper(*args, **kwargs):
        # Initialize variables
        num_retries = 0
        delay = initial_delay

        # Loop until a successful response or max_retries is hit or an exception is raised
        while True:
            try:
                return func(*args, **kwargs)

            # Retry on specific errors
            except errors as e:
                log.error(f"OpenAI API rate-limit error: {e}")

                # Increment retries
                num_retries += 1

                # Check if max retries has been reached
                if num_retries > max_retries:
                    raise Exception(
                        f"Maximum number of retries ({max_retries}) exceeded."
                    )

                # Increment the delay
                delay *= exponential_base * (1 + jitter * random.random())

                # Sleep for the delay
                time.sleep(delay)

            # Raise exceptions for any errors not specified
            except Exception as e:
                raise e

    return wrapper


# Implement the client and command classes.
class OpenAiWrapProxy(ABC):
    """
    Define OpenAiWrap proxy interface to support passing OpenAiWrap into OpenAiCommand.
    """

    @abstractmethod
    def completions_backoff(self, **kwargs):
        """Wrap client.chat.completions.create with @retry_with_exponential_backoff."""
        pass

    @abstractmethod
    def embeddings_backoff(self, **kwargs):
        """Wrap client.embeddings.create with @retry_with_exponential_backoff."""
        pass

    @abstractmethod
    def execute(self, **kwargs):
        """Submit a command for execution."""
        pass


class OpenAiCommand(ABC):
    """
    Abstract command object for implementing OpenAI API commands.
    """

    def __init__(self, session_id: str = None, cmd_name: str = None):
        """
        Construct a new instance.
        :param session_id: The session_id.
        """

        if session_id is None:
            session_id = Utils.new_session_id()

        if cmd_name is None:
            raise Exception('cmd_name is required')

        self.session_id: str = session_id
        self.cmd_name: str = cmd_name
        self.exec_time: float | None = None
        self.openai_id: str | None = None
        self.openai_finish_reason: str | None = None
        self.openai_logprobs = None
        self.openai_system_fingerprint: str | None  = None
        self.completion_tokens: int | None = None
        self.prompt_tokens: int | None = None
        self.total_tokens: int | None = None
        self.response = None
        self.result = None

    @abstractmethod
    def run(self, client: OpenAiWrapProxy):
        """
        Abstract method for executing command logic.
        :param client: The OpenAI API client.
        """
        pass

    @abstractmethod
    def input_key(self):
        """
        Abstract method retrieving the command input key.
        """
        pass

    def output_key(self):
        """
        Implement output_key for base class.
        """
        return {
            'id': self.openai_id,
            'system_fingerprint': self.openai_system_fingerprint,
            'finish_reason': self.openai_finish_reason,
            'completion_tokens': self.completion_tokens,
            'prompt_tokens': self.prompt_tokens,
            'total_tokens': self.total_tokens,
            'exec_time': self.exec_time,
            # 'content': self.result,
        }

    def __str__(self):
        return (f"OpenAiCommand(session_id={self.session_id}" +
                f", cmd_name={self.cmd_name}" +
                f", exec_time={self.exec_time}" +
                f", openai_id={self.openai_id}" +
                f", openai_finish_reason={self.openai_finish_reason}" +
                f", openai_logprobs={self.openai_logprobs}" +
                f", openai_system_fingerprint={self.openai_system_fingerprint}" +
                f", completion_tokens={self.completion_tokens}" +
                f", prompt_tokens={self.prompt_tokens}" +
                f", total_tokens={self.total_tokens}" +
                f", response={self.response}" +
                f", result={self.result}" +
                ")")

    def __repr__(self):
        return (f"OpenAiCommand(session_id={self.session_id!r}" +
                f", cmd_name={self.cmd_name!r}" +
                f", exec_time={self.exec_time!r}" +
                f", openai_id={self.openai_id!r}" +
                f", openai_finish_reason={self.openai_finish_reason!r}" +
                f", openai_logprobs={self.openai_logprobs!r}" +
                f", openai_system_fingerprint={self.openai_system_fingerprint!r}" +
                f", completion_tokens={self.completion_tokens!r}" +
                f", prompt_tokens={self.prompt_tokens!r}" +
                f", total_tokens={self.total_tokens!r}" +
                f", response={self.response!r}" +
                f", result={self.result!r}" +
                ")")


class OpenAiWrap(OpenAiWrapProxy):
    """OpenAI API Wrapper class."""

    def __init__(self, org_id: str, api_key: str, max_retries=3):
        """
        Construct a new instance.

        :param org_id: The OpenAI API organization ID.
        :param api_key: The OpenAI API key.
        :param max_retries: The maximum number of retries due to errors not related to rate limiting.  Default is 3.
        """
        self.org_id = org_id
        self.api_key = api_key
        self.max_retries = max_retries
        self.client = OpenAI(
            organization=org_id,
            api_key=api_key,
        )

    @retry_with_exponential_backoff
    def completions_backoff(self, **kwargs):
        """Wrap client.chat.completions.create with @retry_with_exponential_backoff."""
        return self.client.chat.completions.create(**kwargs)

    @retry_with_exponential_backoff
    def embeddings_backoff(self, **kwargs):
        """Wrap client.embeddings.create with @retry_with_exponential_backoff."""
        return self.client.embeddings.create(**kwargs)

    def execute(self, cmd: OpenAiCommand, retries=0) -> OpenAiCommand:
        """
        Submit a command for execution.
        :param cmd: the OpenAiCommand instance.
        :param retries: The recursive retries counter.  Do not set.
        :return: The completed command instance.
        """
        if retries >= self.max_retries:
            raise Exception(f'{cmd.session_id} | {cmd.cmd_name} | Failed after {self.max_retries} attempts.')

        try:
            log.debug(f"{cmd.session_id} | {cmd.cmd_name} | Request: {cmd}")
            log.debug(f"{cmd.session_id} | {cmd.cmd_name} | Input: {cmd.input_key()}")

            start_time = time.time()
            cmd_result: OpenAiCommand = cmd.run(self)
            end_time = time.time()
            exec_time = end_time - start_time
            cmd_result.exec_time = exec_time

            choice_entry = cmd_result.response.choices[0]

            cmd_result.openai_id = cmd_result.response.id
            cmd_result.openai_finish_reason = choice_entry.finish_reason
            cmd_result.openai_logprobs = choice_entry.logprobs
            cmd_result.openai_system_fingerprint = cmd_result.response.system_fingerprint
            cmd_result.completion_tokens = cmd_result.response.usage.completion_tokens
            cmd_result.prompt_tokens = cmd_result.response.usage.prompt_tokens
            cmd_result.total_tokens = cmd_result.response.usage.total_tokens

            log.debug(f"{cmd.session_id} | {cmd.cmd_name} | Response: {cmd_result}")

            del cmd_result.response

            log.debug(f"{cmd.session_id} | {cmd.cmd_name} | Output: {cmd_result.output_key()}")

            return cmd_result
        except OpenAIError as e:  # Handle OpenAI-specific errors
            retries = retries + 1
            log.error(f"{cmd.session_id} | {cmd.cmd_name} | Attempt #{retries} | OpenAI API error: {e}")
            return self.execute(cmd, retries)
        except RequestException as e:  # Handle network-related errors
            retries = retries + 1
            log.error(f"{cmd.session_id} | {cmd.cmd_name} | Attempt #{retries} | Request error: {e}")
            return self.execute(cmd, retries)
        except Exception as e:  # Handle other unexpected errors
            retries = retries + 1
            log.error(f"{cmd.session_id} | {cmd.cmd_name} | Attempt #{retries} | An unexpected error occurred: {e}")
            return self.execute(cmd, retries)


class CreateEmbeddings(OpenAiCommand):
    """
    Command object for OpenAI create embeddings requests.
    """

    def __init__(self,
                 *,
                 session_id: str,
                 utterances: list[str] = None,
                 engine: str = "text-similarity-babbage-001"
                 ):
        """
        Construct a new instance.
        :param session_id: The session ID.
        :param utterances: The utterances.
        :param engine: The embeddings engine.  Default is "text-similarity-babbage-001".
        """
        super().__init__(
            session_id,
            'CreateEmbeddings'
        )
        if utterances is None:
            raise Exception("utterances is required")

        self.utterances = utterances
        self.engine = engine

    def input_key(self):
        return {
            'engine': self.engine,
            'utterances': self.utterances
        }

    def run(self, client: OpenAiWrap) -> OpenAiCommand:
        response = client.embeddings_backoff(
            input=self.utterances,
            engine=self.engine
        )
        self.response = response
        self.result = [embedding['embedding'] for embedding in response['data']]
        return self


class CreateCompletions(OpenAiCommand):
    """
    Command object for OpenAI chat completion requests.
    """

    def __init__(self,
                 *,
                 session_id: str,
                 user_prompt: str = None,
                 model: str = "gpt-4-1106-preview",
                 sys_prompt: str = "You are a helpful assistant"
                 ):
        """
        Construct a new instance.
        :param session_id: The session ID.
        :param user_prompt: The chat completion user prompt.
        :param sys_prompt: The chat completion system prompt.  The default is "You are a helpful assistant".
        :param model: The chat completion model.  The default is "gpt-4-1106-preview".
        """
        super().__init__(
            session_id,
            'CreateCompletions'
        )
        if user_prompt is None:
            raise Exception("user_prompt is required")

        self.user_prompt = user_prompt
        self.sys_prompt = sys_prompt
        self.model = model

    def input_key(self):
        return {
            'user_prompt': len(self.user_prompt),
            'sys_prompt': len(self.sys_prompt),
            'model': self.model
        }

    def run(self, client: OpenAiWrap) -> OpenAiCommand:
        """
        Execute the command logic.
        :param client: The OpenAiWrap instance.
        :return: the completed command instance.
        """
        final_messages = [
            {"role": "system", "content": self.sys_prompt},
            {"role": "user", "content": self.user_prompt}
        ]
        response = client.completions_backoff(
            messages=final_messages,
            model=self.model,
        )
        self.response = response
        self.result = response.choices[0].message.content
        return self

    def __str__(self):
        return (f"CreateCompletions(super={super().__str__()}" +
                f", model={self.model}" +
                f", sys_prompt={self.sys_prompt}" +
                f", user_prompt={self.user_prompt}" +
                ")")

    def __repr__(self):
        return (f"CreateCompletions(super={super().__repr__()}" +
                f", model={self.model!r}" +
                f", sys_prompt={self.sys_prompt!r}" +
                f", user_prompt={self.user_prompt!r}" +
                ")")
