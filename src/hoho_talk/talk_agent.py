import json
import logging
from copy import deepcopy
from typing import Optional

from ollama import Client

from .data import (
    AgentResponse,
    BlockType,
    ContextBlock,
    ConversationMessage,
    CriticResponse,
)
from .tools import ToolRegistry
from .utils import dedup_tool_calls, format_conversation, parse_json_response

_logger = logging.getLogger(__name__)


class OllamaAgent:

    def __init__(self, client: Optional[Client] = None):
        if client is None:
            client = Client()
        self._client = client


class OllamaTalkAgent(OllamaAgent):
    def __init__(
        self,
        name,
        persona,
        client=None,
        model="qwq:latest",
        revision_trials=3,
        historical_context_blocks: list[ContextBlock] = None,
        extra_sys_prompt: Optional[str] = None,
    ):
        super().__init__(client)
        if historical_context_blocks is None:
            historical_context_blocks = []
        else:
            historical_context_blocks = deepcopy(historical_context_blocks)
            for block in historical_context_blocks:
                block.message_id = None
        self.__model = model[:]
        self.__name = name[:]
        self.__persona = persona[:]
        example_blocks = [
            ContextBlock(block_type=block_type.value, block_content="...")
            for block_type in BlockType
        ]
        block_example_str = "\n".join([str(block) for block in example_blocks])
        self.__sys_prompt = f"""\
You are a helpful assistant.
Your task is to give a possible text response by a person (the 'respondent') given his/her persona.

When derieving the text response, also come up with a rationale of it.
The text response MUST be in the same language as the conversation.

You might be provided with the context of the conversation which consists of multiple blocks.
For example:
```
{block_example_str}
```
"""
        self.__sys_prompt += (
            "The description of the context block types are as follows:\n"
        )
        for block in example_blocks:
            block.model_dump
            self.__sys_prompt += f"""\
- {block.block_type!r}: {block.block_type.description}
"""

        #         self.__sys_prompt += """\

        # You also have access to the tool, `insert_context_block`, which allows you to update the context blocks.
        # Use the `insert_context_block` tool when there is significant information in the conversation which is not included in the context blocks."""
        if extra_sys_prompt is not None:
            self.__sys_prompt += "\n\n" + extra_sys_prompt
        self.__context_blocks = historical_context_blocks
        self.__revision_trials = revision_trials

    @property
    def persona(self):
        return self.__persona

    @property
    def model(self):
        return self.__model

    @property
    def name(self):
        return self.__name

    def get_response(
        self,
        conversation: list[ConversationMessage],
        temperature=0.2,
    ):
        final_response = self.__revise_by_critic(
            conversation=conversation,
            agent_response=self.__get_agent_response(conversation, temperature),
        )
        return final_response

    def __get_agent_response(
        self, conversation: list[ConversationMessage], temperature: float
    ):
        conversation_str = format_conversation(conversation)
        _logger.debug("conversation:\n%s", conversation_str)
        messages = [
            {"role": "system", "content": self.__compile_sys_prompt()},
            {
                "role": "user",
                "content": f"The person you will represent in the conversation is {self.__name}.",
            },
            {
                "role": "user",
                "content": f"""\
The persona of {self.__name} is as following, delimited by ```:
```
{self.__persona}
```
""",
            },
            {
                "role": "user",
                "content": f"""
The conversation by far is as following:
```
{conversation_str}
```
""",
            },
            {
                "role": "user",
                "content": f"""\
Write me your response in JSON, which complies with the following schema:
```json
{json.dumps(AgentResponse.to_simple_json_schema(), indent=4)}
```
""",
            },
        ]
        chat_response = self._client.chat(
            model=self.__model,
            messages=messages,
            options={"temperature": temperature},
            # format=AgentResponse.model_json_schema(),
        )
        _logger.debug("chat response: %s", chat_response.message.content)
        return AgentResponse(
            **parse_json_response(chat_response.message.content.strip())
        )
        # return AgentResponse.model_validate_json(chat_response.message.content.strip())

    def __revise_by_critic(
        self,
        conversation: list[ConversationMessage],
        agent_response: AgentResponse,
    ) -> AgentResponse:
        revised_response = agent_response
        critic_agent = OllamaCriticAgent(client=self._client, model=self.__model)
        with OllamaReviseAgent(client=self._client, model=self.__model) as revise_agent:
            for _ in range(self.__revision_trials):
                critic_response = critic_agent.critic(
                    revised_response,
                    by=self.__name,
                    persona=self.__persona,
                    conversation=conversation,
                )
                if critic_response.is_aligned:
                    break
                revised_response = revise_agent.revise(
                    revised_response,
                    critic_response=critic_response,
                )
            else:
                _logger.debug(
                    "Does not reach the final revision after %d trials",
                    self.__revision_trials,
                )
        return revised_response

    def __compile_sys_prompt(self):
        sys_prompt = f"""\
{self.__sys_prompt}

"""
        for block in self.__context_blocks:
            sys_prompt += "The conversation context:\n"
            sys_prompt += f"""\
{block}
"""
        return sys_prompt.strip()


class OllamaCriticAgent(OllamaAgent):

    def __init__(self, client=None, model="qwq:latest"):
        super().__init__(client)
        self.__model = model[:]

    def critic(
        self,
        agent_response: AgentResponse,
        by: str,
        persona: str,
        conversation: list[ConversationMessage],
        temperature=0.1,
    ) -> CriticResponse:
        sys_prompt = """\
You will be given a response by a person and his/her persona.
Your task is to evaluate the response is aligned with his/her persona.
"""
        conversation_str = format_conversation(conversation)
        messages = [
            {
                "role": "system",
                "content": sys_prompt,
            },
            {
                "role": "user",
                "content": f"""\
The conversation by far is as following:
```
{conversation_str}
```
""",
            },
            {
                "role": "user",
                "content": f"""\
One possilbe response , which is in JSON format,  by {by} is as follows:
```
{agent_response.model_dump_json(indent=4)}
```
""",
            },
            {
                "role": "user",
                "content": f"""\
{by}'s persona is as follows:
```
{persona}
```
""",
            },
            {
                "role": "user",
                "content": f"""\
Considering the conversation so far and the response by {by}, evaluate if the response to the conversation is aligned with his/her persona.
""",
            },
            {"role": "user", "content": "Are you done with your judgement?"},
            {"role": "assistant", "content": "Yes."},
            {
                "role": "user",
                "content": f"""\
Write your judgement in JSON, which complies with the following schema:
```json
{json.dumps(CriticResponse.to_simple_json_schema(), indent=4)}
```
""",
            },
            {"role": "assistant", "content": "Ok, this is my judgement:"},
        ]
        response = self._client.chat(
            model=self.__model,
            messages=messages,
            options={"temperature": temperature},
        )
        critic_response = CriticResponse(
            **parse_json_response(response.message.content.strip())
        )
        _logger.debug(
            "critic response (%s): %s",
            critic_response.is_aligned,
            critic_response.rationale,
        )
        return critic_response


class OllamaReviseAgent(OllamaAgent):
    def __init__(self, client=None, model="qwq:latest"):
        super().__init__(client)
        self.__model = model[:]
        self.__sys_prompt = """\
Your task is to revise the given response according to user's critics and suggestions.

The revision should be in the same language as the original response.
"""
        self.__prev_critic_response_pairs: list[
            tuple[
                CriticResponse, AgentResponse, AgentResponse
            ]  # (critic, original, revised)
        ] = []

    def revise(
        self,
        agent_response: AgentResponse,
        critic_response: CriticResponse,
    ) -> AgentResponse:
        response_json_str = agent_response.model_dump_json(indent=4)
        messages = [
            {
                "role": "system",
                "content": self.__sys_prompt,
            }
        ]
        if self.__prev_critic_response_pairs:
            num_revisions = len(self.__prev_critic_response_pairs)
            prev_revision_str = ("--" * 10).join(
                f"""
Critic:
{prev_critic.rationale}

Original Response:
{prev_ori.model_dump_json(indent=4)}

Revised Response:
{prev_revision.model_dump_json(indent=4)}
"""
                for (
                    prev_critic,
                    prev_ori,
                    prev_revision,
                ) in self.__prev_critic_response_pairs
                if not prev_critic.is_aligned
            )
            messages.extend(
                [
                    {
                        "role": "user",
                        "content": f"You've revised the response for {num_revisions} times already but still have room for improvement.",
                    },
                    {
                        "role": "user",
                        "content": f"""\
Here are the previous critics and your revisions:
{'--' * 10}
{prev_revision_str}
{'--' * 10}
""",
                    },
                    {
                        "role": "user",
                        "content": f"""\
For this time, the response for revision is as following:

{response_json_str}""",
                    },
                ]
            )
        else:
            messages.append(
                {
                    "role": "user",
                    "content": f"""\
The response for revision is as following:

{response_json_str}""",
                }
            )
        messages.extend(
            [
                {
                    "role": "user",
                    "content": f"""\
After I reviewed the response, I found it not aligned with the persona of the person in the conversation.
This is my rationale:
{critic_response.rationale!r}
""",
                },
                {
                    "role": "user",
                    "content": f"""\
My suggestion on the revision of the response is as following:
{critic_response.suggest_change!r}
""",
                },
                {
                    "role": "user",
                    "content": "Write me the revised response according to my suggestion in JSON.",
                },
                {
                    "role": "assistant",
                    "content": "Here you go:",
                },
            ]
        )
        response = self._client.chat(
            model=self.__model,
            messages=messages,
            options={"temperature": 0.1},
        )
        revised_agent_response = AgentResponse(
            **parse_json_response(response.message.content.strip())
        )
        self.__prev_critic_response_pairs.append(
            (critic_response, agent_response, revised_agent_response)
        )
        return revised_agent_response

    def _reset(self):
        self.__prev_critic_response_pairs = []

    def __enter__(self):
        self._reset()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._reset()
        return False
