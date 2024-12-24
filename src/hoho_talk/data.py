from enum import Enum
from typing import Type, Union

from pydantic import BaseModel, Field


class _SimpleJsonSchemaMixin:

    @classmethod
    def to_simple_json_schema(cls: Type[BaseModel]):
        ori_schema = cls.model_json_schema()
        schema = {
            "type": "object",
            "properties": ori_schema["properties"],
        }
        if required := ori_schema.get("required"):
            schema["required"] = required
        return schema


class ConversationMessage(BaseModel):
    by: str
    content: str


class ConversationContext(BaseModel):
    conversation: list[ConversationMessage] = Field(default_factory=list)

    def add_message(self, by: str, content: str):
        self.conversation.append(ConversationMessage(by=by, content=content))
        return self

    def __enter__(self):
        self.conversation = []
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.conversation = []
        return False


class AgentResponse(_SimpleJsonSchemaMixin, BaseModel):

    mood: str = Field(description="the mood of the respondent")
    tone: str = Field(description="the voice tone of the response")
    sentiment: str = Field(
        description="the sentiment of the response (one of 'positive', 'negative' or 'neutral')"
    )
    rationale: str = Field(
        description="the rationale of the response by the respondent. It should be of the first person perspective."
    )
    text_response: str = Field(
        description="the response text. It should be in the same language as the conversation."
    )


class CriticResponse(_SimpleJsonSchemaMixin, BaseModel):
    is_aligned: bool = Field(
        description="if the user response is aligned with his/her persona"
    )
    rationale: str = Field(description="your rationale on your judgement")
    suggest_change: Union[str, None] = Field(
        description="The suggest change to the response if it is not aligned with the persona. The value should be null if it's aligned."
    )
