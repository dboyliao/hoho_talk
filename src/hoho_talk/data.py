from enum import Enum
from typing import Optional, Type, Union
from uuid import uuid4

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
    message_id: str = Field(default_factory=lambda: f"msg-{uuid4()}")
    mood: Optional[str] = None
    tone: Optional[str] = None
    sentiment: Optional[str] = None

    def __str__(self):
        prefix = (
            f"{self.by}"
            if self.mood is None or self.tone is None or self.sentiment is None
            else f"{self.by} (mood: '{self.mood} ({self.sentiment.lower()})', tone: {self.tone.lower()!r})"
        )
        return f"{prefix}: {self.content}"


class BlockType(str, Enum):
    """
    The type of the block in the context.

    - `context`: the context block, which is the general context of the conversation.
    - `memory`: the memory block, which is the memory of the respondent in the conversation.
    """

    context = "context"
    memory = "memory"

    @property
    def description(self):
        return {
            "context": "the context block, which is the general context of the conversation.",
            "memory": "the memory block, which is the memory of the respondent in the conversation.",
        }[self.value]


class ContextBlock(BaseModel):
    block_type: BlockType
    block_content: str
    message_id: Union[str, None] = None

    def bind(self, message: ConversationMessage):
        """
        Bind the context block with a message.
        It establishes the relationship between the context block and the message.
        For example, a memory block bound to a message means that the memory block is the memory of the agent when the message is sent.
        """
        if self.message_id is not None:
            raise ValueError(
                f"The context block is already bound to a message: {self.message_id}"
            )
        self.message_id = message.message_id

    def __str__(self):
        self_str = (
            f"""\
<{self.block_type}>
  {self.block_content}
</{self.block_type}>
"""
            if self.message_id is None  # it's considered as a historical block
            else f"""\
<{self.block_type}>
(note: this block is formed when the conversation progresses to the message {self.message_id})
{self.block_content}
</{self.block_type}>
"""
        )
        return self_str


class ConversationContext(BaseModel):
    conversation_id: str = Field(default_factory=lambda: f"conv-{uuid4()}")
    conversation: list[ConversationMessage] = Field(default_factory=list)

    def add_message(self, by: str, content: str, mood=None, tone=None, sentiment=None):
        self.conversation.append(
            ConversationMessage(
                by=by, content=content, mood=mood, tone=tone, sentiment=sentiment
            )
        )
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
