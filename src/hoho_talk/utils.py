import json
import re

from .data import ConversationMessage

_TAILING_COMMA_PATTERN = re.compile(r",\n?}\n?$")


def parse_json_response(response_str: str, delimiter: str = "```") -> dict:
    escaped_delimiter = re.escape(delimiter)
    pattern = rf"(?:{escaped_delimiter})?(?:json)?([\s\S]*)(?:{escaped_delimiter})?"

    match = re.search(pattern, response_str.strip())
    if match:
        json_str = match.group(1).strip(delimiter)
        json_str = _TAILING_COMMA_PATTERN.sub("}", json_str)
    else:
        json_str = response_str
    return json.loads(json_str.split(delimiter)[0])


def format_conversation(conversation: list[ConversationMessage]):
    return "\n".join(f"{m.by}: {m.content}" for m in conversation)
