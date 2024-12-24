import argparse
import datetime as dt
import json
import os
from pathlib import Path

from .data import ConversationContext
from .talk_agent import OllamaTalkAgent


def main(
    name: str,
    whoami: str,
    persona_file: Path,
    load_conversation: Path,
    save_conversation: bool,
    model: str = "qwq:latest",
):
    conversation = []
    if load_conversation is not None and load_conversation.exists():
        with load_conversation.open("r") as fid:
            conversation = json.load(fid)["conversation"]
    with persona_file.open("r") as f:
        persona = f.read()
    agent = OllamaTalkAgent(persona=persona, name=name, model=model)
    time_str = dt.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    conv_dir = f"conv-{whoami}-{name}".replace(" ", "_")
    if not os.path.exists(conv_dir):
        os.mkdir(conv_dir)
    conv_logs = f"{conv_dir}/{time_str}.txt"
    print(
        "'submit' or empty input to submit your message and get response; 'quit' or 'q' to exit;\n"
    )
    with ConversationContext() as ctx:
        for msg in conversation:
            ctx.add_message(by=msg["by"], content=msg["content"])
        for msg in ctx.conversation:
            print(f"{msg.by}: {msg.content}\n")
        while True:
            user_input = input(f"{whoami}: ").strip()
            match user_input.lower():
                case "quit" | "q":
                    if not save_conversation:
                        save_conversation = input(
                            "Save conversation? (y/n): "
                        ).strip().lower() in ["y", "yes"]
                    break
                case "submit" | "":
                    print("\r", end="")
                    agent_response = _safe_get_agent_response(agent, ctx)
                    print(
                        f"{name} (mood: '{agent_response.mood}({agent_response.sentiment})', tone: {agent_response.tone!r}): {agent_response.text_response}"
                    )
                    ctx.add_message(by=agent.name, content=agent_response.text_response)
                case "s":
                    save_conversation = True
                case _:
                    ctx.add_message(by=whoami, content=user_input)
        if save_conversation and ctx.conversation:
            with open(conv_logs, "w") as fid:
                for msg in ctx.conversation:
                    fid.write(f"{msg.by}: {msg.content}\n")
            with open(conv_logs.replace(".txt", ".json"), "w") as fid:
                fid.write(ctx.model_dump_json(indent=4))


def _safe_get_agent_response(agent: OllamaTalkAgent, ctx: ConversationContext):
    while True:
        try:
            return agent.get_response(ctx.conversation, temperature=0.6)
        except Exception:
            ...


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hoho Talk CLI")
    parser.add_argument("--name", help="the name of the agent", required=True)
    parser.add_argument("-m", "--model", help="the model to use", default="qwq:latest")
    parser.add_argument("-w", "--whoami", help="your name", required=True)
    parser.add_argument(
        "-p",
        "--persona-file",
        type=Path,
        required=True,
        help="the text file of the persona of the agent",
    )
    parser.add_argument(
        "-l",
        "--load-conversation",
        help="the conversation file to load",
        type=Path,
    )
    parser.add_argument(
        "-s",
        "--save-conversation",
        action="store_true",
        help="save the conversation after exit",
    )
    kwargs = vars(parser.parse_args())
    main(**kwargs)
