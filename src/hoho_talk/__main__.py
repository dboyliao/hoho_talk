import argparse
import datetime as dt
import json
import os
from pathlib import Path
from typing import Optional

import click

from .data import ContextBlock, ConversationContext
from .talk_agent import OllamaTalkAgent


def main(
    name: str,
    whoami: str,
    persona_file: Path,
    load_conversation: Optional[Path] = None,
    context_blocks_file: Optional[Path] = None,
    model: str = "qwq:latest",
    save_directory: Optional[str] = None,
):
    conversation = []
    if load_conversation is not None and load_conversation.exists():
        with load_conversation.open("r") as fid:
            conversation = json.load(fid)["conversation"]
    context_blocks: list[ContextBlock] = []
    if context_blocks_file is not None and context_blocks_file.exists():
        with context_blocks_file.open("r") as fid:
            context_blocks = [
                ContextBlock.model_validate_json(cb) for cb in json.load(fid)
            ]
    with persona_file.open("r") as f:
        persona = f.read()
    agent = OllamaTalkAgent(
        persona=persona,
        name=name,
        model=model,
        historical_context_blocks=context_blocks,
    )
    time_str = dt.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    conv_dir = "conv" if save_directory is None else save_directory
    names_prefix = f"{whoami}-{name}".replace(" ", "_")
    if not os.path.exists(conv_dir):
        os.mkdir(conv_dir)
    conv_logs = f"{conv_dir}/{names_prefix}-{time_str}.txt"
    click.secho(
        "'submit' or empty input to submit your message and get response; 'quit' or 'q' to exit;\n",
        bold=True,
    )
    with ConversationContext() as ctx:
        for msg in conversation:
            ctx.add_message(**msg)
        for msg in ctx.conversation:
            click.echo(f"{msg}\n")
        while True:
            user_input = input(f"{whoami}: ").strip()
            match user_input.lower():
                case "quit" | "q":
                    save_conversation = input(
                        "Save conversation? (y/n): "
                    ).strip().lower() in ["y", "yes"]
                    break
                case "submit" | "":
                    agent_response = _safe_get_agent_response(agent, ctx)
                    ctx.add_message(
                        by=agent.name,
                        content=agent_response.text_response,
                        mood=agent_response.mood,
                        tone=agent_response.tone,
                        sentiment=agent_response.sentiment,
                    )
                    click.echo(f"{ctx.conversation[-1]}")
                case "s":
                    save_conversation = True
                case _:
                    ctx.add_message(by=whoami, content=user_input)
        if save_conversation and ctx.conversation:
            with open(conv_logs, "w") as fid:
                for msg in ctx.conversation:
                    fid.write(f"{msg}\n")
            with open(conv_logs.replace(".txt", ".json"), "w") as fid:
                fid.write(ctx.model_dump_json(indent=4))


def _safe_get_agent_response(agent: OllamaTalkAgent, ctx: ConversationContext):
    while True:
        try:
            return agent.get_response(ctx.conversation, temperature=0.6)
        except Exception:
            ...


def get_parser():
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
        "-c",
        "--context-blocks-file",
        help="the context blocks file to load",
        type=Path,
    )
    parser.add_argument(
        "-s",
        "--save-directory",
        help="the directory to save the conversation logs/records",
    )
    return parser


def run_main():
    parser = get_parser()
    kwargs = vars(parser.parse_args())
    try:
        main(**kwargs)
    except KeyboardInterrupt:
        click.secho("Bye!", bold=True)


if __name__ == "__main__":
    run_main()
