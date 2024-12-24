# Setup
```bash
$ uv venv
$ uv sync --dev
$ uv pip install -e .
```

# Example
```bash
$ uv run python -m hoho_talk --persona-file examples/persona/kol_persona.txt  --name "KOL" --whoami "Boss (CEO of BigBig Company)"
```

![猴猴demo](image/hoho_demo.png)

# To Do
- [ ] Agent memory
- [ ] Conversation context support 