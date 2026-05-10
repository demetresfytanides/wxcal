"""
Resolve the right Motus model client and normalised model name.

Local Anthropic (default):   set ANTHROPIC_API_KEY, leave OPENAI_API_KEY unset.
Local OpenAI-compatible:     set OPENAI_API_KEY, pass --llm with an OpenAI model name.
Motus Cloud:                 credentials injected automatically via OPENAI_BASE_URL.
"""

from __future__ import annotations

import os
import sys

_ANTHROPIC_HINTS = ("claude", "anthropic/")


def make_client(model: str) -> tuple:
    """Return (client, model_name) ready to pass to ReActAgent."""
    use_openai = (
        os.getenv("MOTUS_CLOUD")
        or os.getenv("OPENAI_BASE_URL")
        or os.getenv("OPENAI_API_KEY")
    )

    if use_openai:
        # Warn if the model name looks like an Anthropic model — it won't work
        # against the OpenAI endpoint and will produce a cryptic 404.
        if any(h in model.lower() for h in _ANTHROPIC_HINTS):
            print(
                "\n[wxCal] ERROR: OPENAI_API_KEY is set but the model name looks like "
                f"an Anthropic model ('{model}').\n"
                "Please re-run with an OpenAI model, for example:\n\n"
                "    python orchestrator.py ... --llm gpt-4o\n"
                "    python orchestrator.py ... --llm gpt-4.1\n\n"
                "Or unset OPENAI_API_KEY and set ANTHROPIC_API_KEY to use Claude directly,\n"
                "or deploy to Motus Cloud where credentials are handled automatically.\n",
                file=sys.stderr,
            )
            sys.exit(1)
        from motus.models import OpenAIChatClient
        return OpenAIChatClient(), model
    else:
        from motus.models import AnthropicChatClient
        if "/" in model:
            model = model.split("/", 1)[1]
        return AnthropicChatClient(), model
