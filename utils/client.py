"""
Resolve the right Motus model client and normalised model name.

Set MOTUS_CLOUD=1 (or let the platform inject OPENAI_BASE_URL) to route
through OpenRouter / Motus Cloud proxy (OpenAIChatClient, needs
"anthropic/model" format).  Leave both unset for direct Anthropic API
(AnthropicChatClient, needs bare "model" name).
"""

from __future__ import annotations

import os


def make_client(model: str) -> tuple:
    """Return (client, model_name) ready to pass to ReActAgent."""
    if os.getenv("MOTUS_CLOUD") or os.getenv("OPENAI_BASE_URL"):
        from motus.models import OpenAIChatClient
        if "/" not in model:
            model = f"anthropic/{model}"
        return OpenAIChatClient(), model
    else:
        from motus.models import AnthropicChatClient
        if "/" in model:
            model = model.split("/", 1)[1]
        return AnthropicChatClient(), model
