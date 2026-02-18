"""Controller for the AI Copilot — manages Claude API calls with tool use.

Runs API calls on a background QRunnable so the GUI stays responsive.
The agentic loop handles up to MAX_TOOL_TURNS of tool-use round-trips.
"""

from __future__ import annotations

import json
import logging
import os
import traceback
from typing import Any

from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal, Slot

from portopt.engine.copilot_tools import TOOL_SCHEMAS, execute_tool
from portopt.utils.credentials import ANTHROPIC_API_KEY, get_credential

logger = logging.getLogger(__name__)

MAX_TOOL_TURNS = 5
MODEL = "claude-sonnet-4-20250514"


# ── System Prompt Builder ─────────────────────────────────────────────

def _build_system_prompt(context: dict) -> str:
    """Build a dynamic system prompt that includes current portfolio state."""
    parts = [
        "You are Meridian Copilot, an AI assistant built into a quantitative "
        "portfolio optimization terminal.  You help the user understand their "
        "portfolio, run analyses, and interpret results.",
        "",
        "You have access to tools that can compute real metrics from the user's "
        "actual portfolio data.  Always use the tools rather than guessing.  "
        "Present numbers clearly and concisely.",
        "",
    ]

    weights = context.get("weights")
    if weights:
        sorted_w = sorted(weights.items(), key=lambda x: -x[1])[:10]
        holdings = ", ".join(f"{s} ({w:.1%})" for s, w in sorted_w)
        parts.append(f"Current portfolio ({len(weights)} assets): {holdings}")

    result = context.get("result")
    if result:
        parts.append(
            f"Last optimization: {result.method} | "
            f"E[R]={result.expected_return:.2%} | "
            f"Vol={result.volatility:.2%} | "
            f"Sharpe={result.sharpe_ratio:.3f}"
        )

    prices = context.get("prices")
    if prices is not None:
        parts.append(f"Price data: {len(prices)} days, {len(prices.columns)} assets")

    return "\n".join(parts)


# ── Background Worker ─────────────────────────────────────────────────

class _WorkerSignals(QObject):
    """Signals emitted by the copilot worker."""
    response_chunk = Signal(str)       # text chunk for streaming display
    tool_use_started = Signal(str)     # tool name being called
    response_complete = Signal(str)    # full final text
    error = Signal(str)


class _CopilotWorker(QRunnable):
    """Runs the Claude API call + agentic tool loop on a background thread."""

    def __init__(
        self,
        api_key: str,
        messages: list[dict],
        system_prompt: str,
        context: dict,
    ):
        super().__init__()
        self.api_key = api_key
        self.messages = messages
        self.system_prompt = system_prompt
        self.context = context
        self.signals = _WorkerSignals()
        self.setAutoDelete(True)

    @Slot()
    def run(self):
        try:
            import anthropic
        except ImportError:
            self.signals.error.emit(
                "The 'anthropic' package is not installed.\n"
                "Install it with: pip install anthropic"
            )
            return

        try:
            client = anthropic.Anthropic(api_key=self.api_key)
            messages = list(self.messages)  # copy to avoid mutation
            full_text = ""

            for turn in range(MAX_TOOL_TURNS):
                response = client.messages.create(
                    model=MODEL,
                    max_tokens=4096,
                    system=self.system_prompt,
                    messages=messages,
                    tools=TOOL_SCHEMAS,
                )

                # Process content blocks
                assistant_content = response.content
                text_parts = []
                tool_uses = []

                for block in assistant_content:
                    if block.type == "text":
                        text_parts.append(block.text)
                        self.signals.response_chunk.emit(block.text)
                    elif block.type == "tool_use":
                        tool_uses.append(block)

                # If no tool use, we're done
                if not tool_uses:
                    full_text = "".join(text_parts)
                    break

                # Append assistant message with all content
                messages.append({
                    "role": "assistant",
                    "content": [_block_to_dict(b) for b in assistant_content],
                })

                # Execute each tool and build tool_result messages
                tool_results = []
                for tool_block in tool_uses:
                    tool_name = tool_block.name
                    tool_input = tool_block.input or {}
                    self.signals.tool_use_started.emit(tool_name)

                    logger.info("Copilot calling tool: %s(%s)", tool_name, tool_input)
                    result = execute_tool(tool_name, tool_input, self.context)
                    result_text = json.dumps(result, default=str)

                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tool_block.id,
                        "content": result_text,
                    })

                messages.append({"role": "user", "content": tool_results})

            else:
                # Exhausted tool turns — combine whatever text we got
                full_text = full_text or "(Reached maximum analysis steps)"

            self.signals.response_complete.emit(full_text)

        except Exception as e:
            tb = traceback.format_exc()
            logger.error("Copilot worker error: %s\n%s", e, tb)
            self.signals.error.emit(str(e))


def _block_to_dict(block: Any) -> dict:
    """Convert an anthropic content block to a plain dict for message history."""
    if block.type == "text":
        return {"type": "text", "text": block.text}
    elif block.type == "tool_use":
        return {
            "type": "tool_use",
            "id": block.id,
            "name": block.name,
            "input": block.input,
        }
    return {"type": block.type}


# ── Controller ────────────────────────────────────────────────────────

class CopilotController(QObject):
    """Manages copilot state, conversation history, and API calls."""

    # Signals forwarded from worker
    response_chunk = Signal(str)
    tool_use_started = Signal(str)
    response_complete = Signal(str)
    error = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._history: list[dict] = []
        self._context: dict = {}
        self._worker: _CopilotWorker | None = None

    def set_context(
        self,
        prices=None,
        weights=None,
        mu=None,
        cov=None,
        result=None,
    ):
        """Update the portfolio context available to tools."""
        self._context = {
            "prices": prices,
            "weights": weights,
            "mu": mu,
            "cov": cov,
            "result": result,
        }

    def send_message(self, text: str):
        """Send a user message to Claude and start the background worker."""
        api_key = self._get_api_key()
        if not api_key:
            self.error.emit(
                "No Anthropic API key configured.\n"
                "Set it via AI → API Key... or the ANTHROPIC_API_KEY environment variable."
            )
            return

        # Add user message to history
        self._history.append({"role": "user", "content": text})

        system_prompt = _build_system_prompt(self._context)

        worker = _CopilotWorker(
            api_key=api_key,
            messages=list(self._history),
            system_prompt=system_prompt,
            context=self._context,
        )
        worker.signals.response_chunk.connect(self.response_chunk)
        worker.signals.tool_use_started.connect(self.tool_use_started)
        worker.signals.response_complete.connect(self._on_complete)
        worker.signals.error.connect(self._on_error)

        self._worker = worker  # prevent GC
        QThreadPool.globalInstance().start(worker)

    def clear_history(self):
        """Reset conversation history."""
        self._history.clear()

    def _on_complete(self, full_text: str):
        """Store assistant response in history and forward signal."""
        if full_text:
            self._history.append({"role": "assistant", "content": full_text})
        self.response_complete.emit(full_text)

    def _on_error(self, msg: str):
        """Forward error and don't store failed response."""
        self.error.emit(msg)

    @staticmethod
    def _get_api_key() -> str | None:
        """Resolve API key from keyring → env var → .env file."""
        # 1. Keyring
        key = get_credential(ANTHROPIC_API_KEY)
        if key:
            return key

        # 2. Environment variable
        key = os.environ.get("ANTHROPIC_API_KEY")
        if key:
            return key

        # 3. dotenv (.env file)
        try:
            from dotenv import load_dotenv
            load_dotenv()
            key = os.environ.get("ANTHROPIC_API_KEY")
            if key:
                return key
        except ImportError:
            pass

        return None
