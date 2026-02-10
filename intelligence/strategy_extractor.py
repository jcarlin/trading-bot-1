"""RBI Agent: extract trading strategies from text descriptions.

Takes natural language strategy descriptions (e.g., from research papers,
forum posts, wallet analysis) and produces validated BaseStrategy Python code
ready for backtesting.
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

# Imports that are never allowed in generated strategy code
DANGEROUS_IMPORTS = frozenset({
    "os", "subprocess", "sys", "shutil", "socket", "requests",
})

_EXTRACT_RULES_SYSTEM = """You are a trading strategy analyst. Extract the trading rules from the text.
Return a JSON object with these exact keys:
- "name": short snake_case strategy name
- "description": one-sentence summary
- "entry_logic": precise entry conditions
- "exit_logic": precise exit conditions
- "parameters": dict of parameter names to default values (numbers)

Return ONLY valid JSON, no explanation."""

_GENERATE_CODE_SYSTEM = """You are a Python trading strategy developer. Generate a BaseStrategy subclass.

The class MUST:
1. Subclass BaseStrategy from strategy.base
2. Import Signal from core.models and SignalType from core.types
3. Implement setup(self, df) to compute indicators and store in self.indicators
4. Implement generate_signal(self, index, df) returning a Signal object
5. Use self.params for configurable parameters
6. Use self.hold_signal(df, index) for HOLD signals
7. Use pandas and numpy only (no other external libraries)

Return ONLY the Python code, no explanation. Wrap in ```python ... ```."""


@dataclass
class ExtractedStrategy:
    """Result of extracting a strategy from text."""
    name: str
    description: str
    entry_logic: str
    exit_logic: str
    parameters: dict
    generated_code: str
    confidence: float
    source: str
    source_ref: str


class StrategyExtractor:
    """Extracts trading strategies from text using an LLM provider.

    Pipeline: parse_text -> extract_rules -> generate_code -> validate_code
    """

    def __init__(self, llm_provider, config: dict = None):
        self.llm_provider = llm_provider
        config = config or {}
        self.max_retries = config.get("max_retries", 2)

    def parse_text(self, raw_text: str) -> str:
        """Clean and normalize raw input text."""
        # Remove URLs
        text = re.sub(r'https?://\S+', '', raw_text)
        # Collapse whitespace
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def extract_rules(self, text: str) -> Optional[dict]:
        """Use LLM to extract entry/exit rules and parameters as JSON."""
        if not self.llm_provider or not self.llm_provider.is_available():
            return None
        result = self.llm_provider.complete_json(
            _EXTRACT_RULES_SYSTEM,
            f"Extract trading rules from this text:\n\n{text}",
        )
        if result is None:
            return None
        # Validate required keys
        required = {"name", "description", "entry_logic", "exit_logic", "parameters"}
        if not required.issubset(result.keys()):
            logger.warning("Extracted rules missing keys: %s", required - result.keys())
            return None
        return result

    def generate_code(self, rules: dict) -> Optional[str]:
        """Use LLM to generate BaseStrategy Python code from rules."""
        if not self.llm_provider or not self.llm_provider.is_available():
            return None
        prompt = (
            f"Strategy: {rules.get('name', 'unnamed')}\n"
            f"Description: {rules.get('description', '')}\n"
            f"Entry logic: {rules.get('entry_logic', '')}\n"
            f"Exit logic: {rules.get('exit_logic', '')}\n"
            f"Parameters: {rules.get('parameters', {})}"
        )
        response = self.llm_provider.complete(_GENERATE_CODE_SYSTEM, prompt)
        if response is None:
            return None
        # Extract code from markdown block if present
        if "```python" in response:
            code = response.split("```python")[1].split("```")[0]
        elif "```" in response:
            code = response.split("```")[1].split("```")[0]
        else:
            code = response
        return code.strip()

    def validate_code(self, code: str) -> tuple[bool, list[str]]:
        """Validate generated strategy code.

        Checks:
        1. Syntax correctness via compile()
        2. No dangerous imports (os, subprocess, sys, etc.)

        Returns:
            (is_valid, list_of_errors)
        """
        errors = []

        # Syntax check
        try:
            compile(code, "<generated>", "exec")
        except SyntaxError as e:
            errors.append(f"Syntax error: {e}")

        # Safety check: scan for dangerous imports
        for line in code.splitlines():
            stripped = line.strip()
            if stripped.startswith("import ") or stripped.startswith("from "):
                for dangerous in DANGEROUS_IMPORTS:
                    # Match "import os", "from os import ...", "import os.path"
                    if re.search(rf'\b{dangerous}\b', stripped):
                        errors.append(f"Dangerous import: {dangerous}")

        return (len(errors) == 0, errors)

    def extract_from_text(
        self,
        text: str,
        source: str = "",
        source_ref: str = "",
    ) -> Optional[ExtractedStrategy]:
        """Full extraction pipeline: text -> rules -> code -> validate.

        Returns ExtractedStrategy or None if any step fails.
        """
        cleaned = self.parse_text(text)
        if not cleaned:
            return None

        rules = None
        for attempt in range(1 + self.max_retries):
            rules = self.extract_rules(cleaned)
            if rules is not None:
                break
            logger.info("extract_rules attempt %d failed, retrying", attempt + 1)
        if rules is None:
            return None

        code = None
        for attempt in range(1 + self.max_retries):
            code = self.generate_code(rules)
            if code is not None:
                is_valid, errors = self.validate_code(code)
                if is_valid:
                    break
                logger.info("Generated code invalid (attempt %d): %s", attempt + 1, errors)
                code = None
            else:
                logger.info("generate_code attempt %d returned None", attempt + 1)
        if code is None:
            return None

        return ExtractedStrategy(
            name=rules.get("name", "unnamed"),
            description=rules.get("description", ""),
            entry_logic=rules.get("entry_logic", ""),
            exit_logic=rules.get("exit_logic", ""),
            parameters=rules.get("parameters", {}),
            generated_code=code,
            confidence=0.5,
            source=source,
            source_ref=source_ref,
        )
