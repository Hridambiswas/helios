# agents/verifier.py — Helios Gemini-backed cross-verifier
# Author: Hridam Biswas | Project: Helios

from __future__ import annotations
import json
import logging
from typing import Any

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

from config import cfg
from agents.base import BaseAgent

logger = logging.getLogger("helios.agents.verifier")

_SYSTEM_PROMPT = """\
You are the Verifier agent in Helios — a Gemini-backed second-opinion judge.
The Groq synthesizer has already produced an answer. Your job is an INDEPENDENT
cross-check: read the query, the retrieved context, and the answer, then decide
whether the answer holds up on its own merits.
"""
