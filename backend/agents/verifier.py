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
