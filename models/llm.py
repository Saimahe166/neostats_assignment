"""
LLM provider-agnostic wrapper with OpenAI, Groq, and Gemini support.
"""

from typing import Optional, Dict, Any, List
import os, json, traceback

from config.config import OPENAI_API_KEY, GROQ_API_KEY, GEMINI_API_KEY

# Optional imports protected by try/except so app still loads without every SDK installed.
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

try:
    from groq import Groq
except Exception:
    Groq = None

try:
    import google.generativeai as genai
except Exception:
    genai = None


class LLMClient:
    def __init__(self, provider: str = "openai", model: str = "gpt-4o-mini", temperature: float = 0.3):
        self.provider = provider.lower()
        self.model = model
        self.temperature = float(temperature)

        if self.provider == "openai":
            if not OPENAI_API_KEY:
                raise RuntimeError("OPENAI_API_KEY not set")
            if OpenAI is None:
                raise RuntimeError("openai SDK not available. Add `openai` to requirements.")
            self.client = OpenAI(api_key=OPENAI_API_KEY)

        elif self.provider == "groq":
            if not GROQ_API_KEY:
                raise RuntimeError("GROQ_API_KEY not set")
            if Groq is None:
                raise RuntimeError("groq SDK not available. Add `groq` to requirements.")
            self.client = Groq(api_key=GROQ_API_KEY)

        elif self.provider == "gemini":
            if not GEMINI_API_KEY:
                raise RuntimeError("GEMINI_API_KEY not set")
            if genai is None:
                raise RuntimeError("google-generativeai SDK not available. Add `google-generativeai` to requirements.")
            genai.configure(api_key=GEMINI_API_KEY)
            self.client = genai.GenerativeModel(self.model)

        else:
            raise ValueError(f"Unknown provider: {self.provider}")

    def chat(self, system: str, user: str) -> str:
        """
        Normalized single-turn chat. For multi-turn, you can extend to accept history.
        """
        try:
            if self.provider == "openai":
                rsp = self.client.chat.completions.create(
                    model=self.model,
                    temperature=self.temperature,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                )
                return rsp.choices[0].message.content

            elif self.provider == "groq":
                rsp = self.client.chat.completions.create(
                    model=self.model,
                    temperature=self.temperature,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                )
                return rsp.choices[0].message.content

            elif self.provider == "gemini":
                prompt = f"System: {system}\nUser: {user}"
                rsp = self.client.generate_content(prompt, generation_config={"temperature": self.temperature})
                return rsp.text

        except Exception as e:
            return f"[LLM Error] {e}\n{traceback.format_exc()}"
