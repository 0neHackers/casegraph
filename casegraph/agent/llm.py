"""Gemini, used for what an LLM is good at: choosing the next tool, and writing.

* plan(): given a digest of the evidence so far and the tool catalogue, pick
  the next graph tools to call (native function calling), or stop.
* write(): turn the structured findings + retrieved policy/regulatory context
  (GraphRAG) into the case summary, the SAR narrative, the "what changed"
  note, the stop reason and, for undocumented patterns, a description in the
  agent's own words.

Graph analysis, probabilities and policy actions are NOT delegated to the
model. If no key is configured the agent still runs: planning falls back to a
rule-based planner and writing to templates, and the answer file records
llm="offline" and tokens=0.
"""
from __future__ import annotations

import json
import time

from casegraph import config as C

SYSTEM = """You are CaseGraph, a fraud investigation agent at a bank. You work only from the evidence given to you,
which comes from a TigerGraph knowledge graph (transactions, cards, cardholders, device profiles, billing regions,
closed cases) and from retrieved policy and regulatory text. Never invent IDs, amounts or dates. The V, C, D, M and id
columns are unnamed model features: never claim to know what they mean. A risk score is a reason to look, never a
verdict. Be concise and specific."""


class LLM:
    def __init__(self):
        self.tokens = 0
        self.calls = 0
        # A pool of models: on the free tier each has its own daily request quota, so a 429 moves the
        # agent to the next model instead of retrying one that is exhausted.
        self.models = [m for m in dict.fromkeys([C.GEMINI_MODEL] + C.GEMINI_FALLBACKS) if m]
        self.model = self.models[0]
        self.exhausted: set[str] = set()
        self.used: list[str] = []
        self.client = None
        if C.GEMINI_API_KEY:
            try:
                from google import genai

                self.client = genai.Client(api_key=C.GEMINI_API_KEY)
            except Exception:  # pragma: no cover
                self.client = None

    @property
    def online(self) -> bool:
        return self.client is not None

    def _count(self, resp) -> None:
        u = getattr(resp, "usage_metadata", None)
        if u is not None:
            self.tokens += int(getattr(u, "total_token_count", 0) or 0)
        self.calls += 1

    def _gen(self, contents, config):
        err = None
        for m in self.models:
            if m in self.exhausted:
                continue
            for attempt in range(2):
                try:
                    resp = self.client.models.generate_content(model=m, contents=contents, config=config)
                    self._count(resp)
                    self.model = m
                    self.used.append(m)
                    return resp
                except Exception as e:
                    err = e
                    msg = str(e)
                    if "429" in msg or "RESOURCE_EXHAUSTED" in msg or "404" in msg:
                        self.exhausted.add(m)          # quota gone (or model retired): next model
                        break
                    if "503" in msg or "UNAVAILABLE" in msg:
                        time.sleep(3)
                        if attempt == 1:
                            break
                        continue
                    raise
        raise err or RuntimeError("no Gemini model available")

    # ------------------------------------------------------------------ planning
    def plan(self, digest: str, tool_specs: list[dict], budget: int) -> list[dict]:
        """Return [{'tool': name, 'args': {...}, 'why': str}] (possibly empty = stop)."""
        from google.genai import types

        decls = [types.FunctionDeclaration(name=t["name"], description=t["description"],
                                           parameters=t["parameters"]) for t in tool_specs]
        decls.append(types.FunctionDeclaration(
            name="finish_investigation", description="Stop gathering: the evidence is enough to decide, or more "
                                                     "tool calls would not change the decision.",
            parameters={"type": "object", "properties": {"why": {"type": "string"}}, "required": ["why"]}))
        prompt = (f"{digest}\n\nYou may make up to {budget} more tool calls. Call the tools that would most change or "
                  "confirm the decision (for example: what happened on the OTHER cards that share this device or "
                  "region, a wider window on this card, a connected card's history). Give each call a short reason "
                  "in the 'why' argument. If nothing would change the decision, call finish_investigation.")
        cfg = types.GenerateContentConfig(system_instruction=SYSTEM, temperature=0.1,
                                          tools=[types.Tool(function_declarations=decls)],
                                          automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True))
        resp = self._gen(prompt, cfg)
        out = []
        for fc in (resp.function_calls or [])[:budget]:
            args = dict(fc.args or {})
            if fc.name == "finish_investigation":
                return [{"tool": "finish", "args": {}, "why": args.get("why", "")}]
            out.append({"tool": fc.name, "args": {k: v for k, v in args.items() if k != "why"},
                        "why": args.get("why", "")})
        return out

    # ------------------------------------------------------------------ writing
    def write(self, facts: dict, context: list[str]) -> dict:
        from google.genai import types

        schema = {
            "type": "object",
            "properties": {
                "summary": {"type": "string", "description": "2-6 sentences an analyst can read"},
                "sar_narrative": {"type": "string", "description": "only if sar_required: 6-12 sentences, who/what/"
                                  "when/where/how/why, stands on its own, uses only given IDs, dates, amounts"},
                "what_changed": {"type": "string", "description": "1-2 sentences: why final differs from initial, or 'nothing'"},
                "stop_reason": {"type": "string", "description": "1-2 sentences citing policy section 6"},
                "pattern_description": {"type": "string", "description": "only if pattern is undocumented: 2-3 "
                                        "sentences on what the pattern is, who it affects, how it was found"},
            },
            "required": ["summary", "what_changed", "stop_reason"],
        }
        prompt = ("Write the case-file text for this investigation. Use the facts exactly; do not add IDs that are not "
                  "listed. FinCEN guidance says a SAR narrative must answer who, what, when, where, how and why the "
                  "activity is suspicious, chronologically, in plain language.\n\n"
                  f"RETRIEVED CONTEXT (policy, typologies, regulatory guidance, prior cases):\n" +
                  "\n---\n".join(context) + "\n\nFACTS (JSON):\n" + json.dumps(facts, default=str, indent=1))
        cfg = types.GenerateContentConfig(system_instruction=SYSTEM, temperature=0.2,
                                          response_mime_type="application/json", response_schema=schema)
        resp = self._gen(prompt, cfg)
        try:
            return json.loads(resp.text)
        except Exception:
            return {}
