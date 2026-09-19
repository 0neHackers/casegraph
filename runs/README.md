`traces/<case_id>.json` is the full record behind each answer file: every event, every tool call
(tool, arguments, latency, transport = mcp), the facts handed to the LLM, the episode fingerprint and
the log-odds weight of every finding. The analyst console reads these.
