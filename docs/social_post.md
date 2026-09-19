# Social post (LinkedIn / X)

## LinkedIn

A fraud model scores a transaction 0.90. Block the card?

In the dataset we got at #HackerHouseGoa, most alerts above 0.7 are legitimate, and some real fraud
scores near zero. So for the @TigerGraphDB track, our team 0neHackers built **CaseGraph**: an agent
that doesn't trust the score. It investigates the graph around the alert, decides whether it knows
enough to act, and asks for more evidence when it doesn't.

What's inside:
🕸️ TigerGraph Savanna: 590k transactions, 222k inferred cardholders, 5.5k closed cases as memory
🔌 TigerGraph MCP: every investigation step is an MCP tool call into installed GSQL queries
🧮 A connected-components ring detector in GSQL. It found 28 cards sharing one Samsung device profile, 100% behind an anonymous proxy, in about 0.3 s
🧠 TigerVector + GraphRAG: closed cases searchable by narrative *and* by behavioural fingerprint, plus the bank's policy and FinCEN guidance
📜 Policy as code: exact actions, L1/L2 approval routes, case vs. report, all citing the rule
✍️ Gemini picks the next graph tool and writes the SAR narrative; it never picks the action

The lesson that mattered most: a "customer" in this data was a bucket of hundreds of cardholders.
Once we inferred the actual person, same-holder neighbours of a known fraud turned out to be fraud
67% of the time, against 2.3% for everything else on the card.

Blog: <link to docs/blog.md or published post>
Demo: <link to video>
Code: github.com/0neHackers/casegraph

#TigerGraph #GraphRAG #AgenticAI #FraudDetection #MCP

## X (≤ 280 chars)

Built CaseGraph for @TigerGraphDB at #HackerHouseGoa: a fraud agent that doubts the risk score,
queries the graph via TigerGraph MCP, finds device rings with GSQL connected components, remembers
every case in TigerVector, and asks for evidence before it blocks. 🧵 <link>
