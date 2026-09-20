# CaseGraph: teaching a fraud agent to doubt the risk score

*HH Goa 2026 · TigerGraph track · 0neHackers*

A bank's fraud model fires. The score is 0.90. What should happen next?

If you block the card, you are wrong most of the time: in the dataset we were given, most alerts
scored above 0.7 turn out to be legitimate. If you ignore it, some real fraud scores near zero and
walks straight through. The honest answer is "it depends on what else you can find", and that is
the job we built CaseGraph to do: take an alert, investigate it the way an analyst would, decide
whether there is enough evidence to act, ask for more when there is not, and leave behind a case
file that the next investigation can learn from.

This post covers what we built, how it is put together, how TigerGraph carries it, and what we
learned along the way.

## What we built

CaseGraph is an agent that takes one of three triggers (a model alert, a customer's complaint, or an
analyst's request) and produces three things:

1. **A case**: verdict, calibrated fraud probability, pattern, the affected transactions, how far the
   fraud goes (connected cards and devices), exposure, the evidence with the query each claim came from,
   and the past cases it used as memory. The case is also written into the graph.
2. **A suspicious activity report** when the bank's policy calls for one, written to FinCEN's
   who / what / when / where / how / why standard.
3. **The next best action, twice**: before it asked for evidence and after, with the approval route
   for every action (the agent may only execute `auto` actions; `L1` and `L2` wait for a human).

It ran on the 20 benchmark cases, and on its own it scanned November and December for rings nobody
had alerted on.

## The data had a trap in it

The dataset is the IEEE-CIS fraud data (590,742 card transactions) with the fraud label removed and a
risk score added, plus 5,565 closed investigations from July to October.

Our first hand-investigation went wrong in an instructive way. Customer `C08623` has 1,134
transactions across 52 billing regions. That is not a person; `customer_id` is derived from an issuer
field, so it is a bucket of many cardholders. An agent that asks "has this customer been to this
region before?" gets "yes, of course" every time.

So we inferred the person. Each transaction carries a `D1` column, a time delta in days. Subtract it
from the transaction day and you get a stable anchor: the day the card relationship started. Card +
billing region + anchor day isolates an individual cardholder far better than `customer_id` does.
We made that a vertex, **Holder**, and let the agent investigate at that level.

It is not a guess: we checked it against the bank's own closed cases. Transactions by the *same
Holder* within three days of a confirmed fraud were themselves fraud **67%** of the time. Other
transactions on the same card: **2.3%**.

## Architecture

```
trigger → open case → gather (GSQL via MCP) → LLM planner: more tools? → analyse (signals + ring + Louvain)
        → recall memory (graph links + TigerVector) → uncertain? → request evidence → re-assess
        → final actions (policy engine) → action desk (auto executed, L1/L2 queued)
        → case text (LLM + GraphRAG) → persist case into the graph
```

Four layers, with a clear split of responsibilities:

- **TigerGraph** holds the bank's data, the knowledge and the agent's memory, and does the
  traversal, the pattern queries and the ring detection.
- **Signal analyzers** turn query results into evidence. Each finding carries a claim, the query it
  came from, the entity IDs it rests on, a log-odds weight and an *independence group*.
- **A policy engine** (the bank's Fraud Policy v1.0 as ~250 lines of Python) maps the assessment to
  exact action identifiers, approval routes, the case-versus-report decision and the stopping rule,
  and cites the rule for each.
- **Gemini** chooses which additional graph tools to call (native function calling) and writes the
  summary, the SAR narrative and the explanation of what changed, grounded in retrieved policy and
  regulatory text.

The LLM never decides an action code and never does graph math. It reasons about where to look and
explains what was found.

## How TigerGraph is used

**Schema.** 13 vertex types: the suggested ones (Customer, Card, Transaction, DeviceProfile,
EmailDomain, BillingRegion, ClosedCase) plus Holder, Pattern, DocChunk, InvestigationCase and
CaseEvent. 26 edge types, including a per-card `NEXT` chain. Everything loads over REST in chunks,
so one command works for both Savanna and a local Community Edition container.

**GSQL.** Twelve installed queries. The workhorses:

- `behaviour_profile` answers the first questions an analyst asks (has this cardholder used this
  device, this region, this email, this product before, and how unusual is the amount), strictly
  before the alert.
- `device_neighbors` asks what happened on the *other* cards that used this device profile in the
  window, and which closed cases involved it.
- `prior_cases` is case memory by graph proximity: closed cases and the agent's own cases linked by
  card, holder or device.

**Graph algorithms.** A nightly-style job runs TigerGraph's own algorithm library, `GDBMS_ALGO`,
over an undirected projection: cardholders linked to the device profiles they used, weighted by
transaction count. `community.louvain` finds 1,245 communities and `community.wcc` the connected
components; both are written onto the vertices. During an investigation, `device_community` reads the
alert device's community and how much confirmed fraud it had produced *before the alert*. On HHG-014
the SM-G935F profile sits in a 110-holder, single-device community, which becomes a line of evidence.

For the time-bounded question ("who shared this device this fortnight?") `device_ring` runs label-propagation connected components on the
Holder ↔ DeviceProfile graph inside a time window. Two filters make it useful. It works on Holders,
because an issuer-bucket "card" would bridge unrelated devices. And it only keeps *specific* device
profiles, so "Windows | Chrome" does not glue ten thousand customers together. On benchmark case
HHG-014 it returns, in about 0.3 seconds, a group of 28 cards joined by a single Samsung SM-G935F
profile: 100% of its transactions on a device new to the account, 100% behind an anonymous proxy.

**TigerVector + GraphRAG.** Every closed case has two vectors: an embedding of the analyst's
narrative, and a 16-dimension behavioural fingerprint of the episode (channel mix, amount shape,
device novelty, proxy use, burst tempo, region novelty, closeness to round authorisation limits).
The policy, the pattern typologies and five FinCEN documents are chunked, embedded and linked to the
Pattern vertices they discuss. Retrieval hands the LLM connected context (a policy rule together
with its pattern, a similar case together with its outcome), not raw rows.

**TigerGraph MCP.** Every investigation step is a `run_installed_query` call through the official
`tigergraph-mcp` server; the knowledge build and case vectors go through `upsert_vectors`; case
memory is written with `add_node` and `add_edges`. The UI shows every call with its latency and
transport.

## The agentic part

**Knowing when it is not sure.** The probability starts from a model trained only on the bank's own
closed cases (out-of-time AUC 0.918 against 0.866 for the bank's risk score) and moves with each
finding. The policy's stopping rule needs "at least two independent pieces of evidence", so the agent
counts independent *groups*: two device facts are one line of evidence.

**Asking for more, and changing its mind.** When the evidence is not enough, the agent asks through
the controls the policy allows: customer verification, step-up authentication. The replies are not
in the data, so it assumes the reply the evidence points to and records that assumption: denial when
p ≥ 0.55, confirmation when p ≤ 0.35, and *no reply within 24 hours* in between, which triggers the
policy's holding actions (R4) and escalation (R8). You can force any reply from the CLI or the UI and
watch the recommendation move.

**Memory.** Cases are processed in the order they were opened, and each one is written back into
the graph. When a later alert touches the same device, `prior_cases` returns the earlier case and the
agent cites it.

**Finding what nobody asked about.** The monitor slides a 14-day window across the exam period, runs
the ring query, keeps groups that look coordinated, and opens investigations on them.

**Following the bank's own habits.** In the history, every one of the 900 cleared alerts was closed
only after the cardholder confirmed it. So when a strong alert is contradicted by the graph, the agent
verifies before closing rather than dismissing it.

## What it found

- **HHG-006: an undocumented pattern.** A customer disputed one $482.12 purchase. The agent found four
  online purchases in 30 minutes, $456.96 to $488.04, each just under $500, from devices new to the
  account ($1,906.07 in total). It matched five closed cases the analysts had marked "undocumented"
  with the same shape, filed a report and escalated.
- **HHG-014: a device ring.** The 28-card SM-G935F group above. The same device profile appeared on four
  closed cases in August and September; the analysts never named it. The agent labelled it undocumented,
  described it in its own words, monitored every connected card and filed.
- **HHG-019 and HHG-011: shared origin on other cards.** Neither alert said anything about other cards.
  In both, the device profile had been used on three or four other cards within days, at nearly the
  same amount, each New to its account. Both became reports under the policy's shared-origin rule.
- **Ten of eleven model alerts were legitimate**, with scores up to 0.90. The cardholder's own history
  contradicted them.

## What we learned

- **Look at the entity before you model it.** The single biggest improvement came from realising the
  "customer" was not a person. Everything downstream (behaviour baselines, region novelty, episode
  boundaries) got better once the agent investigated Holders.
- **Let the history set the weights.** Every weight we were unsure about, we checked against the
  closed cases. Two surprises: the fingerprint vote has to be compared to the 84% fraud base rate of
  closed cases, not to 50%; and a Holder's prior confirmed fraud matters a lot even after the model
  has spoken.
- **Specificity is what makes a device a link.** Most device profiles are generic. A ring detector
  without a specificity filter finds one giant component and nothing useful.
- **Keep the LLM on the parts language is good at.** Planning which evidence to fetch next and
  writing a regulator-grade narrative are language tasks. Policy compliance is not; code does it better
  and never forgets the $2,500 routing threshold.

## What we would do with more time

- Learn the evidence weights jointly (a small calibrated model over the signal groups, fitted on the
  October closed cases) instead of setting each one from its own measured lift.
- A real customer channel: send the verification request, wait, resume the case from the graph.
- Region-cluster rings: the same components query on Holder ↔ BillingRegion for card-present skimming.
- Stream new transactions in and let the monitor run continuously.
- A paid LLM tier. The free tier's ~20 requests per model per day forced us to rotate across six models.

---

Code, answer files and the analyst console: **[github.com/0neHackers/casegraph](https://github.com/0neHackers/casegraph)** · built on
TigerGraph Savanna, TigerGraph MCP, TigerVector and Gemini.
