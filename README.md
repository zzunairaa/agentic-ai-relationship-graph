# Agentic AI — Relationship Graph

**Assignment:** Building a Simple Agentic AI with a Relationship Graph  
**Tech Stack:** Python · FastAPI · Groq API (LLaMA 3.3 70B) · HTML/CSS/JS  
**Model:** `llama-3.3-70b-versatile` via Groq (free tier)

---

## Objective

Build a conversational AI assistant that:
- Reads a conversation and extracts relationships between the user and people mentioned
- Maintains a relationship graph that updates as the conversation evolves
- Uses the graph to generate more contextual and empathetic responses

---

## Agent Loop

```
User message
     ↓
LLM extracts relationship info  (agent.py)
     ↓
Hybrid filter validates extraction  (main.py)
     ↓
Graph updated  (graph.py → graph.json)
     ↓
Contextual response generated  (agent.py)
     ↓
Response + graph returned to frontend
```

---

## Graph Representation

Relationships stored as JSON — exactly as required:

```json
{
  "nodes": ["User", "Sara", "Anna"],
  "edges": [
    {"source": "User", "target": "Sara", "relation": "friend"},
    {"source": "User", "target": "Anna", "relation": "sister"}
  ],
  "removed": [
    {"person": "Sara", "relation": "friend"}
  ]
}
```

- `nodes` — all people ever mentioned (kept even after removal for memory)
- `edges` — currently active User→Person relationships only
- `removed` — history of ended relationships (bonus: enables empathy responses)

---

## Assignment Examples — All Working

**Example 1 — Adding a relationship:**
```
User: I have a friend named Sara.

Extraction: {"person": "Sara", "relationship": "friend", "action": "add"}
Graph:      User -- friend --> Sara
Response:   "That's lovely! How long have you known Sara?"
```

**Example 2 — Adding another person:**
```
User: My sister Anna is visiting this week.

Extraction: {"person": "Anna", "relationship": "sister", "action": "add"}
Graph:      User -- friend --> Sara
            User -- sister --> Anna
Response:   "That sounds nice! Are you planning to spend time with Anna?"
```

**Example 3 — Removing a relationship:**
```
User: Sara and I are not friends anymore.

Extraction: {"person": "Sara", "relationship": "friend", "action": "remove"}
Graph:      User -- sister --> Anna
Response:   "I'm sorry to hear that, losing a friendship can be really tough."
```

---

## Project Structure

```
my-agent/
├── agent.py            # LLM calls — extraction + response generation
├── graph.py            # Graph logic — add, remove, persist to file
├── main.py             # FastAPI backend — agent loop + hybrid filter
├── graph.json          # Persistent graph storage (auto-created)
├── extractions.json    # Full extraction history with timestamps
├── requirements.txt    # Python dependencies
└── static/
    └── index.html      # Frontend — chat UI + live graph display
```

---

## How to Run

### 1. Install dependencies
```bash
pip install fastapi uvicorn groq python-dotenv
```

### 2. Set up your API key
Create a `.env` file in the project root:
```
GROQ_API_KEY=your_groq_api_key_here
```
Get a free key at: **https://console.groq.com**

### 3. Start the server
```bash
uvicorn main:app --reload
```

### 4. Open in browser
```
http://localhost:8000
```

---

## Features Implemented

### Core (Assignment Requirements)
| Requirement | Implementation |
|---|---|
| Read user message | FastAPI `/chat` endpoint receives message |
| Extract person name | LLM extracts from natural language |
| Extract relationship type | LLM returns relationship label |
| Extract action (add/remove) | LLM returns add/remove/update |
| Update relationship graph | `graph.py` add/remove functions |
| Generate contextual response | Graph passed to LLM for response |
| JSON graph with nodes + edges | `graph.json` with exact required format |
| Only User→Person relationships | Enforced in both prompt and code |

### Beyond Requirements (Bonus)
| Feature | Description |
|---|---|
| Update action | Handles relationship changes (e.g. colleague → manager) |
| Multiple people per message | Extracts all people mentioned in one message |
| Persistent graph | `graph.json` survives server restarts |
| Relationship history | `removed` list preserves ended relationships |
| Empathy responses | Agent remembers past relationships and responds with care |
| Indirect relationship blocking | Ignores "Hassan's brother" — not User's direct relation |
| No-name blocking | Ignores "I met my sister" — no name given |
| Generic name blocking | Prevents "Brother" being stored as a person's name |
| Duplicate prevention | Same relationship never added twice |
| Extraction log | Every extraction saved to `extractions.json` with timestamp |

---

## Extraction Log Sample (`extractions.json`)

```json
[
  {
    "timestamp": "2026-04-07 10:00:00",
    "message": "I have a friend named Sara.",
    "extraction": [{"person": "Sara", "relationship": "friend", "action": "add"}]
  },
  {
    "timestamp": "2026-04-07 10:01:00",
    "message": "My sister Anna is visiting this week.",
    "extraction": [{"person": "Anna", "relationship": "sister", "action": "add"}]
  },
  {
    "timestamp": "2026-04-07 10:02:00",
    "message": "Sara and I are not friends anymore.",
    "extraction": [{"person": "Sara", "relationship": "friend", "action": "remove"}]
  },
  {
    "timestamp": "2026-04-07 10:03:00",
    "message": "Sara is coming over today.",
    "extraction": []
  }
]
```

Note: The last entry returns `[]` — no new relationship extracted. But the agent remembers Sara was removed and responds with empathy.

---

## Research Techniques Applied

| Technique | Paper | Used For |
|---|---|---|
| Chain of Thought prompting | Wei et al., NeurIPS 2022 | Step-by-step reasoning before extraction |
| Few-shot prompting | Brown et al., NeurIPS 2020 | IGNORE + EXTRACT examples in prompt |
| Semantic Role Labeling | He et al., ACL 2017 | Only extract when User is grammatical subject |
| Hybrid IE architecture | Chiticariu et al., EMNLP 2013 | LLM + rule-based post-processing filter |
| Dependency parsing | Manning et al., ACL 2014 | Possessive pattern detection (`'s`) |
| Knowledge graph dedup | Dong et al., KDD 2014 | Duplicate edge prevention |
| Episodic memory | Tulving, 1972 | Removed relationship history for empathy |
| Affective computing | Picard, MIT Press 1997 | Tone-aware empathetic responses |

---

## Design Decisions

**Why keep removed nodes?**  
Deleting a node loses context. Keeping history allows the agent to respond with empathy when a previously removed person is mentioned again — making the assistant feel more human.

**Why a hybrid architecture?**  
LLMs occasionally hallucinate indirect relationships despite prompt rules. A rule-based post-processing layer in `main.py` catches edge cases the LLM misses — combining the strengths of both neural and symbolic approaches.

**Why Groq?**  
Free tier with fast inference. No credit card needed — ideal for development and testing.

**Why FastAPI?**  
Modern Python framework, clean separation between backend logic and frontend, automatic API documentation.

---

## Limitations

- Only tracks User → Person relationships (by design, per assignment scope)
- Relationship label quality depends on LLM — rare indirect cases may slip through
- Could be extended with Neo4j for larger scale graph storage
- Coreference resolution (he/she/they) would improve accuracy further

---

## References

- Wei, J. et al. (2022). Chain-of-Thought Prompting. NeurIPS.
- Brown, T. et al. (2020). Language Models are Few-Shot Learners. NeurIPS.
- He, L. et al. (2017). Deep Semantic Role Labeling. ACL.
- Chiticariu, L. et al. (2013). Rule-Based IE is Dead! Long Live Rule-Based IE Systems! EMNLP.
- Manning, C. et al. (2014). The Stanford CoreNLP NLP Toolkit. ACL.
- Dong, X. et al. (2014). Knowledge Vault. KDD.
- Tulving, E. (1972). Episodic and Semantic Memory.
- Picard, R. (1997). Affective Computing. MIT Press.
