# Agentic AI — Relationship Graph

A conversational AI assistant that automatically extracts, tracks, and remembers relationships between the user and people mentioned in chat. Built with Python, FastAPI, and LLaMA 3.3 70B via Groq API.

---

## What it does

The agent reads natural language messages and automatically:
- Extracts who the user is talking about and what their relationship is
- Updates a live relationship graph (add, remove, update)
- Generates empathetic, context-aware responses using the graph
- Remembers ended relationships and responds with empathy when they are mentioned again

---

## Agent Loop

```
User message
     ↓
LLM extracts relationship info (agent.py)
     ↓
Hybrid filter validates extraction (main.py)
     ↓
Graph updated (graph.py)
     ↓
Contextual response generated (agent.py)
     ↓
Response + graph returned to frontend
```

---

## Graph Representation

Relationships are stored as a JSON structure:

```json
{
  "nodes": ["User", "Sara", "Anna"],
  "edges": [
    {"source": "User", "target": "Anna", "relation": "sister"},
    {"source": "User", "target": "Sara", "relation": "friend"}
  ],
  "removed": [
    {"person": "Sara", "relation": "friend"}
  ]
}
```

- `nodes` — all people ever mentioned
- `edges` — currently active relationships
- `removed` — history of ended relationships (used for empathy responses)

---

## Example Interactions

**Adding a relationship:**
```
User: I have a friend named Sara.
Extraction: {"person": "Sara", "relationship": "friend", "action": "add"}
Graph: User -- friend --> Sara
Response: "That's lovely! How long have you known Sara?"
```

**Removing a relationship:**
```
User: Sara and I are not friends anymore.
Extraction: {"person": "Sara", "relationship": "friend", "action": "remove"}
Graph: Sara removed from edges
Response: "I'm sorry to hear that, losing a friendship can be really tough."
```

**Memory and empathy:**
```
User: Sara is coming over today.
Extraction: [] (no new relationship)
Response: "I remember things were tough with Sara before —
           how are you feeling about seeing her again?"
```

**Indirect relationship blocked:**
```
User: I met Andre's dad today.
Extraction: [] (possessive pattern detected — not User's direct relation)
Graph: unchanged
```

---

## Project Structure

```
my-agent/
├── agent.py          # LLM calls — extraction and response generation
├── graph.py          # Graph logic — add, remove, persist
├── main.py           # FastAPI backend server
├── graph.json        # Persistent graph storage
├── extractions.json  # Full extraction history with timestamps
├── requirements.txt  # Python dependencies
└── static/
    └── index.html    # Frontend chat interface
```

---

## How to Run

### 1. Install dependencies
```bash
pip install fastapi uvicorn groq python-dotenv
```

### 2. Set up API key
Create a `.env` file:
```
GROQ_API_KEY=your_groq_api_key_here
```
Get a free key at: https://console.groq.com

### 3. Start the server
```bash
uvicorn main:app --reload
```

### 4. Open in browser
```
http://localhost:8000
```

---

## Features

### Core (Assignment Requirements)
- Extracts person name, relationship type, and action (add/remove) from natural language
- Updates relationship graph after every message
- Generates contextual responses using the current graph state

### Beyond Requirements

| Feature | Description |
|---|---|
| Update action | Handles relationship changes (colleague → manager) |
| Multiple extractions | Extracts multiple people from one message |
| Persistent storage | Graph saved to `graph.json`, survives server restarts |
| Relationship history | Removed relationships saved to `removed` list |
| Empathy responses | Detects past relationships and responds with emotional awareness |
| Indirect relationship blocking | Ignores "Hassan's brother" or "friend of my sister" |
| No-name blocking | Ignores "I met my sister" (no name given) |
| Generic name blocking | Prevents "Brother" being stored as a person name |
| Duplicate prevention | Same relationship never added twice |
| Extraction logging | Every extraction saved to `extractions.json` with timestamp |

---

## Research Techniques Used

| Technique | Reference | Applied |
|---|---|---|
| Chain of Thought prompting | Wei et al., NeurIPS 2022 | Step-by-step reasoning before extraction |
| Few-shot prompting | Brown et al., NeurIPS 2020 | IGNORE + EXTRACT examples in prompt |
| Semantic Role Labeling | He et al., ACL 2017 | Subject must be User (I/me/my) |
| Hybrid IE architecture | Chiticariu et al., EMNLP 2013 | LLM + rule-based post-processing |
| Dependency parsing | Manning et al., ACL 2014 | Possessive pattern detection |
| Knowledge graph dedup | Dong et al., KDD 2014 | Duplicate edge prevention |
| Episodic memory | Tulving, 1972 | Removed relationship history for empathy |
| Affective computing | Picard, MIT Press 1997 | Tone-aware empathetic responses |
| Ontology normalization | Suchanek et al., WWW 2007 | Lowercase relation labels |
| Provenance tracking | Buneman et al., PODS 2001 | Timestamped extraction log |

---

## Tech Stack

| Component | Technology |
|---|---|
| Backend | Python, FastAPI |
| LLM | Groq API (LLaMA 3.3 70B) |
| Frontend | HTML, CSS, JavaScript |
| Storage | JSON files |
| Server | Uvicorn |

---

## Design Decisions

**Why Groq?**
Free tier with fast inference — ideal for development and testing without cost barriers.

**Why FastAPI?**
Modern Python framework with automatic API docs, async support, and clean separation between backend logic and frontend.

**Why JSON file storage?**
Keeps the project focused on agent logic rather than database infrastructure. Relationships persist across server restarts without any additional setup.

**Why keep removed nodes in graph?**
Deleting a node loses context. Keeping history allows the agent to respond with empathy when a previously removed person is mentioned — making the assistant feel more human and contextually aware.

**Why a hybrid architecture?**
LLMs occasionally hallucinate indirect relationships despite prompt instructions. A rule-based post-processing layer catches these edge cases — mirroring production NLP systems where neural and symbolic approaches are combined for robustness.

---

## Limitations and Future Work

- Currently only tracks User → Person relationships (by design)
- Relationship labels depend on LLM understanding — rare edge cases may slip through
- Could be extended with a proper graph database (Neo4j) for larger scale
- Coreference resolution could improve handling of pronouns (he/she/they)
- A visual graph diagram (D3.js or Cytoscape) would improve the UI

---

## References

- Wei, J. et al. (2022). Chain-of-Thought Prompting Elicits Reasoning in Large Language Models. NeurIPS.
- Brown, T. et al. (2020). Language Models are Few-Shot Learners. NeurIPS.
- He, L. et al. (2017). Deep Semantic Role Labeling. ACL.
- Chiticariu, L. et al. (2013). Rule-Based Information Extraction is Dead! Long Live Rule-Based Information Extraction Systems! EMNLP.
- Manning, C. et al. (2014). The Stanford CoreNLP Natural Language Processing Toolkit. ACL.
- Dong, X. et al. (2014). Knowledge Vault: A Web-Scale Approach to Probabilistic Knowledge Fusion. KDD.
- Tulving, E. (1972). Episodic and Semantic Memory.
- Picard, R. (1997). Affective Computing. MIT Press.
- Suchanek, F. et al. (2007). YAGO: A Core of Semantic Knowledge. WWW.
- Buneman, P. et al. (2001). Why and Where: A Characterization of Data Provenance. PODS.
- Miller, J. (1998). An Introduction to the Resource Description Framework. D-Lib Magazine.
