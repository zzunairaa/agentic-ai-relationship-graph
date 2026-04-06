# Agentic AI For Relationship Graph

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

Relationships stored as JSON 

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

## Assignment Examples

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

##  How to Run the Project

Before starting, get your **Groq API Key**:  
 https://console.groq.com/keys


### Option 1: Run with Docker (Recommended)
This method ensures the environment is identical to the one used during development and handles all dependencies automatically.

1.  **Build the image:**
    ```bash
    docker build -t relationship-agent .
    ```

2.  **Run the container:** *(Replace `your_key` with your actual Groq API Key)*
    ```bash
    docker run -p 7860:7860 -e GROQ_API_KEY="your_key" relationship-agent
    ```

3.  **Access the App:** Open your browser to: [http://localhost:7860](http://localhost:7860)

---

### Option 2: Run Locally (Manual)
Use this option if you prefer to run the Python script directly on your host machine.

1.  **Install dependencies:**
    ```bash
    pip install fastapi uvicorn groq python-dotenv
    ```

2.  **Set up API key:** Create a `.env` file in the project root directory and add your key:
    ```text
    GROQ_API_KEY=your_groq_api_key_here
    ```

3.  **Start the server:**
    ```bash
    uvicorn main:app --reload --port 8000
    ```

4.  **Access the App:** Open your browser to: [http://localhost:8000](http://localhost:8000)
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
Deleting a node loses context. Keeping history allows the agent to respond with empathy when a previously removed person is mentioned again making the assistant feel more human.

**Why a hybrid architecture?**  
LLMs occasionally hallucinate indirect relationships despite prompt rules. A rule-based post-processing layer in `main.py` catches edge cases the LLM misses combining the strengths of both neural and symbolic approaches.

**Why Groq?**  
Free tier with fast inference. No credit card needed , ideal for development and testing.

**Why FastAPI?**  
Modern Python framework, clean separation between backend logic and frontend, automatic API documentation.

---

## References

- Wei, J. et al. (2022). *Chain-of-Thought Prompting*. NeurIPS.  
  https://arxiv.org/pdf/2201.11903

- Brown, T. et al. (2020). *Language Models are Few-Shot Learners*. NeurIPS.  
  https://arxiv.org/pdf/2005.14165

- He, L. et al. (2017). *Deep Semantic Role Labeling*. ACL.  
  https://aclanthology.org/P17-1044.pdf

- Chiticariu, L. et al. (2013). *Rule-Based IE is Dead! Long Live Rule-Based IE Systems!* EMNLP.  
  https://aclanthology.org/D13-1079.pdf

- Manning, C. et al. (2014). *The Stanford CoreNLP NLP Toolkit*. ACL.  
  https://nlp.stanford.edu/pubs/StanfordCoreNlp2014.pdf

- Dong, X. et al. (2014). *Knowledge Vault*. KDD.  
  https://www.cs.ubc.ca/~murphyk/papers/kv-kdd14.pdf

- Tulving, E. (1972). *Episodic and Semantic Memory*.  
  https://alicekim.ca/12.EpSem72.pdf

- Picard, R. (1997). *Affective Computing*. MIT Press.  
  https://arl.human.cornell.edu/linked%20docs/Picard%20Affective%20Computing.pdf
