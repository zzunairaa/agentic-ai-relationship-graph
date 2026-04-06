# main.py
# FastAPI backend server — implements the full agentic loop
# Architecture: Extract → Filter → Update Graph → Generate Response
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from graph import load_graph, add_relationship, remove_relationship, remove_all_relationships, save_extraction
from agent import extract_relationship, generate_response

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

class MessageRequest(BaseModel):
    message: str

@app.get("/")
def root():
    return FileResponse("static/index.html")

@app.post("/chat")
def chat(request: MessageRequest):
    # Always load fresh from file — ensures graph consistency across requests
    graph = load_graph()

    user_message = request.message.strip()
    if not user_message:
        return {
            "response": "Please type something!",
            "extraction": [],
            "graph": graph
        }

    # Step 1: Extract ALL relationships using LLM
    # Uses Chain of Thought prompting (Wei et al., NeurIPS 2022)
    # and Semantic Role Labeling principles (He et al., ACL 2017)
    extractions = extract_relationship(user_message, graph)

    # Step 2: Save only unique extractions that actually changed the graph
    # Deduplication follows knowledge graph principles (Dong et al., KDD 2014)
    filtered_extractions = []
    for extraction in extractions:
        person = (extraction.get("person") or "").strip().capitalize()
        relationship = (extraction.get("relationship") or "").strip().lower()
        already_exists = any(
            e["target"].lower() == person.lower() and
            e["relation"].lower() == relationship.lower()
            for e in graph["edges"]
        )
        if not already_exists:
            filtered_extractions.append(extraction)

    if filtered_extractions:
        save_extraction(user_message, filtered_extractions)

    # Step 3: Update graph with hybrid filtering pipeline
    for extraction in extractions:
        action = extraction.get("action", "none")

        # Normalize person name to canonical form
        # Ontology normalization principle (Suchanek et al., WWW 2007)
        person = extraction.get("person")
        if person:
            person = person.strip().capitalize()

        # Block generic relationship words used as person names
        # Based on NER common noun vs proper noun distinction
        # (Nadeau & Sekine, Lingvisticae Investigationes 2007)
        generic_names = [
            "brother", "sister", "friend", "colleague", "manager",
            "mentor", "neighbor", "roommate", "classmate", "cousin",
            "partner", "boyfriend", "girlfriend", "mother", "father",
            "mom", "dad", "uncle", "aunt", "boss", "coworker"
        ]
        if person and person.lower() in generic_names:
            continue

        
        # Normalize relationship label to canonical lowercase form
        # Ontology normalization (Suchanek et al., WWW 2007)
        relationship = extraction.get("relationship")
        if relationship:
            relationship = relationship.strip().lower()
        if relationship in ["null", "none", ""]:
            relationship = None

        # Hybrid Architecture: Rule-based post-processing filter
        # Applied after neural extraction to improve precision
        # (Chiticariu et al., EMNLP 2013 — hybrid IE systems)
        # Possessive pattern detection based on dependency parsing
        # nmod:poss relations indicate indirect mentions
        # (Manning et al., ACL 2014 — Stanford CoreNLP)
        if person and ("'s" in person or "s'" in person):
            continue

        # Block indirect relationship labels using lexical pattern matching
        # Prevents hallucinated indirect relations (e.g. "brother's friend")
        # Post-hoc filtering approach (Lockard et al., ACL 2020)
        if relationship and any(word in relationship.lower() for word in [
            "father of", "mother of", "friend of", "dad of",
            "brother of", "sister of", "colleague of", "parent of",
            "brother's", "sister's", "friend's", "colleague's",
            "cousin's", "neighbor's", "roommate's", "classmate's",
            "partner's", "manager's", "mentor's", "uncle's", "aunt's"
        ]):
            continue

        # old_relationship kept for backward compatibility with update action
        old_relationship = extraction.get("old_relationship")
        if old_relationship:
            old_relationship = old_relationship.strip().lower()
        if old_relationship in ["null", "none", ""]:
            old_relationship = None

        if action == "add" and person and relationship:
            # Idempotent add — skip if relation already exists
            # Knowledge graph deduplication (Dong et al., KDD 2014)
            already_exists = any(
                e["target"].lower() == person.lower() and
                e["relation"].lower() == relationship.lower()
                for e in graph["edges"]
            )
            if not already_exists:
                graph = add_relationship(graph, person, relationship)

        elif action == "remove" and person:
            # Try to remove specific relationship first
            # Fall back to removing all if relation unknown
            if relationship:
                graph = remove_relationship(graph, person, relationship)
            elif old_relationship:
                graph = remove_relationship(graph, person, old_relationship)
            else:
                graph = remove_all_relationships(graph, person)

        elif action == "update" and person and relationship:
            # Check current active edges first
            existing_relation = next(
                (e["relation"] for e in graph["edges"] 
                if e["target"].lower() == person.lower()),
                None
            )
            # Also check removed list for previously ended relationships
            removed_relation = next(
                (r["relation"] for r in graph.get("removed", [])
                if r["person"].lower() == person.lower()),
                None
            )
            # Only update if relationship actually changed
            # Prevents redundant graph mutations
            if existing_relation and existing_relation.lower() != relationship.lower():
                graph = remove_all_relationships(graph, person)
                graph = add_relationship(graph, person, relationship)
            elif removed_relation:
                #Re-add previously removed person with new relationship
                graph = add_relationship(graph, person, relationship)

    # Step 4: Generate contextual empathetic response using updated graph
    # Graph-conditioned response generation
    response = generate_response(user_message, extractions, graph)

    return {
        "response": response,
        "extraction": extractions,
        "graph": graph
    }

@app.get("/graph")
def get_graph():
    # Always load fresh from file to reflect latest state
    return load_graph()