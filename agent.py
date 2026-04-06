# agent.py
# Handles all LLM communication — relationship extraction and response generation
# Uses Groq API with LLaMA 3.3 70B model

import json
import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
MODEL = "llama-3.3-70b-versatile"


def extract_relationship(user_message, current_graph):
    """
    Extracts relationship information from a user message using LLM.

    Techniques used:
    - Chain of Thought prompting (Wei et al., NeurIPS 2022)
      Forces LLM to reason step-by-step before extracting
    - Few-shot prompting with negative examples (Brown et al., NeurIPS 2020)
      IGNORE and EXTRACT examples guide the model
    - Semantic Role Labeling principle (He et al., ACL 2017)
      Only extracts when User (I/me/my) is the grammatical subject
    - Constrained output format — JSON only, no free text
    """

    prompt = f"""
You are an AI that extracts relationship information from messages.

Current relationship graph:
{json.dumps(current_graph, indent=2)}

User message: "{user_message}"

Before extracting, reason through these steps:
1. Who is the grammatical subject of this sentence? (I/me/my OR someone else?)
2. Is the User (I/me/my) directly claiming this relationship?
3. Does the person have a real name? (not just "my sister" or "my friend")
4. Is this a direct User→Person relationship or indirect (X's friend, Y's dad)?
5. Only if ALL conditions are met → extract it

Extract ALL relationships mentioned and respond ONLY with a JSON list like this:
[
  {{
    "person": "name of person",
    "relationship": "type of relationship",
    "action": "add or remove or update"
  }}
]

Rules:
- Extract ALL people and relationships mentioned in one message
- action "add" → user is forming a new relationship
- action "remove" → user is ending a relationship
- action "update" → user is changing an existing relationship 
- Do NOT use "update" if the relationship has not actually changed
- Do NOT use "add" if the person already exists in the graph with the same relationship
- Check the current graph before extracting — if person and relationship already exist → return []
- If you use "update" for a person, do NOT also add a "remove" for the same person
- Never return duplicate entries for the same person in one response
- If no relationship is mentioned → return empty list []
- ONLY extract relationships where the User is directly related to the person (e.g., "my friend", "my sister", "my colleague")
- The user must explicitly say the relationship — do NOT infer or assume
- ONLY extract a person if they have a real name — do NOT extract if the person has no name
- "I met my sister" → [] because no name is given
- "I met my sister Emma" → extract Emma as sister
- If the relationship involves another person (e.g., "my sister's friend", "someone who works with my sister", "cousin's acquaintance") → DO NOT extract it
- If the sentence does NOT explicitly describe a direct relationship with the User → return []
- Ignore relationships between two other people (e.g., "Sara and Noah are friends") → return []
- NEVER infer or create indirect relationship labels like "neighbor's brother" or "friend's colleague"
- If someone introduces the User to another person → IGNORE that person unless User explicitly states their own relationship with them
- "X introduced me to Y" → return [] for Y unless User says "Y is my friend/colleague/etc"
- A relationship is ONLY valid if the grammatical subject is "I", "me", or "my"
- If the subject is another person (she, he, they, her, his, their) → IGNORE completely
- If the person's name contains a possessive (e.g. "Rafay's dad", "Anna's friend") → IGNORE
- "X told me about Y" → return [] for Y always
- "X mentioned their Z named Y" → return [] for Y always  
- "X talked about Y" → return [] for Y always
- Only extract when the User is the direct subject claiming the relationship
- Keep relationship labels short and simple: friend, sister, brother, colleague, manager, mentor, neighbor, roommate, classmate, cousin, partner, boyfriend, girlfriend
- Respond with ONLY the JSON list, nothing else

Examples of what to IGNORE (return []):
- "I met someone who works with my sister" → []
- "There's a guy my cousin talks about named Ali" → []
- "My brother's friend Max came over" → []
- "Sara and Noah are best friends" → []
- "My colleague introduced me to his friend" → []
- "Hassan introduced me to his brother Ali" → []
- "My neighbor introduced me to someone new" → []
- "Zara introduced me to some of her friends" → []
- "I met my sister today" → [] (no name given)
- "I talked to my friend yesterday" → [] (no name given)
- "I met my brother today" → [] (no name given)
- "I spent the day with Anna, she is my sister" (if Anna is already sister in graph) → []
- "Tariq is still my best friend" (if Tariq is already best friend in graph) → []
- "she told me about her father Hasnain" → [] (subject is "she" not User)
- "he mentioned his colleague Sara" → [] (subject is "he" not User)
- "I met Rafay's dad today" → [] (possessive of another person)
- "Rafay talked about his family" → [] (subject is Rafay not User)
- "she introduced me to her brother" → [] (subject is "she", brother belongs to her)

Examples of what to EXTRACT:
- "I have a friend named Sara" → [{{"person": "Sara", "relationship": "friend", "action": "add"}}]
- "My sister Anna is visiting" → [{{"person": "Anna", "relationship": "sister", "action": "add"}}]
- "I met Max today and we became friends" → [{{"person": "Max", "relationship": "friend", "action": "add"}}]
- "Sara and I are not friends anymore" → [{{"person": "Sara", "relationship": "friend", "action": "remove"}}]
- "James is now my manager not my colleague" → [{{"person": "James", "relationship": "manager", "action": "update"}}]
- "Hassan introduced me to his brother Ali, and Ali and I became friends" → [{{"person": "Ali", "relationship": "friend", "action": "add"}}]
"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0  # Temperature 0 for deterministic extraction
    )

    raw = response.choices[0].message.content.strip()

    # Strip markdown code blocks if LLM wraps response
    clean = raw.replace("```json", "").replace("```", "").strip()

    # Safely locate JSON list boundaries — handles extra LLM chatter
    start = clean.find("[")
    end = clean.rfind("]") + 1

    if start == -1 or end == 0:
        return []

    # Safe JSON parsing — returns empty list on failure
    try:
        return json.loads(clean[start:end])
    except:
        return []


def generate_response(user_message, extractions, current_graph):
    """
    Generates a contextual, empathetic response using the relationship graph.

    Techniques used:
    - Graph-conditioned response generation
      Graph state is passed as context to the LLM
    - Episodic memory for empathy (Tulving, 1972)
      Removed relationships trigger empathetic tone
    - Tone-aware prompting based on action type
      Different instructions for add/remove/update/existing/removed
    """

    # Determine what relationship actions occurred in this message
    actions = [e.get("action") for e in extractions if e.get("action")]
    has_add = "add" in actions
    has_remove = "remove" in actions
    has_update = "update" in actions

    # Check if mentioned people already exist in graph nodes
    # Uses nodes (not edges) to catch removed people too
    existing_people = [
        n.lower() for n in current_graph.get("nodes", [])
        if n.lower() != "user"
    ]
    is_existing_mention = any(
        (e.get("person") or "").lower() in existing_people
        for e in extractions
    )

    # Build removed people dict — only include people NOT in active edges
    # Prevents false empathy for people who are still active in the graph
    active_people = [e["target"].lower() for e in current_graph.get("edges", [])]
    removed_people = {
        r["person"].lower(): r["relation"]
        for r in current_graph.get("removed", [])
        if r["person"].lower() not in active_people
    }

    # Check both extractions AND raw message text for removed people mentions
    # Raw message check handles cases where extraction returned []
    is_removed_mention = (
        any(
            (e.get("person") or "").lower() in removed_people
            for e in extractions
        ) or
        any(
            name in user_message.lower()
            for name in removed_people
        )
    )

    # Build context string describing removed relationships
    # Passed to LLM so it can respond with appropriate empathy
    removed_context = ""
    if is_removed_mention:
        for e in extractions:
            name = (e.get("person") or "").lower()
            if name in removed_people:
                removed_context += f"- {e.get('person')} was previously a {removed_people[name]} but that relationship ended.\n"
        for name, relation in removed_people.items():
            if name in user_message.lower() and name not in removed_context.lower():
                removed_context += f"- {name.capitalize()} was previously a {relation} but that relationship ended.\n"

    # Select tone instruction based on relationship action
    # Removed mention checked FIRST — takes priority over all other tones
    # This ensures empathy triggers even when extraction returns []
    if is_removed_mention:
        # Empathy tone for previously removed relationships
        # Inspired by affective computing principles (Picard, MIT Press 1997)
        tone_instruction = f"""
The user is mentioning someone whose relationship with them has ended in the past:
{removed_context}
Respond with emotional awareness and gentle empathy.
Acknowledge that things were difficult between them before.
Do NOT pretend you don't know about the past.
Use soft language like 'I remember things were tough with them' or 'how are you feeling about seeing them again?'
Be warm, caring and supportive.
"""

    elif not extractions:
        # No relationship detected — general friendly response
        tone_instruction = "Just reply in a friendly, natural, warm way."

    elif has_remove and has_add:
        # Mixed action — acknowledge both loss and new connection
        tone_instruction = """
The user has both added someone new and lost/ended another relationship.
Acknowledge both warmly — celebrate the new connection and show empathy for the loss.
Be balanced, warm and human.
"""
    elif has_remove:
        # Relationship ended — respond with empathy
        tone_instruction = """
The user has ended or lost a relationship.
Respond with empathy and emotional awareness.
Use soft language like 'that must be difficult' or 'I understand things can change'.
Never state facts bluntly like 'X is no longer your friend'.
Be warm, gentle and supportive.
"""
    elif has_update:
        # Relationship changed — acknowledge softly
        tone_instruction = """
The user has changed a relationship.
Acknowledge the change softly and naturally.
Use understanding language like 'it sounds like things have shifted' or 'relationships evolve over time'.
Be warm and non-judgmental.
"""
    elif has_add and not is_existing_mention:
        # New person added — warm and curious
        tone_instruction = """
The user has added a new person to their life.
Respond positively and warmly.
Ask a light, friendly follow-up question about that person.
Show genuine interest.
"""
    elif has_add and is_existing_mention:
        # Existing person mentioned with add action — respond naturally
        tone_instruction = """
The user is talking about someone already in their life.
Respond naturally and warmly.
Do not treat this as meeting someone new.
"""
    elif is_existing_mention:
        # Known person mentioned — respond with context awareness
        tone_instruction = """
The user is mentioning someone already in their life.
Respond naturally and warmly, without assuming any change.
Do not treat this as a relationship update or loss.
"""
    else:
        tone_instruction = "Just reply in a friendly, natural, warm way."

    prompt = f"""
You are a warm, empathetic AI assistant that genuinely cares about the user.

Current relationship graph:
{json.dumps(current_graph, indent=2)}

User just said: "{user_message}"

Relationships extracted: {json.dumps(extractions)}

Tone instructions:
{tone_instruction}

General rules:
- Never be blunt or robotic
- Never say things like "I have updated the graph" or "I have noted this"
- Sound like a caring friend, not an AI assistant
- Keep response to 1-2 sentences maximum
- Use natural, human language
"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.8  # Higher temperature for natural varied responses
    )

    return response.choices[0].message.content.strip()


# Function summary:
# extract_relationship() → sends message to Groq LLM, returns JSON extraction list
# generate_response()    → builds tone-aware prompt, returns empathetic response string
#
# References:
# - Wei et al. (2022) Chain of Thought Prompting. NeurIPS.
# - Brown et al. (2020) Few-shot Learners. NeurIPS.
# - He et al. (2017) Deep Semantic Role Labeling. ACL.
# - Picard, R. (1997) Affective Computing. MIT Press.
# - Tulving, E. (1972) Episodic and Semantic Memory.