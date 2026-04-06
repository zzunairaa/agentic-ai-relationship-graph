# graph.py
# Manages the relationship graph — storage, retrieval, and mutation
# Handles graph.json and extractions.json persistence

import json
import os

GRAPH_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "graph.json")

# Load graph from file if it exists, otherwise return default empty graph
def load_graph():
    if os.path.exists(GRAPH_FILE):
        with open(GRAPH_FILE, "r") as f:
            return json.load(f)
    return {
        "nodes": ["User"],
        "edges": [],
        "removed": []
    }

def save_graph(graph):
    # Ensure removed list always exists before saving
    if "removed" not in graph:
        graph["removed"] = []
    with open(GRAPH_FILE, "w") as f:
        json.dump(graph, f, indent=2)

def add_relationship(graph, person, relation):
    # Add person to nodes if not already present
    if person not in graph["nodes"]:
        graph["nodes"].append(person)

    # Add edge only if it doesn't already exist — prevents duplicates
    exists = any(
        e["source"] == "User" and
        e["target"] == person and
        e["relation"] == relation
        for e in graph["edges"]
    )

    if not exists:
        graph["edges"].append({
            "source": "User",
            "target": person,
            "relation": relation
        })

    # Keep removed history intact — never delete past relationships
    # This enables empathy responses when removed person is mentioned later

    save_graph(graph)
    return graph

def remove_relationship(graph, person, relation):
# Save to removed list before deleting — preserves relationship history
    if "removed" not in graph:
        graph["removed"] = []
    already_removed = any(
        r["person"].lower() == person.lower()
        for r in graph["removed"]
    )
    if not already_removed:
        graph["removed"].append({
            "person": person,
            "relation": relation
        })
    # Remove the matching edge
    graph["edges"] = [
        e for e in graph["edges"]
        if not (
            e["source"] == "User" and
            e["target"] == person and
            e["relation"] == relation
        )
    ]

    # Remove person from nodes if no more edges
    still_connected = any(
        e["target"] == person
        for e in graph["edges"]
    )
    # Node is kept in graph even after edge removal
    # This preserves identity for future empathy responses
    save_graph(graph)
    return graph

def remove_all_relationships(graph, person):
    # Save first known relation to removed list before deleting
    if "removed" not in graph:
        graph["removed"] = []
    existing_relations = [
        e["relation"] for e in graph["edges"]
        if e["target"] == person
    ]
    already_removed = any(
        r["person"].lower() == person.lower()
        for r in graph["removed"]
    )
    if not already_removed and existing_relations:
        graph["removed"].append({
            "person": person,
            "relation": existing_relations[0]
        })
    # Remove all edges for this person
    graph["edges"] = [
        e for e in graph["edges"]
        if e["target"] != person
    ]

# Node preserved for memory continuity — not removed from nodes list
    save_graph(graph)
    return graph

EXTRACTIONS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "extractions.json")

def save_extraction(user_message, extractions):
      # Load existing extraction log
    if os.path.exists(EXTRACTIONS_FILE):
        try:
            with open(EXTRACTIONS_FILE, "r") as f:
                all_extractions = json.load(f)
        except:
            all_extractions = []
    else:
        all_extractions = []

    # Append new extraction entry with timestamp
    from datetime import datetime
    all_extractions.append({
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "message": user_message,
        "extraction": extractions
    })

    with open(EXTRACTIONS_FILE, "w") as f:
        json.dump(all_extractions, f, indent=2)

# Function summary:
# load_graph()               → reads graph.json, returns default if missing
# save_graph()               → writes graph.json after every mutation
# add_relationship()         → adds User→Person edge, skips if duplicate
# remove_relationship()      → removes specific edge, preserves node and history
# remove_all_relationships() → removes all edges for person, preserves node
# save_extraction()          → appends extraction to extractions.json with timestamp