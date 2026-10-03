import json
from typing import List, Tuple
from ollama import Client as OllamaClient
from models import GraphEdge, GraphNode, normalize_entity_name


class OllamaService:
    """Owns all Ollama interaction and prompt construction."""

    def __init__(self, host: str):
        self.client = OllamaClient(host=host)

    def extract_graph_elements(
        self, document_content: str, document_name: str, model_name: str
    ) -> Tuple[List[GraphNode], List[GraphEdge]]:
        prompt = f"""You are a structural parser. Extract core entities and relations from the text.
Return strictly valid JSON with this schema:
{{
  "nodes": [
    {{"name": "Entity Name", "type": "PERSON/ORGANIZATION/CONCEPT/TECHNOLOGY/EVENT", "summary": "Brief explanation"}}
  ],
  "edges": [
    {{"source": "Source Name", "target": "Target Name", "type": "RELATIONSHIP_TYPE", "context": "How they connect"}}
  ]
}}

Text to analyze:
{document_content}
"""
        completion = self.client.chat(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            format="json",
        )
        parsed = json.loads(completion["message"]["content"])

        nodes: List[GraphNode] = []
        entity_lookup = {}
        for raw in parsed.get("nodes", []):
            name = str(raw["name"]).strip()
            key = normalize_entity_name(name)
            entity_lookup[key] = name
            nodes.append(
                GraphNode(
                    entity_key=key,
                    display_name=name,
                    category=str(raw.get("type", "CONCEPT")).upper(),
                    summary=str(raw.get("summary", "")).strip(),
                    origin_document=document_name,
                    text_snippet=document_content[:250] + ("..." if len(document_content) > 250 else ""),
                )
            )

        edges: List[GraphEdge] = []
        for raw in parsed.get("edges", []):
            source = str(raw["source"]).strip()
            target = str(raw["target"]).strip()
            edges.append(
                GraphEdge(
                    source_entity_key=normalize_entity_name(source),
                    target_entity_key=normalize_entity_name(target),
                    source_name=source,
                    target_name=target,
                    edge_type=str(raw.get("type", "RELATED_TO")).upper(),
                    context=str(raw.get("context", "")).strip(),
                    origin_document=document_name,
                )
            )
        return nodes, edges

    def answer(self, model_name: str, prompt: str) -> str:
        response = self.client.chat(model=model_name, messages=[{"role": "user", "content": prompt}])
        return response["message"]["content"]
