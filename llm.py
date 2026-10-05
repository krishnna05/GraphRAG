import json
from typing import Any, Dict, List, Tuple
from urllib.parse import urlsplit

from ollama import Client as OllamaClient
from models import GraphEdge, GraphNode, normalize_entity_name


def _parse_json_object(content: str, model_name: str) -> Dict[str, Any]:
    if not content or not content.strip():
        raise ValueError(f"Ollama returned an empty response for model '{model_name}'.")

    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        object_start = content.find("{")
        if object_start == -1:
            raise ValueError(f"Ollama did not return a JSON object for model '{model_name}'.") from None

        try:
            parsed, _ = json.JSONDecoder().raw_decode(content[object_start:])
        except json.JSONDecodeError as exc:
            raise ValueError(f"Ollama returned malformed JSON for model '{model_name}'.") from exc

    if not isinstance(parsed, dict):
        raise ValueError(f"Ollama returned a non-object JSON response for model '{model_name}'.")

    return parsed


class OllamaService:
    """Owns all Ollama interaction and prompt construction."""

    def __init__(self, host: str):
        self.client = OllamaClient(host=host)
        self.supports_structured_outputs = urlsplit(host).hostname != "ollama.com"

    def extract_graph_elements(
        self, document_content: str, document_name: str, model_name: str
    ) -> Tuple[List[GraphNode], List[GraphEdge]]:
        prompt = f"""You are a structural parser. Extract core entities and relations from the text.
Return one valid JSON object only: no markdown fences or prose before or after it.
Use this schema:
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
        chat_options = {"format": "json"} if self.supports_structured_outputs else {}
        completion = self.client.chat(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            **chat_options,
        )
        response_content = completion["message"]["content"]
        parsed = _parse_json_object(response_content, model_name)

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
