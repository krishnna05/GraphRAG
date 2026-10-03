from dataclasses import dataclass
from typing import List
import re

def normalize_entity_name(name: str) -> str:
    """Return a stable global key for entity resolution across documents."""
    value = re.sub(r"[^a-z0-9]+", " ", name.lower()).strip()
    return re.sub(r"\s+", " ", value)

@dataclass(frozen=True)
class GraphNode:
    entity_key: str
    display_name: str
    category: str
    summary: str
    origin_document: str
    text_snippet: str

@dataclass(frozen=True)
class GraphEdge:
    source_entity_key: str
    target_entity_key: str
    source_name: str
    target_name: str
    edge_type: str
    context: str
    origin_document: str

@dataclass(frozen=True)
class SourceAttribution:
    claim_text: str
    doc_source: str
    reference_text: str
    score: float
    path_trace: List[str]

@dataclass(frozen=True)
class AttributedAnswer:
    content: str
    sources: List[SourceAttribution]
    execution_steps: List[str]