import re
from graph import Neo4jGraphStore
from llm import OllamaService
from models import AttributedAnswer, SourceAttribution


def run_attributed_graph_rag(user_query: str, graph_store: Neo4jGraphStore, llm: OllamaService, model_name: str = "llama3.2") -> AttributedAnswer:
    trace_steps = [f"🔎 Querying Neo4j Graph Index for: '{user_query}'"]
    initial_matches = graph_store.query_nodes_by_text(user_query)
    if not initial_matches:
        return AttributedAnswer(
            content="No matching nodes found in the knowledge graph to answer your query.",
            sources=[], execution_steps=trace_steps,
        )

    trace_steps.append(f"📊 Identified {len(initial_matches)} potential starting points")
    combined_contexts = []
    for match in initial_matches[:3]:
        trace_steps.append(f"🔗 Traversing paths from node: {match['name']}")
        for nb in graph_store.fetch_hops_context(match["entity_key"], depth=2):
            combined_contexts.append(nb)
            trace_steps.append(f"  → Found connected element: {nb['name']}")

    context_builder = []
    provenance_reference = {}
    for idx, ctx in enumerate(combined_contexts):
        ref_id = f"[{idx + 1}]"
        context_builder.append(f"{ref_id} Entity: {ctx['name']} - {ctx['summary']}")
        provenance_reference[ref_id] = ctx

    compiled_context = "\n".join(context_builder)
    trace_steps.append(f"📝 Compiled context from {len(context_builder)} nodes")

    prompt = f"""You are a query processor. Answer the user question based only on the graph context below.
Every factual claim must include one or more source references using [N] notation.
Do not invent references.

GRAPH CONTEXT:
{compiled_context}

QUESTION: {user_query}
"""
    try:
        answer_text = llm.answer(model_name, prompt)
        trace_steps.append("✅ Generated attributed text from local LLM")
        attributions = []
        for citation in sorted(set(re.findall(r"\[(\d+)\]", answer_text)), key=int):
            key = f"[{citation}]"
            if key not in provenance_reference:
                continue
            item = provenance_reference[key]
            sources = item.get("sources") or ["Unknown document"]
            attributions.append(
                SourceAttribution(
                    claim_text=f"Claim linked to citation {key}",
                    doc_source=", ".join(sources),
                    reference_text=item.get("chunk", ""),
                    score=1.0,
                    path_trace=[f"GraphNode: {item['name']}"],
                )
            )
        trace_steps.append(f"🔒 Resolved {len(attributions)} citation references")
        return AttributedAnswer(answer_text, attributions, trace_steps)
    except Exception as exc:
        return AttributedAnswer(f"An error occurred during text generation: {exc}", [], trace_steps)
