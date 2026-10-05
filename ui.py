import streamlit as st

from config import AVAILABLE_MODELS, DEFAULT_MODEL, NEO4J_PASSWORD, NEO4J_URI, NEO4J_USER, OLLAMA_HOST
from graph import Neo4jGraphStore
from llm import OllamaService
from rag import run_attributed_graph_rag

def main() -> None:
    st.set_page_config(page_title="GraphRAG", page_icon="🔍", layout="wide")
    st.title("GraphRAG: Knowledge Graph-based Retrieval-Augmented Generation")
    st.markdown("Developed by Krishna Vishwakarma | Powered by Neo4j and Ollama LLMs")
    st.divider()
    st.markdown("""
    This system implements **Knowledge Graph-based Retrieval-Augmented Generation (GraphRAG)**:
    - **Multi-hop reasoning**: Traverses interconnected elements across source files.
    - **Citations provenance**: Maps generated [N] references to retrieved graph context.
    - **Audit trails**: Displays retrieval and traversal steps.
    """)

    st.sidebar.markdown("### **GraphRAG**")
    st.sidebar.divider()
    st.sidebar.header("⚙️ Graph Configuration")
    if NEO4J_PASSWORD:
        db_uri = NEO4J_URI
        db_user = NEO4J_USER
        db_pass = NEO4J_PASSWORD
        st.sidebar.caption("Neo4j connection is managed by the app deployment.")
    else:
        db_uri = st.sidebar.text_input("Neo4j Database URI", NEO4J_URI)
        db_user = st.sidebar.text_input("Neo4j Username", NEO4J_USER)
        db_pass = st.sidebar.text_input("Neo4j Password", type="password")
    model_choice = st.sidebar.selectbox("Local Inference Model", AVAILABLE_MODELS, index=AVAILABLE_MODELS.index(DEFAULT_MODEL) if DEFAULT_MODEL in AVAILABLE_MODELS else 0)

    if "has_nodes" not in st.session_state:
        st.session_state.has_nodes = False
        st.session_state.processed_docs = []

    tab_docs, tab_query, tab_viz = st.tabs(["📄 Document Ingestion", "❓ Ask Assistant", "🔬 Database Status"])
    demo_documents = {
        "TaskBoard Application Specification": """
        TaskBoard is a project management application that helps teams organize work into
        projects and tasks. Project owners assign tasks to team members, set due dates, and
        track progress on a shared dashboard. The application sends notifications when a task
        is assigned or its status changes, and stores project activity for team members to review.
        """,
        "TaskBoard Application Workflow": """
        In TaskBoard, a team creates a project and adds tasks for planned work. Each task has
        one assignee, a due date, and a status of To Do, In Progress, or Done. When a project
        owner assigns a task, the notification service alerts the assignee. Team members can
        filter the dashboard by project or status, while project owners review overdue tasks
        and the activity history to follow progress.
        """,
    }

    with tab_docs:
        st.header("1. Extract Knowledge Graph elements")
        selected_template = st.selectbox("Select document template:", list(demo_documents.keys()))
        document_payload = st.text_area("Input Document Text:", demo_documents[selected_template], height=200)
        user_doc_name = st.text_input("Specify Document Label:", selected_template)
        if st.button("🔨 Extract & Ingest Nodes"):
            store = None
            try:
                with st.spinner("Extracting elements and relationships using LLM..."):
                    store = Neo4jGraphStore(db_uri, db_user, db_pass)
                    llm = OllamaService(OLLAMA_HOST)
                    nodes, edges = llm.extract_graph_elements(document_payload, user_doc_name, model_choice)
                    store.insert_nodes(nodes)
                    store.insert_edges(edges)
                st.success(f"Successfully loaded {len(nodes)} entities and {len(edges)} relationships.")
                with st.expander("Ingested Entities Details"):
                    for node in nodes:
                        st.write(f"🔹 **{node.display_name}** ({node.category}): {node.summary}")
                with st.expander("Ingested Relationships Details"):
                    for edge in edges:
                        st.write(f"🔗 {edge.source_name} --[{edge.edge_type}]--> {edge.target_name}: {edge.context}")
                st.session_state.has_nodes = True
                if user_doc_name not in st.session_state.processed_docs:
                    st.session_state.processed_docs.append(user_doc_name)
            except Exception as exc:
                st.error(f"Failed to ingest: {exc}")
                st.info("Ensure Neo4j is online and the selected Ollama model is available.")
            finally:
                if store:
                    store.shutdown()

    with tab_query:
        st.header("2. Attributed Query Search")
        if not st.session_state.has_nodes:
            st.warning("Please index documents in the Knowledge Graph before asking questions.")
        else:
            st.info(f"Loaded Indexes: {', '.join(st.session_state.processed_docs)}")
        user_query = st.text_input("Enter Question:", "How does TaskBoard notify team members when a task is assigned?")
        if st.button("🔎 Search Attributed Answer"):
            store = None
            try:
                with st.spinner("Traversing graph and resolving query..."):
                    store = Neo4jGraphStore(db_uri, db_user, db_pass)
                    rag_result = run_attributed_graph_rag(user_query, store, OllamaService(OLLAMA_HOST), model_choice)
                st.subheader("🔬 Path Traversals & Trace Logs")
                for step in rag_result.execution_steps:
                    st.text(step)
                st.subheader("💬 Generated Answer")
                st.write(rag_result.content)
                st.subheader("📚 Source Attributions")
                if rag_result.sources:
                    for idx, source in enumerate(rag_result.sources):
                        with st.expander(f"Source [{idx + 1}]: {source.doc_source}"):
                            st.write(f"**Document Name:** {source.doc_source}")
                            st.write(f"**Context Text:** {source.reference_text}")
                            st.write(f"**Reasoning Trace:** {' ➔ '.join(source.path_trace)}")
                else:
                    st.info("No citation references matched the generated answer.")
            except Exception as exc:
                st.error(f"Error querying: {exc}")
            finally:
                if store:
                    store.shutdown()

    with tab_viz:
        st.header("3. Graph DB Statistics")
        if st.button("📊 Query Database Sizes"):
            store = None
            try:
                store = Neo4jGraphStore(db_uri, db_user, db_pass)
                stats = store.get_stats()
                left, right = st.columns(2)
                left.metric("Total Nodes", stats["nodes"])
                right.metric("Total Edges", stats["edges"])
            except Exception as exc:
                st.error(f"Could not connect: {exc}")
            finally:
                if store:
                    store.shutdown()
        if st.button("🗑️ Clear Database Graph"):
            store = None
            try:
                store = Neo4jGraphStore(db_uri, db_user, db_pass)
                store.reset_database()
                st.session_state.has_nodes = False
                st.session_state.processed_docs = []
                st.success("Successfully reset all database records.")
            except Exception as exc:
                st.error(f"Failed to clear database: {exc}")
            finally:
                if store:
                    store.shutdown()

if __name__ == "__main__":
    main()