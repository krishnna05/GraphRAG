from typing import Dict, List
from neo4j import GraphDatabase
from models import GraphEdge, GraphNode

class Neo4jGraphStore:
    """Encapsulates Neo4j schema, writes, retrieval and graph traversal."""

    def __init__(self, connection_url: str, auth_user: str, auth_pass: str):
        self.db_driver = GraphDatabase.driver(connection_url, auth=(auth_user, auth_pass))
        self.ensure_schema()

    def shutdown(self) -> None:
        self.db_driver.close()

    def ensure_schema(self) -> None:
        with self.db_driver.session() as session:
            session.run(
                """
                CREATE CONSTRAINT graph_node_entity_key IF NOT EXISTS
                FOR (n:GraphNode) REQUIRE n.entity_key IS UNIQUE
                """
            )
            session.run(
                """
                CREATE INDEX graph_node_name IF NOT EXISTS
                FOR (n:GraphNode) ON (n.name)
                """
            )

    def reset_database(self) -> None:
        with self.db_driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")

    def insert_nodes(self, nodes: List[GraphNode]) -> None:
        if not nodes:
            return
        with self.db_driver.session() as session:
            session.execute_write(self._insert_nodes_tx, nodes)

    @staticmethod
    def _insert_nodes_tx(tx, nodes: List[GraphNode]) -> None:
        tx.run(
            """
            UNWIND $nodes AS node
            MERGE (e:GraphNode {entity_key: node.entity_key})
            SET e.name = CASE WHEN e.name IS NULL OR e.name = '' THEN node.display_name ELSE e.name END,
                e.category = CASE WHEN e.category IS NULL OR e.category = '' THEN node.category ELSE e.category END,
                e.summary = CASE WHEN e.summary IS NULL OR e.summary = '' THEN node.summary ELSE e.summary END,
                e.origin_documents = CASE
                    WHEN e.origin_documents IS NULL THEN [node.origin_document]
                    WHEN node.origin_document IN e.origin_documents THEN e.origin_documents
                    ELSE e.origin_documents + node.origin_document
                END,
                e.text_snippet = CASE WHEN e.text_snippet IS NULL OR e.text_snippet = '' THEN node.text_snippet ELSE e.text_snippet END
            """,
            nodes=[node.__dict__ for node in nodes],
        )

    def insert_edges(self, edges: List[GraphEdge]) -> None:
        if not edges:
            return
        with self.db_driver.session() as session:
            session.execute_write(self._insert_edges_tx, edges)

    @staticmethod
    def _insert_edges_tx(tx, edges: List[GraphEdge]) -> None:
        tx.run(
            """
            UNWIND $edges AS edge
            MATCH (x:GraphNode {entity_key: edge.source_entity_key})
            MATCH (y:GraphNode {entity_key: edge.target_entity_key})
            MERGE (x)-[r:LINKED_TO {type: edge.edge_type, origin_document: edge.origin_document}]->(y)
            SET r.context = edge.context
            """,
            edges=[edge.__dict__ for edge in edges],
        )

    def query_nodes_by_text(self, text_query: str) -> List[Dict]:
        with self.db_driver.session() as session:
            cursor = session.run(
                """
                MATCH (n:GraphNode)
                WHERE toLower(n.name) CONTAINS toLower($search)
                   OR toLower(n.summary) CONTAINS toLower($search)
                RETURN n.entity_key AS entity_key,
                       n.name AS name,
                       n.summary AS description,
                       n.origin_documents AS sources,
                       n.text_snippet AS chunk,
                       n.category AS type
                LIMIT 12
                """,
                search=text_query,
            )
            return [dict(record) for record in cursor]

    def fetch_hops_context(self, start_entity_key: str, depth: int = 2) -> List[Dict]:
        depth = max(1, min(depth, 4))
        query = f"""
            MATCH path = (root:GraphNode {{entity_key: $entity_key}})-[*1..{depth}]-(neighbor:GraphNode)
            RETURN DISTINCT neighbor.entity_key AS entity_key,
                   neighbor.name AS name,
                   neighbor.summary AS summary,
                   neighbor.origin_documents AS sources,
                   neighbor.text_snippet AS chunk,
                   [rel IN relationships(path) | rel.context] AS path_contexts
            LIMIT 25
        """
        with self.db_driver.session() as session:
            cursor = session.run(query, entity_key=start_entity_key)
            return [dict(record) for record in cursor]

    def get_stats(self) -> Dict[str, int]:
        with self.db_driver.session() as session:
            nodes = session.run("MATCH (n:GraphNode) RETURN count(n) AS cnt").single()["cnt"]
            edges = session.run("MATCH ()-[r:LINKED_TO]->() RETURN count(r) AS cnt").single()["cnt"]
            return {"nodes": nodes, "edges": edges}
