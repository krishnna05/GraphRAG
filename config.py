import os

NEO4J_URI = os.getenv(
    "NEO4J_URI",
    "bolt://localhost:7687"
)

NEO4J_USER = os.getenv(
    "NEO4J_USER",
    "neo4j"
)

NEO4J_PASSWORD = os.getenv(
    "NEO4J_PASSWORD",
    "password"
)

OLLAMA_HOST = os.getenv(
    "OLLAMA_HOST",
    "http://localhost:11434"
)

DEFAULT_MODEL = os.getenv(
    "OLLAMA_MODEL",
    "llama3.2"
)

AVAILABLE_MODELS = [
    "llama3.2",
    "mistral",
    "phi3",
]