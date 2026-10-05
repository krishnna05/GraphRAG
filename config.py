import os

import streamlit as st
from streamlit.errors import StreamlitSecretNotFoundError


def _get_setting(name: str, default: str) -> str:
    value = os.getenv(name)
    if value:
        return value

    try:
        return str(st.secrets[name])
    except (KeyError, StreamlitSecretNotFoundError):
        return default


def _running_in_docker() -> bool:
    return os.path.exists("/.dockerenv") or os.getenv("IS_DOCKER", "").lower() in {"1", "true", "yes"}


def _normalize_host_url(value: str, default: str, scheme: str) -> str:
    if not value:
        return default

    cleaned = value.strip()
    if "0.0.0.0" in cleaned or cleaned.startswith("[::]") or cleaned.startswith("::"):
        return default

    if cleaned.startswith("http://") or cleaned.startswith("https://"):
        return cleaned
    if cleaned.startswith(("bolt://", "bolt+s://", "neo4j://", "neo4j+s://")):
        return cleaned
    if cleaned.startswith("/"):
        return default

    return f"{scheme}{cleaned}" if not cleaned.startswith(scheme) else cleaned


DEFAULT_NEO4J_URI = "bolt://neo4j:7687" if _running_in_docker() else "bolt://localhost:7687"
DEFAULT_OLLAMA_HOST = "http://ollama:11434" if _running_in_docker() else "http://localhost:11435"

NEO4J_URI = _normalize_host_url(_get_setting("NEO4J_URI", DEFAULT_NEO4J_URI), DEFAULT_NEO4J_URI, "bolt://")

NEO4J_USER = _get_setting(
    "NEO4J_USER",
    "neo4j",
)

NEO4J_PASSWORD = _get_setting(
    "NEO4J_PASSWORD",
    "",
)

OLLAMA_HOST = _normalize_host_url(_get_setting("OLLAMA_HOST", DEFAULT_OLLAMA_HOST), DEFAULT_OLLAMA_HOST, "http://")

DEFAULT_MODEL = _get_setting(
    "OLLAMA_MODEL",
    "llama3.2",
)

ollama_api_key = _get_setting("OLLAMA_API_KEY", "")
if ollama_api_key:
    os.environ["OLLAMA_API_KEY"] = ollama_api_key

if OLLAMA_HOST.rstrip("/") == "https://ollama.com":
    AVAILABLE_MODELS = ["gemma4:31b"]
else:
    AVAILABLE_MODELS = ["llama3.2", "mistral", "phi3"]