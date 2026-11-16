"""Optional client for a locally running Ollama service."""

from __future__ import annotations

import json
import http.client
import math
import numbers
import socket
from urllib.parse import urlparse
from typing import Any

from neurograph.errors import IntelligenceError, ValidationError


class OllamaClient:
    """Call Ollama locally with distinct connect and response timeouts."""

    def __init__(
        self,
        *,
        model: str = "llama3:latest",
        embedding_model: str = "nomic-embed-text",
        base_url: str = "http://localhost:11434",
        timeout: float = 240.0,
        connection_timeout: float = 10.0,
        keep_alive: str = "10m",
    ) -> None:
        if not isinstance(base_url, str):
            raise ValidationError("Ollama base_url must be a local HTTP URL.")
        normalized_url = base_url.strip().rstrip("/")
        if not normalized_url:
            raise ValidationError("Ollama base_url must be a local HTTP URL.")
        try:
            parsed = urlparse(normalized_url)
            port = parsed.port
        except ValueError as exc:
            raise ValidationError("Ollama base_url is malformed.") from exc
        if parsed.scheme != "http" or parsed.hostname not in {"localhost", "127.0.0.1", "::1"}:
            raise ValidationError("Ollama base_url must point to a local HTTP service.")
        if (
            parsed.username
            or parsed.password
            or parsed.path not in ("", "/")
            or parsed.query
            or parsed.fragment
        ):
            raise ValidationError("Ollama base_url must be a plain local service origin.")
        if port is not None and not 1 <= port <= 65535:
            raise ValidationError("Ollama service port must be between 1 and 65535.")
        if (
            not isinstance(model, str)
            or not model.strip()
            or not isinstance(embedding_model, str)
            or not embedding_model.strip()
        ):
            raise ValidationError("Ollama model names must not be empty.")
        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, numbers.Real)
            or not math.isfinite(timeout)
            or timeout <= 0
        ):
            raise ValidationError("Ollama timeout must be greater than zero.")
        if (
            isinstance(connection_timeout, bool)
            or not isinstance(connection_timeout, numbers.Real)
            or not math.isfinite(connection_timeout)
            or connection_timeout <= 0
        ):
            raise ValidationError("Ollama connection_timeout must be greater than zero.")
        if not isinstance(keep_alive, str) or not keep_alive.strip():
            raise ValidationError("Ollama keep_alive must not be empty.")
        self.model = model
        self.embedding_model = embedding_model
        self.base_url = normalized_url
        self.timeout = timeout
        self.connection_timeout = connection_timeout
        self.keep_alive = keep_alive
        self._host = parsed.hostname
        self._port = port or 80

    def embed(self, text: str) -> list[float]:
        """Return a local embedding vector for text."""
        if not isinstance(text, str):
            raise ValidationError("Embedding input must be text.")
        if not text.strip():
            raise ValidationError("Cannot create an embedding for empty text.")
        response = self._post("/api/embed", {"model": self.embedding_model, "input": text})
        embeddings = response.get("embeddings")
        if not isinstance(embeddings, list) or not embeddings or not isinstance(embeddings[0], list):
            raise IntelligenceError("Ollama returned no embedding vector.")
        try:
            vector = [
                float(value)
                for value in embeddings[0]
                if not isinstance(value, bool) and isinstance(value, numbers.Real)
            ]
        except (TypeError, ValueError) as exc:
            raise IntelligenceError("Ollama returned an invalid embedding vector.") from exc
        if len(vector) != len(embeddings[0]) or not vector or any(
            not math.isfinite(value) for value in vector
        ):
            raise IntelligenceError("Ollama returned an invalid embedding vector.")
        return vector

    def generate(self, prompt: str) -> str:
        """Generate a non-streaming response with the configured local chat model."""
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValidationError("Chat prompt must not be empty.")
        response = self._post(
            "/api/chat",
            {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "keep_alive": self.keep_alive,
            },
        )
        message = response.get("message")
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, str) or not content.strip():
            raise IntelligenceError("Ollama returned an empty chat response.")
        return content.strip()

    def _post(self, endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
        connection = http.client.HTTPConnection(
            self._host, self._port, timeout=self.connection_timeout
        )
        connected = False
        try:
            connection.connect()
            connected = True
            if connection.sock is not None:
                connection.sock.settimeout(self.timeout)
            connection.request(
                "POST",
                endpoint,
                body=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
            )
            response = connection.getresponse()
            raw_body = response.read(2048 if response.status >= 400 else -1)
            details = raw_body.decode("utf-8", errors="replace").strip()
            if response.status >= 400:
                body = details
                try:
                    error_body = json.loads(details)
                    if isinstance(error_body, dict):
                        body = str(error_body.get("error", details))
                except json.JSONDecodeError:
                    pass
                model_name = payload.get("model")
                model_missing = (
                    response.status == 404
                    and isinstance(model_name, str)
                    and "model" in body.casefold()
                    and "not found" in body.casefold()
                )
                reason = (
                    f"Ollama model {model_name!r} was not found"
                    if model_missing
                    else f"Ollama request to {endpoint} failed with HTTP {response.status} "
                    f"({response.reason})"
                )
                if body:
                    reason += f": {body}"
                raise IntelligenceError(reason)
            body = json.loads(details)
        except (socket.timeout, TimeoutError) as exc:
            if connected:
                reason = f"Ollama response timed out after {self.timeout:g} seconds"
            else:
                reason = (
                    f"Could not connect to local Ollama within "
                    f"{self.connection_timeout:g} seconds"
                )
            raise IntelligenceError(f"{reason}: {exc}") from exc
        except OSError as exc:
            phase = "communicate with" if connected else "connect to"
            raise IntelligenceError(f"Could not {phase} local Ollama service: {exc}") from exc
        except (http.client.HTTPException, UnicodeError, json.JSONDecodeError) as exc:
            raise IntelligenceError(f"Invalid response from local Ollama service: {exc}") from exc
        finally:
            connection.close()
        if not isinstance(body, dict):
            raise IntelligenceError("Ollama returned an invalid response.")
        if "error" in body:
            raise IntelligenceError(f"Ollama error: {body['error']}")
        return body
