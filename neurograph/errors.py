"""Exceptions raised by NeuroGraph's core engine."""


class GraphError(Exception):
    """Base class for expected NeuroGraph errors."""


class ValidationError(GraphError, ValueError):
    """Raised when graph data is invalid."""


class NotFoundError(GraphError, LookupError):
    """Raised when a requested node or edge does not exist."""


class DuplicateError(GraphError):
    """Raised when an ID conflicts with existing graph data."""


class IntelligenceError(GraphError):
    """Raised when an optional intelligence operation cannot be completed."""
