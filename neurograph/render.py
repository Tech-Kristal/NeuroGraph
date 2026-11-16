"""HTML rendering for the details and analysis panels (no Qt imports).

Output targets Qt's rich-text subset, so it sticks to simple tables, paragraphs
and inline styles. Concept names are links using the ``ng://`` scheme that the
main window understands.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import html
import json
from typing import Any, Iterable, Mapping, Sequence
from urllib.parse import quote

from neurograph.models import Edge, Node
from neurograph.theme import Theme

MAX_ROWS = 60


@dataclass(frozen=True)
class Link:
    """One relationship row in the details panel."""

    direction: str  # "in" or "out"
    relation: str
    node_id: str
    node_name: str
    edge_id: str


@dataclass(frozen=True)
class NodeDetails:
    """Everything the details panel shows for one concept."""

    node: Node
    category: str
    color: str
    incoming: tuple[Link, ...]
    outgoing: tuple[Link, ...]
    similar: tuple[tuple[float, str, str], ...]
    cluster: tuple[int, int] | None  # (1-based index, member count)


def esc(value: Any) -> str:
    return html.escape(str(value), quote=False).replace("\n", "<br>")


def node_href(node_id: str) -> str:
    return "ng://node/" + quote(node_id, safe="")


def node_link(node_id: str, name: str) -> str:
    return f'<a href="{node_href(node_id)}">{esc(name)}</a>'


def node_labels(nodes: Sequence[Node]) -> dict[str, str]:
    """Display labels for pickers; duplicate names get a short ID suffix."""
    counts = Counter(node.name for node in nodes)
    return {
        node.id: node.name if counts[node.name] == 1 else f"{node.name} ({node.id[:8]})"
        for node in nodes
    }


def document(theme: Theme, body: str) -> str:
    return (
        "<html><head><style>"
        f"a {{ color: {theme.accent}; text-decoration: none; }}"
        "</style></head>"
        f'<body style="color:{theme.text};">{body}</body></html>'
    )


def _p(text: str, *, top: int = 0, bottom: int = 0, extra: str = "") -> str:
    return (
        f'<p style="margin-top:{top}px; margin-bottom:{bottom}px; '
        f'margin-left:0px; margin-right:0px; {extra}">{text}</p>'
    )


def _section(theme: Theme, title: str) -> str:
    return _p(esc(title), top=18, bottom=4, extra=f"color:{theme.text_muted}; font-weight:600;")


def _dot(color: str) -> str:
    return f'<span style="color:{color};">&#9679;</span>'


def message(theme: Theme, title: str, detail: str = "") -> str:
    body = _p(esc(title), top=4, extra=f"font-weight:600;")
    if detail:
        body += _p(esc(detail), top=4, extra=f"color:{theme.text_muted};")
    return document(theme, body)


def _value(value: Any) -> str:
    if isinstance(value, str):
        return esc(value)
    text = json.dumps(value, ensure_ascii=False)
    if len(text) > 300:
        text = text[:300] + "…"
    return f"<code>{esc(text)}</code>"


def properties_table(theme: Theme, properties: Mapping[str, Any]) -> str:
    shown = {k: v for k, v in properties.items() if not (k == "type" and isinstance(v, str))}
    if not shown:
        return _p("No additional properties.", extra=f"color:{theme.text_muted};")
    rows = "".join(
        f'<tr><td valign="top" style="color:{theme.text_muted}; padding:2px 14px 2px 0px;">{esc(key)}</td>'
        f'<td valign="top" style="padding:2px 0px;">{_value(value)}</td></tr>'
        for key, value in shown.items()
    )
    return f'<table cellspacing="0" cellpadding="0" width="100%">{rows}</table>'


def _links_table(theme: Theme, links: Sequence[Link]) -> str:
    if not links:
        return _p("None.", extra=f"color:{theme.text_muted};")
    arrow = {"in": "←", "out": "→"}
    rows = "".join(
        f'<tr><td valign="top" style="color:{theme.text_muted}; padding:2px 8px 2px 0px;">{arrow[link.direction]}</td>'
        f'<td valign="top" style="padding:2px 0px;">{node_link(link.node_id, link.node_name)}'
        f'&nbsp;&nbsp;<span style="color:{theme.text_muted};">{esc(link.relation)}</span></td></tr>'
        for link in links[:MAX_ROWS]
    )
    extra = ""
    if len(links) > MAX_ROWS:
        extra = _p(f"… and {len(links) - MAX_ROWS} more", extra=f"color:{theme.text_muted};")
    return f'<table cellspacing="0" cellpadding="0" width="100%">{rows}</table>{extra}'


def empty_selection(theme: Theme) -> str:
    return message(
        theme,
        "Nothing selected",
        "Select a concept or relationship in the graph or the concept list to see its details.",
    )


def node_html(theme: Theme, details: NodeDetails) -> str:
    node = details.node
    degree = len(details.incoming) + len(details.outgoing)
    noun = "connection" if degree == 1 else "connections"
    nid = quote(node.id, safe="")
    actions = " &nbsp;·&nbsp; ".join(
        (
            f'<a href="ng://edit/{nid}">Edit</a>',
            f'<a href="ng://pathfrom/{nid}">Path from here</a>',
            f'<a href="ng://pathto/{nid}">Path to here</a>',
        )
    )
    body = _p(esc(node.name), extra=f"font-size:13pt; font-weight:600;")
    body += _p(
        f"{_dot(details.color)} {esc(details.category)} &nbsp;·&nbsp; {degree} {noun}",
        top=3,
        extra=f"color:{theme.text_muted};",
    )
    body += _p(esc(node.id), top=2, extra=f"color:{theme.text_disabled}; font-size:8pt;")
    body += _p(actions, top=8)
    body += _section(theme, "Properties") + properties_table(theme, node.properties)
    body += _section(theme, f"Incoming ({len(details.incoming)})") + _links_table(theme, details.incoming)
    body += _section(theme, f"Outgoing ({len(details.outgoing)})") + _links_table(theme, details.outgoing)
    body += _section(theme, "Structural similarity")
    if details.similar:
        rows = "".join(
            f'<tr><td style="color:{theme.text_muted}; padding:2px 12px 2px 0px;">{score:.2f}</td>'
            f"<td>{node_link(other_id, name)}</td></tr>"
            for score, name, other_id in details.similar
        )
        body += f'<table cellspacing="0" cellpadding="0">{rows}</table>'
    else:
        body += _p("No concepts share neighbours with this one.", extra=f"color:{theme.text_muted};")
    body += _section(theme, "Cluster")
    if details.cluster:
        index, size = details.cluster
        body += _p(f"Cluster {index} · {size} concepts")
    else:
        body += _p("Not clustered.", extra=f"color:{theme.text_muted};")
    return document(theme, body)


def edge_html(theme: Theme, edge: Edge, source: Node, target: Node) -> str:
    body = _p(esc(edge.relation), extra="font-size:13pt; font-weight:600;")
    body += _p("Relationship", top=3, extra=f"color:{theme.text_muted};")
    body += _p(esc(edge.id), top=2, extra=f"color:{theme.text_disabled}; font-size:8pt;")
    body += _section(theme, "From") + _p(node_link(source.id, source.name))
    body += _section(theme, "To") + _p(node_link(target.id, target.name))
    body += _section(theme, "Properties") + properties_table(theme, edge.properties)
    return document(theme, body)


# --- analysis results -------------------------------------------------------


def paths_block(theme: Theme, paths: Sequence[Any]) -> str:
    if not paths:
        return _p("No directed relationship path found.", top=4, extra=f"color:{theme.text_muted};")
    out = ""
    for index, path in enumerate(paths):
        hops = path.length
        head = (
            f"<b>Path {index + 1}</b> <span style=\"color:{theme.text_muted};\">· "
            f"{hops} {'hop' if hops == 1 else 'hops'}</span> &nbsp; "
            f'<a href="ng://path/{index}">Show on graph</a>'
        )
        lines = [node_link(path.nodes[0].id, path.nodes[0].name)]
        for node, edge, nxt in zip(path.nodes, path.edges, path.nodes[1:]):
            arrow = "↓" if edge.source_id == node.id else "↑"
            lines.append(f'<span style="color:{theme.text_muted};">&nbsp;&nbsp;{arrow} {esc(edge.relation)}</span>')
            lines.append(node_link(nxt.id, nxt.name))
        out += _p(head, top=10, bottom=2) + _p("<br>".join(lines), bottom=2)
    return out


def matches_block(theme: Theme, matches: Sequence[Any]) -> str:
    if not matches:
        return _p("No concepts in the graph.", top=4, extra=f"color:{theme.text_muted};")
    rows = "".join(
        f'<tr><td style="color:{theme.text_muted}; padding:2px 12px 2px 0px;">{m.score:.3f}</td>'
        f"<td>{node_link(m.node.id, m.node.name)}</td></tr>"
        for m in matches
    )
    return _p("<b>Semantic matches</b>", top=10, bottom=4) + f'<table cellspacing="0" cellpadding="0">{rows}</table>'


def clusters_block(theme: Theme, clusters: Sequence[Any]) -> str:
    if not clusters:
        return _p("The graph has no concepts.", top=4, extra=f"color:{theme.text_muted};")
    out = ""
    for index, cluster in enumerate(clusters[:MAX_ROWS]):
        names = ", ".join(node_link(n.id, n.name) for n in cluster.nodes[:40])
        if len(cluster.nodes) > 40:
            names += f", … and {len(cluster.nodes) - 40} more"
        stats = (
            f"{len(cluster.nodes)} concepts · {cluster.internal_edge_count} relationships · "
            f"density {cluster.density:.2f}"
        )
        head = (
            f"<b>Cluster {index + 1}</b> <span style=\"color:{theme.text_muted};\">· {stats}</span>"
            f' &nbsp; <a href="ng://cluster/{index}">Highlight</a>'
        )
        out += _p(head, top=10, bottom=2) + _p(names, bottom=2)
    if len(clusters) > MAX_ROWS:
        out += _p(f"… and {len(clusters) - MAX_ROWS} more clusters", extra=f"color:{theme.text_muted};")
    return out


def inferences_block(theme: Theme, suggestions: Sequence[Any], names: Mapping[str, str]) -> str:
    if not suggestions:
        return _p("No transitive suggestions found.", top=4, extra=f"color:{theme.text_muted};")
    out = _p(
        "<b>Suggested links</b> "
        f'<span style="color:{theme.text_muted};">· unconfirmed, review before adding</span>',
        top=10,
        bottom=2,
    )
    for item in suggestions[:MAX_ROWS]:
        chain = []
        for edge in item.evidence:
            chain.append(f"{esc(names.get(edge.source_id, edge.source_id))} –{esc(edge.relation)}→")
        chain.append(esc(names.get(item.evidence[-1].target_id, item.evidence[-1].target_id)))
        out += _p(
            f"{node_link(item.source.id, item.source.name)} "
            f'<span style="color:{theme.text_muted};">{esc(item.relation)} →</span> '
            f"{node_link(item.target.id, item.target.name)}",
            top=8,
        )
        out += _p(" ".join(chain), extra=f"color:{theme.text_muted}; font-size:8pt;")
    if len(suggestions) > MAX_ROWS:
        out += _p(f"… and {len(suggestions) - MAX_ROWS} more", extra=f"color:{theme.text_muted};")
    return out


def text_block(theme: Theme, title: str, text: str) -> str:
    return _p(f"<b>{esc(title)}</b>", top=12, bottom=2) + _p(esc(text))


def results_document(theme: Theme, blocks: Iterable[str]) -> str:
    joined = "".join(blocks)
    if not joined:
        joined = message(theme, "No results yet", "Run an analysis above to see results here.")
        return joined
    return document(theme, joined)
