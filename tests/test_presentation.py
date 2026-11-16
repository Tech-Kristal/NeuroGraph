"""Tests for the Qt-free theme, layout and rendering modules."""

import re

from neurograph.layout import calculate_layout, curve_offsets, node_radius
from neurograph.models import Edge, Node
from neurograph.render import Link, NodeDetails, node_html, node_labels, node_link
from neurograph.theme import DARK, LIGHT, build_stylesheet, category_colors


def test_stylesheets_have_no_unresolved_placeholders():
    for theme in (DARK, LIGHT):
        assert not re.search(r"\$\w+", build_stylesheet(theme))


def test_category_colors_are_stable_and_concept_is_neutral():
    colors = category_colors(["topic", "concept", "person"], DARK)
    assert colors["concept"] == DARK.node_neutral
    assert colors["person"] != colors["topic"]
    assert colors == category_colors(["person", "topic", "concept"], DARK)


def test_parallel_and_opposing_edges_are_fanned_out():
    edges = [Edge("a", "b", "x", id="1"), Edge("b", "a", "y", id="2"), Edge("a", "c", "z", id="3")]
    offsets = curve_offsets(edges)
    assert offsets["1"] != offsets["2"]
    assert offsets["3"] == 0


def test_node_radius_grows_but_is_capped():
    assert node_radius(0) < node_radius(9) < node_radius(10_000) == 26.0


def test_layout_keeps_known_positions_and_places_new_nodes_nearby():
    nodes = [Node("A", id="a"), Node("B", id="b")]
    first = calculate_layout(nodes, [Edge("a", "b", "r")])
    nodes.append(Node("C", id="c"))
    second = calculate_layout(nodes, [Edge("a", "b", "r"), Edge("c", "a", "r")], first)
    assert second["a"] == first["a"] and second["b"] == first["b"]
    ax, ay = first["a"]
    cx, cy = second["c"]
    assert ((cx - ax) ** 2 + (cy - ay) ** 2) ** 0.5 < 200


def test_layout_handles_empty_and_single_node():
    assert calculate_layout([], []) == {}
    assert calculate_layout([Node("A", id="a")], []) == {"a": (0.0, 0.0)}


def test_details_html_escapes_content_and_encodes_links():
    node = Node("<b>Bold</b>", {"type": "topic", "note": "a & b"}, id="id with/slash")
    details = NodeDetails(
        node, "topic", "#ffffff", (Link("in", "is_a", "other", "<i>x</i>", "e1"),), (), (), None
    )
    html = node_html(DARK, details)
    assert "<b>Bold</b>" not in html and "&lt;b&gt;Bold&lt;/b&gt;" in html
    assert "a &amp; b" in html
    assert "ng://node/other" in html
    assert node_link("id with/slash", "n") == '<a href="ng://node/id%20with%2Fslash">n</a>'


def test_duplicate_names_are_disambiguated_in_pickers():
    labels = node_labels([Node("Same", id="11111111-x"), Node("Same", id="22222222-x"), Node("Other", id="z")])
    assert labels["z"] == "Other"
    assert labels["11111111-x"] != labels["22222222-x"]
