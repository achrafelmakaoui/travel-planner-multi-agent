"""
Simple smoke test: verifies that the graph compiles without errors.
Run: python -m tests.test_graph
"""
from graph import build_graph


def test_graph_compiles():
    graph = build_graph()
    assert graph is not None
    print("OK: graph compiled successfully.")


if __name__ == "__main__":
    test_graph_compiles()
