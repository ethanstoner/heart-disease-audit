from heart_audit.survey import SEED, screening_order


def _item(repo, path, fork=False):
    return {"repo": repo, "path": path, "fork": fork, "sha": "s", "ref": "r", "html_url": "u"}


def test_forks_dropped_and_notebooks_grouped_in_path_order():
    items = [
        _item("a/x", "z.ipynb"), _item("a/x", "b.ipynb"), _item("a/x", "b.ipynb"),
        _item("b/y", "n.ipynb", fork=True),
        _item("c/z", "m.ipynb"),
    ]
    order = dict(screening_order(items))
    assert set(order) == {"a/x", "c/z"}
    assert [i["path"] for i in order["a/x"]] == ["b.ipynb", "z.ipynb"]


def test_repository_order_is_seeded_and_input_order_independent():
    items = [_item(f"r/{i:02d}", "n.ipynb") for i in range(50)]
    first = [repo for repo, _ in screening_order(items)]
    assert [repo for repo, _ in screening_order(list(reversed(items)))] == first
    assert first != sorted(first)
    assert SEED == 20260922
