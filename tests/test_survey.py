from heart_audit.survey import SEED, prefilter, screening_order


def _item(repo, path, fork=False):
    return {"repo": repo, "path": path, "fork": fork, "sha": "s", "ref": "r", "html_url": "u"}


def test_prefilter_drops_forks_and_keeps_first_path_per_repo():
    items = [
        _item("a/x", "z.ipynb"), _item("a/x", "b.ipynb"),
        _item("b/y", "n.ipynb", fork=True),
        _item("c/z", "m.ipynb"),
    ]
    kept = prefilter(items)
    assert [(i["repo"], i["path"]) for i in kept] == [("a/x", "b.ipynb"), ("c/z", "m.ipynb")]


def test_prefilter_drops_exact_duplicate_results():
    items = [_item("a/x", "b.ipynb"), _item("a/x", "b.ipynb")]
    assert len(prefilter(items)) == 1


def test_screening_order_is_deterministic_and_input_order_independent():
    items = [_item(f"r/{i:02d}", "n.ipynb") for i in range(50)]
    first = screening_order(items)
    assert screening_order(list(reversed(items))) == first
    assert [i["repo"] for i in first] != [i["repo"] for i in items]
    assert SEED == 20260922
