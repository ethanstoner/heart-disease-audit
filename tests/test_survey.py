import pandas as pd

from heart_audit.survey import SEED, modal_pipeline, screening_order


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


def test_modal_pipeline_mode_and_first_in_screening_order_tiebreak():
    coding = pd.DataFrame({
        "eval_design": ["cv", "single_split", "single_split", "cv"],   # tie: cv screened first
        "test_size": [0.3, 0.2, 0.2, None],
        "models": ["random_forest;knn", "random_forest", "random_forest;knn", "svm"],
    })
    modal, ties = modal_pipeline(coding)
    assert modal["eval_design"] == "cv"
    assert modal["test_size"] == 0.2
    assert ties == ["eval_design"]
    assert modal["models"] == ["knn", "random_forest"]


def test_models_fall_back_to_top_four_with_ties():
    coding = pd.DataFrame({"models": ["a;b", "c;d", "e", "a;c;e", "f"]})
    modal, _ = modal_pipeline(coding)
    # a, c, e appear twice; b, d, f once -> fewer than 2 reach 50%; top 4 incl. ties at 4th
    assert modal["models"] == ["a", "b", "c", "d", "e", "f"]
