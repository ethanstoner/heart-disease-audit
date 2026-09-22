from heart_audit.seeds import MASTER_SEED, derive


def test_derive_is_deterministic_and_named():
    assert derive("control", 3) == derive("control", 3)
    assert derive("control", 3) != derive("bootstrap", 3)
    assert derive("control", 5)[:3] == derive("control", 3)
    assert all(0 <= s < 2**31 for s in derive("control", 1000))
    assert len(set(derive("control", 1000))) == 1000
    assert MASTER_SEED == 20260922
