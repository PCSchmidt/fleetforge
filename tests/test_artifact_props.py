from hypothesis import HealthCheck, given, settings, strategies as st

from fleetforge import artifact as A
from test_artifact import CARD, MODEL, NOW, SBOM, SIGNER, TARGET, bundle, device

S = settings(max_examples=300, deadline=None, suppress_health_check=list(HealthCheck))


@given(st.lists(st.tuples(st.integers(1, 50), st.binary(max_size=64), st.sampled_from(["detector", "tracker"])), max_size=25))
@S
def test_the_floor_only_rises_and_nothing_below_it_is_ever_accepted(ops):
    d = device()
    for seq, extra, name in ops:
        b = bundle(seq, model=MODEL + extra, name=name)
        before = d.floor.get(name, 0)
        r = d.verify(b, NOW)
        if r.ok:
            assert seq >= before
            d.commit(b)
        else:
            assert r.code == "rollback" and seq <= before
        assert d.floor.get(name, 0) >= before


@given(st.binary(max_size=2000), st.binary(max_size=100), st.binary(max_size=500), st.binary(max_size=100), st.binary(max_size=100))
@S
def test_arbitrary_bytes_never_crash_and_are_never_accepted(m, sig, model, card, sbom):
    assert not device().verify(A.Bundle(m, sig, model, card, sbom), NOW).ok


@given(
    st.text("abcdefghijklmnopqrstuvwxyz0123456789", min_size=1, max_size=20),
    st.integers(1, 10**9),
    st.integers(0, 10**6),
    st.integers(1, 10**6),
    st.binary(max_size=300),
)
@S
def test_any_well_formed_artifact_signed_by_a_trusted_key_is_accepted(name, seq, min_agent_slack, window, model):
    d = device(agent=min_agent_slack + 1)
    b = A.make_bundle(SIGNER, "k1", name, seq, "v", TARGET, model, CARD, SBOM, NOW - 10, NOW - 10 + window + 20, min_agent_slack)
    assert d.verify(b, NOW).ok
