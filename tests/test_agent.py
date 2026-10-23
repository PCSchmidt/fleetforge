import random

import pytest

from fleetforge import agent as AG
from fleetforge import sweep as S
from fleetforge.agent import Agent, Bricked
from fleetforge.artifact import make_bundle
from fleetforge.link import Server
from fleetforge.simfs import PATTERNS, PowerCut, SimFS

W = S.make_world()


def fresh():
    return S.clone(W.base)


def agent(fs):
    return Agent(fs, W.root_pub, S.TARGET)


def signed(seq, target=S.TARGET, seed=77, signer="signer", created=S.NOW - 10, not_after=S.NOW + 10**5):
    return make_bundle(S._key(signer), "k1", S.NAME, seq, f"v{seq}", target, S._blob(seed, len(W.b2.model)), b"card", b"{}", created, not_after)


# ---------------------------------------------------------------------------------------------- the simulated store
def test_simfs_unsynced_writes_are_lost_synced_ones_survive():
    fs = SimFS()
    fs.write("a", 0, b"hello")
    fs.write("b", 0, b"world")
    fs.sync("b")
    after = fs.crash("lose")
    assert after.read("b") == b"world" and after.read("a") == b""
    assert fs.crash("keep").read("a") == b"hello"


def test_simfs_rename_before_sync_can_leave_an_empty_file():
    fs = SimFS()
    fs.write("tmp", 0, b"data")
    fs.rename("tmp", "final")  # no sync first
    assert fs.read("final") == b"data"
    assert fs.crash("lose").read("final") == b""
    fs2 = SimFS()
    fs2.write("tmp", 0, b"data")
    fs2.sync("tmp")
    fs2.rename("tmp", "final")
    assert fs2.crash("lose").read("final") == b"data"


def test_simfs_a_cut_on_a_write_is_torn_and_a_cut_on_other_ops_is_not_applied():
    fs = SimFS()
    fs.cut_at = 1
    with pytest.raises(PowerCut):
        fs.write("a", 0, b"xyz")
    assert fs.crash("keep").read("a") == b"xyz" and fs.crash("lose").read("a") == b""
    fs = SimFS()
    fs.write("a", 0, b"1")
    fs.sync("a")
    fs.cut_at = fs.ops + 1
    with pytest.raises(PowerCut):
        fs.delete("a")
    assert fs.read("a") == b"1"  # the delete did not happen


def test_simfs_patterns_half_and_alt():
    fs = SimFS()
    for i in range(4):
        fs.write("f", i, bytes([65 + i]))
    assert fs.crash("half").read("f") == b"AB"
    assert fs.crash("alt").read("f") == b"A\x00C"
    assert len(PATTERNS) == 4


# ---------------------------------------------------------------------------------------------- normal flow
def test_a_good_update_is_staged_then_promoted_on_confirm():
    fs = fresh()
    a = agent(fs)
    assert a.boot() == "A"
    assert a.update(W.server(), S.NOW) == "pending"
    st = a._load()
    assert st["pending"]["slot"] == "B" and st["floor"]["detector"] == 1  # the floor has not moved yet
    a = agent(fs)
    assert a.boot() == "B"
    assert a.confirm(True) == "promoted"
    st = a._load()
    assert st["active"] == "B" and st["previous"] == "A" and st["floor"]["detector"] == 2 and st["pending"] is None
    assert agent(fs).update(W.server(), S.NOW) == "up_to_date"


def test_the_running_slot_is_never_written_during_an_update():
    fs = fresh()
    before = {p: bytes(fs.dur[p]) for p in fs.dur if p.startswith("slots/A/")}
    agent(fs).update(W.server(), S.NOW)
    assert {p: bytes(fs.dur[p]) for p in fs.dur if p.startswith("slots/A/")} == before
    assert not any(h.split(" ")[1].startswith("slots/A") for h in fs.history if h.split(" ")[0] in ("write", "delete", "truncate", "sync") or h.startswith("rename slots/A"))


def test_an_unhealthy_model_is_rolled_back_remembered_and_not_retried():
    fs = fresh()
    agent(fs).update(W.server(), S.NOW)
    a = agent(fs)
    a.boot()
    assert a.confirm(False) == "rolled_back"
    st = a._load()
    assert st["active"] == "A" and st["floor"]["detector"] == 1 and W.h2 in st["bad"]
    server = W.server()
    assert agent(fs).update(server, S.NOW) == "rejected:bad_model"
    assert server.bytes_sent < 2000  # only the small manifest and signature were fetched


def test_a_crash_looping_model_is_dropped_after_max_tries():
    fs = fresh()
    agent(fs).update(W.server(), S.NOW)
    runs = [agent(fs).boot() for _ in range(AG.MAX_TRIES)]
    assert runs == ["B"] * AG.MAX_TRIES
    assert agent(fs).boot() == "A"
    st = agent(fs)._load()
    assert st["pending"] is None and W.h2 in st["bad"]


def test_unacceptable_updates_are_refused_before_the_model_is_downloaded():
    fs = fresh()
    cases = {
        "rejected:wrong_target": signed(2, target="other-board"),
        "rejected:rollback": signed(1),
        "rejected:expired": signed(3, created=S.NOW - 10**6, not_after=S.NOW - 5),
        "rejected:bad_signature": signed(3, signer="someone-else"),
    }
    for want, bundle in cases.items():
        server = Server(bundle)
        assert agent(fs).update(server, S.NOW) == want
        assert server.bytes_sent < 2000, want  # the model was never fetched


def test_a_tampered_model_is_caught_at_the_signed_chunk_index_before_any_of_it_is_fetched():
    bad = type(W.b2)(W.b2.manifest, W.b2.signature, bytes([W.b2.model[0] ^ 1]) + W.b2.model[1:], W.b2.model_card, W.b2.sbom)
    fs = fresh()
    server = Server(bad)
    assert agent(fs).update(server, S.NOW) == "rejected:chunk_index"
    assert server.bytes_sent < 3000  # manifest, signature and index only
    assert agent(fs)._load()["pending"] is None
    assert agent(fs).update(W.server(), S.NOW) == "pending"  # a good copy afterwards works


def test_a_server_with_the_right_index_but_bad_chunks_never_gets_a_model_staged():
    class Liar(Server):
        def read_chunk(self, i, compress=True):
            payload, packed = super().read_chunk(i, compress)
            if i == 2:
                payload = payload[:-1] + bytes([payload[-1] ^ 1])  # always damaged, so retrying cannot help
            return payload, packed

    fs = fresh()
    a = agent(fs)
    assert a.update(Liar(W.b2), S.NOW) == "link_down"
    assert a.stats["bad_chunks"] > 0 and agent(fs)._load()["pending"] is None
    assert agent(fs).update(W.server(), S.NOW) == "pending"  # resumes from the good prefix with an honest server


def test_a_new_server_bundle_mid_download_restarts_cleanly():
    fs = fresh()
    fs.cut_at = 25
    with pytest.raises(PowerCut):
        agent(fs).update(W.server(), S.NOW)
    fs = fs.crash("keep")
    assert agent(fs).update(Server(signed(3)), S.NOW) == "pending"
    a = agent(fs)
    a.boot()
    a.confirm(True)
    assert a._load()["floor"]["detector"] == 3


# ---------------------------------------------------------------------------------------------- the link
def test_resume_does_not_refetch_the_good_prefix():
    fs = fresh()
    fs.cut_at = 30
    s1 = W.server()
    with pytest.raises(PowerCut):
        agent(fs).update(s1, S.NOW)
    fs = fs.crash("lose")
    s2 = W.server()
    assert agent(fs).update(s2, S.NOW) == "pending"
    assert s2.bytes_sent < len(W.b2.model)  # less than a full download
    assert s1.bytes_sent + s2.bytes_sent < 1.5 * len(W.b2.model)


def test_a_torn_tail_is_detected_and_repaired_on_resume():
    fs = fresh()
    fs.cut_at = 28  # inside a chunk write
    with pytest.raises(PowerCut):
        agent(fs).update(W.server(), S.NOW)
    fs = fs.crash("keep")  # the torn chunk's bytes survive without its journal line
    assert agent(fs).update(W.server(), S.NOW) == "pending"


@pytest.mark.parametrize("seed", range(40))
def test_flaky_links_end_in_success_or_a_clean_retryable_state(seed):
    fs = fresh()
    server = Server(W.b2, random.Random(seed), p_drop=0.3, p_short=0.3, p_corrupt=0.03)
    for _ in range(80):
        res = agent(fs).update(server, S.NOW)
        # a byte flipped in the small manifest, signature or chunk index in transit is refused and retried, never blacklisted
        assert res in ("pending", "link_down", "rejected:manifest_malformed", "rejected:bad_signature", "rejected:chunk_index") or res.startswith("verify_failed:"), res
        assert agent(fs)._load()["bad"] == []
        assert not S.invariants(fs, W)
        if res == "pending":
            break
    else:
        pytest.fail("no success in 80 attempts")
    a = agent(fs)
    a.boot()
    a.confirm(True)
    assert a._load()["floor"]["detector"] == 2


def test_resuming_saves_bandwidth_on_a_lossy_link():
    def total(resume):
        sent = 0
        for seed in range(15):
            fs = fresh()
            server = Server(W.b2, random.Random(seed), p_drop=0.5)
            res = ""
            for _ in range(300):
                a = agent(fs)
                a.resume = resume
                res = a.update(server, S.NOW)
                if res == "pending":
                    break
            assert res == "pending"
            sent += server.bytes_sent
        return sent

    assert total(True) < total(False)


# ---------------------------------------------------------------------------------------------- power cuts
@pytest.mark.parametrize("mode", ["good", "bad", "crashloop"])
def test_every_single_power_cut_leaves_a_valid_model_and_the_update_completes(mode):
    r = S.single_cut_sweep(W, mode)
    assert r.runs >= 4 * r.ops and not r.failures, r.failures[:3]


def test_double_cuts_during_recovery_also_leave_a_valid_model():
    r = S.double_cut_sweep(W, "good", patterns=("lose",))
    assert r.runs > 1000 and not r.failures, r.failures[:3]


def test_the_naive_in_place_updater_is_bricked_by_most_cut_points():
    r = S.naive_sweep(W)
    assert len(r.failures) > 0.5 * r.runs


def test_repeated_power_loss_during_a_trial_drops_even_a_good_model_but_never_the_device():
    fs = fresh()
    agent(fs).update(W.server(), S.NOW)
    for _ in range(AG.MAX_TRIES + 1):  # the device keeps losing power before the supervisor can confirm
        agent(fs).boot()
    st = agent(fs)._load()
    assert st["pending"] is None and W.h2 in st["bad"] and st["active"] == "A"
    assert agent(fs).boot() == "A"
    assert agent(fs).update(Server(signed(3, seed=99)), S.NOW) == "pending"  # a corrected model with a new seq is accepted


def test_a_corrupted_active_model_is_reported_not_run():
    fs = fresh()
    fs.dur["slots/A/model"][5] ^= 1
    with pytest.raises(Bricked):
        agent(fs).boot()


def test_a_second_update_never_touches_a_pending_trial_slot():
    fs = fresh()
    agent(fs).update(W.server(), S.NOW)
    pending_files = {p: bytes(v) for p, v in fs.dur.items() if p.startswith("slots/B/")}
    assert agent(fs).update(Server(signed(3)), S.NOW) == "pending"
    assert {p: bytes(v) for p, v in fs.dur.items() if p.startswith("slots/B/")} == pending_files


def test_a_slot_that_is_damaged_while_pending_is_dropped_at_boot_and_the_device_stays_on_the_old_model():
    fs = fresh()
    agent(fs).update(W.server(), S.NOW)
    fs.dur["slots/B/model"][7] ^= 1
    assert agent(fs).boot() == "A"
    st = agent(fs)._load()
    assert st["pending"] is None and st["bad"] == []  # damage to the slot is not blamed on the model


def test_bit_rot_in_the_partial_download_is_found_by_the_journal_and_only_that_part_is_refetched():
    fs = fresh()
    fs.cut_at = 32
    s1 = W.server()
    with pytest.raises(PowerCut):
        agent(fs).update(s1, S.NOW)
    fs = fs.crash("keep")
    part = fs.dur["slots/B/model.part"]
    assert len(part) >= 2 * AG.CHUNK
    part[AG.CHUNK + 3] ^= 1  # damage the second chunk after it was journalled
    s2 = W.server()
    assert agent(fs).update(s2, S.NOW) == "pending"  # first call, no full restart needed
    assert s2.bytes_sent < len(W.b2.model)


# ---------------------------------------------------------------------------------------------- chunks: reuse, compression, damage
def test_only_the_chunks_that_changed_are_fetched():
    w = S.make_world(8 * AG.CHUNK, reuse=0.75)  # the first 6 of 8 chunks are the running model's
    fs = S.clone(w.base)
    server = w.server()
    a = Agent(fs, w.root_pub, S.TARGET)
    assert a.update(server, S.NOW) == "pending"
    assert a.stats == {"fetched": 2, "reused": 6, "bad_chunks": 0}
    assert server.bytes_sent < 2.5 * AG.CHUNK + 3000  # two chunks (random bytes do not compress) plus the small parts
    b = Agent(fs, w.root_pub, S.TARGET)
    assert b.boot() == "B"
    b.confirm(True)
    assert b._load()["floor"]["detector"] == 2


def test_a_compressible_model_costs_less_than_its_size_on_the_wire():
    zeros = make_bundle(S._key("signer"), "k1", S.NAME, 3, "v3", S.TARGET, (b"weights" * 600)[: 8 * AG.CHUNK], b"card", b"{}", S.NOW - 10, S.NOW + 10**5)
    fs = fresh()
    server = Server(zeros)
    assert agent(fs).update(server, S.NOW) == "pending"
    assert server.bytes_sent < 0.2 * len(zeros.model) + 3000
    assert agent(fs).audit()["state"]["pending"]["seq"] == 3


def test_incompressible_chunks_are_sent_raw_not_inflated():
    s = W.server()
    payload, packed = s.read_chunk(0)
    assert not packed and len(payload) == AG.CHUNK


def test_a_decompression_bomb_is_refused_without_expanding():
    import lzma

    bomb = lzma.compress(bytes(50_000_000), preset=9)
    assert len(bomb) < 100_000
    assert AG._decode(bomb, True, AG.CHUNK) is None  # claims to be one chunk, would expand to 50 MB
    good = lzma.compress(b"x" * AG.CHUNK)
    assert AG._decode(good, True, AG.CHUNK) == b"x" * AG.CHUNK
    assert AG._decode(good[:-3], True, AG.CHUNK) is None  # truncated stream
    assert AG._decode(good, True, AG.CHUNK - 1) is None  # wrong length for this chunk
    assert AG._decode(b"abc", False, 4) is None and AG._decode(b"abcd", False, 4) == b"abcd"


def test_one_damaged_chunk_costs_one_chunk_not_the_whole_download():
    class OnceBad(Server):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            self.hit = False

        def read_chunk(self, i, compress=True):
            payload, packed = super().read_chunk(i, compress)
            if i == 3 and not self.hit:
                self.hit = True
                payload = bytes([payload[0] ^ 1]) + payload[1:]
            return payload, packed

    fs = fresh()
    server = OnceBad(W.b2)
    a = agent(fs)
    assert a.update(server, S.NOW) == "pending"  # first call, no restart
    n = -(-len(W.b2.model) // AG.CHUNK)
    assert a.stats["bad_chunks"] == 1 and a.stats["fetched"] == n
    assert server.bytes_sent < len(W.b2.model) + 2 * AG.CHUNK + 3000  # one chunk re-sent, not the model


def test_power_cuts_with_chunk_reuse_still_leave_a_valid_model():
    w = S.make_world(reuse=0.5)
    r = S.single_cut_sweep(w, "good")
    assert r.runs >= 4 * r.ops and not r.failures, r.failures[:3]


def test_a_long_download_on_a_flaky_link_is_not_cut_short_by_a_total_failure_budget():
    w = S.make_world(200 * AG.CHUNK)  # 200 chunks, so a 20% drop rate means about 40 failed requests in all
    fs = S.clone(w.base)
    server = Server(w.b2, random.Random(5), p_drop=0.2)
    calls = 0
    while True:
        calls += 1
        res = Agent(fs, w.root_pub, S.TARGET).update(server, S.NOW)
        if res == "pending" or calls > 10:
            break
    assert res == "pending" and calls <= 2  # six *consecutive* failures are unlikely at a 20% drop rate
