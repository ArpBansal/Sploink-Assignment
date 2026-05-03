from __future__ import annotations

import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from solution import load_sessions, classify, _content_hash, featurize


def _ev(sid, step, action, target="x", status="success", ts=None,
        inp=None, out=None, latency=100.0):
    if ts is None:
        ts = float(step)
    if inp is None:
        inp = f"in::{step}"
    if out is None:
        out = f"out::{step}::{status}"
    meta = {"status": status, "latency_ms": latency}
    if target is not None:
        meta["target"] = target
    return {
        "session_id": sid,
        "timestamp": ts,
        "step": step,
        "action": action,
        "input": inp,
        "output": out,
        "metadata": meta,
    }


def _write_jsonl(events):
    fd, path = tempfile.mkstemp(suffix=".jsonl")
    with os.fdopen(fd, "w") as f:
        for e in events:
            f.write(json.dumps(e) + "\n")
    return path


def _make_progressing(sid, n=40):
    import random as _r
    rng = _r.Random(0)
    events = []
    visited = []
    for i in range(n):
        if visited and rng.random() < 0.35:
            target = rng.choice(visited)
        else:
            target = f"src/m_{i % 8}.py"
            visited.append(target)
        action = "write_file" if i >= n // 2 else "read_file"
        status = "success" if i >= 5 or (i % 3 != 0) else "failure"
        out = f"output::{action}::ok={status=='success'}"
        events.append(_ev(sid, i, action, target=target, status=status, out=out))
    return events


def _make_looping(sid, n=40):
    events = []
    body = [("read_file", "src/loop_0.py"),
            ("llm_call", "src/loop_0.py"),
            ("write_file", "src/loop_1.py"),
            ("test_run", "src/loop_2.py")]
    canned = ["state=A", "state=B", "state=A", "state=B"]
    for i in range(n):
        action, target = body[i % len(body)]
        out = canned[i % len(canned)]
        events.append(_ev(sid, i, action, target=target, status="success", out=out))
    return events


def _make_failing(sid, n=40):
    events = []
    bad_target = "build/target_0.o"
    for i in range(n):
        action = "retry" if i % 2 == 0 else "test_run"
        status = "failure" if i >= 5 else "success"
        events.append(_ev(sid, i, action, target=bad_target, status=status,
                          out=f"err=ExitCode{i}"))
    return events


def _ok(name):
    print(f"  PASS  {name}")


def _fail(name, msg):
    print(f"  FAIL  {name}  --  {msg}")
    raise AssertionError(f"{name}: {msg}")


def test_baseline_unmodified():
    name = "baseline (no-noise progressing session classified as progressing)"
    sid = "test_baseline"
    events = _make_progressing(sid, 40)
    path = _write_jsonl(events)
    try:
        sessions, dropped = load_sessions(path)
        assert dropped == 0, f"unexpected dedup count: {dropped}"
        label, _ = classify(sessions[sid])
        if label != "progressing":
            _fail(name, f"expected progressing, got {label}")
        _ok(name)
    finally:
        os.unlink(path)


def test_duplicate_events():
    name = "duplicate events: deduped, label unchanged"
    sid = "test_dup"
    events = _make_failing(sid, 40)
    duped = events + [dict(e) for e in events]
    path_clean = _write_jsonl(events)
    path_dup = _write_jsonl(duped)
    try:
        s_clean, dropped_clean = load_sessions(path_clean)
        s_dup, dropped_dup = load_sessions(path_dup)
        if dropped_dup != len(events):
            _fail(name, f"expected {len(events)} duplicates dropped, got {dropped_dup}")
        if len(s_dup[sid]) != len(s_clean[sid]):
            _fail(name, f"event count mismatch: clean={len(s_clean[sid])} dup={len(s_dup[sid])}")
        label_clean, _ = classify(s_clean[sid])
        label_dup, _ = classify(s_dup[sid])
        if label_clean != label_dup:
            _fail(name, f"label changed under duplication: {label_clean} -> {label_dup}")
        _ok(name)
    finally:
        os.unlink(path_clean)
        os.unlink(path_dup)


def test_step_collisions_preserved():
    name = "step collisions: siblings preserved, not deduped"
    sid = "test_collide"
    events = _make_looping(sid, 40)
    siblings = []
    for e in events[:10]:
        twin = dict(e)
        twin["timestamp"] = e["timestamp"] + 0.001
        twin["output"] = e["output"] + "::sibling"
        siblings.append(twin)
    combined = events + siblings
    path = _write_jsonl(combined)
    try:
        sessions, dropped = load_sessions(path)
        if dropped != 0:
            _fail(name, f"siblings were incorrectly deduped: dropped={dropped}")
        if len(sessions[sid]) != len(combined):
            _fail(name, f"event count {len(sessions[sid])} != input {len(combined)}")
        label, _ = classify(sessions[sid])
        if label != "looping":
            _fail(name, f"expected looping, got {label}")
        _ok(name)
    finally:
        os.unlink(path)


def test_late_events_via_hybrid_sort():
    name = "late events: timestamp inversion, hybrid sort recovers correct order"
    sid = "test_late"
    events = _make_progressing(sid, 40)
    for i in range(0, len(events), 5):
        events[i]["timestamp"] -= 100.0
    path = _write_jsonl(events)
    try:
        sessions, _ = load_sessions(path)
        steps = [e["step"] for e in sessions[sid]]
        if steps != sorted(steps):
            _fail(name, f"events not sorted by step after load: {steps[:10]}...")
        label, _ = classify(sessions[sid])
        if label != "progressing":
            _fail(name, f"expected progressing, got {label}")
        _ok(name)
    finally:
        os.unlink(path)


def test_missing_metadata():
    name = "missing metadata: latency / target / status missing on subset, no crash"
    sid = "test_missing"
    events = _make_failing(sid, 40)
    for i, e in enumerate(events):
        if i % 4 == 0:
            e["metadata"].pop("latency_ms", None)
        if i % 4 == 1:
            e["metadata"].pop("target", None)
        if i % 4 == 2:
            e["metadata"].pop("status", None)
    path = _write_jsonl(events)
    try:
        sessions, _ = load_sessions(path)
        f = featurize(sessions[sid])
        if not (0.0 <= f.success_rate_overall <= 1.0):
            _fail(name, f"success_rate out of range: {f.success_rate_overall}")
        label, _ = classify(sessions[sid])
        if label not in ("failing", "looping"):
            _fail(name, f"expected failing or looping, got {label}")
        _ok(name)
    finally:
        os.unlink(path)


def test_interleaved_sessions():
    name = "interleaved sessions: globally shuffled file, each grouped and labeled"
    sid_a = "test_inter_a"
    sid_b = "test_inter_b"
    sid_c = "test_inter_c"
    events_a = _make_progressing(sid_a, 30)
    events_b = _make_looping(sid_b, 30)
    events_c = _make_failing(sid_c, 30)
    combined = []
    i = 0
    while i < max(len(events_a), len(events_b), len(events_c)):
        for src in (events_a, events_b, events_c):
            if i < len(src):
                combined.append(src[i])
        i += 1
    path = _write_jsonl(combined)
    try:
        sessions, _ = load_sessions(path)
        if set(sessions.keys()) != {sid_a, sid_b, sid_c}:
            _fail(name, f"session grouping wrong: {set(sessions.keys())}")
        labels = {sid: classify(evs)[0] for sid, evs in sessions.items()}
        expected = {sid_a: "progressing", sid_b: "looping", sid_c: "failing"}
        for sid, exp in expected.items():
            if labels[sid] != exp:
                _fail(name, f"{sid} expected {exp} got {labels[sid]}")
        _ok(name)
    finally:
        os.unlink(path)


def test_burst_of_events():
    name = "burst: 100+ events sharing a near-identical timestamp"
    sid = "test_burst"
    events = []
    for i in range(120):
        events.append(_ev(sid, i, "run_command", target="src/x.py",
                          status="success", ts=1700.0 + i * 1e-5))
    path = _write_jsonl(events)
    try:
        sessions, _ = load_sessions(path)
        if len(sessions[sid]) != 120:
            _fail(name, f"expected 120 events, got {len(sessions[sid])}")
        steps = [e["step"] for e in sessions[sid]]
        if steps != sorted(steps):
            _fail(name, "burst events not in step order after load")
        label, _ = classify(sessions[sid])
        if label not in CLASSES_ALL:
            _fail(name, f"label not in known classes: {label}")
        _ok(name)
    finally:
        os.unlink(path)


def test_silent_stall_detected():
    name = "silent stall: long gap mid-session, has_silent_stall fires"
    sid = "test_stall"
    events = _make_progressing(sid, 40)
    for i in range(20, len(events)):
        events[i]["timestamp"] += 60.0
    path = _write_jsonl(events)
    try:
        sessions, _ = load_sessions(path)
        f = featurize(sessions[sid])
        if not f.has_silent_stall:
            _fail(name, f"has_silent_stall=False; max_gap={f.max_gap_seconds:.2f}")
        if f.max_gap_seconds < 25.0:
            _fail(name, f"max_gap_seconds {f.max_gap_seconds:.2f} below threshold")
        _ok(name)
    finally:
        os.unlink(path)


def test_late_event_inside_burst():
    name = "combined: burst + step-collisions + duplicates in one session"
    sid = "test_combined"
    base = _make_looping(sid, 60)
    for ev in base[20:35]:
        ev["timestamp"] = 1000.0
    siblings = []
    for ev in base[:8]:
        sib = dict(ev)
        sib["timestamp"] += 1e-4
        sib["output"] += "::sibling"
        siblings.append(sib)
    duped = [dict(ev) for ev in base[:5]]
    combined = base + siblings + duped
    path = _write_jsonl(combined)
    try:
        sessions, dropped = load_sessions(path)
        if dropped != 5:
            _fail(name, f"expected 5 duplicates dropped, got {dropped}")
        expected_count = len(base) + len(siblings)
        if len(sessions[sid]) != expected_count:
            _fail(name, f"expected {expected_count} unique events, got {len(sessions[sid])}")
        label, _ = classify(sessions[sid])
        if label != "looping":
            _fail(name, f"expected looping, got {label}")
        _ok(name)
    finally:
        os.unlink(path)


CLASSES_ALL = ["progressing", "looping", "drifting", "failing"]


TESTS = [
    test_baseline_unmodified,
    test_duplicate_events,
    test_step_collisions_preserved,
    test_late_events_via_hybrid_sort,
    test_missing_metadata,
    test_interleaved_sessions,
    test_burst_of_events,
    test_silent_stall_detected,
    test_late_event_inside_burst,
]


def main():
    print("Running edge-case demonstration tests:")
    print("-" * 60)
    failures = 0
    for t in TESTS:
        try:
            t()
        except AssertionError:
            failures += 1
        except Exception as e:
            print(f"  ERROR {t.__name__}: {e}")
            failures += 1
    print("-" * 60)
    print(f"Total: {len(TESTS)}    Failures: {failures}")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
