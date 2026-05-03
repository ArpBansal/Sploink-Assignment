from __future__ import annotations

import argparse
import json
import math
import os
import random
from dataclasses import dataclass
from typing import Callable

ACTIONS = [
    "read_file", "write_file", "run_command",
    "llm_call", "test_run", "retry", "branch",
]

PROGRESSING_TARGETS = [f"src/module_{i}.py" for i in range(20)]
LOOP_TARGETS = [f"src/loop_{i}.py" for i in range(3)]
DRIFT_TARGETS = [f"area_{i}/file.py" for i in range(40)]
FAIL_TARGETS = [f"build/target_{i}.o" for i in range(5)]

CLASSES = ["progressing", "looping", "drifting", "failing"]


@dataclass
class PhaseConfig:
    label: str
    n_steps: int
    base_ts: float
    seed: int
    p_missing_meta: float = 0.05


def _meta(status, latency, target, rng, p_missing):
    m = {"status": status, "latency_ms": round(latency, 2)}
    if target is not None:
        m["target"] = target
    if rng.random() < p_missing:
        candidates = [k for k in m.keys() if k != "status"]
        if candidates:
            m.pop(rng.choice(candidates), None)
    return m


def _event(sid, ts, step, action, inp, out, meta):
    return {
        "session_id": sid,
        "timestamp": ts,
        "step": step,
        "action": action,
        "input": inp,
        "output": out,
        "metadata": meta,
    }


def gen_progressing_phase(sid, cfg, start_step):
    rng = random.Random(cfg.seed)
    events = []
    ts = cfg.base_ts
    visited = []
    for i in range(cfg.n_steps):
        step = start_step + i
        progress_frac = i / max(1, cfg.n_steps - 1)
        p_success = 0.6 + 0.37 * progress_frac
        status = "success" if rng.random() < p_success else "failure"
        if visited and rng.random() < 0.2:
            target = rng.choice(visited)
        else:
            target = rng.choice(PROGRESSING_TARGETS)
            visited.append(target)
        if progress_frac < 0.4:
            action = rng.choices(
                ["read_file", "llm_call", "branch", "run_command"],
                weights=[4, 4, 1, 2],
            )[0]
        else:
            action = rng.choices(
                ["write_file", "test_run", "run_command", "llm_call"],
                weights=[4, 4, 2, 2],
            )[0]
        latency = rng.gauss(180, 40) if action != "llm_call" else rng.gauss(900, 200)
        latency = max(10.0, latency)
        events.append(_event(
            sid, ts, step, action,
            f"input::{action}::{step}",
            f"output::{action}::ok={status=='success'}",
            _meta(status, latency, target, rng, cfg.p_missing_meta),
        ))
        ts += rng.uniform(0.4, 1.6)
    return events, ts


def gen_looping_phase(sid, cfg, start_step):
    rng = random.Random(cfg.seed)
    events = []
    ts = cfg.base_ts
    body = [
        ("read_file", LOOP_TARGETS[0]),
        ("llm_call", LOOP_TARGETS[0]),
        ("write_file", LOOP_TARGETS[1]),
        ("test_run", LOOP_TARGETS[2]),
    ]
    canned = ["state=A", "state=B", "state=A", "state=B"]
    for i in range(cfg.n_steps):
        step = start_step + i
        action, target = body[i % len(body)]
        status = "success" if rng.random() < 0.85 else "failure"
        latency = max(20.0, rng.gauss(220, 60))
        out = canned[i % len(canned)]
        events.append(_event(
            sid, ts, step, action,
            f"input::loop::{i % len(body)}", out,
            _meta(status, latency, target, rng, cfg.p_missing_meta),
        ))
        ts += rng.uniform(0.3, 1.2)
    return events, ts


def gen_drifting_phase(sid, cfg, start_step):
    rng = random.Random(cfg.seed)
    events = []
    ts = cfg.base_ts
    third = max(1, cfg.n_steps // 3)
    pools = [
        rng.sample(DRIFT_TARGETS, 8),
        rng.sample(DRIFT_TARGETS, 8),
        rng.sample(DRIFT_TARGETS, 8),
    ]
    for i in range(cfg.n_steps):
        step = start_step + i
        phase_idx = min(2, i // third)
        target = rng.choice(pools[phase_idx])
        action = rng.choice(ACTIONS)
        status = "success" if rng.random() < 0.7 else "failure"
        latency = max(20.0, rng.gauss(260, 80))
        out = f"goal=phase_{phase_idx}::work={rng.randint(0, 9999)}"
        events.append(_event(
            sid, ts, step, action,
            f"input::drift::phase{phase_idx}::{step}", out,
            _meta(status, latency, target, rng, cfg.p_missing_meta),
        ))
        ts += rng.uniform(0.4, 1.5)
    return events, ts


def gen_failing_phase(sid, cfg, start_step):
    rng = random.Random(cfg.seed)
    events = []
    ts = cfg.base_ts
    bad_target = rng.choice(FAIL_TARGETS)
    for i in range(cfg.n_steps):
        step = start_step + i
        progress_frac = i / max(1, cfg.n_steps - 1)
        p_failure = 0.35 + 0.55 * progress_frac
        action = rng.choices(
            ["run_command", "test_run", "retry", "llm_call", "read_file"],
            weights=[4, 4, 5, 1, 1],
        )[0]
        target = bad_target if action in ("retry", "run_command", "test_run") \
            else rng.choice(FAIL_TARGETS)
        status = "failure" if rng.random() < p_failure else "success"
        latency = max(50.0, rng.gauss(800, 300))
        events.append(_event(
            sid, ts, step, action,
            f"input::fail::{step}",
            f"err=ExitCode{rng.randint(1, 127)}" if status == "failure" else "ok",
            _meta(status, latency, target, rng, cfg.p_missing_meta),
        ))
        ts += rng.uniform(0.3, 1.2)
    return events, ts


PHASE_GEN: dict[str, Callable] = {
    "progressing": gen_progressing_phase,
    "looping": gen_looping_phase,
    "drifting": gen_drifting_phase,
    "failing": gen_failing_phase,
}


def build_session(sid, structure, base_ts, seed):
    rng = random.Random(seed)
    all_events = []
    cur_step = 0
    cur_ts = base_ts
    phase_records = []
    for phase_label, n_steps in structure:
        cfg = PhaseConfig(
            label=phase_label,
            n_steps=n_steps,
            base_ts=cur_ts,
            seed=rng.randint(0, 2**31 - 1),
        )
        evs, cur_ts = PHASE_GEN[phase_label](sid, cfg, cur_step)
        all_events.extend(evs)
        phase_records.append({
            "label": phase_label,
            "start_step": cur_step,
            "end_step": cur_step + n_steps - 1,
            "n_steps": n_steps,
        })
        cur_step += n_steps
    return all_events, phase_records


def soft_label_from_phases(phase_records, total_steps):
    weights = {c: 0.0 for c in CLASSES}
    for p in phase_records:
        weights[p["label"]] += p["n_steps"]
    return {c: weights[c] / total_steps for c in CLASSES}


def hard_label_second_half(phase_records, total_steps):
    half = total_steps // 2
    weights = {c: 0.0 for c in CLASSES}
    for p in phase_records:
        overlap_lo = max(p["start_step"], half)
        overlap_hi = min(p["end_step"], total_steps - 1)
        if overlap_hi >= overlap_lo:
            weights[p["label"]] += (overlap_hi - overlap_lo + 1)
    return max(CLASSES, key=lambda c: weights[c])


def make_pure_structure(label, total_steps):
    return [(label, total_steps)]


def make_phase_transition_structure(rng, total_steps):
    direction = rng.choice(["recovery", "degradation"])
    degraded_choices = ["looping", "drifting", "failing"]
    degraded = rng.choice(degraded_choices)
    split_frac = rng.uniform(0.35, 0.65)
    first_n = max(5, int(total_steps * split_frac))
    second_n = total_steps - first_n
    if direction == "recovery":
        return [(degraded, first_n), ("progressing", second_n)]
    else:
        return [("progressing", first_n), (degraded, second_n)]


def make_locally_mixed_structure(rng, total_steps):
    primary_choices = ["progressing", "looping", "drifting", "failing"]
    primary = rng.choice(primary_choices)
    other_choices = [c for c in primary_choices if c != primary]
    intruder = rng.choice(other_choices)
    intruder_len = max(3, int(total_steps * rng.uniform(0.10, 0.20)))
    pre = max(3, int(total_steps * rng.uniform(0.25, 0.55)))
    post = total_steps - pre - intruder_len
    if post < 3:
        post = 3
        pre = max(3, total_steps - intruder_len - post)
    return [(primary, pre), (intruder, intruder_len), (primary, post)]


def inject_noise(events, rng,
                 p_duplicate=0.04,
                 p_late=0.05,
                 p_burst=0.25,
                 p_step_collision=0.03,
                 p_silent_stall=0.30,
                 force_silent_stall=False):
    out = list(events)
    dups = []
    for ev in out:
        if rng.random() < p_duplicate:
            dups.append(dict(ev))
    out.extend(dups)
    for ev in list(out):
        if rng.random() < p_step_collision:
            twin = dict(ev)
            twin["timestamp"] = ev["timestamp"] + 0.001
            twin["output"] = ev["output"] + "::sibling"
            out.append(twin)
    for ev in out:
        if rng.random() < p_late:
            ev["timestamp"] -= rng.uniform(2.0, 8.0)
    if rng.random() < p_burst and len(out) > 20:
        i = rng.randint(0, len(out) - 15)
        burst_ts = out[i]["timestamp"]
        for j in range(i, min(i + rng.randint(8, 14), len(out))):
            out[j]["timestamp"] = burst_ts + (j - i) * 1e-4
    if force_silent_stall or rng.random() < p_silent_stall:
        if len(out) > 4:
            i = rng.randint(len(out) // 2, max(len(out) // 2 + 1, len(out) - 2))
            gap = rng.uniform(30.0, 120.0)
            for j in range(i, len(out)):
                out[j]["timestamp"] += gap
    return out


def assign_session_kind(rng):
    r = rng.random()
    if r < 0.60:
        return "pure"
    if r < 0.85:
        return "phase_transition"
    return "locally_mixed"


def build_dataset(out_path, seed, per_class=100,
                  steps_min=25, steps_max=80):
    rng = random.Random(seed)
    all_events = []
    manifest = []
    counter = 0
    target_pure_per_class = int(per_class * 0.60)
    pure_counts = {c: 0 for c in CLASSES}

    sessions_to_build = []
    for c in CLASSES:
        for _ in range(target_pure_per_class):
            sessions_to_build.append(("pure", c))
    remaining = per_class * len(CLASSES) - len(sessions_to_build)
    for _ in range(remaining):
        kind = "phase_transition" if rng.random() < (0.25 / 0.40) else "locally_mixed"
        sessions_to_build.append((kind, None))

    rng.shuffle(sessions_to_build)

    for kind, hint in sessions_to_build:
        sid = f"sess_{counter:05d}"
        counter += 1
        n_steps = rng.randint(steps_min, steps_max)
        if kind == "pure":
            structure = make_pure_structure(hint, n_steps)
        elif kind == "phase_transition":
            structure = make_phase_transition_structure(rng, n_steps)
        else:
            structure = make_locally_mixed_structure(rng, n_steps)

        base_ts = rng.uniform(1_700_000_000, 1_700_500_000)
        session_seed = rng.randint(0, 2**31 - 1)
        events, phase_records = build_session(sid, structure, base_ts, session_seed)

        soft = soft_label_from_phases(phase_records, n_steps)
        hard = hard_label_second_half(phase_records, n_steps)

        events = inject_noise(events, rng)
        all_events.extend(events)
        manifest.append({
            "session_id": sid,
            "label": hard,
            "soft_label": soft,
            "kind": kind,
            "phases": phase_records,
            "n_events": len(events),
            "n_steps": n_steps,
        })

    rng.shuffle(all_events)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        for ev in all_events:
            f.write(json.dumps(ev) + "\n")

    truth_path = os.path.join(os.path.dirname(out_path) or ".", "labels.jsonl")
    with open(truth_path, "w") as f:
        for row in manifest:
            f.write(json.dumps(row) + "\n")

    print(f"wrote {len(all_events)} events across {len(manifest)} sessions to {out_path}")
    print(f"wrote ground-truth labels to {truth_path}")
    kind_counts = {}
    label_counts = {c: 0 for c in CLASSES}
    for m in manifest:
        kind_counts[m["kind"]] = kind_counts.get(m["kind"], 0) + 1
        label_counts[m["label"]] += 1
    print(f"  session kinds: {kind_counts}")
    print(f"  hard labels (second-half-dominant): {label_counts}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data/sessions.jsonl")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--per-class", type=int, default=100)
    args = p.parse_args()
    build_dataset(args.out, args.seed, args.per_class)


if __name__ == "__main__":
    main()
