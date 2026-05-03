from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass

CLASSES = ["progressing", "looping", "drifting", "failing"]
DEGRADATION_PRIORITY = ["failing", "looping", "drifting", "progressing"]


def _content_hash(ev):
    payload = json.dumps({
        "s": ev.get("session_id"),
        "k": ev.get("step"),
        "a": ev.get("action"),
        "i": ev.get("input"),
        "o": ev.get("output"),
    }, sort_keys=True)
    return hashlib.sha1(payload.encode()).hexdigest()


def load_sessions(path):
    by_session = defaultdict(list)
    seen_hashes = defaultdict(set)
    dropped_dups = 0
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ev = json.loads(line)
            sid = ev.get("session_id")
            if sid is None:
                continue
            h = _content_hash(ev)
            if h in seen_hashes[sid]:
                dropped_dups += 1
                continue
            seen_hashes[sid].add(h)
            by_session[sid].append(ev)
    for sid, evs in by_session.items():
        evs.sort(key=lambda e: (e.get("step", 0), e.get("timestamp", 0.0)))
    return by_session, dropped_dups


@dataclass
class Features:
    n_steps: int
    success_rate_overall: float
    success_rate_first_half: float
    success_rate_second_half: float
    monotone_success: float
    cycle_pressure: float
    unique_node_ratio: float
    output_entropy: float
    target_entropy: float
    phase_shift: float
    retry_density: float
    failure_clustering: float
    max_gap_seconds: float
    has_silent_stall: bool


def _entropy(counts):
    total = sum(counts.values())
    if total == 0:
        return 0.0
    return -sum((c / total) * math.log2(c / total) for c in counts.values() if c > 0)


def _js_divergence(p, q):
    keys = set(p) | set(q)
    if not keys:
        return 0.0
    sp = sum(p.values()) or 1
    sq = sum(q.values()) or 1
    pn = {k: p.get(k, 0) / sp for k in keys}
    qn = {k: q.get(k, 0) / sq for k in keys}
    m = {k: 0.5 * (pn[k] + qn[k]) for k in keys}

    def kl(a, b):
        s = 0.0
        for k in keys:
            if a[k] > 0 and b[k] > 0:
                s += a[k] * math.log2(a[k] / b[k])
        return s
    return 0.5 * kl(pn, m) + 0.5 * kl(qn, m)


def featurize(events):
    n = len(events)
    if n == 0:
        return Features(0, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, False)

    statuses = [(e.get("metadata") or {}).get("status") for e in events]
    successes = [s == "success" for s in statuses if s in ("success", "failure")]
    if successes:
        sr_all = sum(successes) / len(successes)
        h = len(successes) // 2 or 1
        sr_first = sum(successes[:h]) / h
        sr_second = sum(successes[h:]) / max(1, len(successes) - h)
    else:
        sr_all = sr_first = sr_second = 0.5
    monotone = sr_second - sr_first

    nodes = []
    for e in events:
        target = (e.get("metadata") or {}).get("target", "<none>")
        nodes.append((e.get("action", "?"), target))

    unique_node_ratio = len(set(nodes)) / max(1, n)

    if len(nodes) >= 3:
        triples = Counter(zip(nodes, nodes[1:], nodes[2:]))
        repeated = sum(c for c in triples.values() if c >= 2)
        cycle_pressure = repeated / max(1, sum(triples.values()))
    else:
        cycle_pressure = 0.0

    out_counter = Counter(e.get("output", "") for e in events)
    output_entropy = _entropy(out_counter)

    target_counter = Counter(t for (_, t) in nodes if t != "<none>")
    target_entropy = _entropy(target_counter)

    half = n // 2 or 1
    p_first = Counter(t for (_, t) in nodes[:half] if t != "<none>")
    p_second = Counter(t for (_, t) in nodes[half:] if t != "<none>")
    phase_shift = _js_divergence(p_first, p_second)

    retry_count = sum(1 for e in events if e.get("action") == "retry")
    retry_density = retry_count / max(1, n)

    longest = cur = 0
    for s in statuses:
        if s == "failure":
            cur += 1
            longest = max(longest, cur)
        else:
            cur = 0
    failure_clustering = longest / max(1, n)

    timestamps = [e.get("timestamp", 0.0) for e in events]
    if len(timestamps) >= 2:
        gaps = [b - a for a, b in zip(timestamps, timestamps[1:]) if b >= a]
        max_gap = max(gaps) if gaps else 0.0
    else:
        max_gap = 0.0
    has_stall = max_gap > 25.0

    return Features(
        n_steps=n,
        success_rate_overall=sr_all,
        success_rate_first_half=sr_first,
        success_rate_second_half=sr_second,
        monotone_success=monotone,
        cycle_pressure=cycle_pressure,
        unique_node_ratio=unique_node_ratio,
        output_entropy=output_entropy,
        target_entropy=target_entropy,
        phase_shift=phase_shift,
        retry_density=retry_density,
        failure_clustering=failure_clustering,
        max_gap_seconds=max_gap,
        has_silent_stall=has_stall,
    )


def score_features(f):
    scores = {c: 0.0 for c in CLASSES}

    if f.retry_density > 0.30:
        scores["failing"] += 2.0
    if f.failure_clustering > 0.10:
        scores["failing"] += 1.5
    if f.success_rate_overall < 0.45:
        scores["failing"] += 1.0
    if f.monotone_success < -0.15:
        scores["failing"] += 0.8

    if f.cycle_pressure > 0.55:
        scores["looping"] += 2.0
    if f.unique_node_ratio < 0.20:
        scores["looping"] += 1.5
    if f.output_entropy < 2.5:
        scores["looping"] += 1.0
    if f.retry_density > 0.30:
        scores["looping"] -= 1.0

    if f.output_entropy > 4.5:
        scores["drifting"] += 2.0
    if f.phase_shift > 0.40 and f.output_entropy > 4.0:
        scores["drifting"] += 1.5
    if f.target_entropy > 3.4 and abs(f.monotone_success) < 0.05:
        scores["drifting"] += 1.0

    if f.monotone_success > 0.05:
        scores["progressing"] += 1.5
    if f.success_rate_overall > 0.65 and f.retry_density < 0.10:
        scores["progressing"] += 1.0
    if f.cycle_pressure < 0.40 and f.output_entropy < 4.0 and \
       f.failure_clustering < 0.10:
        scores["progressing"] += 1.0

    return scores


def classify(events):
    full_feats = featurize(events)
    full_scores = score_features(full_feats)

    n = len(events)
    if n >= 4:
        second_half = events[n // 2:]
        sh_feats = featurize(second_half)
        sh_scores = score_features(sh_feats)
    else:
        sh_feats = full_feats
        sh_scores = full_scores

    best_sh = max(sh_scores.values()) if sh_scores else 0.0
    sh_winners = [c for c in CLASSES if sh_scores[c] == best_sh]

    if best_sh <= 0:
        full_best = max(full_scores.values()) if full_scores else 0.0
        if full_best <= 0:
            label = "progressing"
        else:
            full_winners = [c for c in CLASSES if full_scores[c] == full_best]
            label = min(full_winners, key=lambda c: DEGRADATION_PRIORITY.index(c))
    elif len(sh_winners) == 1:
        label = sh_winners[0]
    else:
        label = min(sh_winners, key=lambda c: DEGRADATION_PRIORITY.index(c))

    return label, {
        "full_scores": full_scores,
        "second_half_scores": sh_scores,
        "full_features": full_feats,
        "second_half_features": sh_feats,
    }


def evaluate(predictions, truth_path):
    if not truth_path:
        return None
    truth = {}
    with open(truth_path) as f:
        for line in f:
            row = json.loads(line)
            truth[row["session_id"]] = row

    cm = {a: {p: 0 for p in CLASSES} for a in CLASSES}
    correct = 0
    total = 0
    by_kind = defaultdict(lambda: {"correct": 0, "total": 0})
    for sid, pred in predictions:
        if sid not in truth:
            continue
        actual = truth[sid]["label"]
        kind = truth[sid].get("kind", "unknown")
        cm[actual][pred] += 1
        total += 1
        by_kind[kind]["total"] += 1
        if pred == actual:
            correct += 1
            by_kind[kind]["correct"] += 1

    print(f"\nAccuracy: {correct}/{total} = {correct / total:.3f}")
    print("\nConfusion matrix (rows = actual, cols = predicted):")
    header = "actual \\ pred  | " + " | ".join(f"{c:>11}" for c in CLASSES)
    print(header)
    print("-" * len(header))
    for a in CLASSES:
        row = " | ".join(f"{cm[a][p]:>11d}" for p in CLASSES)
        print(f"{a:<14} | {row}")

    print("\nPer-class precision / recall / F1:")
    for c in CLASSES:
        tp = cm[c][c]
        fp = sum(cm[a][c] for a in CLASSES if a != c)
        fn = sum(cm[c][p] for p in CLASSES if p != c)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        print(f"  {c:<12}  P={prec:.3f}  R={rec:.3f}  F1={f1:.3f}")

    print("\nAccuracy by session kind:")
    for kind in sorted(by_kind.keys()):
        s = by_kind[kind]
        if s["total"]:
            print(f"  {kind:<18} {s['correct']}/{s['total']} = {s['correct']/s['total']:.3f}")

    return {"accuracy": correct / total if total else 0.0, "confusion": cm}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--events", default="data/sessions.jsonl")
    p.add_argument("--truth", default="data/labels.jsonl")
    p.add_argument("--out", default="predictions.csv")
    p.add_argument("--features-out", default="features.csv")
    args = p.parse_args()

    sessions, dropped = load_sessions(args.events)
    print(f"loaded {sum(len(v) for v in sessions.values())} unique events "
          f"across {len(sessions)} sessions (dropped {dropped} exact duplicates)")

    rows = []
    feature_rows = []
    for sid, evs in sessions.items():
        label, debug = classify(evs)
        rows.append((sid, label))
        feature_rows.append((sid, debug["full_features"]))

    rows.sort()
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["session_id", "predicted_label"])
        w.writerows(rows)
    print(f"wrote {len(rows)} predictions to {args.out}")

    if args.features_out:
        with open(args.features_out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "session_id", "n_steps",
                "success_rate_overall", "monotone_success",
                "cycle_pressure", "unique_node_ratio", "output_entropy",
                "target_entropy", "phase_shift",
                "retry_density", "failure_clustering",
                "max_gap_seconds", "has_silent_stall",
            ])
            for sid, ft in feature_rows:
                w.writerow([
                    sid, ft.n_steps,
                    f"{ft.success_rate_overall:.3f}",
                    f"{ft.monotone_success:.3f}",
                    f"{ft.cycle_pressure:.3f}",
                    f"{ft.unique_node_ratio:.3f}",
                    f"{ft.output_entropy:.3f}",
                    f"{ft.target_entropy:.3f}",
                    f"{ft.phase_shift:.3f}",
                    f"{ft.retry_density:.3f}",
                    f"{ft.failure_clustering:.3f}",
                    f"{ft.max_gap_seconds:.2f}",
                    int(ft.has_silent_stall),
                ])

    evaluate(rows, args.truth)


if __name__ == "__main__":
    main()
