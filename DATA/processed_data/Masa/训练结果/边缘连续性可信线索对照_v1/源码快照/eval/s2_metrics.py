"""S2 metrics: planned tasks, including missing/aborted tasks, are the denominator."""
from collections import Counter
import statistics

DISTANCES = (4, 5, 6, 7, 8)
VLM_ARMS = ("G", "E", "G_no_region", "G_gray_target")
RULE_ARMS = ("Frontier", "FixedRegion")


def summarize(episodes, records):
    expected = {ep.episode_id: ep for ep in episodes}
    actual = {r["episode_id"]: r for r in records}
    if len(expected) != len(episodes) or len(actual) != len(records):
        raise ValueError("duplicate episode IDs")
    if not set(actual) <= set(expected):
        raise ValueError("unexpected episode IDs")

    def group(eps):
        done = [actual[e.episode_id] for e in eps
                if actual.get(e.episode_id, {}).get("completion") == "completed"]
        successes = sum(bool(r["evaluation"]["success"]) for r in done)
        observed_sg = sum(r["evaluation"]["sg"] for r in done)
        rows = [actual[e.episode_id] for e in eps if e.episode_id in actual]
        actions = [a for r in rows for a in r["actions"]]
        repeat = sum(bool(a["revisited"]) for a in actions)
        return {"planned": len(eps), "recorded": len(rows), "completed": len(done),
                "successes": successes, "q": len(done) / len(eps),
                "sr_gate": successes / len(eps),
                "sg_gate": sum(actual[e.episode_id]["evaluation"]["sg"]
                    if actual.get(e.episode_id, {}).get("completion") == "completed"
                    else 2 * (e.grid_size - 1) for e in eps) / len(eps),
                "sr_nav": successes / len(done) if done else None,
                "sg_nav": observed_sg / len(done) if done else None,
                "repeat_rate": repeat / len(actions) if actions else None,
                "executed_actions": len(actions),
                "terminal_states": dict(Counter(r["completion"] for r in rows))}

    result = group(episodes)
    result["by_distance"] = {str(d): group([e for e in episodes if e.dist == d])
                             for d in DISTANCES if any(e.dist == d for e in episodes)}
    result["by_area"] = {a: group([e for e in episodes if e.area == a])
                         for a in sorted({e.area for e in episodes})}
    result["distance_complete"] = set(result["by_distance"]) == set(map(str, DISTANCES))
    result["sr_macro"] = statistics.mean(g["sr_gate"] for g in result["by_distance"].values())
    result["sg_macro"] = statistics.mean(g["sg_gate"] for g in result["by_distance"].values())
    return result


def acceptance(jobs):
    """Jobs keyed by ARM_r0..r2; deterministic rules are registered once."""
    required = [f"{a}_r{r}" for a in (*VLM_ARMS, "Random") for r in range(3)]
    required += [f"{a}_r0" for a in RULE_ARMS]
    if any(k not in jobs or not jobs[k].get("audit_passed") for k in required):
        return {"status": "pending", "passed": False,
                "reason": "All 17 registered jobs need complete records and passing offline audits."}
    summaries = {k: jobs[k]["metrics"] for k in required}
    if any(s["recorded"] != s["planned"] for s in summaries.values()):
        return {"status": "pending", "passed": False, "reason": "Missing terminal records."}
    rule = max(RULE_ARMS, key=lambda a: (summaries[f"{a}_r0"]["sr_macro"],
                                        -summaries[f"{a}_r0"]["sg_macro"]))
    baseline = summaries[f"{rule}_r0"]
    gs = [summaries[f"G_r{r}"] for r in range(3)]
    avg_sr = statistics.mean(s["sr_macro"] for s in gs)
    avg_sg = statistics.mean(s["sg_macro"] for s in gs)
    eps = 1e-12
    checks = {
        "all_jobs_q_ge_99pct": all(s["q"] >= .99 for s in summaries.values()),
        "all_distance_tiers_present": all(s["distance_complete"] for s in summaries.values()),
        "G_mean_SR_ge_60pct": avg_sr + eps >= .60,
        "G_mean_SG_le_1_8": avg_sg <= 1.8 + eps,
        "G_rule_gain_ge_5pp": avg_sr - baseline["sr_macro"] + eps >= .05,
        "G_SG_not_worse_than_rule": avg_sg <= baseline["sg_macro"] + eps,
        "G_each_R_le_10pct": all(s["repeat_rate"] is not None and s["repeat_rate"] <= .10 + eps for s in gs),
        "G_SR_range_le_10pp": max(s["sr_macro"] for s in gs) - min(s["sr_macro"] for s in gs) <= .10 + eps,
        "G_beats_rule_in_two_rounds": sum(s["sr_macro"] > baseline["sr_macro"] + eps for s in gs) >= 2,
    }
    for d in DISTANCES:
        checks[f"G_C{d}_minimum_SR"] = statistics.mean(s["by_distance"][str(d)]["sr_gate"] for s in gs) + eps >= (.4 if d >= 7 else .3)
    for arm in ("G_no_region", "G_gray_target"):
        alt = [summaries[f"{arm}_r{r}"] for r in range(3)]
        checks[f"G_vs_{arm}_gain_ge_5pp"] = avg_sr - statistics.mean(s["sr_macro"] for s in alt) + eps >= .05
        checks[f"G_vs_{arm}_two_positive_rounds"] = sum(g["sr_macro"] > a["sr_macro"] + eps for g, a in zip(gs, alt)) >= 2
        checks[f"G_vs_{arm}_SG_not_worse"] = avg_sg <= statistics.mean(s["sg_macro"] for s in alt) + eps
    return {"status": "passed" if all(checks.values()) else "failed",
            "passed": all(checks.values()), "checks": checks, "strongest_rule": rule,
            "G_mean_SR": avg_sr, "G_mean_SG": avg_sg,
            "next_stage": "S3 frozen test only" if all(checks.values()) else "stay in development; no expansion"}
