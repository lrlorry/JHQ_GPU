#!/usr/bin/env python3
"""summary.json for the progressive primary screening pilot: the five Phase C gates per workload
and policy, the novelty test, and the decision, derived from analysis_summary.json and
cost_model.json (no new numbers).

  python3 make_summary.py <results dir>

Gates, at each policy's validation-chosen operating point, final top-T*:
  1 non-inferior    test bootstrap 95% CI lower end of the recall@10 difference vs full(T*) >= -0.001
  2 bytes           ideal primary bytes <= 0.70 B_old
  3 tail            test P1 of the per-query difference >= -0.1 and worst >= -0.2
  4 feasible        metadata + scratch <= 0.10 B_old and realistic bytes (32-B sectors + metadata +
                    scratch) < B_old
  5 cost            predicted end-to-end time saving >= 10% at b = 8 or b = 1000, reported for the
                    optimistic (O) and the central (C) scenario of cost_model.py
The pre-declared procedure is the ideal-bytes operating point; the realistic-bytes operating point
is the secondary rule added after the first run, reported beside it.
"""
import json
import os
import sys

POLS = ("A0", "A0c", "A1", "A1c", "A2", "A3")


def main():
    R = sys.argv[1]
    A = json.load(open(os.path.join(R, "analysis_summary.json")))
    C = json.load(open(os.path.join(R, "cost_model.json")))
    out = dict(pilot="Query-Adaptive Progressive Primary Screening", branch="exp/progressive-primary-pilot",
               evidence_labels={"offline": "FP64 offline evaluation on exported codes and candidates (CPU)",
                                "derived": "cost model from measured v65/v62 stage times (no new kernel)",
                                "gpu_measured": "none -- the gate did not pass, no prototype was built"},
               rules=A["rules"], gates_definition=__doc__.split("Gates,")[1].strip(), workloads={})
    for tag, w in A["workloads"].items():
        Ts = w["T_star"]
        g = dict(ck=w["ck"], T_star=Ts, n_candidates_mean=w["n_cand_mean"],
                 recall_full_ck_test=w["recall_full"][str(w["ck"])]["test"],
                 recall_full_Tstar_test=w["recall_full"][str(Ts)]["test"],
                 checks=dict(levels=w["levels"], prim_reconstruction_max_rel_dev=w["prim_reconstruction_max_rel_dev"],
                             sim_equals_v65_final=w["sim_equals_v65_final"],
                             topck_positions_match_v65=w["topck_positions_match_v65"]),
                 list_sign_means=w["mu"], novelty=dict(), gates=dict())
        for rule in ("ideal", "realistic"):
            nv = w["novelty"][rule]
            g["novelty"][rule] = dict(margin=nv["margin"], novel=nv["novel"],
                                      A0=None if nv["A0"] is None else dict(p=nv["A0"]["p"], L_over_ck=nv["A0"]["L_over_ck"]),
                                      best_other=None if nv["best_novel"] is None else
                                      dict(policy=nv["best_novel"]["policy"], p=nv["best_novel"]["p"], L_over_ck=nv["best_novel"]["L_over_ck"]))
            g["gates"][rule] = {}
            for pol in POLS:
                o = w["operating_points"][rule].get("%s/T%d" % (pol, Ts))
                if o is None:
                    continue
                cm = {b: C["workloads"][tag][b]["ops"][rule][pol]["price"] for b in ("b8", "b1000")}
                gain = {sc: {b: cm[b][sc]["gain"] for b in cm} for sc in ("U", "O", "C")}
                gates = {"1_non_inferior": o["non_inferior"], "2_bytes": o["ideal_bytes_ratio"] <= 0.70,
                         "3_tail": (o["p1"] >= -0.1 and o["worst"] >= -0.2),
                         "4_feasible": (o["metadata_ratio"] + o["scratch_ratio"] <= 0.10 and o["total_bytes_ratio"] < 1.0),
                         "5_cost_O": max(gain["O"].values()) >= 0.10, "5_cost_C": max(gain["C"].values()) >= 0.10}
                g["gates"][rule][pol] = dict(p=o["p"], L_over_ck=o["L_over_ck"], lam=o["lam"], delta=o["delta"], ci95=o["ci95"],
                                             ideal_bytes=o["ideal_bytes_ratio"], realistic_bytes=o["total_bytes_ratio"],
                                             predicted_time_saving=gain, gates=gates,
                                             all_pass_O=all(v for k, v in gates.items() if k != "5_cost_C"),
                                             all_pass_C=all(v for k, v in gates.items() if k != "5_cost_O"))
        out["workloads"][tag] = g
    pre = {t: [p for p, v in g["gates"]["ideal"].items() if v["all_pass_O"]] for t, g in out["workloads"].items()}
    pre_c = {t: [p for p, v in g["gates"]["ideal"].items() if v["all_pass_C"]] for t, g in out["workloads"].items()}
    sec_c = {t: [p for p, v in g["gates"]["realistic"].items() if v["all_pass_C"]] for t, g in out["workloads"].items()}
    out["decision"] = dict(
        verdict="NO-GO",
        pre_declared=dict(policies_passing_all_gates_scenario_O=pre, policies_passing_all_gates_scenario_C=pre_c,
                          novelty={t: g["novelty"]["ideal"]["novel"] for t, g in out["workloads"].items()}),
        secondary=dict(policies_passing_all_gates_scenario_C=sec_c,
                       novelty={t: g["novelty"]["realistic"]["novel"] for t, g in out["workloads"].items()}),
        reason=("arxiv and vogue fail gate 5 in both the optimistic and the central scenario at every non-inferior "
                "operating point (arxiv's pre-declared novelty holds only in ideal bytes: A1c 0.508 vs A0 0.586, while "
                "its realistic bytes are worse, 0.961 vs 0.856); openai3 passes gates 1-5 (scenario O) but its fixed-prefix A0 does as well as the "
                "query-adaptive policies at the pre-declared operating point (novelty margin 0), i.e. it is the "
                "v21/v39 prefix cascade with a global top-L. Under the secondary realistic-bytes rule A1 halves L "
                "on openai3 (8 vs 16 ck) and the derived central saving rises from 12-14% (A0) to 17-24% (A1): a "
                "local, post-hoc, derived finding on one workload, not a GO."),
        gpu_minutes_used=1.0, prototype_built=False)
    json.dump(out, open(os.path.join(R, "summary.json"), "w"), indent=1, default=float)
    print(json.dumps(out["decision"], indent=1, default=float))


if __name__ == "__main__":
    main()
