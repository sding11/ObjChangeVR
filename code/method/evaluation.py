import os, re, unicodedata
import pandas as pd
from difflib import SequenceMatcher
from collections import Counter
from typing import List, Dict, Tuple

# ================== Configuration ==================
DATA_ROOT = "./dataset"
TARGET_FILENAME = "results.csv"   
SCENE_TYPES = ["market", "fastfood", "architecture", "museum", "village"]

GROUP_MAP = { "market":"Short", "fastfood":"Short", "architecture":"Short",
              "museum":"Long", "village":"Long" }

TAU = 0.80                            # Threshold for EM@τ
WRITE_EXCEL = True
EXCEL_OUTPUT = "./evaluation.xlsx"

# F1 configuration
COMPUTE_CLASS_F1 = True
EXCLUDE_UNKNOWN_GT = True             # Whether to exclude samples with unknown GT when computing F1
LABELS = ["disappeared", "never", "always been there"]

# Whether to print AvgSim for diagnostic purposes
SHOW_AVG_SIM_IN_PRINT = True
# =====================================================

def normalize_minimal(s: str) -> str:
    """Lightweight string normalization for robust comparison."""
    if s is None: return ""
    s = str(s)
    s = unicodedata.normalize("NFKC", s)
    s = s.strip().strip('"').strip("'")
    s = re.sub(r"\s+", " ", s)
    return s.lower()

def extract_judgment(text: str) -> str:
    """
    Extract the primary judgment clause from an answer, removing leading 'Answer:'
    and truncating before justification (e.g., 'because ...').
    """
    if text is None: return ""
    t = normalize_minimal(text)
    t = re.sub(r'^(answer:\s*)', '', t)
    t = re.split(r"\b(because)\b", t, maxsplit=1)[0]
    m = re.split(r"[.!?。？！]", t, maxsplit=1)
    return (m[0] if m else t).strip()

def em_strict(ans: str, gen: str) -> int:
    """Strict exact match on normalized full strings."""
    return int(normalize_minimal(ans) == normalize_minimal(gen))

def em_tau(ans: str, gen: str, tau: float) -> Tuple[float, int]:
    """
    Soft EM via string similarity over extracted judgments.
    Returns (similarity_score, em_at_tau_flag).
    """
    a = extract_judgment(ans)
    g = extract_judgment(gen)
    score = SequenceMatcher(None, a, g).ratio()
    return score, int(score >= tau)

# =========================
# Label mapping and F1
# =========================

def classify_answer(answer: str) -> str:
    """Map free-form answers into one of the 3 normalized labels (or 'unknown')."""
    if isinstance(answer, str):
        a = normalize_minimal(answer)
        if "disappear" in a:
            return "disappeared"
        if "never there" in a or a == "never":
            return "never"
        if "always been" in a or "always there" in a:
            return "always been there"
    return "unknown"

def precision_recall_f1_from_lists(y_true: List[str], y_pred: List[str], labels: List[str]):
    """
    Compute per-class precision/recall/F1 as well as micro/macro/weighted F1
    for the given label set.
    """
    tp, fp, fn, support = Counter(), Counter(), Counter(), Counter()
    for t, p in zip(y_true, y_pred):
        if t in labels:
            support[t] += 1
        if t == p and t in labels:
            tp[t] += 1
        else:
            if p in labels:
                fp[p] += 1
            if t in labels:
                fn[t] += 1

    per_class = {}
    for c in labels:
        _tp, _fp, _fn = tp[c], fp[c], fn[c]
        prec = _tp / (_tp + _fp) if (_tp + _fp) > 0 else 0.0
        rec  = _tp / (_tp + _fn) if (_tp + _fn) > 0 else 0.0
        f1   = (2*prec*rec / (prec+rec)) if (prec+rec) > 0 else 0.0
        per_class[c] = {"precision": prec, "recall": rec, "f1": f1, "support": support[c]}

    sum_tp, sum_fp, sum_fn = sum(tp.values()), sum(fp.values()), sum(fn.values())
    micro_prec = sum_tp / (sum_tp + sum_fp) if (sum_tp + sum_fp) > 0 else 0.0
    micro_rec  = sum_tp / (sum_tp + sum_fn) if (sum_tp + sum_fn) > 0 else 0.0
    micro_f1   = (2*micro_prec*micro_rec / (micro_prec+micro_rec)) if (micro_prec+micro_rec) > 0 else 0.0

    macro_f1 = sum(per_class[c]["f1"] for c in labels) / len(labels) if labels else 0.0
    total_support = sum(per_class[c]["support"] for c in labels)
    weighted_f1 = (sum(per_class[c]["f1"] * per_class[c]["support"] for c in labels) / total_support) if total_support > 0 else 0.0
    return per_class, micro_f1, macro_f1, weighted_f1

# =========================
# Evaluation
# =========================

def list_scene_files() -> List[Dict]:
    """Collect result CSVs across scenes and trajectory folders."""
    items = []
    for scene in SCENE_TYPES:
        scene_dir = os.path.join(DATA_ROOT, scene)
        if not os.path.isdir(scene_dir): continue
        sub_dirs = sorted([d for d in os.listdir(scene_dir) if d.isdigit()], key=lambda x: int(x))
        for sub in sub_dirs:
            csv_path = os.path.join(scene_dir, sub, TARGET_FILENAME)
            if os.path.exists(csv_path):
                items.append({"scene_type": scene, "folder": sub, "csv": csv_path})
    return items

def evaluate_file(csv_path: str, tau: float) -> Dict:
    """Evaluate a single result CSV and return metrics plus per-sample diagnostics."""
    df = pd.read_csv(csv_path)
    if "Answer" not in df.columns or "GeneratedAnswer" not in df.columns:
        raise ValueError(f"{csv_path} missing required columns Answer/GeneratedAnswer")

    # Strict EM
    strict_vec = [em_strict(a, g) for a, g in zip(df["Answer"], df["GeneratedAnswer"])]
    strict_em = sum(strict_vec) / len(strict_vec) if len(strict_vec) else 0.0

    # EM@τ based on similarity of extracted judgments
    scores, tau_vec = [], []
    ans_judge, gen_judge = [], []
    for a, g in zip(df["Answer"], df["GeneratedAnswer"]):
        s, flag = em_tau(a, g, tau)
        scores.append(s); tau_vec.append(flag)
        ans_judge.append(extract_judgment(a))
        gen_judge.append(extract_judgment(g))
    em_tau_rate = sum(tau_vec) / len(tau_vec) if len(tau_vec) else 0.0
    avg_score = sum(scores) / len(scores) if len(scores) else 0.0

    # Class-level F1 over normalized labels
    y_true, y_pred = [], []
    if COMPUTE_CLASS_F1:
        df_cls = df.copy()
        df_cls["GT_Class"] = df_cls["Answer"].map(classify_answer)
        df_cls["PR_Class"] = df_cls["GeneratedAnswer"].map(classify_answer)
        if EXCLUDE_UNKNOWN_GT:
            df_cls = df_cls[df_cls["GT_Class"] != "unknown"]
        y_true = df_cls["GT_Class"].tolist()
        y_pred = df_cls["PR_Class"].tolist()
        per_class, micro_f1, macro_f1, weighted_f1 = precision_recall_f1_from_lists(y_true, y_pred, LABELS)
    else:
        per_class = {c: {"precision": "", "recall": "", "f1": "", "support": ""} for c in LABELS}
        micro_f1 = macro_f1 = weighted_f1 = ""

    per_sample = pd.DataFrame({
        "Answer": df["Answer"],
        "GeneratedAnswer": df["GeneratedAnswer"],
        "Ans_Judgment": ans_judge,
        "Gen_Judgment": gen_judge,
        "StrictEM": strict_vec,
        "SimScore": scores,
        f"EM@{tau:.2f}": tau_vec,
    })

    return {
        "samples": len(df),
        "strict_em": strict_em,
        "em_tau": em_tau_rate,
        "avg_sim_score": avg_score,
        "per_sample": per_sample,

        "per_class": per_class,
        "micro_f1": micro_f1,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,

        "y_true": y_true,
        "y_pred": y_pred,
    }

def merge_group(records: List[Dict]) -> Dict:
    """Aggregate metrics across multiple files, weighted by sample counts."""
    total = sum(r["samples"] for r in records)
    if total == 0:
        return {"samples": 0, "strict_em": 0.0, "em_tau": 0.0, "avg_sim_score": 0.0,
                "per_class": {c: {"precision":0,"recall":0,"f1":0,"support":0} for c in LABELS},
                "micro_f1": 0.0, "macro_f1": 0.0, "weighted_f1": 0.0}

    strict_em = sum(r["strict_em"] * r["samples"] for r in records) / total
    em_tau_rate = sum(r["em_tau"] * r["samples"] for r in records) / total
    avg_sim = sum(r["avg_sim_score"] * r["samples"] for r in records) / total

    if COMPUTE_CLASS_F1:
        y_true_all, y_pred_all = [], []
        for r in records:
            y_true_all.extend(r["y_true"])
            y_pred_all.extend(r["y_pred"])
        per_class, micro_f1, macro_f1, weighted_f1 = precision_recall_f1_from_lists(y_true_all, y_pred_all, LABELS)
    else:
        per_class = {c: {"precision":0,"recall":0,"f1":0,"support":0} for c in LABELS}
        micro_f1 = macro_f1 = weighted_f1 = 0.0

    return {
        "samples": total,
        "strict_em": strict_em,
        "em_tau": em_tau_rate,
        "avg_sim_score": avg_sim,
        "per_class": per_class,
        "micro_f1": micro_f1,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
    }

def main():
    files = list_scene_files()
    if not files:
        print(f"No {TARGET_FILENAME} found. Please verify DATA_ROOT / SCENE_TYPES / TARGET_FILENAME settings.")
        return

    per_file_summary = []
    per_file_details = {}
    groups = {"Short": [], "Long": []}

    # Evaluate each file individually
    for item in files:
        scene, folder, path = item["scene_type"], item["folder"], item["csv"]
        group = GROUP_MAP.get(scene, "Short")
        stats = evaluate_file(path, TAU)
        groups[group].append(stats)

        row = {
            "scene_type": scene, "group": group, "folder": folder, "csv_path": path,
            "samples": stats["samples"],
            "StrictEM": stats["strict_em"], f"EM@{TAU:.2f}": stats["em_tau"],
            "AvgSimScore": stats["avg_sim_score"],
        }
        if COMPUTE_CLASS_F1:
            row.update({"micro_F1": stats["micro_f1"], "macro_F1": stats["macro_f1"], "weighted_F1": stats["weighted_f1"]})
        per_file_summary.append(row)
        per_file_details[f"{scene}_{folder}"] = stats["per_sample"]

    # Print group-level metrics: Short / Long / Average
    print("\n===== Group Metrics =====")
    for gname in ["Short", "Long"]:
        g = merge_group(groups[gname]) if groups[gname] else None
        if not g or g["samples"] == 0:
            print(f"[{gname}] no data")
            continue
        line = (f"[{gname}] Samples={g['samples']}  "
                f"StrictEM={g['strict_em']:.4f}  EM@{TAU:.2f}={g['em_tau']:.4f}")
        if SHOW_AVG_SIM_IN_PRINT:
            line += f"  AvgSim={g['avg_sim_score']:.4f}"
        if COMPUTE_CLASS_F1:
            line += (f"  micro-F1={g['micro_f1']:.4f}  "
                     f"macro-F1={g['macro_f1']:.4f}  weighted-F1={g['weighted_f1']:.4f}")
        print(line)

    # Overall average across all groups
    all_records = (groups["Short"] if groups["Short"] else []) + (groups["Long"] if groups["Long"] else [])
    avg_stats = merge_group(all_records) if all_records else None
    if avg_stats and avg_stats["samples"] > 0:
        line = (f"[Average] Samples={avg_stats['samples']}  "
                f"StrictEM={avg_stats['strict_em']:.4f}  EM@{TAU:.2f}={avg_stats['em_tau']:.4f}")
        if SHOW_AVG_SIM_IN_PRINT:
            line += f"  AvgSim={avg_stats['avg_sim_score']:.4f}"
        if COMPUTE_CLASS_F1:
            line += (f"  micro-F1={avg_stats['micro_f1']:.4f}  "
                     f"macro-F1={avg_stats['macro_f1']:.4f}  weighted-F1={avg_stats['weighted_f1']:.4f}")
        print(line)
    else:
        print("[Average] no data")

    # Write detailed results to Excel
    if WRITE_EXCEL:
        with pd.ExcelWriter(EXCEL_OUTPUT, engine="openpyxl") as writer:
            rows = []
            for gname in ["Short","Long"]:
                g = merge_group(groups[gname]) if groups[gname] else None
                if not g: continue
                row = {"group": gname, "samples": g["samples"],
                       "StrictEM": g["strict_em"], f"EM@{TAU:.2f}": g["em_tau"],
                       "AvgSimScore": g["avg_sim_score"]}
                if COMPUTE_CLASS_F1:
                    row.update({"micro_F1": g["micro_f1"], "macro_F1": g["macro_f1"], "weighted_F1": g["weighted_f1"]})
                rows.append(row)

            if avg_stats:
                row = {"group": "Average", "samples": avg_stats["samples"],
                       "StrictEM": avg_stats["strict_em"], f"EM@{TAU:.2f}": avg_stats["em_tau"],
                       "AvgSimScore": avg_stats["avg_sim_score"]}
                if COMPUTE_CLASS_F1:
                    row.update({"micro_F1": avg_stats["micro_f1"], "macro_F1": avg_stats["macro_f1"], "weighted_F1": avg_stats["weighted_f1"]})
                rows.append(row)

            pd.DataFrame(rows).to_excel(writer, index=False, sheet_name="Groups_Summary")

            # Per-file summary
            pd.DataFrame(per_file_summary).to_excel(writer, index=False, sheet_name="Files_Summary")

            for sheet_name, df_det in per_file_details.items():
                safe = sheet_name[:31]
                df_det.to_excel(writer, index=False, sheet_name=safe)

        print(f"\nDetailed results written to: {EXCEL_OUTPUT}")

if __name__ == "__main__":
    main()
