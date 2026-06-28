import argparse
import csv
import json
import os
import sys
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, *args, **kwargs):
        return iterable


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gflownet.models import HierarchicalGFlowNet
from gflownet.utils.query_sketch import QueryCoverageSketch, f1_lower_bound
from train_streaming import (
    _derive_greedy_cap_cached,
    _make_greedy_cache_key,
    build_global_qcs_original,
    compute_global_stats,
    configure_runtime_caches,
    load_preprocessed_trajectory_record,
    normalize_trajectory,
    repair_combined_trajectory,
    run_chunk_inference_const,
    run_chunk_inference_prs,
    scan_valid_indices,
)
from profiling_utils import finalize_profile, increment_profile_counter, init_profile_collector, profile_scope


DEFAULT_TARGET_F1_LIST = [0.45, 0.55, 0.65, 0.75, 0.85, 0.95]
FORMAL_DATASETS = [
    {"dataset_label": "Geolife", "traj_path": "TrajData/Geolife_out", "start_idx": 7000, "end_idx": 7500},
    {"dataset_label": "Tdrive", "traj_path": "Tdrive/taxiout", "start_idx": 7000, "end_idx": 7500},
    {"dataset_label": "WT", "traj_path": "WT/WTout", "start_idx": 1000, "end_idx": 1500},
    {"dataset_label": "AIS", "traj_path": "AIS/AISout", "start_idx": 1000, "end_idx": 1500},
]

SUMMARY_FIELDNAMES = [
    "dataset_label", "traj_path", "start_idx", "end_idx", "checkpoint", "config_name", "mode",
    "target_f1", "valid_count", "avg_f1", "avg_precision", "avg_recall", "avg_cr", "global_total_cr",
    "success_rate", "prs_fallback_rate", "prs_repair_avg_deleted", "prs_avg_candidates",
    "prs_avg_feasible", "safe_feasible_rate", "best_margin_avg", "cr_p50", "cr_p90", "cr_p95",
]

PER_TRAJ_FIELDNAMES = [
    "traj_idx", "num_original_points", "num_kept_points", "f1", "precision", "recall", "cr",
    "cap_cr", "cr_gap", "chunk_count", "prs_fallback_chunks", "prs_safe_feasible_chunks",
    "prs_best_margin_avg", "repair_deleted_points",
]


def parse_f1_list(f1_text) -> List[float]:
    if f1_text is None:
        return []
    if isinstance(f1_text, (list, tuple)):
        return [float(x) for x in f1_text]
    f1_text = str(f1_text).strip()
    if len(f1_text) == 0:
        return []
    return [float(x.strip()) for x in f1_text.split(",") if x.strip()]


def build_parser(add_help: bool = True) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=add_help)
    parser.add_argument("--traj_path", type=str, default="TrajData/Geolife_out")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--start_idx", type=int, default=7000)
    parser.add_argument("--end_idx", type=int, default=7500)
    parser.add_argument("--dataset_label", type=str, default="")
    parser.add_argument("--config_name", type=str, default="manual")
    parser.add_argument("--config_json", type=str, default="")
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--target_f1", type=float, default=0.90)
    parser.add_argument("--target_f1_list", type=str, default="0.45,0.55,0.65,0.75,0.85,0.95")
    parser.add_argument(
        "--f1_threshold",
        type=float,
        default=None,
        help="Helper cap threshold. If omitted, it syncs to current target_f1.",
    )
    parser.add_argument("--target_compression", type=float, default=0.06)
    parser.add_argument("--cr_cap_source", type=str, default="greedy", choices=["greedy", "target"])
    parser.add_argument("--min_keep_ratio", type=float, default=0.20)
    parser.add_argument("--infer_k", type=int, default=8)
    parser.add_argument("--infer_k_max", type=int, default=16)
    parser.add_argument("--infer_temperature", type=float, default=0.65)
    parser.add_argument("--infer_dual_lambda", type=float, default=1.0)
    parser.add_argument("--action_pool_size", type=int, default=64)
    parser.add_argument("--action_pool_explore_ratio", type=float, default=0.15)
    parser.add_argument("--multifidelity_disable", action="store_true")
    parser.add_argument("--multifidelity_topk_exact", type=int, default=2)
    parser.add_argument("--multifidelity_proxy_grid_size", type=int, default=24)
    parser.add_argument("--multifidelity_proxy_stride", type=int, default=4)
    parser.add_argument("--f1_safe_margin", type=float, default=0.01)
    parser.add_argument("--prs_exact_lowcr_topk", type=int, default=2)
    parser.add_argument("--prs_exact_reward_topk", type=int, default=1)
    parser.add_argument("--prs_exact_f1_topk", type=int, default=1)
    parser.add_argument("--repair_delete_ratio", type=float, default=0.30)
    parser.add_argument("--repair_max_delete", type=int, default=1000000)
    parser.add_argument("--repair_jaccard_dedup", type=float, default=0.95)
    parser.add_argument("--repair_mode", type=str, default="hybrid", choices=["exact", "hybrid"])
    parser.add_argument("--repair_skip_below_kept", type=int, default=128)
    parser.add_argument("--repair_min_slack", type=float, default=0.02)
    parser.add_argument("--repair_shortlist_topk", type=int, default=64)
    parser.add_argument("--repair_proxy_grid_size", type=int, default=24)
    parser.add_argument("--repair_proxy_margin", type=float, default=0.01)
    parser.add_argument("--stop_logit_bias", type=float, default=0.0)
    parser.add_argument("--stop_logit_logn_coef", type=float, default=0.0)
    parser.add_argument("--traj_cache_size", type=int, default=512)
    parser.add_argument("--greedy_cache_size", type=int, default=4096)
    parser.add_argument("--global_stats_sample_size", type=int, default=200)
    parser.add_argument("--summary_csv", type=str, default="checkpoints/verify_results/verify_summary.csv")
    parser.add_argument("--per_traj_csv_dir", type=str, default="checkpoints/verify_results/per_traj")
    parser.add_argument("--write_per_traj_csv", action="store_true")
    parser.add_argument("--show_progress", action="store_true")
    parser.add_argument(
        "--save_traj_idx",
        type=int,
        default=-1,
        help="Trajectory index to export simplified result; negative disables export",
    )
    parser.add_argument("--save_traj_dir", type=str, default="checkpoints/verify_streaming_exports")
    parser.add_argument("--legacy_single_path", action="store_true")
    parser.add_argument("--debug_output", action="store_true")
    parser.add_argument("--profile_json", type=str, default="")
    parser.add_argument("--profile_label", type=str, default="")
    parser.add_argument("--with_repair", dest="with_repair", action="store_true")
    parser.add_argument("--no_repair", dest="with_repair", action="store_false")
    parser.set_defaults(with_repair=True)
    parser.add_argument("--no_skip_existing", dest="skip_existing", action="store_false")
    parser.set_defaults(skip_existing=True)
    return parser


def postprocess_args(args: argparse.Namespace) -> argparse.Namespace:
    if getattr(args, "save_traj_idx", None) is not None and int(args.save_traj_idx) < 0:
        args.save_traj_idx = None
    args.target_f1 = float(args.target_f1)
    args.start_idx = int(args.start_idx)
    args.end_idx = int(args.end_idx)
    args.hidden_dim = int(args.hidden_dim)
    args.target_compression = float(args.target_compression)
    args.min_keep_ratio = float(np.clip(args.min_keep_ratio, 0.0, 1.0))
    args.infer_k = max(1, int(args.infer_k))
    args.infer_k_max = max(args.infer_k, int(args.infer_k_max))
    args.infer_temperature = max(1e-6, float(args.infer_temperature))
    args.infer_dual_lambda = float(args.infer_dual_lambda)
    args.action_pool_size = max(0, int(args.action_pool_size))
    args.action_pool_explore_ratio = float(np.clip(args.action_pool_explore_ratio, 0.0, 1.0))
    args.multifidelity_topk_exact = max(1, int(args.multifidelity_topk_exact))
    args.multifidelity_proxy_grid_size = max(0, int(args.multifidelity_proxy_grid_size))
    args.multifidelity_proxy_stride = max(1, int(args.multifidelity_proxy_stride))
    args.f1_safe_margin = max(0.0, float(args.f1_safe_margin))
    args.prs_exact_lowcr_topk = max(0, int(args.prs_exact_lowcr_topk))
    args.prs_exact_reward_topk = max(0, int(args.prs_exact_reward_topk))
    args.prs_exact_f1_topk = max(0, int(args.prs_exact_f1_topk))
    args.repair_delete_ratio = float(np.clip(args.repair_delete_ratio, 0.0, 1.0))
    args.repair_max_delete = max(0, int(args.repair_max_delete))
    args.repair_jaccard_dedup = float(np.clip(args.repair_jaccard_dedup, 0.0, 1.0))
    args.repair_mode = str(getattr(args, "repair_mode", "hybrid")).strip().lower()
    if args.repair_mode not in ("exact", "hybrid"):
        args.repair_mode = "hybrid"
    args.repair_skip_below_kept = max(0, int(getattr(args, "repair_skip_below_kept", 128)))
    args.repair_min_slack = max(0.0, float(getattr(args, "repair_min_slack", 0.02)))
    args.repair_shortlist_topk = max(1, int(getattr(args, "repair_shortlist_topk", 64)))
    args.repair_proxy_grid_size = max(4, int(getattr(args, "repair_proxy_grid_size", 24)))
    args.repair_proxy_margin = max(0.0, float(getattr(args, "repair_proxy_margin", 0.01)))
    args.repair_enable = bool(getattr(args, "with_repair", False))
    args.traj_cache_size = max(0, int(args.traj_cache_size))
    args.greedy_cache_size = max(0, int(args.greedy_cache_size))
    args.global_stats_sample_size = max(1, int(args.global_stats_sample_size))
    if getattr(args, "dataset_label", "") is None:
        args.dataset_label = ""
    if getattr(args, "config_name", "") is None:
        args.config_name = "manual"
    return args


def clone_args(args: argparse.Namespace, overrides: Optional[Dict[str, object]] = None) -> argparse.Namespace:
    cloned = argparse.Namespace(**vars(args).copy())
    if overrides:
        for key, value in overrides.items():
            setattr(cloned, key, value)
    return postprocess_args(cloned)


def apply_config_json(args: argparse.Namespace, json_path: str) -> argparse.Namespace:
    if not json_path:
        return args
    with open(json_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    overrides = payload.get("overrides", payload)
    for key, value in overrides.items():
        if hasattr(args, key):
            setattr(args, key, value)
    if payload.get("config_name") and getattr(args, "config_name", "") == "manual":
        args.config_name = str(payload["config_name"])
    return postprocess_args(args)


def _normalize_key_path(path: str) -> str:
    return os.path.normcase(os.path.abspath(path))


def _safe_slug(text: str) -> str:
    text = str(text).strip().replace("\\", "_").replace("/", "_")
    out = []
    for ch in text:
        if ch.isalnum() or ch in ("-", "_", "."):
            out.append(ch)
        else:
            out.append("_")
    return "".join(out).strip("_") or "unnamed"


def _summary_key_parts(
    traj_path: str,
    start_idx: int,
    end_idx: int,
    checkpoint: str,
    config_name: str,
    mode: str,
    target_f1: float,
) -> Tuple[object, ...]:
    return (
        _normalize_key_path(traj_path),
        int(start_idx),
        int(end_idx),
        _normalize_key_path(checkpoint),
        str(config_name),
        str(mode),
        f"{float(target_f1):.6f}",
    )


def summary_key_from_row(row: Dict[str, object]) -> Tuple[object, ...]:
    return _summary_key_parts(
        traj_path=row["traj_path"],
        start_idx=int(row["start_idx"]),
        end_idx=int(row["end_idx"]),
        checkpoint=row["checkpoint"],
        config_name=row["config_name"],
        mode=row["mode"],
        target_f1=float(row["target_f1"]),
    )


def summary_key_from_args(args: argparse.Namespace, config_name: str, target_f1: float) -> Tuple[object, ...]:
    mode = "const" if bool(getattr(args, "legacy_single_path", False)) else "prs"
    return _summary_key_parts(
        traj_path=args.traj_path,
        start_idx=args.start_idx,
        end_idx=args.end_idx,
        checkpoint=args.checkpoint,
        config_name=config_name,
        mode=mode,
        target_f1=target_f1,
    )


def load_existing_summary_keys(summary_csv: str) -> set:
    if not summary_csv or (not os.path.exists(summary_csv)):
        return set()
    keys = set()
    with open(summary_csv, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row:
                continue
            keys.add(summary_key_from_row(row))
    return keys


def append_csv_row(path: str, fieldnames: Sequence[str], row: Dict[str, object]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def resolve_target_f1_list(args: argparse.Namespace, target_f1_list: Optional[Sequence[float]] = None) -> List[float]:
    if target_f1_list is not None:
        return [float(x) for x in target_f1_list]
    targets = parse_f1_list(getattr(args, "target_f1_list", None))
    if not targets:
        targets = [float(args.target_f1)]
    return [float(x) for x in targets]


def _build_compressed_qcs(
    raw_trajectory: np.ndarray,
    selected_indices: Iterable[int],
    qcs_original: QueryCoverageSketch,
) -> Tuple[QueryCoverageSketch, List[int]]:
    qcs_compressed = QueryCoverageSketch(
        qcs_original.grid_size,
        qcs_original.x_range,
        qcs_original.y_range,
        qcs_original.t_range,
    )
    uniq_indices = sorted(
        int(i) for i in set(int(x) for x in selected_indices) if 0 <= int(i) < len(raw_trajectory)
    )
    if uniq_indices:
        qcs_compressed.add_point(raw_trajectory[uniq_indices[0]])
        for i in range(1, len(uniq_indices)):
            qcs_compressed.add_segment(raw_trajectory[uniq_indices[i - 1]], raw_trajectory[uniq_indices[i]])
    return qcs_compressed, uniq_indices


def calculate_global_qcs_metrics(
    raw_trajectory: np.ndarray,
    selected_indices: Iterable[int],
    local_stats: Optional[dict] = None,
    qcs_original: Optional[QueryCoverageSketch] = None,
) -> Dict[str, object]:
    if qcs_original is None:
        qcs_original = build_global_qcs_original(raw_trajectory, local_stats=local_stats)

    qcs_compressed, uniq_indices = _build_compressed_qcs(raw_trajectory, selected_indices, qcs_original)
    intersection_size = len(qcs_compressed.intersection(qcs_original))
    compressed_size = qcs_compressed.size()
    original_size = qcs_original.size()

    precision = intersection_size / max(1, compressed_size) if compressed_size > 0 else 0.0
    recall = intersection_size / max(1, original_size) if original_size > 0 else 0.0
    f1 = f1_lower_bound(qcs_compressed, qcs_original)
    f1_from_pr = 0.0
    if precision + recall > 0:
        f1_from_pr = 2.0 * precision * recall / (precision + recall)

    return {
        "indices": uniq_indices,
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
        "f1_from_pr": float(f1_from_pr),
        "intersection_size": int(intersection_size),
        "compressed_qcs_size": int(compressed_size),
        "original_qcs_size": int(original_size),
        "qcs_original": qcs_original,
    }


def maybe_export_traj_result(
    args: argparse.Namespace,
    target_f1: float,
    traj_idx: int,
    traj: np.ndarray,
    indices: Sequence[int],
    metrics: Dict[str, object],
    cr: float,
    mode: str,
) -> None:
    if args.save_traj_idx is None:
        return
    if int(traj_idx) != int(args.save_traj_idx):
        return

    out_dir = args.save_traj_dir
    os.makedirs(out_dir, exist_ok=True)

    uniq_indices = sorted(set(int(i) for i in indices))
    if len(uniq_indices) > 0:
        simp_traj = traj[np.array(uniq_indices, dtype=np.int64)]
    else:
        dim = int(traj.shape[1]) if getattr(traj, "ndim", 0) == 2 else 3
        simp_traj = np.zeros((0, dim), dtype=np.float32)

    txt_path = os.path.join(out_dir, f"traj_{traj_idx}_simplified.txt")
    write_mode = "a" if os.path.exists(txt_path) else "w"
    with open(txt_path, write_mode, encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write(f"traj_idx: {int(traj_idx)}\n")
        f.write(f"target_f1: {float(target_f1):.4f}\n")
        f.write(f"mode: {mode}\n")
        f.write(f"num_original_points: {int(len(traj))}\n")
        f.write(f"num_kept_points: {int(len(uniq_indices))}\n")
        f.write(f"compression_ratio: {float(cr):.8f}\n")
        f.write(f"f1: {float(metrics['f1']):.8f}\n")
        f.write(f"precision: {float(metrics['precision']):.8f}\n")
        f.write(f"recall: {float(metrics['recall']):.8f}\n")
        f.write(f"checkpoint: {args.checkpoint}\n")
        f.write(f"range: [{int(args.start_idx)}, {int(args.end_idx)})\n")
        f.write("selected_indices:\n")
        if uniq_indices:
            f.write(",".join(str(i) for i in uniq_indices) + "\n")
        else:
            f.write("(empty)\n")
        f.write("simplified_points(x y t):\n")
        if len(simp_traj) > 0:
            np.savetxt(f, simp_traj, fmt="%.8f")
        else:
            f.write("(empty)\n")
        f.write("\n")

    print(f"[Saved] Trajectory {traj_idx} simplified txt: {txt_path}")


def create_model(args: argparse.Namespace, device: str) -> HierarchicalGFlowNet:
    model = HierarchicalGFlowNet(
        input_dim=3,
        hidden_dim=args.hidden_dim,
        num_layers=2,
        dropout=0.1,
    ).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.eval()
    return model


def prepare_eval_context(args: argparse.Namespace) -> Dict[str, object]:
    with profile_scope("simplify", "simplify_prepare", "prepare_eval_context_total"):
        configure_runtime_caches(
            traj_cache_size=args.traj_cache_size,
            greedy_cache_size=args.greedy_cache_size,
        )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        print(f"Scanning trajectories {args.start_idx}-{args.end_idx} from {args.traj_path}...")
        with profile_scope("simplify", "simplify_prepare", "verify_scan_valid_indices"):
            valid_indices = scan_valid_indices(args.traj_path, args.start_idx, args.end_idx)
        print(f"Found {len(valid_indices)} valid trajectories.")
        if not valid_indices:
            return {"device": device, "valid_indices": [], "global_stats": None, "model": None}

        sample_size = min(args.global_stats_sample_size, len(valid_indices))
        with profile_scope("simplify", "simplify_prepare", "verify_compute_global_stats"):
            global_stats = compute_global_stats(args.traj_path, valid_indices, sample_size=sample_size)
        print(f"Loading checkpoint: {args.checkpoint}")
        with profile_scope("simplify", "simplify_prepare", "verify_load_checkpoint"):
            model = create_model(args, device)
        return {
            "device": device,
            "valid_indices": valid_indices,
            "global_stats": global_stats,
            "model": model,
        }


def _effective_helper_f1_threshold(args: argparse.Namespace, target_f1: float) -> float:
    if getattr(args, "f1_threshold", None) is None:
        return float(target_f1)
    return float(args.f1_threshold)


def evaluate_single_trajectory(
    model,
    args: argparse.Namespace,
    target_f1: float,
    traj_idx: int,
    global_stats: dict,
    device: str,
) -> Tuple[Dict[str, object], Dict[str, object]]:
    with profile_scope("simplify", "simplify_inference", "per_traj_load_preprocess"):
        traj_record = load_preprocessed_trajectory_record(args.traj_path, traj_idx)
    if traj_record is None:
        return {}, {}

    traj = traj_record["traj"]
    bounds = traj_record["bounds"]
    chunk_ranges = traj_record["chunk_ranges"]
    qcs_original = traj_record.get("global_qcs_original")
    if qcs_original is None:
        qcs_original = build_global_qcs_original(traj, local_stats=bounds)
        traj_record["global_qcs_original"] = qcs_original

    worker_args = clone_args(args, {"f1_threshold": _effective_helper_f1_threshold(args, target_f1)})

    all_selected_indices = set()
    all_cap_indices = set()
    chunk_count = 0
    prs_fallback_chunks = 0
    prs_safe_feasible_chunks = 0
    prs_best_margin_sum = 0.0
    prs_total_candidates = 0
    prs_total_feasible = 0
    traj_repair_deleted = 0.0

    for start, end in chunk_ranges:
        chunk = traj[start:end]
        with profile_scope("simplify", "simplify_inference", "per_chunk_normalize"):
            chunk_norm = normalize_trajectory(chunk, stats=global_stats)
        keep_start = start == 0
        keep_end = end == len(traj)
        greedy_cache_key = _make_greedy_cache_key(
            worker_args.traj_path,
            traj_idx,
            start,
            end,
            keep_start,
            keep_end,
            worker_args,
        )
        with profile_scope("simplify", "simplify_inference", "per_chunk_greedy_cap"):
            _, greedy_indices, greedy_max_keep, greedy_cr_cap, _ = _derive_greedy_cap_cached(
                greedy_cache_key,
                chunk,
                chunk_norm,
                keep_start,
                keep_end,
                worker_args,
                global_stats,
                device,
                relax_factor=1.0,
            )
        min_keep = max(2, int(np.ceil(greedy_max_keep * worker_args.min_keep_ratio)))
        min_keep = min(min_keep, greedy_max_keep)

        if worker_args.legacy_single_path:
            chunk_indices, chunk_info = run_chunk_inference_const(
                model=model,
                chunk=chunk,
                chunk_norm=chunk_norm,
                keep_start=keep_start,
                keep_end=keep_end,
                max_keep=greedy_max_keep,
                min_keep=min_keep,
                cr_cap_ratio=greedy_cr_cap,
                args=worker_args,
                global_stats=global_stats,
                device=device,
            )
        else:
            chunk_indices, chunk_info = run_chunk_inference_prs(
                model=model,
                chunk=chunk,
                chunk_norm=chunk_norm,
                keep_start=keep_start,
                keep_end=keep_end,
                max_keep=greedy_max_keep,
                min_keep=min_keep,
                cr_cap_ratio=greedy_cr_cap,
                args=worker_args,
                global_stats=global_stats,
                device=device,
                infer_k=worker_args.infer_k,
                infer_k_max=worker_args.infer_k_max,
                profile_phase="simplify",
                profile_major_module="simplify_inference",
            )

        chunk_count += 1
        prs_fallback_chunks += int(1 if chunk_info.get("prs_fallback", False) else 0)
        prs_safe_feasible_chunks += int(1 if chunk_info.get("prs_best_safe_feasible", False) else 0)
        prs_best_margin_sum += float(chunk_info.get("prs_best_margin", 0.0))
        prs_total_candidates += int(chunk_info.get("prs_candidate_count", 0))
        prs_total_feasible += int(chunk_info.get("prs_feasible_count", 0))
        all_selected_indices.update(int(j) + start for j in chunk_indices)
        all_cap_indices.update(int(j) + start for j in greedy_indices)

    unique_indices = sorted(idx_g for idx_g in all_selected_indices if 0 <= idx_g < len(traj))
    unique_cap_indices = sorted(idx_g for idx_g in all_cap_indices if 0 <= idx_g < len(traj))
    if worker_args.repair_enable and (not worker_args.legacy_single_path):
        with profile_scope("simplify", "simplify_inference", "per_traj_repair"):
            unique_indices, traj_repair_deleted, repair_stats = repair_combined_trajectory(
                raw_trajectory=traj,
                selected_indices=unique_indices,
                tau=float(worker_args.f1_threshold),
                args=worker_args,
                bounds=bounds,
                qcs_original=qcs_original,
            )
        for key, value in (repair_stats or {}).items():
            if int(value) != 0:
                increment_profile_counter(str(key), int(value))
    with profile_scope("simplify", "simplify_inference", "per_traj_global_metrics"):
        metrics = calculate_global_qcs_metrics(traj, unique_indices, local_stats=bounds, qcs_original=qcs_original)
        cr = len(unique_indices) / max(1, len(traj))
        cap_cr = len(unique_cap_indices) / max(1, len(traj))

    maybe_export_traj_result(
        worker_args,
        target_f1,
        traj_idx,
        traj,
        unique_indices,
        metrics,
        cr,
        mode="const" if worker_args.legacy_single_path else "prs",
    )

    traj_row = {
        "traj_idx": int(traj_idx),
        "num_original_points": int(len(traj)),
        "num_kept_points": int(len(unique_indices)),
        "indices": list(unique_indices),
        "f1": float(metrics["f1"]),
        "precision": float(metrics["precision"]),
        "recall": float(metrics["recall"]),
        "cr": float(cr),
        "cap_cr": float(cap_cr),
        "cr_gap": float(cr - cap_cr),
        "chunk_count": int(chunk_count),
        "prs_fallback_chunks": int(prs_fallback_chunks),
        "prs_safe_feasible_chunks": int(prs_safe_feasible_chunks),
        "prs_best_margin_avg": float(prs_best_margin_sum / max(1, chunk_count)),
        "repair_deleted_points": int(traj_repair_deleted),
        "f1_from_pr": float(metrics["f1_from_pr"]),
    }
    chunk_stats = {
        "chunk_count": int(chunk_count),
        "prs_fallback_chunks": int(prs_fallback_chunks),
        "prs_safe_feasible_chunks": int(prs_safe_feasible_chunks),
        "prs_best_margin_sum": float(prs_best_margin_sum),
        "prs_total_candidates": int(prs_total_candidates),
        "prs_total_feasible": int(prs_total_feasible),
        "repair_deleted_points": float(traj_repair_deleted),
    }
    return traj_row, chunk_stats


def write_per_traj_csv(
    per_traj_csv_dir: str,
    dataset_label: str,
    config_name: str,
    start_idx: int,
    end_idx: int,
    target_f1: float,
    rows: List[Dict[str, object]],
) -> str:
    os.makedirs(per_traj_csv_dir, exist_ok=True)
    file_name = (
        f"{_safe_slug(dataset_label)}_{start_idx}_{end_idx}_"
        f"{_safe_slug(config_name)}_f1_{target_f1:.2f}.csv"
    ).replace(".", "p")
    csv_path = os.path.join(per_traj_csv_dir, file_name)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=PER_TRAJ_FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in PER_TRAJ_FIELDNAMES})
    return csv_path


def evaluate_target_f1(
    model,
    args: argparse.Namespace,
    target_f1: float,
    valid_indices: Sequence[Tuple[int, dict]],
    global_stats: dict,
    device: str,
    config_name: str,
) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
    per_traj_rows: List[Dict[str, object]] = []
    total_f1 = 0.0
    total_precision = 0.0
    total_recall = 0.0
    total_cr = 0.0
    total_points_compressed = 0
    total_points_original = 0
    valid_count = 0
    success_count = 0
    total_chunks = 0
    total_fallback_chunks = 0
    total_safe_feasible_chunks = 0
    total_best_margin_sum = 0.0
    total_candidates = 0
    total_feasible = 0
    total_repair_deleted = 0.0

    with profile_scope("simplify", "simplify_inference", "evaluate_target_total"):
        iterator = tqdm(valid_indices, desc=f"{config_name} @ F1={target_f1:.2f}", disable=not args.show_progress)
        for idx, _ in iterator:
            traj_row, chunk_stats = evaluate_single_trajectory(
                model=model,
                args=args,
                target_f1=target_f1,
                traj_idx=int(idx),
                global_stats=global_stats,
                device=device,
            )
            if not traj_row:
                continue

            per_traj_rows.append(traj_row)
            total_f1 += float(traj_row["f1"])
            total_precision += float(traj_row["precision"])
            total_recall += float(traj_row["recall"])
            total_cr += float(traj_row["cr"])
            total_points_compressed += int(traj_row["num_kept_points"])
            total_points_original += int(traj_row["num_original_points"])
            valid_count += 1
            if float(traj_row["f1"]) >= float(target_f1):
                success_count += 1

            total_chunks += int(chunk_stats["chunk_count"])
            total_fallback_chunks += int(chunk_stats["prs_fallback_chunks"])
            total_safe_feasible_chunks += int(chunk_stats["prs_safe_feasible_chunks"])
            total_best_margin_sum += float(chunk_stats["prs_best_margin_sum"])
            total_candidates += int(chunk_stats["prs_total_candidates"])
            total_feasible += int(chunk_stats["prs_total_feasible"])
            total_repair_deleted += float(chunk_stats["repair_deleted_points"])

    dataset_label = args.dataset_label or os.path.normpath(args.traj_path)
    mode = "const" if bool(args.legacy_single_path) else "prs"
    cr_values = [float(row["cr"]) for row in per_traj_rows]
    summary = {
        "dataset_label": dataset_label,
        "traj_path": args.traj_path,
        "start_idx": int(args.start_idx),
        "end_idx": int(args.end_idx),
        "checkpoint": args.checkpoint,
        "config_name": config_name,
        "mode": mode,
        "target_f1": float(target_f1),
        "valid_count": int(valid_count),
        "avg_f1": float(total_f1 / max(1, valid_count)),
        "avg_precision": float(total_precision / max(1, valid_count)),
        "avg_recall": float(total_recall / max(1, valid_count)),
        "avg_cr": float(total_cr / max(1, valid_count)),
        "global_total_cr": float(total_points_compressed / max(1, total_points_original)),
        "success_rate": float(success_count / max(1, valid_count)),
        "prs_fallback_rate": float(total_fallback_chunks / max(1, total_chunks)),
        "prs_repair_avg_deleted": float(total_repair_deleted / max(1, valid_count)),
        "prs_avg_candidates": float(total_candidates / max(1, total_chunks)),
        "prs_avg_feasible": float(total_feasible / max(1, total_chunks)),
        "safe_feasible_rate": float(total_safe_feasible_chunks / max(1, total_chunks)),
        "best_margin_avg": float(total_best_margin_sum / max(1, total_chunks)),
        "cr_p50": float(np.percentile(cr_values, 50)) if cr_values else 0.0,
        "cr_p90": float(np.percentile(cr_values, 90)) if cr_values else 0.0,
        "cr_p95": float(np.percentile(cr_values, 95)) if cr_values else 0.0,
    }
    return summary, per_traj_rows


def print_summary(summary: Dict[str, object], per_traj_csv: str = "") -> None:
    print("\n" + "=" * 72)
    print(
        f"[{summary['config_name']}] {summary['dataset_label']} "
        f"[{summary['start_idx']},{summary['end_idx']}) | "
        f"target_f1={float(summary['target_f1']):.2f} | mode={summary['mode']}"
    )
    print(
        f"  Avg F1={float(summary['avg_f1']):.4f} | "
        f"Precision={float(summary['avg_precision']):.4f} | "
        f"Recall={float(summary['avg_recall']):.4f}"
    )
    print(
        f"  Avg CR/global={float(summary['avg_cr'])*100:.2f}%/"
        f"{float(summary['global_total_cr'])*100:.2f}% | "
        f"Success={float(summary['success_rate'])*100:.2f}% | "
        f"CR p50/p90/p95={float(summary['cr_p50'])*100:.2f}%/"
        f"{float(summary['cr_p90'])*100:.2f}%/"
        f"{float(summary['cr_p95'])*100:.2f}%"
    )
    print(
        f"  PRS fallback={float(summary['prs_fallback_rate'])*100:.2f}% | "
        f"repair_del/traj={float(summary['prs_repair_avg_deleted']):.2f} | "
        f"candidates/feasible={float(summary['prs_avg_candidates']):.2f}/"
        f"{float(summary['prs_avg_feasible']):.2f}"
    )
    print(
        f"  SafeFeasible={float(summary['safe_feasible_rate'])*100:.2f}% | "
        f"BestMarginAvg={float(summary['best_margin_avg']):.4f}"
    )
    if per_traj_csv:
        print(f"  Per-traj CSV: {per_traj_csv}")
    print("=" * 72)


def run_verification_targets(
    args: argparse.Namespace,
    config_name: Optional[str] = None,
    target_f1_list: Optional[Sequence[float]] = None,
    summary_csv: Optional[str] = None,
    context: Optional[Dict[str, object]] = None,
) -> List[Dict[str, object]]:
    args = postprocess_args(args)
    config_name = config_name or args.config_name or "manual"
    summary_csv = summary_csv or args.summary_csv
    target_values = resolve_target_f1_list(args, target_f1_list=target_f1_list)
    existing_keys = load_existing_summary_keys(summary_csv) if args.skip_existing else set()

    pending_targets = [
        target for target in target_values
        if summary_key_from_args(args, config_name, target) not in existing_keys
    ]
    if not pending_targets:
        print(
            f"[Skip] All targets already exist for {config_name} on "
            f"{args.traj_path}[{args.start_idx},{args.end_idx})."
        )
        return []

    if context is None:
        context = prepare_eval_context(args)
    valid_indices = context.get("valid_indices") or []
    if not valid_indices:
        print("No valid trajectories found.")
        return []

    model = context["model"]
    global_stats = context["global_stats"]
    device = context["device"]
    summaries: List[Dict[str, object]] = []

    for target_f1 in target_values:
        key = summary_key_from_args(args, config_name, target_f1)
        if args.skip_existing and key in existing_keys:
            print(f"[Skip] Existing summary found for target_f1={target_f1:.2f} ({config_name}).")
            continue

        summary, per_traj_rows = evaluate_target_f1(
            model=model,
            args=args,
            target_f1=target_f1,
            valid_indices=valid_indices,
            global_stats=global_stats,
            device=device,
            config_name=config_name,
        )
        with profile_scope("simplify", "result_output", "write_summary_csv"):
            append_csv_row(summary_csv, SUMMARY_FIELDNAMES, summary)
        per_traj_csv = ""
        if args.write_per_traj_csv:
            with profile_scope("simplify", "result_output", "write_per_traj_csv"):
                per_traj_csv = write_per_traj_csv(
                    per_traj_csv_dir=args.per_traj_csv_dir,
                    dataset_label=summary["dataset_label"],
                    config_name=config_name,
                    start_idx=args.start_idx,
                    end_idx=args.end_idx,
                    target_f1=target_f1,
                    rows=per_traj_rows,
                )
        print_summary(summary, per_traj_csv=per_traj_csv)
        summaries.append(summary)
        existing_keys.add(key)

    return summaries


def _coerce_row_numbers(row: Dict[str, object]) -> Dict[str, object]:
    out = dict(row)
    numeric_fields = {
        "target_f1",
        "valid_count",
        "avg_f1",
        "avg_precision",
        "avg_recall",
        "avg_cr",
        "global_total_cr",
        "success_rate",
        "prs_fallback_rate",
        "prs_repair_avg_deleted",
        "prs_avg_candidates",
        "prs_avg_feasible",
        "safe_feasible_rate",
        "best_margin_avg",
        "cr_p50",
        "cr_p90",
        "cr_p95",
    }
    for key in numeric_fields:
        if key in out:
            out[key] = float(out[key])
    if "valid_count" in out:
        out["valid_count"] = int(round(float(out["valid_count"])))
    return out


def select_best_ablation(
    rows: Sequence[Dict[str, object]],
    primary_target_f1: float = 0.95,
    fallback_target_f1: float = 0.85,
    success_close_tol: float = 0.01,
    f1_close_tol: float = 0.002,
    weak_success_floor: float = 0.05,
) -> Optional[Dict[str, object]]:
    if not rows:
        return None
    typed_rows = [_coerce_row_numbers(row) for row in rows]

    def select_rows(target_value: float) -> List[Dict[str, object]]:
        return [
            row for row in typed_rows
            if abs(float(row["target_f1"]) - float(target_value)) <= 1e-9
        ]

    primary_rows = select_rows(primary_target_f1)
    selected_target = primary_target_f1
    selected_rows = primary_rows
    if (not primary_rows) or max(float(row["success_rate"]) for row in primary_rows) < weak_success_floor:
        selected_target = fallback_target_f1
        selected_rows = select_rows(fallback_target_f1)

    if not selected_rows:
        return None

    def better(a: Dict[str, object], b: Dict[str, object]) -> bool:
        a_success = float(a["success_rate"])
        b_success = float(b["success_rate"])
        if a_success > b_success + success_close_tol:
            return True
        if b_success > a_success + success_close_tol:
            return False

        a_f1 = float(a["avg_f1"])
        b_f1 = float(b["avg_f1"])
        if a_f1 > b_f1 + f1_close_tol:
            return True
        if b_f1 > a_f1 + f1_close_tol:
            return False

        a_cr = float(a["global_total_cr"])
        b_cr = float(b["global_total_cr"])
        if a_cr < b_cr - 1e-12:
            return True
        if b_cr < a_cr - 1e-12:
            return False

        return str(a["config_name"]) < str(b["config_name"])

    best_row = selected_rows[0]
    for row in selected_rows[1:]:
        if better(row, best_row):
            best_row = row

    return {
        "selected_target_f1": float(selected_target),
        "fallback_used": bool(abs(float(selected_target) - float(primary_target_f1)) > 1e-9),
        "best_row": best_row,
    }


def print_final_cr_summary(rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        return
    print("\n" + "#" * 72)
    print("Summary: Global Total CR by Target F1")
    print("#" * 72)
    for row in rows:
        print(
            f"{row['config_name']} | Target F1={float(row['target_f1']):.2f} -> "
            f"AvgF1={float(row['avg_f1']):.4f}, "
            f"P={float(row['avg_precision']):.4f}, "
            f"R={float(row['avg_recall']):.4f}, "
            f"GlobalCR={float(row['global_total_cr'])*100:.4f}%"
        )


def main() -> None:
    parser = build_parser(add_help=True)
    args = parser.parse_args()
    args = postprocess_args(args)
    args = apply_config_json(args, args.config_json)
    collector = init_profile_collector(
        profile_json=args.profile_json,
        script_name="verify_streaming.py",
        profile_label=args.profile_label,
        extra_metadata={
            "traj_path": args.traj_path,
            "start_idx": int(args.start_idx),
            "end_idx": int(args.end_idx),
            "checkpoint": args.checkpoint,
            "target_f1_list": resolve_target_f1_list(args),
        },
    )
    if collector is not None:
        collector.set_metadata(
            torch_cuda_available=bool(torch.cuda.is_available()),
        )
    summaries = run_verification_targets(args)
    print_final_cr_summary(summaries)
    profile_path = finalize_profile(args.profile_json)
    if profile_path:
        print(f"[Profile] Saved profiling JSON: {profile_path}")


if __name__ == "__main__":
    main()
