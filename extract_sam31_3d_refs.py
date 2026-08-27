#!/usr/bin/env python3
import argparse
import contextlib
import csv
import io
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from sam3.model_builder import build_sam3_predictor


def to_numpy(value):
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return value


def infer_image(predictor, bgr: np.ndarray, prompt: str, threshold: float):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb)
    inference_state = predictor.model.init_state(
        resource_path=[image],
        offload_video_to_cpu=True,
        async_loading_frames=False,
    )
    try:
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            _, outputs = predictor.model.add_prompt(
                inference_state=inference_state,
                frame_idx=0,
                text_str=prompt,
                output_prob_thresh=threshold,
                rel_coordinates=True,
            )
        return to_numpy(outputs.get("out_binary_masks"))
    finally:
        inference_state.clear()


def deproject(u: np.ndarray, v: np.ndarray, z_m: np.ndarray, intr: dict) -> np.ndarray:
    x = (u - float(intr["ppx"])) / float(intr["fx"]) * z_m
    y = (v - float(intr["ppy"])) / float(intr["fy"]) * z_m
    return np.stack([x, y, z_m], axis=-1)


def robust_mask_depth(mask: np.ndarray, depth_u16: np.ndarray, depth_scale: float) -> tuple[np.ndarray, np.ndarray]:
    valid_mask = mask.astype(bool) & (depth_u16 > 0)
    if int(valid_mask.sum()) < 20:
        kernel = np.ones((5, 5), np.uint8)
        valid_mask = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1).astype(bool) & (depth_u16 > 0)
    z = depth_u16[valid_mask].astype(np.float32) * depth_scale
    if len(z) == 0:
        return valid_mask, z
    lo, hi = np.percentile(z, [10, 90])
    clipped = valid_mask.copy()
    clipped[valid_mask] = (z >= lo) & (z <= hi)
    return clipped, depth_u16[clipped].astype(np.float32) * depth_scale


def mask_angle_deg(mask: np.ndarray) -> float | None:
    ys, xs = np.where(mask)
    if len(xs) < 5:
        return None
    pts = np.stack([xs.astype(np.float32), ys.astype(np.float32)], axis=1)
    pts -= pts.mean(axis=0, keepdims=True)
    cov = pts.T @ pts / max(len(pts) - 1, 1)
    vals, vecs = np.linalg.eigh(cov)
    axis = vecs[:, int(np.argmax(vals))]
    return float(np.degrees(np.arctan2(axis[1], axis[0])))


def mask_stats(mask: np.ndarray, bgr: np.ndarray, depth_u16: np.ndarray, depth_scale: float, intr: dict) -> dict:
    if mask.ndim == 3:
        mask = mask[0]
    mask = mask.astype(bool)
    area = int(mask.sum())
    if area <= 0:
        return {"area_px": 0}

    ys, xs = np.where(mask)
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    cx_px = float(xs.mean())
    cy_px = float(ys.mean())
    mean_bgr = bgr[mask].mean(axis=0)
    mean_gray = float(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)[mask].mean())

    valid_mask, z_values = robust_mask_depth(mask, depth_u16, depth_scale)
    if len(z_values) > 0:
        v_valid, u_valid = np.where(valid_mask)
        z_valid = depth_u16[valid_mask].astype(np.float32) * depth_scale
        xyz = deproject(u_valid.astype(np.float32), v_valid.astype(np.float32), z_valid, intr)
        xyz_median = np.median(xyz, axis=0).astype(float).tolist()
        z_median = float(np.median(z_values))
        center_xyz = deproject(
            np.array([cx_px], dtype=np.float32),
            np.array([cy_px], dtype=np.float32),
            np.array([z_median], dtype=np.float32),
            intr,
        )[0].astype(float).tolist()
    else:
        xyz_median = None
        z_median = None
        center_xyz = None

    return {
        "area_px": area,
        "bbox_xyxy": [x0, y0, x1, y1],
        "center_px": [cx_px, cy_px],
        "center_xyz_m": center_xyz,
        "xyz_median_m": xyz_median,
        "depth_median_m": z_median,
        "valid_depth_px": int(valid_mask.sum()),
        "mean_bgr": [float(x) for x in mean_bgr.tolist()],
        "mean_gray": mean_gray,
        "angle_deg": mask_angle_deg(mask),
    }


def select_masks(label: str, stats: list[dict], args) -> list[int]:
    valid = [i for i, s in enumerate(stats) if s.get("area_px", 0) > 0 and s.get("center_xyz_m") is not None]
    if label == "silver_metal_hinge":
        valid = [i for i in valid if stats[i]["area_px"] >= args.hinge_min_area]
        if not valid:
            return []
        return [max(valid, key=lambda i: stats[i]["area_px"])]
    if label == "screw_holes":
        selected = []
        for i in valid:
            s = stats[i]
            if s["area_px"] < args.hole_min_area:
                continue
            if s["area_px"] > args.hole_max_area:
                continue
            if s["mean_gray"] > args.hole_max_gray:
                continue
            selected.append(i)
        selected.sort(key=lambda i: (stats[i]["center_px"][1], stats[i]["center_px"][0]))
        return selected
    return valid


def draw_overlay(bgr: np.ndarray, stats: list[dict], selected: list[int], label: str) -> np.ndarray:
    vis = bgr.copy()
    for i, s in enumerate(stats):
        if s.get("area_px", 0) <= 0:
            continue
        x0, y0, x1, y1 = s["bbox_xyxy"]
        color = (80, 80, 255) if i not in selected else (80, 255, 80)
        cv2.rectangle(vis, (x0, y0), (x1, y1), color, 2)
        cx, cy = s["center_px"]
        cv2.circle(vis, (int(cx), int(cy)), 4, color, -1)
        text = f"{i} a={s['area_px']} g={s['mean_gray']:.0f}"
        cv2.putText(vis, text, (x0, max(16, y0 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(vis, text, (x0, max(16, y0 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    cv2.putText(vis, f"{label} selected={len(selected)}", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(vis, f"{label} selected={len(selected)}", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
    return vis


def latest_session() -> Path:
    candidates = sorted(Path("datasets/prompt_refs").glob("*/manifest.csv"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise RuntimeError("No prompt ref sessions found.")
    return candidates[-1].parent.resolve()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("session_dir", nargs="?", default=None)
    parser.add_argument("--checkpoint", default="checkpoints/sam3.1_multiplex.pt")
    parser.add_argument("--threshold", type=float, default=0.15)
    parser.add_argument("--labels", nargs="*", default=["silver_metal_hinge"])
    parser.add_argument("--out-name", default="sam31_3d")
    parser.add_argument("--hinge-min-area", type=int, default=1500)
    parser.add_argument("--hole-min-area", type=int, default=60)
    parser.add_argument("--hole-max-area", type=int, default=2200)
    parser.add_argument("--hole-max-gray", type=float, default=135.0)
    args = parser.parse_args()

    session_dir = latest_session() if args.session_dir is None else Path(args.session_dir).expanduser().resolve()
    out_dir = session_dir / args.out_name
    overlay_dir = out_dir / "overlays"
    overlay_dir.mkdir(parents=True, exist_ok=True)

    with (session_dir / "intrinsics.json").open("r", encoding="utf-8") as f:
        intrinsics = json.load(f)
    depth_scale = float(intrinsics["depth_scale_m_per_unit"])
    # Depth frames were aligned to color before saving, so use color intrinsics here.
    color_intr = intrinsics["color"]

    with (session_dir / "manifest.csv").open("r", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["label"] in set(args.labels)]

    checkpoint = Path(args.checkpoint).expanduser().resolve()
    print(f"session={session_dir}")
    print(f"loading SAM3.1 checkpoint={checkpoint}")
    muted = io.StringIO()
    with contextlib.redirect_stdout(muted):
        predictor = build_sam3_predictor(
            version="sam3.1",
            checkpoint_path=str(checkpoint),
            use_fa3=False,
            use_rope_real=False,
            async_loading_frames=False,
            default_output_prob_thresh=args.threshold,
        )

    csv_path = out_dir / "objects_3d.csv"
    json_path = out_dir / "objects_3d.json"
    records = []
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        fields = [
            "label",
            "prompt",
            "index",
            "mask_index",
            "selected_rank",
            "area_px",
            "center_u",
            "center_v",
            "x_m",
            "y_m",
            "z_m",
            "depth_median_m",
            "mean_gray",
            "angle_deg",
            "bbox_xyxy",
            "overlay_path",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for idx, row in enumerate(rows, 1):
            label = row["label"]
            prompt = row["prompt"]
            sample_index = row["index"]
            bgr = cv2.imread(row["color_path"], cv2.IMREAD_COLOR)
            depth = cv2.imread(row["depth_u16_path"], cv2.IMREAD_UNCHANGED)
            if bgr is None or depth is None:
                raise RuntimeError(f"Failed to read sample {row}")

            masks = infer_image(predictor, bgr, prompt, args.threshold)
            if masks is None:
                masks = np.zeros((0, bgr.shape[0], bgr.shape[1]), dtype=bool)
            stats = [mask_stats(m, bgr, depth, depth_scale, color_intr) for m in masks]
            selected = select_masks(label, stats, args)

            label_overlay = overlay_dir / label
            label_overlay.mkdir(parents=True, exist_ok=True)
            overlay_path = label_overlay / f"{int(sample_index):03d}.jpg"
            cv2.imwrite(str(overlay_path), draw_overlay(bgr, stats, selected, label))

            print(f"[{idx:02d}/{len(rows)}] {label}/{sample_index}: masks={len(stats)} selected={len(selected)}", flush=True)
            for rank, mask_i in enumerate(selected):
                s = stats[mask_i]
                cx, cy = s["center_px"]
                xyz = s["center_xyz_m"]
                record = {
                    "label": label,
                    "prompt": prompt,
                    "index": int(sample_index),
                    "mask_index": int(mask_i),
                    "selected_rank": int(rank),
                    "area_px": int(s["area_px"]),
                    "center_u": float(cx),
                    "center_v": float(cy),
                    "x_m": float(xyz[0]),
                    "y_m": float(xyz[1]),
                    "z_m": float(xyz[2]),
                    "depth_median_m": None if s["depth_median_m"] is None else float(s["depth_median_m"]),
                    "mean_gray": float(s["mean_gray"]),
                    "angle_deg": s["angle_deg"],
                    "bbox_xyxy": s["bbox_xyxy"],
                    "overlay_path": str(overlay_path),
                    "all_masks": stats,
                }
                records.append(record)
                writer.writerow(
                    {
                        "label": record["label"],
                        "prompt": record["prompt"],
                        "index": record["index"],
                        "mask_index": record["mask_index"],
                        "selected_rank": record["selected_rank"],
                        "area_px": record["area_px"],
                        "center_u": f"{record['center_u']:.3f}",
                        "center_v": f"{record['center_v']:.3f}",
                        "x_m": f"{record['x_m']:.5f}",
                        "y_m": f"{record['y_m']:.5f}",
                        "z_m": f"{record['z_m']:.5f}",
                        "depth_median_m": f"{record['depth_median_m']:.5f}" if record["depth_median_m"] is not None else "",
                        "mean_gray": f"{record['mean_gray']:.2f}",
                        "angle_deg": "" if record["angle_deg"] is None else f"{record['angle_deg']:.2f}",
                        "bbox_xyxy": json.dumps(record["bbox_xyxy"]),
                        "overlay_path": record["overlay_path"],
                    }
                )

    json_path.write_text(json.dumps(records, indent=2), encoding="utf-8")
    del predictor
    torch.cuda.empty_cache()
    print(f"csv={csv_path}")
    print(f"json={json_path}")
    print(f"overlays={overlay_dir}")


if __name__ == "__main__":
    main()
