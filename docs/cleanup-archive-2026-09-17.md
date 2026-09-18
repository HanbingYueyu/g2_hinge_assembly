# Repository cleanup archive — 2026-09-17

Archive destination: `/home/flyfuture/traditional_auto_flow_g2_archive/2026-09-17/`.
Repository source: `/home/flyfuture/g2_hinge_assembly/`.

## Archived content

- `logs/`: regular files last modified before `2026-09-10 00:00:00` local time. Recent logs remain in the repository.
- `training_runs/`: files originally under repository `runs/`, retaining relative paths. The following tracked runtime weights remain in the repository:
  - `runs/right_d405_new_target_seg_torch/positive_only_v5/best.pt`
  - `runs/right_d405_shaft_seg_torch/mixed_auto_p4_v7/best.pt`

## Archived experimental entry points

Archive `scripts/g2_adapter/` retains these files without modification:

- `live_left_d405_yolo_sam_review_web.py`: historical YOLO + SAM3 review page; the current flow uses its resident YOLO detector and VLM checks.
- `right_hole_target_sam3_preview.py`: historical SAM3 prompt preview, not part of the current AprilTag/right-arm flow.
- `run_left_step_rgb.py`: standalone STEP/RGB experimental preview, not part of current placement/teach entry points.

These tools are archived as unused by the current documented operating commands, not asserted to be universally useless. They have no imports or invocations from the remaining Python scripts/tests/launchers. Shared modules and calibration extraction tools remain available. Historical instructions in `MIGRATION.md`, `g2_adapter/README.md` and `g2_adapter/LEFT_STEP_RGB.md` may reference these entries; restore before following those commands.

Restore by moving each file from archive `scripts/g2_adapter/<filename>` back to repository `g2_adapter/<filename>`, without overwriting newer files. Run restored scripts in the original repository; the archive alone is not a standalone installation.

## Retired bilateral alignment flow

The operator confirmed that the old bilateral XY alignment flow is no longer used. These six files are archived unchanged under `scripts/g2_adapter/`:

- `run_bilateral_alignment_gdk.py`
- `bilateral_servo.py`
- `bilateral_depth_accumulator.py`
- `test_run_bilateral_alignment_gdk.py`
- `test_bilateral_servo.py`
- `test_bilateral_depth_accumulator.py`

The alignment-only model-selection test is removed from `test_stage_services.py`; current stage injection and model-selection tests remain. `bilateral_rgbd_geometry.py` and its tests remain because the active grasp chain imports its geometry functions. Historical alignment instructions require restoring the archived files to their original paths before use.

## Retired motion-kinematics experiment and historical results

At the operator's request, `dianpi/` is entirely excluded from this cleanup, including all vendor demos and platform libraries.

- `g2_adapter/kinematics.py` and `g2_adapter/test_kinematics.py` are archived unchanged under `scripts/g2_adapter/`. No remaining production module imports this standalone FK/Jacobian experiment.
- Repository `reference_outputs/` is moved intact to archive `reference_outputs/`. These are historical experiment results, not the active VLM captures or runtime configuration.

Restore the two Python files to their original `g2_adapter/` paths as described above. Restore historical results by moving archive `reference_outputs/` back to repository `reference_outputs/`; if a newer directory exists, merge selected files without overwriting them.

## Retained operational content

Models, datasets, checkpoints, trajectories (including worktree recordings), calibration/configuration, credentials, environments, Git history, both worktrees, active runtime outputs and remaining independent tooling remain untouched. Only the explicitly listed archived files and directories are removed from the active repository paths.

## Recovery

For any archived log, move it from archive `logs/<filename>` back to repository `logs/<filename>`. For any archived training file, move it from archive `training_runs/<relative path>` back to repository `runs/<relative path>`. Create its parent directory first. Do not overwrite newer files when restoring.

Archiving on the same filesystem tidies the repository but does not reclaim disk space. The file lists can be inspected directly in the archive. This document describes the scope; the completion report states which operations actually succeeded.
