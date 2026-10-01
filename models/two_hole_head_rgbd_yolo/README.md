# Head RGB-D two-hole detector and localization

This model uses **only the paired head RGB-D streams**: GDK `kHeadColor`
(640×400 RGB) and `kHeadDepth` (640×400 Z16). It does not use a stereo camera.

| Class | Name | Target |
| --- | --- | --- |
| 0 | `irregular_table_hole` | Irregular white tabletop hole |
| 1 | `round_hinge_tip_hole` | Round hole at the silver hinge tip |

The fixed native RGB crop is `[x0,y0,x1,y1] = [105,125,345,315]`, or
240×190 pixels. Training used 60 new head RGB-D captures under changing
colored lights. The RGB data produced 240 training crops from 48 source frames
and 12 unaugmented validation crops from 12 other source frames. YOLO11n used
`imgsz=960`, `batch=8`, AdamW `lr0=0.001`, no mosaic or flips, and early
stopping after 29 epochs (best epoch 4). The 12-frame same-scene validation
achieved mAP50 0.995 for both classes.

Four of the original 60 saved RGB-D pairs had color/depth timestamp gaps over
50 ms. Their RGB images were suitable for detector training; depth from those
pairs is not used for 3D localization. The capture command now skips such
pairs, and the live localization command enforces the same 50 ms limit.

Run read-only 3D localization:

```bash
cd /home/flyfuture/g2_hinge_assembly
./scripts/run_gdk_sam3_python.sh -m g2_adapter.localize_two_holes_head_rgbd_gdk --count 5 --interval 1
```

The program detects on `kHeadColor`, uses the GDK
`kHeadDepthToHeadColor` extrinsic to align depth **only to that RGB-D color
stream**, then maps each detected center through
`T_head_link3_head_color` and image-time `T_base_link_head_link3`. JSON output
includes optical-camera, `head_link3`, and `base_link` coordinates and the
vector from the table hole to the hinge-tip hole. GDK's TF lookup result is
inverted at the API boundary, as required by the installed GDK convention.
RGB/depth pairs over 50 ms apart and TF samples over 100 ms from image time
are rejected. No robot motion APIs are called.

The depth value is a robust median of visible surfaces inside each detection
box. It approximates the hole center's supporting surface; the empty interior
does not itself have a physical depth return. A separate 12-sample live check
detected both holes in every sample; the observed coordinate ranges were up
to about 4 mm. These samples used the same fixed scene and do not establish
absolute spatial accuracy under different robot poses or target placements.
