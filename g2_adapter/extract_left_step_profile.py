"""Extract this hinge's seven hole axes from STEP, retaining entity provenance.

The grasp point is explicitly supplied in CAD millimetres: it is never inferred
from a YOLO box or silently guessed from the STEP bounding box.
"""
import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np


HOLE_SURFACES = [(485, 6352), (2366, 2051), (6232,), (4796,),
                 (13259, 4565), (2388, 734), (9314, 144)]


def extract(step, surfaces=HOLE_SURFACES):
    raw = Path(step).read_bytes()
    text = raw.decode("latin1")
    if not re.search(r"SI_UNIT\s*\(\s*\.MILLI\.\s*,\s*\.METRE\.\s*\)", text):
        raise ValueError("expected millimetre STEP")
    entities = dict(re.findall(r"#(\d+)\s*=\s*(.*?);", text, re.S))

    def refs(value):
        return re.findall(r"#(\d+)", value)

    def xyz(entity):
        return np.array([float(x) for x in re.search(r"\(\s*([^()]*)\)\s*\)", entities[entity]).group(1).split(",")])

    holes = []
    for group in surfaces:
        centres, rim_points = [], []
        for surface in group:
            value = entities[str(surface)]
            if not value.startswith("CYLINDRICAL_SURFACE"):
                raise ValueError("STEP topology changed; reselect hole surfaces")
            placement = refs(entities[refs(value)[0]])
            axis = xyz(placement[1])
            if abs(axis[2]) < .999:
                raise ValueError("unexpected hole axis")
            centres.append(xyz(placement[0]))
            seen = set()

            def walk(entity):
                if entity in seen:
                    return
                seen.add(entity)
                entry = entities[entity]
                if entry.startswith("VERTEX_POINT"):
                    rim_points.append(xyz(refs(entry)[0]))
                for reference in refs(entry):
                    walk(reference)

            for entity, entry in entities.items():
                if entry.startswith("ADVANCED_FACE") and str(surface) in refs(entry):
                    walk(entity)
        # Slot centre = midpoint between its two circular end axes.
        centre = np.mean(centres, axis=0)
        rim = np.asarray(rim_points)
        # Top rim elevation, compensated for the small tilt of the STEP axes.
        elevations = rim[:, 2] - (rim[:, 0] - centre[0]) * (-axis[0] / axis[2])
        z = float(elevations.max())
        centre += axis * ((z - centre[2]) / axis[2])
        holes.append(centre / 1000)
    return {"source_step": str(step), "source_sha256": hashlib.sha256(raw).hexdigest(),
            "hole_surface_entities": surfaces, "hole_centers_m": np.asarray(holes).tolist(),
            "verified": False,
            "notes": "CAD axis centres at top hole rims; validate image slot-centre correspondence and grasp point before motion."}


def extract_marked(step, reference=None):
    """Transfer verified hole topology only after checking both exports' axes.

    The target is the midpoint of the two marked faces' vertex extents, not
    an area centroid. This defines the centre between the two jaw contacts.
    """
    raw = Path(step).read_bytes()
    entities = dict(re.findall(r"#(\d+)\s*=\s*(.*?);", raw.decode("latin1"), re.S))

    def refs(value):
        return re.findall(r"#(\d+)", value)

    def xyz(table, key):
        return np.array([float(x) for x in re.search(r"\(\s*([^()]*)\)\s*\)", table[key]).group(1).split(",")])

    def cylinder(table, key):
        value = table[key]
        placement = refs(table[refs(value)[0]])
        return xyz(table, placement[0]), xyz(table, placement[1]), float(value.rsplit(",", 1)[-1].strip().rstrip(")"))

    cylinders = {key: cylinder(entities, key) for key, value in entities.items() if value.startswith("CYLINDRICAL_SURFACE")}
    # Axis samples from the original export, mm. Used only to identify surfaces;
    # full-precision geometry is then read from the current STEP.
    seeds = [
        [(-272.966330,1272.129440,20.851272,3.5),(-272.966330,1274.005993,20.851272,3.5)],
        [(-233.798810,1281.532729,20.852819,3.5),(-234.650748,1283.204751,20.852786,3.5)],
        [(-272.966069,1284.909051,14.251272,3)],
        [(-239.600372,1292.919446,14.252590,3)],
        [(-272.966330,1295.812108,20.851272,3.5),(-272.966330,1297.504479,20.851272,3.5)],
        [(-244.550517,1302.634141,20.852395,3.5),(-245.318837,1304.142055,20.852364,3.5)],
        [(-258.347695,1297.564164,14.651850,3.5),(-258.581141,1298.536534,14.651840,3.5)],
    ]
    mapping = []
    for group in seeds:
        mapped = []
        for sample in group:
            point, radius = np.asarray(sample[:3]), sample[3]
            axis = np.array([-3.94966960644e-5, 0, .99999999922])
            matches = [candidate for candidate, (p, n, r) in cylinders.items()
                       if abs(radius - r) < 1e-5 and abs(np.dot(axis, n)) > .999999
                       and np.linalg.norm(np.cross(p - point, axis)) < .001]
            if not matches:
                raise ValueError(f"new STEP does not match reference hole axis {sample}")
            mapped.append(matches[0])
        mapping.append(mapped)
    profile = extract(step, surfaces=mapping)

    red = set()
    for key, value in entities.items():
        if value.startswith("COLOUR_RGB"):
            rgb = [float(x) for x in value[value.index(",") + 1:].rstrip().rstrip(")").split(",")]
            if rgb[0] > .8 and max(rgb[1:]) < .2:
                red.add(key)

    def contains_red(key, seen=None):
        seen = set() if seen is None else seen
        if key in red:
            return True
        if key in seen:
            return False
        seen.add(key)
        return any(contains_red(r, seen) for r in refs(entities[key]))

    faces = []
    for key, value in entities.items():
        if value.startswith(("OVER_RIDING_STYLED_ITEM", "STYLED_ITEM")):
            links = refs(value)
            if links and contains_red(links[0]):
                faces.extend(r for r in links[1:] if entities[r].startswith("ADVANCED_FACE"))
    faces = sorted(set(faces))
    if len(faces) != 2:
        raise ValueError(f"expected two red jaw contact faces, got {faces}")
    centres = []
    for face in faces:
        points, seen = [], set()

        def walk(key):
            if key in seen:
                return
            seen.add(key)
            value = entities[key]
            if value.startswith("VERTEX_POINT"):
                points.append(xyz(entities, refs(value)[0]))
            for link in refs(value):
                walk(link)

        walk(face)
        points = np.asarray(points)
        centres.append((points.min(axis=0) + points.max(axis=0)) / 2)
    profile.update(source_step=str(step), source_sha256=hashlib.sha256(raw).hexdigest(),
                   new_hole_surface_entities=mapping, red_face_entities=faces,
                   red_face_extent_centers_mm=np.asarray(centres).tolist(),
                   grasp_center_m=(np.mean(centres, axis=0) / 1000).tolist(),
                   grasp_center_definition="midpoint_of_two_red_face_vertex_extent_centers",
                   notes="Hole axes cross-checked against original STEP. Target is between red contacts. Physical RGB/TCP verification pending.")
    return profile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--step", type=Path, default=Path("models/顶铰链.stp"))
    parser.add_argument("--grasp-center-mm", help="Optional explicit CAD X,Y,Z override; default extracts red faces")
    parser.add_argument("--output", type=Path, default=Path("g2_adapter/left_step_profile.json"))
    args = parser.parse_args()
    profile = extract_marked(args.step)
    if args.grasp_center_mm:
        target = np.array([float(x) for x in args.grasp_center_mm.split(",")])
        if target.shape != (3,) or not np.isfinite(target).all():
            parser.error("grasp centre must be three finite millimetre coordinates")
        profile["grasp_center_m"] = (target / 1000).tolist()
        profile["grasp_center_definition"] = "explicit_override"
    args.output.write_text(json.dumps(profile, ensure_ascii=False, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
