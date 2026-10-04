"""Measure the red hinge-tip bore and four marked plate slots from 顶铰链.stp.

This STEP file uses millimetres. The marked plate holes are oblong slots:
each has two r=3.5 mm cylindrical end surfaces. Their planar centre is the
midpoint of the two end-circle centres. A fifth, unmarked slot lies 3.6 mm
below the four marked ones and is deliberately excluded.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

DEFAULT_STEP = Path(__file__).resolve().parents[1] / "models/顶铰链.stp"
ASSEMBLY_STEP = Path("/home/flyfuture/下载/306W改.STEP")
# Measured from the matching component in /home/flyfuture/下载/306W改.STEP.
# The outer r=20 mm and inner r=11 mm circles share this centre. The red r=6
# bore matches DEFAULT_STEP to <0.001 mm, so both STEP exports use the same
# model XY coordinates. The assembly file is 161 MB and is not copied here.
ASSEMBLY_STEP_SHA256 = "ca6a0b663f49fd72ff98fbfd263e42eb8d536ba648c2f4ca4d55ca25d535f2eb"
ASSEMBLY_RED_XY_MM = (-273.9159402086825, 1362.358972978954)
ASSEMBLY_TABLE_XY_MM = (-248.06551103157, 1254.684844249108)
_ENTITY = re.compile(r"#(\d+)\s*=\s*([A-Z0-9_]+)\s*\((.*?)\);", re.S)
_NUMBER = r"[-+0-9.Ee]+"
_POINT = re.compile(r"\(\s*(" + _NUMBER + r"(?:\s*,\s*" + _NUMBER + r")*)\s*\)")


@dataclass(frozen=True)
class HingeGeometry:
    step_sha256: str
    red_xy_mm: tuple[float, float]
    green_xy_mm: tuple[float, float]
    next_holes_xy_mm: tuple[tuple[str, tuple[float, float]], ...]
    plate_z_mm: float
    red_green_distance_mm: float
    assembly_table_xy_mm: tuple[float, float]
    assembly_step_sha256: str

    def as_dict(self) -> dict:
        return {
            "step_sha256": self.step_sha256,
            "unit": "mm",
            "red_hinge_tip_xy_mm": list(self.red_xy_mm),
            "green_taught_hole_xy_mm": list(self.green_xy_mm),
            "other_three_holes_xy_mm": {
                name: list(xy) for name, xy in self.next_holes_xy_mm
            },
            "plate_hole_z_mm": self.plate_z_mm,
            "red_to_green_distance_mm": self.red_green_distance_mm,
            "recognized_table_hole_xy_mm": list(self.assembly_table_xy_mm),
            "recognized_red_to_table_distance_mm": math.dist(self.red_xy_mm, self.assembly_table_xy_mm),
            "assembly_step_sha256": self.assembly_step_sha256,
            "next_hole_offsets_from_green_mm": {
                name: [xy[0] - self.green_xy_mm[0], xy[1] - self.green_xy_mm[1]]
                for name, xy in self.next_holes_xy_mm
            },
        }


def _numbers(entity: tuple[str, str]) -> tuple[float, ...]:
    match = _POINT.search(entity[1])
    if match is None:
        raise ValueError(f"STEP entity has no numeric vector: {entity[0]}")
    return tuple(float(value) for value in match.group(1).split(","))


def measure_top_hinge(path: Path = DEFAULT_STEP) -> HingeGeometry:
    source = path.expanduser().resolve()
    raw = source.read_bytes()
    if ASSEMBLY_STEP.is_file():
        digest = hashlib.sha256()
        with ASSEMBLY_STEP.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != ASSEMBLY_STEP_SHA256:
            raise ValueError("Assembly STEP changed; remeasure red and table anchor centres")
    content = raw.decode("latin1")
    if "SI_UNIT(.MILLI.,.METRE.)" not in content:
        raise ValueError("Expected millimetre STEP geometry")
    entities = {
        int(number): (kind, body)
        for number, kind, body in _ENTITY.findall(content)
    }
    surfaces: list[tuple[float, tuple[float, float, float]]] = []
    for kind, body in entities.values():
        if kind != "CYLINDRICAL_SURFACE":
            continue
        match = re.search(r"#(\d+)\s*,\s*(" + _NUMBER + r")", body)
        if match is None:
            continue
        radius = float(match.group(2))
        if not (abs(radius - 3.5) < 1e-6 or abs(radius - 6.0) < 1e-6):
            continue
        axis_entity = entities[int(match.group(1))]
        refs = [int(value) for value in re.findall(r"#(\d+)", axis_entity[1])]
        if len(refs) < 2:
            raise ValueError("Cylinder has no centre and axis")
        centre = _numbers(entities[refs[0]])
        axis = _numbers(entities[refs[1]])
        if len(centre) != 3 or len(axis) != 3:
            raise ValueError("Expected three-dimensional STEP cylinder")
        if abs(abs(axis[2]) - 1.0) < 1e-5:
            surfaces.append((radius, centre))
    red_samples = [
        (centre[0], centre[1])
        for radius, centre in surfaces if abs(radius - 6.0) < 1e-6
    ]
    if not red_samples:
        raise ValueError("No axial r=6 mm red tip bore found")
    red = (
        sum(point[0] for point in red_samples) / len(red_samples),
        sum(point[1] for point in red_samples) / len(red_samples),
    )
    if max(math.dist(red, point) for point in red_samples) > 0.01:
        raise ValueError("Axial r=6 mm red tip surfaces disagree on their centre")
    if math.dist(red, ASSEMBLY_RED_XY_MM) > 0.01:
        raise ValueError("Standalone hinge and assembly STEP red bore centres do not align")
    slot_ends = [
        centre for radius, centre in surfaces if abs(radius - 3.5) < 1e-6
    ]
    if len(slot_ends) != 10:
        raise ValueError(f"Expected ten r=3.5 mm slot end surfaces, found {len(slot_ends)}")
    slot_centres = []
    while slot_ends:
        first = slot_ends.pop(0)
        matches = [
            other for other in slot_ends
            if math.dist(first[:2], other[:2]) < 2.5
            and abs(first[2] - other[2]) < 0.01
        ]
        if len(matches) != 1:
            raise ValueError(f"Could not identify a unique oblong slot pair: {first}")
        second = matches[0]
        slot_ends.remove(second)
        slot_centres.append(tuple((a + b) / 2 for a, b in zip(first, second)))
    # Four yellow-marked top-face slots share z≈18.65 mm. The central,
    # unmarked slot lies at z≈15.05 mm in this model.
    top = [point for point in slot_centres if 18.60 < point[2] < 18.70]
    lower = [point for point in slot_centres if 15.00 < point[2] < 15.10]
    if len(top) != 4 or len(lower) != 1:
        raise ValueError("STEP top/lower slot layers do not match the marked four-hole plate")
    ordered = sorted(top, key=lambda point: point[0])
    top_left, top_right = sorted(ordered[:2], key=lambda point: point[1])
    bottom_middle = ordered[2]
    green = ordered[3]
    points = {
        "bottom_middle": bottom_middle,
        "top_right": top_right,
        "top_left": top_left,
    }
    green_xy = (green[0], green[1])
    return HingeGeometry(
        step_sha256=hashlib.sha256(raw).hexdigest(),
        red_xy_mm=red,
        green_xy_mm=green_xy,
        next_holes_xy_mm=tuple(
            (name, (point[0], point[1])) for name, point in points.items()
        ),
        plate_z_mm=sum(point[2] for point in top) / 4,
        red_green_distance_mm=math.dist(red, green_xy),
        assembly_table_xy_mm=ASSEMBLY_TABLE_XY_MM,
        assembly_step_sha256=ASSEMBLY_STEP_SHA256,
    )


if __name__ == "__main__":
    print(json.dumps(measure_top_hinge().as_dict(), ensure_ascii=False, indent=2))
