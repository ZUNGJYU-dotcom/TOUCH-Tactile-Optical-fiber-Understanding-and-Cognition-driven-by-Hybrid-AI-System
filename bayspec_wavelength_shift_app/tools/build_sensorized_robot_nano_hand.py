"""Build the TOUCH hand GLB with four real SolidWorks-cut fingertip parts.

Each original closed fingertip component is replaced by its registered CAD
counterpart. The replacement geometry contains a local planar contact seat and
a closed 0.8 mm capsule recess; no triangle-only surface deletion is used.
"""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path
from typing import Any

import numpy as np


GLB_MAGIC = b"glTF"
JSON_CHUNK = 0x4E4F534A
BIN_CHUNK = 0x004E4942
STL_RECORD_DTYPE = np.dtype(
    [
        ("normal", "<f4", (3,)),
        ("vertices", "<f4", (3, 3)),
        ("attribute", "<u2"),
    ]
)


def parse_glb(path: Path) -> tuple[dict[str, Any], bytearray]:
    payload = path.read_bytes()
    magic, version, total_length = struct.unpack_from("<4sII", payload, 0)
    if magic != GLB_MAGIC or version != 2 or total_length != len(payload):
        raise ValueError(f"Unsupported or corrupt GLB: {path}")

    offset = 12
    document: dict[str, Any] | None = None
    binary: bytearray | None = None
    while offset < total_length:
        chunk_length, chunk_type = struct.unpack_from("<II", payload, offset)
        offset += 8
        chunk = payload[offset : offset + chunk_length]
        offset += chunk_length
        if chunk_type == JSON_CHUNK:
            document = json.loads(chunk.decode("utf-8").rstrip(" \0"))
        elif chunk_type == BIN_CHUNK:
            binary = bytearray(chunk)
    if document is None or binary is None:
        raise ValueError("GLB must contain JSON and BIN chunks")
    return document, binary


def accessor_array(
    document: dict[str, Any],
    binary: bytearray,
    accessor_index: int,
    dtype: str,
    width: int,
) -> np.ndarray:
    accessor = document["accessors"][accessor_index]
    view = document["bufferViews"][accessor["bufferView"]]
    byte_offset = int(view.get("byteOffset", 0)) + int(accessor.get("byteOffset", 0))
    count = int(accessor["count"]) * width
    return np.frombuffer(binary, dtype=dtype, count=count, offset=byte_offset).reshape(
        -1, width
    )


def load_binary_stl(path: Path) -> np.ndarray:
    with path.open("rb") as handle:
        handle.read(80)
        triangle_count = struct.unpack("<I", handle.read(4))[0]
        records = np.fromfile(handle, dtype=STL_RECORD_DTYPE, count=triangle_count)
    if len(records) != triangle_count:
        raise ValueError(
            f"{path}: expected {triangle_count} triangles, got {len(records)}"
        )
    return records["vertices"].astype(np.float64)


def connected_index_components(indices: np.ndarray) -> list[np.ndarray]:
    parent = np.arange(len(indices), dtype=np.int32)
    rank = np.zeros(len(indices), dtype=np.int8)

    def find(item: int) -> int:
        root = item
        while parent[root] != root:
            root = int(parent[root])
        while parent[item] != item:
            next_item = int(parent[item])
            parent[item] = root
            item = next_item
        return root

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root == right_root:
            return
        if rank[left_root] < rank[right_root]:
            left_root, right_root = right_root, left_root
        parent[right_root] = left_root
        if rank[left_root] == rank[right_root]:
            rank[left_root] += 1

    vertex_owner: dict[int, int] = {}
    for triangle_index, triangle in enumerate(indices):
        for vertex_index in triangle:
            key = int(vertex_index)
            previous = vertex_owner.get(key)
            if previous is None:
                vertex_owner[key] = triangle_index
            else:
                union(triangle_index, previous)

    groups: dict[int, list[int]] = {}
    for triangle_index in range(len(indices)):
        groups.setdefault(find(triangle_index), []).append(triangle_index)
    return [np.asarray(group, dtype=np.int32) for group in groups.values()]


def connected_stl_components(
    triangles: np.ndarray,
    tolerance: float = 1e-5,
) -> list[np.ndarray]:
    parent = np.arange(len(triangles), dtype=np.int32)
    rank = np.zeros(len(triangles), dtype=np.int8)

    def find(item: int) -> int:
        root = item
        while parent[root] != root:
            root = int(parent[root])
        while parent[item] != item:
            next_item = int(parent[item])
            parent[item] = root
            item = next_item
        return root

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root == right_root:
            return
        if rank[left_root] < rank[right_root]:
            left_root, right_root = right_root, left_root
        parent[right_root] = left_root
        if rank[left_root] == rank[right_root]:
            rank[left_root] += 1

    vertex_owner: dict[tuple[int, int, int], int] = {}
    quantized = np.rint(triangles / tolerance).astype(np.int64)
    for triangle_index, triangle in enumerate(quantized):
        for vertex in triangle:
            key = tuple(int(value) for value in vertex)
            previous = vertex_owner.get(key)
            if previous is None:
                vertex_owner[key] = triangle_index
            else:
                union(triangle_index, previous)

    groups: dict[int, list[int]] = {}
    for triangle_index in range(len(triangles)):
        groups.setdefault(find(triangle_index), []).append(triangle_index)
    return [np.asarray(group, dtype=np.int32) for group in groups.values()]


def component_bounds(
    triangles: np.ndarray,
    component: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    points = triangles[component].reshape(-1, 3)
    minimum = points.min(axis=0)
    maximum = points.max(axis=0)
    return 0.5 * (minimum + maximum), maximum - minimum


def select_hand_component(
    triangles: np.ndarray,
    components: list[np.ndarray],
    expected: dict[str, Any],
    unavailable: set[int],
) -> tuple[int, float, np.ndarray, np.ndarray]:
    expected_center = np.asarray(expected["center_model"], dtype=np.float64)
    expected_extent = np.asarray(expected["extent_model"], dtype=np.float64)
    expected_triangles = int(expected["triangle_count"])
    best: tuple[float, int, np.ndarray, np.ndarray] | None = None
    for component_id, component in enumerate(components):
        if component_id in unavailable:
            continue
        center, extent = component_bounds(triangles, component)
        center_cost = float(np.linalg.norm(center - expected_center))
        extent_cost = float(np.linalg.norm(extent - expected_extent))
        triangle_cost = abs(len(component) - expected_triangles) / max(
            1.0, expected_triangles
        )
        cost = center_cost + 0.4 * extent_cost + 4.0 * triangle_cost
        if best is None or cost < best[0]:
            best = (cost, component_id, center, extent)
    if best is None or best[0] > 3.0:
        raise RuntimeError(
            "Could not locate the expected closed fingertip component "
            f"(best cost={best[0] if best else float('nan'):.3f})."
        )
    return best[1], best[0], best[2], best[3]


def select_trial_fingertip(
    triangles: np.ndarray,
    export_translation: np.ndarray,
    expected_local_center: np.ndarray,
) -> tuple[np.ndarray, int]:
    aligned = triangles - export_translation
    components = connected_stl_components(aligned)
    component_id = min(
        range(len(components)),
        key=lambda index: float(
            np.linalg.norm(
                component_bounds(aligned, components[index])[0] - expected_local_center
            )
        ),
    )
    center, _ = component_bounds(aligned, components[component_id])
    if float(np.linalg.norm(center - expected_local_center)) > 0.25:
        raise RuntimeError(
            "The SolidWorks trial STL fingertip component could not be identified."
        )
    return aligned[components[component_id]], component_id


def transform_triangles(
    triangles: np.ndarray,
    registration: dict[str, Any],
) -> np.ndarray:
    scale = float(registration["scale"])
    rotation = np.asarray(registration["rotation"], dtype=np.float64)
    translation = np.asarray(registration["translation"], dtype=np.float64)
    return scale * (triangles @ rotation.T) + translation


def deduplicate_triangles(
    triangles: np.ndarray,
    decimals: int = 6,
) -> tuple[np.ndarray, np.ndarray]:
    rounded = np.round(triangles.reshape(-1, 3), decimals=decimals)
    vertices, inverse = np.unique(rounded, axis=0, return_inverse=True)
    return vertices.astype("<f4"), inverse.astype("<u4").reshape(-1, 3)


def pad4(payload: bytes) -> bytes:
    return payload + b"\0" * ((4 - len(payload) % 4) % 4)


def rebuild_mesh_binary(
    document: dict[str, Any],
    indices: np.ndarray,
    positions: np.ndarray,
    colors: np.ndarray,
) -> bytearray:
    primitive = document["meshes"][0]["primitives"][0]
    index_accessor = document["accessors"][int(primitive["indices"])]
    position_accessor = document["accessors"][
        int(primitive["attributes"]["POSITION"])
    ]
    color_accessor = document["accessors"][int(primitive["attributes"]["COLOR_0"])]
    index_view = document["bufferViews"][int(index_accessor["bufferView"])]
    position_view = document["bufferViews"][int(position_accessor["bufferView"])]
    color_view = document["bufferViews"][int(color_accessor["bufferView"])]

    index_bytes = np.ascontiguousarray(indices.reshape(-1), dtype="<u4").tobytes()
    position_bytes = np.ascontiguousarray(positions, dtype="<f4").tobytes()
    color_bytes = np.ascontiguousarray(colors, dtype="<u1").tobytes()
    index_payload = pad4(index_bytes)
    position_payload = pad4(position_bytes)
    color_payload = pad4(color_bytes)

    index_view["byteOffset"] = 0
    index_view["byteLength"] = len(index_bytes)
    position_view["byteOffset"] = len(index_payload)
    position_view["byteLength"] = len(position_bytes)
    color_view["byteOffset"] = len(index_payload) + len(position_payload)
    color_view["byteLength"] = len(color_bytes)

    flat_indices = indices.reshape(-1)
    index_accessor["count"] = int(flat_indices.size)
    index_accessor["min"] = [int(flat_indices.min())]
    index_accessor["max"] = [int(flat_indices.max())]
    position_accessor["count"] = int(len(positions))
    position_accessor["min"] = positions.min(axis=0).astype(float).tolist()
    position_accessor["max"] = positions.max(axis=0).astype(float).tolist()
    color_accessor["count"] = int(len(colors))

    binary = bytearray(index_payload + position_payload + color_payload)
    document["buffers"][0]["byteLength"] = len(binary)
    return binary


def write_glb(path: Path, document: dict[str, Any], binary: bytearray) -> None:
    document["buffers"][0]["byteLength"] = len(binary)
    json_bytes = json.dumps(
        document,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    json_bytes += b" " * ((4 - len(json_bytes) % 4) % 4)
    binary_bytes = pad4(bytes(binary))
    total_length = 12 + 8 + len(json_bytes) + 8 + len(binary_bytes)
    payload = bytearray(struct.pack("<4sII", GLB_MAGIC, 2, total_length))
    payload.extend(struct.pack("<II", len(json_bytes), JSON_CHUNK))
    payload.extend(json_bytes)
    payload.extend(struct.pack("<II", len(binary_bytes), BIN_CHUNK))
    payload.extend(binary_bytes)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def build_sensorized_hand(
    document: dict[str, Any],
    binary: bytearray,
    manifest: dict[str, Any],
    slot_directory: Path,
) -> tuple[bytearray, dict[str, Any]]:
    primitive = document["meshes"][0]["primitives"][0]
    index_accessor_index = int(primitive["indices"])
    position_accessor_index = int(primitive["attributes"]["POSITION"])
    color_accessor_index = int(primitive["attributes"]["COLOR_0"])
    index_accessor = document["accessors"][index_accessor_index]
    color_accessor = document["accessors"][color_accessor_index]
    if int(index_accessor["componentType"]) != 5125:
        raise ValueError("Expected UNSIGNED_INT hand indices")
    if (
        int(color_accessor["componentType"]) != 5121
        or color_accessor.get("type") != "VEC4"
    ):
        raise ValueError("Expected normalized four-channel unsigned-byte colors")

    indices = accessor_array(
        document,
        binary,
        index_accessor_index,
        "<u4",
        1,
    ).reshape(-1, 3)
    positions = accessor_array(
        document,
        binary,
        position_accessor_index,
        "<f4",
        3,
    ).copy()
    colors = accessor_array(
        document,
        binary,
        color_accessor_index,
        "<u1",
        4,
    ).copy()
    triangles = positions[indices].astype(np.float64)
    components = connected_index_components(indices)

    removed = np.zeros(len(indices), dtype=bool)
    used_components: set[int] = set()
    appended_positions: list[np.ndarray] = []
    appended_indices: list[np.ndarray] = []
    slot_audit: dict[str, Any] = {}
    next_vertex = len(positions)

    for finger_id, config in manifest["fingers"].items():
        component_id, match_cost, target_center, target_extent = select_hand_component(
            triangles,
            components,
            config["target_hand_component"],
            used_components,
        )
        used_components.add(component_id)
        target_component = components[component_id]
        removed[target_component] = True

        stl_path = slot_directory / str(config["stl"])
        trial_triangles = load_binary_stl(stl_path)
        local_tip, trial_component_id = select_trial_fingertip(
            trial_triangles,
            np.asarray(config["solidworks_export_translation_mm"], dtype=np.float64),
            np.asarray(config["local_fingertip_center_mm"], dtype=np.float64),
        )
        mapped_tip = transform_triangles(local_tip, config["registration"])
        new_vertices, new_indices = deduplicate_triangles(mapped_tip)
        appended_positions.append(new_vertices)
        appended_indices.append(new_indices + next_vertex)
        next_vertex += len(new_vertices)

        registration = config["registration"]
        rotation = np.asarray(registration["rotation"], dtype=np.float64)
        scale = float(registration["scale"])
        local_slot_center = np.asarray(config["local_slot_center_mm"], dtype=np.float64)
        slot_center = (
            scale * (local_slot_center @ rotation.T)
            + np.asarray(registration["translation"], dtype=np.float64)
        )
        local_longitudinal = np.asarray(
            config.get("local_longitudinal_axis", [0.0, 1.0, 0.0]),
            dtype=np.float64,
        )
        local_outward = np.asarray(
            config.get("local_outward_normal", [0.0, 0.0, 1.0]),
            dtype=np.float64,
        )
        longitudinal = rotation @ local_longitudinal
        longitudinal /= np.linalg.norm(longitudinal)
        outward = rotation @ local_outward
        outward /= np.linalg.norm(outward)
        slot_audit[finger_id] = {
            "replacement_method": "closed_solidworks_cad_component",
            "hand_component_id": component_id,
            "hand_component_match_cost": match_cost,
            "removed_triangle_count": int(len(target_component)),
            "inserted_triangle_count": int(len(new_indices)),
            "inserted_vertex_count": int(len(new_vertices)),
            "trial_stl_component_id": int(trial_component_id),
            "target_component_center_model": target_center.round(6).tolist(),
            "target_component_extent_model": target_extent.round(6).tolist(),
            "registration_scale": scale,
            "registration_trimmed_rmse_mm": float(
                registration["trimmed_rmse_mm"]
            ),
            "slot_center_model": slot_center.round(6).tolist(),
            "longitudinal_axis_model": longitudinal.round(8).tolist(),
            "outward_normal_model": outward.round(8).tolist(),
            "slot_length_model_mm": float(config["slot_length_mm"]) * scale,
            "slot_width_model_mm": float(config["slot_width_mm"]) * scale,
            "slot_depth_model_mm": float(manifest["slot_depth_mm"]) * scale,
            "flat_plane_local_z_mm": float(config["flat_plane_z_mm"]),
            "slot_floor_local_z_mm": float(config["slot_floor_z_mm"]),
            "local_longitudinal_axis": local_longitudinal.round(8).tolist(),
            "local_outward_normal": local_outward.round(8).tolist(),
            "plane_slope_x_dz_dx": float(config["plane_slope_x_dz_dx"]),
            "plane_slope_y_dz_dy": float(config["plane_slope_y_dz_dy"]),
        }

    kept_indices = indices[~removed]
    output_positions = np.vstack([positions, *appended_positions]).astype("<f4")
    output_indices = np.vstack([kept_indices, *appended_indices]).astype("<u4")
    base_color = np.median(colors.astype(np.float64), axis=0).round().astype("<u1")
    added_vertex_count = len(output_positions) - len(positions)
    output_colors = np.vstack(
        [
            colors,
            np.repeat(base_color[None, :], added_vertex_count, axis=0),
        ]
    ).astype("<u1")
    output_binary = rebuild_mesh_binary(
        document,
        output_indices,
        output_positions,
        output_colors,
    )
    audit = {
        "source_triangle_count": int(len(indices)),
        "output_triangle_count": int(len(output_indices)),
        "removed_triangle_count": int(removed.sum()),
        "inserted_triangle_count": int(
            sum(len(values) for values in appended_indices)
        ),
        "source_vertex_count": int(len(positions)),
        "output_vertex_count": int(len(output_positions)),
        "slots": slot_audit,
        "geometry_status": "solidworks_tilted_flush_max_area_recesses_integrated",
        "geometry_semantics": (
            "Four closed fingertip CAD components with enlarged fitted tangent "
            "seats and equal-depth 0.8 mm sensor pockets; existing modified "
            "thumb preserved."
        ),
        "manufacturing_status": "provisional_trial_dimensions",
    }
    return output_binary, audit


def main() -> int:
    project_root = Path(__file__).resolve().parents[2]
    model_directory = (
        project_root
        / "bayspec_wavelength_shift_app"
        / "frontend"
        / "assets"
        / "models"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=model_directory / "robot_nano_hand_body.glb",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=model_directory / "robot_nano_hand_sensorized.glb",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=model_directory / "four_finger_cad_slots" / "manifest.json",
    )
    parser.add_argument(
        "--slot-directory",
        type=Path,
        default=model_directory / "four_finger_cad_slots",
    )
    parser.add_argument("--audit", type=Path, default=None)
    args = parser.parse_args()

    document, binary = parse_glb(args.input)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    output_binary, audit = build_sensorized_hand(
        document,
        binary,
        manifest,
        args.slot_directory,
    )
    write_glb(args.output, document, output_binary)

    def audit_path_value(path: Path) -> str:
        try:
            return path.resolve().relative_to(project_root).as_posix()
        except ValueError:
            return str(path.resolve())

    audit.update(
        {
            "source": audit_path_value(args.input),
            "output": audit_path_value(args.output),
            "manifest": audit_path_value(args.manifest),
        }
    )
    audit_path = args.audit or args.output.with_suffix(".audit.json")
    audit_path.write_text(
        json.dumps(audit, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(audit, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
