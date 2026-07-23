"""Build a Robot Nano Hand GLB with four fingertip sensor recesses.

The source hand is a single merged indexed mesh.  This utility removes only the
outward-facing triangles inside the configured fingertip footprints and keeps
all vertex/color buffers unchanged.  Runtime sensor inserts then occupy those
openings and share the existing thumb sensor deformation geometry.
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

DEFAULT_FINGERS: dict[str, dict[str, Any]] = {
    "index": {
        "center_model": [79.087, 62.010, -39.369],
        "longitudinal_axis_model": [0.0, 1.0, 0.0],
        "outward_normal_model": [-0.274, -0.275, -0.922],
        "slot_length_mm": 15.0,
        "slot_width_mm": 9.0,
        "slot_depth_tolerance_mm": 2.2,
    },
    "middle": {
        "center_model": [55.573, 82.611, -41.918],
        "longitudinal_axis_model": [-0.392, 0.920, 0.0],
        "outward_normal_model": [-0.217, -0.559, -0.800],
        "slot_length_mm": 15.0,
        "slot_width_mm": 9.0,
        "slot_depth_tolerance_mm": 2.4,
    },
    "ring": {
        "center_model": [23.129, 101.826, -19.171],
        "longitudinal_axis_model": [-0.076, 0.997, 0.0],
        "outward_normal_model": [-0.072, -0.196, -0.978],
        "slot_length_mm": 15.0,
        "slot_width_mm": 9.0,
        "slot_depth_tolerance_mm": 2.2,
    },
    "little": {
        "center_model": [-9.320, 92.498, -24.449],
        "longitudinal_axis_model": [-0.087, 0.996, 0.0],
        "outward_normal_model": [0.027, -0.241, -0.970],
        "slot_length_mm": 15.0,
        "slot_width_mm": 9.0,
        "slot_depth_tolerance_mm": 2.4,
    },
}


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
    return np.frombuffer(binary, dtype=dtype, count=count, offset=byte_offset).reshape(-1, width)


def normalized(vector: np.ndarray) -> np.ndarray:
    length = float(np.linalg.norm(vector))
    if length <= 1e-9:
        raise ValueError("Cannot normalize a zero-length vector")
    return vector / length


def slot_basis(config: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    outward = normalized(np.asarray(config["outward_normal_model"], dtype=np.float64))
    longitudinal = np.asarray(config["longitudinal_axis_model"], dtype=np.float64)
    longitudinal = longitudinal - outward * float(np.dot(longitudinal, outward))
    longitudinal = normalized(longitudinal)
    transverse = normalized(np.cross(outward, longitudinal))
    return longitudinal, transverse, outward


def load_finger_config(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is not None:
        try:
            import yaml  # type: ignore

            loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            configured = loaded.get("finger_sensor_array", {}).get("fingers", {})
            selected = {
                finger_id: dict(values)
                for finger_id, values in configured.items()
                if finger_id != "thumb" and values.get("enabled", True)
            }
            if selected:
                return selected
        except (ImportError, OSError, ValueError, TypeError):
            pass
    return {finger_id: dict(values) for finger_id, values in DEFAULT_FINGERS.items()}


def carve_fingertip_slots(
    document: dict[str, Any],
    binary: bytearray,
    fingers: dict[str, dict[str, Any]],
    carve_scale: float,
    normal_threshold: float,
) -> dict[str, Any]:
    primitive = document["meshes"][0]["primitives"][0]
    index_accessor_index = int(primitive["indices"])
    position_accessor_index = int(primitive["attributes"]["POSITION"])
    index_accessor = document["accessors"][index_accessor_index]
    if int(index_accessor["componentType"]) != 5125:
        raise ValueError("Expected UNSIGNED_INT indices")

    indices = accessor_array(document, binary, index_accessor_index, "<u4", 1).reshape(-1, 3)
    positions = accessor_array(document, binary, position_accessor_index, "<f4", 3)
    triangles = positions[indices]
    centroids = triangles.mean(axis=1)
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    lengths = np.linalg.norm(normals, axis=1)
    valid_normals = lengths > 1e-9
    normals[valid_normals] /= lengths[valid_normals, None]

    removed = np.zeros(indices.shape[0], dtype=bool)
    slot_audit: dict[str, Any] = {}
    for finger_id, config in fingers.items():
        center = np.asarray(config["center_model"], dtype=np.float64)
        longitudinal, transverse, outward = slot_basis(config)
        delta = centroids - center
        longitudinal_distance = delta @ longitudinal
        transverse_distance = delta @ transverse
        plane_distance = delta @ outward
        normal_alignment = normals @ outward
        half_length = float(config.get("slot_length_mm", 15.0)) * 0.5 * carve_scale
        half_width = float(config.get("slot_width_mm", 9.0)) * 0.5 * carve_scale
        depth_tolerance = float(config.get("slot_depth_tolerance_mm", 2.4))
        ellipse = (
            np.square(longitudinal_distance / max(half_length, 1e-6))
            + np.square(transverse_distance / max(half_width, 1e-6))
        )
        finger_removed = (
            (~removed)
            & (ellipse <= 1.0)
            & (np.abs(plane_distance) <= depth_tolerance)
            & (normal_alignment >= normal_threshold)
        )
        removed |= finger_removed
        slot_audit[finger_id] = {
            "removed_triangle_count": int(finger_removed.sum()),
            "center_model": center.round(6).tolist(),
            "longitudinal_axis_model": longitudinal.round(8).tolist(),
            "transverse_axis_model": transverse.round(8).tolist(),
            "outward_normal_model": outward.round(8).tolist(),
            "carved_length_mm": round(half_length * 2, 4),
            "carved_width_mm": round(half_width * 2, 4),
            "depth_tolerance_mm": depth_tolerance,
        }

    filtered = np.ascontiguousarray(indices[~removed].reshape(-1), dtype="<u4")
    index_view = document["bufferViews"][index_accessor["bufferView"]]
    index_offset = int(index_view.get("byteOffset", 0)) + int(index_accessor.get("byteOffset", 0))
    original_byte_length = int(index_view["byteLength"])
    filtered_bytes = filtered.tobytes()
    if len(filtered_bytes) > original_byte_length:
        raise ValueError("Filtered index buffer unexpectedly exceeds the original buffer")
    binary[index_offset : index_offset + len(filtered_bytes)] = filtered_bytes
    binary[index_offset + len(filtered_bytes) : index_offset + original_byte_length] = b"\0" * (
        original_byte_length - len(filtered_bytes)
    )
    index_accessor["count"] = int(filtered.size)
    index_accessor["min"] = [int(filtered.min()) if filtered.size else 0]
    index_accessor["max"] = [int(filtered.max()) if filtered.size else 0]
    index_view["byteLength"] = len(filtered_bytes)

    return {
        "source_triangle_count": int(indices.shape[0]),
        "output_triangle_count": int(filtered.size // 3),
        "removed_triangle_count": int(removed.sum()),
        "slots": slot_audit,
    }


def write_glb(path: Path, document: dict[str, Any], binary: bytearray) -> None:
    document["buffers"][0]["byteLength"] = len(binary)
    json_bytes = json.dumps(document, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    json_bytes += b" " * ((4 - len(json_bytes) % 4) % 4)
    binary_bytes = bytes(binary)
    binary_bytes += b"\0" * ((4 - len(binary_bytes) % 4) % 4)
    total_length = 12 + 8 + len(json_bytes) + 8 + len(binary_bytes)
    payload = bytearray(struct.pack("<4sII", GLB_MAGIC, 2, total_length))
    payload.extend(struct.pack("<II", len(json_bytes), JSON_CHUNK))
    payload.extend(json_bytes)
    payload.extend(struct.pack("<II", len(binary_bytes), BIN_CHUNK))
    payload.extend(binary_bytes)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def main() -> int:
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=project_root / "bayspec_wavelength_shift_app/frontend/assets/models/robot_nano_hand_body.glb",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=project_root / "bayspec_wavelength_shift_app/frontend/assets/models/robot_nano_hand_sensorized.glb",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=project_root / "config/thumb_holder_scene.yaml",
    )
    parser.add_argument("--carve-scale", type=float, default=0.96)
    parser.add_argument("--normal-threshold", type=float, default=0.20)
    parser.add_argument("--audit", type=Path, default=None)
    args = parser.parse_args()

    document, binary = parse_glb(args.input)
    fingers = load_finger_config(args.config)
    audit = carve_fingertip_slots(
        document,
        binary,
        fingers,
        carve_scale=max(0.70, min(1.10, args.carve_scale)),
        normal_threshold=max(-1.0, min(1.0, args.normal_threshold)),
    )
    write_glb(args.output, document, binary)
    audit.update(
        {
            "source": str(args.input),
            "output": str(args.output),
            "config": str(args.config),
            "geometry_status": "four_fingertip_recesses_carved",
        }
    )
    audit_path = args.audit or args.output.with_suffix(".audit.json")
    audit_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(audit, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
