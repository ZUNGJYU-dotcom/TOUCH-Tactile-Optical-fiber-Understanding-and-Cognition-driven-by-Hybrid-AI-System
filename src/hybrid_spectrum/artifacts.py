"""Fail-closed verification for persisted hybrid-spectrum model bundles."""

from __future__ import annotations

import json
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib

from .dataset import sha256_file


SUPPORTED_BUNDLE_SCHEMA_VERSION = "hybrid_spectral_model_bundle_v2"


def _feature_schema_sha256(columns: list[str]) -> str:
    return hashlib.sha256("\n".join(columns).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ModelArtifactVerification:
    verified: bool
    blockers: tuple[str, ...]
    manifest_path: str
    model_path: str
    model_relative_path: str | None
    model_sha256: str | None
    expected_model_sha256: str | None
    config_sha256: str | None
    expected_config_sha256: str | None
    feature_schema_sha256: str | None
    model_id: str | None


def _normalized_relative_path(path: Path, root: Path) -> str | None:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return None


def verify_model_artifact(
    model_path: Path,
    manifest_path: Path,
    config_path: Path,
) -> tuple[ModelArtifactVerification, dict[str, Any]]:
    """Verify bytes and manifest contract without deserializing the model."""

    model_path = Path(model_path)
    manifest_path = Path(manifest_path)
    config_path = Path(config_path)
    blockers: list[str] = []
    manifest: dict[str, Any] = {}
    if not model_path.is_file():
        blockers.append("model_artifact_missing")
    if not manifest_path.is_file():
        blockers.append("model_manifest_missing")
    if not config_path.is_file():
        blockers.append("runtime_config_missing")
    if blockers:
        return (
            ModelArtifactVerification(
                verified=False,
                blockers=tuple(blockers),
                manifest_path=str(manifest_path.resolve()),
                model_path=str(model_path.resolve()),
                model_relative_path=None,
                model_sha256=None,
                expected_model_sha256=None,
                config_sha256=None,
                expected_config_sha256=None,
                feature_schema_sha256=None,
                model_id=None,
            ),
            manifest,
        )

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        blockers.append("model_manifest_unreadable")
        manifest = {}
    if manifest.get("manifest_schema_version") != "hybrid_spectral_artifact_manifest_v1":
        blockers.append("unsupported_model_manifest_schema")
    if manifest.get("bundle_schema_version") != SUPPORTED_BUNDLE_SCHEMA_VERSION:
        blockers.append("unsupported_model_bundle_schema_manifest")

    relative_model_path = _normalized_relative_path(model_path, manifest_path.parent)
    if relative_model_path is None:
        blockers.append("model_outside_manifest_root")
    artifacts = manifest.get("artifacts", [])
    artifact = next(
        (
            item
            for item in artifacts
            if str(item.get("relative_path", "")).replace("\\", "/")
            == relative_model_path
        ),
        None,
    )
    if artifact is None:
        blockers.append("model_not_listed_in_manifest")

    model_hash = sha256_file(model_path)
    config_hash = sha256_file(config_path)
    expected_model_hash = str(artifact.get("sha256")) if artifact else None
    expected_config_hash = manifest.get("config_sha256")
    if not expected_model_hash or model_hash != expected_model_hash:
        blockers.append("model_artifact_sha256_mismatch")
    if artifact is not None:
        try:
            expected_size = int(artifact.get("size_bytes"))
        except (TypeError, ValueError):
            blockers.append("model_artifact_size_missing")
        else:
            if model_path.stat().st_size != expected_size:
                blockers.append("model_artifact_size_mismatch")
    if not expected_config_hash or config_hash != expected_config_hash:
        blockers.append("model_manifest_config_sha256_mismatch")
    feature_schema_hash = manifest.get("feature_schema_sha256")
    if not feature_schema_hash:
        blockers.append("model_manifest_feature_schema_missing")

    blockers = list(dict.fromkeys(blockers))
    return (
        ModelArtifactVerification(
            verified=not blockers,
            blockers=tuple(blockers),
            manifest_path=str(manifest_path.resolve()),
            model_path=str(model_path.resolve()),
            model_relative_path=relative_model_path,
            model_sha256=model_hash,
            expected_model_sha256=expected_model_hash,
            config_sha256=config_hash,
            expected_config_sha256=(
                str(expected_config_hash) if expected_config_hash is not None else None
            ),
            feature_schema_sha256=(
                str(feature_schema_hash) if feature_schema_hash is not None else None
            ),
            model_id=str(artifact.get("model_id")) if artifact else None,
        ),
        manifest,
    )


def load_verified_model_bundle(
    model_path: Path,
    manifest_path: Path,
    config_path: Path,
) -> tuple[dict[str, Any], ModelArtifactVerification]:
    """Deserialize only after the persisted artifact passes byte verification."""

    verification, manifest = verify_model_artifact(
        model_path,
        manifest_path,
        config_path,
    )
    if not verification.verified:
        raise RuntimeError(
            "Model artifact verification failed before deserialization: "
            + ", ".join(verification.blockers)
        )
    bundle = joblib.load(model_path)
    if not isinstance(bundle, dict):
        raise RuntimeError("Verified model bundle is not a dictionary")
    metadata = bundle.get("metadata", {}) or {}
    post_load_blockers: list[str] = []
    if bundle.get("model") is None:
        post_load_blockers.append("model_object_missing")
    if metadata.get("model_id") != verification.model_id:
        post_load_blockers.append("model_id_manifest_mismatch")
    if metadata.get("bundle_schema_version") != manifest.get("bundle_schema_version"):
        post_load_blockers.append("model_bundle_schema_manifest_mismatch")
    if metadata.get("config_sha256") != verification.config_sha256:
        post_load_blockers.append("model_metadata_config_sha256_mismatch")
    if metadata.get("feature_schema_sha256") != manifest.get("feature_schema_sha256"):
        post_load_blockers.append("model_metadata_feature_schema_mismatch")
    root_feature_columns = bundle.get("feature_columns")
    metadata_feature_columns = metadata.get("feature_columns")
    if not isinstance(root_feature_columns, list) or not root_feature_columns:
        post_load_blockers.append("model_feature_columns_missing")
    elif root_feature_columns != metadata_feature_columns:
        post_load_blockers.append("model_feature_columns_metadata_mismatch")
    elif _feature_schema_sha256([str(value) for value in root_feature_columns]) != manifest.get(
        "feature_schema_sha256"
    ):
        post_load_blockers.append("model_feature_columns_schema_hash_mismatch")
    if bundle.get("labels") != metadata.get("labels"):
        post_load_blockers.append("model_labels_metadata_mismatch")
    if bundle.get("evaluation_validity") != metadata.get("evaluation_validity"):
        post_load_blockers.append("model_evaluation_validity_metadata_mismatch")
    if bool(bundle.get("real_model_deployable", False)) != bool(
        metadata.get("real_model_deployable", False)
    ):
        post_load_blockers.append("model_deployment_flag_metadata_mismatch")
    if post_load_blockers:
        raise RuntimeError(
            "Verified model metadata contract failed: " + ", ".join(post_load_blockers)
        )
    return bundle, verification
