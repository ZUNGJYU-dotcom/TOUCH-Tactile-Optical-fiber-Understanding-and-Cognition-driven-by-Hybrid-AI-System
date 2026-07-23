import * as THREE from "three";

const DEFAULT_THUMB_COLOR = "#d9e1e7";
const DEFAULT_SLOT_COLOR = "#8fd0dc";
const DEFAULT_SLOT_LINE = "#37a8c7";

function applyMaterialStyle(object, options = {}) {
  const opacity = Number(options.opacity ?? 0.34);
  const color = options.color || DEFAULT_THUMB_COLOR;
  const wireframe = Boolean(options.wireframe);
  const depthWrite = options.depthWrite ?? opacity > 0.8;
  const roughness = Number(options.roughness ?? 0.86);
  const metalness = Number(options.metalness ?? 0);
  const side = options.side || THREE.FrontSide;
  object.traverse((child) => {
    if (!child.isMesh) return;
    if (child.geometry && !child.geometry.getAttribute("normal")) {
      child.geometry.computeVertexNormals();
    }
    child.material = new THREE.MeshStandardMaterial({
      color,
      transparent: true,
      opacity,
      roughness,
      metalness,
      wireframe,
      depthWrite,
      side,
    });
    child.castShadow = false;
    child.receiveShadow = true;
  });
}

function applyGeometryCentering(geometry) {
  geometry.computeBoundingBox();
  geometry.computeVertexNormals();
  const box = geometry.boundingBox;
  if (!box) return;
  const center = new THREE.Vector3();
  const size = new THREE.Vector3();
  box.getCenter(center);
  box.getSize(size);
  const maxAxis = Math.max(size.x, size.y, size.z, 1);
  geometry.translate(-center.x, -center.y, -center.z);
  geometry.scale(5.4 / maxAxis, 5.4 / maxAxis, 5.4 / maxAxis);
}

function meshFromStlGeometry(geometry, options = {}) {
  applyGeometryCentering(geometry);
  const material = new THREE.MeshStandardMaterial({
    color: options.color || DEFAULT_THUMB_COLOR,
    transparent: true,
    opacity: Number(options.opacity ?? 0.42),
    roughness: 0.82,
    metalness: 0,
    side: THREE.DoubleSide,
    wireframe: Boolean(options.wireframe),
    depthWrite: Number(options.opacity ?? 0.42) > 0.8,
  });
  const mesh = new THREE.Mesh(geometry, material);
  mesh.name = "thumb_holder_stl_mesh";
  mesh.castShadow = false;
  mesh.receiveShadow = true;
  const group = new THREE.Group();
  group.name = "thumb_holder_stl";
  group.add(mesh);
  group.userData.modelLoadStatus = "stl_loaded";
  return group;
}

export function createFallbackThumbHolder(config = {}) {
  const visual = config.visual_style || {};
  const transform = config.thumb_model_transform || {};
  const group = new THREE.Group();
  group.name = "fallback_thumb_holder";

  const thumbMaterial = new THREE.MeshStandardMaterial({
    color: visual.thumb_material_color || DEFAULT_THUMB_COLOR,
    transparent: true,
    opacity: Number(transform.opacity ?? 0.34),
    roughness: 0.88,
    metalness: 0,
  });
  const slotMaterial = new THREE.MeshStandardMaterial({
    color: visual.thumb_slot_color || DEFAULT_SLOT_COLOR,
    transparent: true,
    opacity: 0.30,
    roughness: 0.72,
    metalness: 0,
    side: THREE.DoubleSide,
  });
  const lineMaterial = new THREE.LineBasicMaterial({
    color: visual.slot_outline_color || DEFAULT_SLOT_LINE,
    transparent: true,
    opacity: 0.86,
  });

  const pad = new THREE.Mesh(new THREE.CapsuleGeometry(1.05, 4.15, 18, 34), thumbMaterial);
  pad.name = "thumb_pad_body";
  pad.rotation.z = Math.PI / 2;
  pad.scale.set(1.55, 0.72, 0.36);
  pad.position.set(0, -0.28, 0);
  group.add(pad);

  const base = new THREE.Mesh(new THREE.BoxGeometry(5.45, 0.26, 2.85), thumbMaterial);
  base.name = "thumb_holder_base";
  base.position.set(0, -0.78, 0);
  group.add(base);

  const slot = new THREE.Mesh(new THREE.BoxGeometry(4.25, 0.055, 2.85), slotMaterial);
  slot.name = "sensor_slot_floor";
  slot.position.set(0, -0.065, 0.015);
  group.add(slot);

  const slotFrame = new THREE.Group();
  slotFrame.name = "sensor_slot_outline";
  const w = 4.48;
  const d = 3.05;
  const y = -0.028;
  const points = [
    new THREE.Vector3(-w / 2, y, -d / 2),
    new THREE.Vector3(w / 2, y, -d / 2),
    new THREE.Vector3(w / 2, y, d / 2),
    new THREE.Vector3(-w / 2, y, d / 2),
    new THREE.Vector3(-w / 2, y, -d / 2),
  ];
  slotFrame.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(points), lineMaterial));
  group.add(slotFrame);

  group.userData.modelLoadStatus = "fallback_placeholder";
  group.userData.sensorSlotLocalPosition = { x: 0, y: 0.06, z: 0.02 };
  group.userData.note = "Real thumb model not loaded. Replace with thumb_holder.glb or thumb_holder.stl.";
  return group;
}

export async function loadThumbHolderModel(config = {}) {
  const sceneConfig = config.thumb_holder_scene || {};
  const visual = config.visual_style || {};
  const transform = config.thumb_model_transform || {};
  const assetUrl = sceneConfig.model_asset_url || "";
  const stlAssetUrl = sceneConfig.fallback_asset_url || "/static/assets/models/thumb_holder.stl";
  const fallbackEnabled = sceneConfig.fallback_placeholder_enabled !== false;

  try {
    if (!assetUrl) {
      throw new Error("thumb_holder.glb not configured; using STL fallback");
    }
    const assetProbe = await fetch(assetUrl, { method: "HEAD", cache: "no-store" });
    if (!assetProbe.ok) {
      throw new Error("thumb_holder.glb not found");
    }
    const { GLTFLoader } = await import("three/addons/loaders/GLTFLoader.js");
    const loader = new GLTFLoader();
    const gltf = await new Promise((resolve, reject) => {
      loader.load(assetUrl, resolve, undefined, reject);
    });
    const object = gltf.scene || gltf.scenes?.[0];
    if (!object) throw new Error("GLB loaded without a scene");
    object.name = "thumb_holder_glb";
    applyMaterialStyle(object, {
      color: visual.thumb_material_color || DEFAULT_THUMB_COLOR,
      opacity: Number(transform.opacity ?? 0.34),
      wireframe: Boolean(transform.wireframe),
    });
    object.userData.modelLoadStatus = "glb_loaded";
    object.userData.assetUrl = assetUrl;
    return {
      object,
      status: "glb_loaded",
      assetUrl,
      fallback: false,
      message: "thumb_holder.glb loaded",
    };
  } catch (error) {
    try {
      const stlProbe = await fetch(stlAssetUrl, { method: "HEAD", cache: "no-store" });
      if (!stlProbe.ok) {
        throw new Error("thumb_holder.stl not found");
      }
      const { STLLoader } = await import("three/addons/loaders/STLLoader.js");
      const loader = new STLLoader();
      const geometry = await new Promise((resolve, reject) => {
        loader.load(stlAssetUrl, resolve, undefined, reject);
      });
      const object = meshFromStlGeometry(geometry, {
        color: visual.thumb_material_color || DEFAULT_THUMB_COLOR,
        opacity: Number(transform.opacity ?? 0.42),
        wireframe: Boolean(transform.wireframe),
      });
      object.userData.assetUrl = stlAssetUrl;
      return {
        object,
        status: "stl_loaded",
        assetUrl: stlAssetUrl,
        fallback: false,
        message: "thumb_holder.stl loaded",
      };
    } catch (stlError) {
      if (!fallbackEnabled) {
        return {
          object: null,
          status: "blocked_conversion_required",
          assetUrl,
          fallback: false,
          message: `${error?.message || String(error)}; ${stlError?.message || String(stlError)}`,
        };
      }
    }
    if (!fallbackEnabled) {
      return {
        object: null,
        status: "blocked_conversion_required",
        assetUrl,
        fallback: false,
        message: error?.message || String(error),
      };
    }
    const object = createFallbackThumbHolder(config);
    return {
      object,
      status: "fallback_placeholder",
      assetUrl,
      fallback: true,
      message: error?.message || "GLB not available; using fallback placeholder",
    };
  }
}

export async function loadRobotNanoHandModel(config = {}) {
  const sceneConfig = config.whole_hand_scene || {};
  const visual = config.visual_style || {};
  const assetUrl = sceneConfig.asset_url || "/static/assets/models/robot_nano_hand_body.glb";

  try {
    const assetProbe = await fetch(assetUrl, { method: "HEAD", cache: "no-store" });
    if (!assetProbe.ok) {
      throw new Error("Robot Nano Hand body asset not found");
    }
    const { GLTFLoader } = await import("three/addons/loaders/GLTFLoader.js");
    const loader = new GLTFLoader();
    const gltf = await new Promise((resolve, reject) => {
      loader.load(assetUrl, resolve, undefined, reject);
    });
    const object = gltf.scene || gltf.scenes?.[0];
    if (!object) throw new Error("Robot Nano Hand GLB loaded without a scene");
    object.name = "robot_nano_hand_body";
    applyMaterialStyle(object, {
      color: visual.whole_hand_material_color || "#d6e0e6",
      opacity: Number(sceneConfig.body_opacity ?? 0.42),
      wireframe: false,
      depthWrite: false,
      roughness: 0.52,
      metalness: 0.12,
      side: THREE.DoubleSide,
    });
    object.userData.modelLoadStatus = "glb_loaded";
    object.userData.assetUrl = assetUrl;
    object.userData.sourceRepository = sceneConfig.source_repository_url || "";
    object.userData.sourceLicense = sceneConfig.source_license || "MIT";
    return {
      object,
      status: "glb_loaded",
      assetUrl,
      message: "Robot Nano Hand body loaded",
    };
  } catch (error) {
    return {
      object: null,
      status: "whole_hand_asset_unavailable",
      assetUrl,
      message: error?.message || String(error),
    };
  }
}
