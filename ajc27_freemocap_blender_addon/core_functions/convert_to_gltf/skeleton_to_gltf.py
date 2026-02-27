# AI Generated
"""
skeleton_to_gltf.py
--------------------
Converts a single frame of joint transforms into a minimal glTF 2.0 JSON
payload representing a skeleton pose.

Input
-----
A dict mapping joint name → Transform, where Transform has:
    .position  : (x, y, z)         global position, in METERS
    .rotation  : (x, y, z, w)      global orientation as a quaternion

Output
------
A self-contained glTF 2.0 JSON string with:
  - One Armature wrapper node
  - One node per joint, with translation/rotation set to the CURRENT POSE
    (global→local converted)
  - A skin referencing all joint nodes (required for Godot to produce
    a Skeleton3D on import rather than plain Node3D nodes)
  - Identity inverse bind matrices (no mesh, so IBMs are meaningless,
    but the skin entry requires them)
  - NO animation block
  - NO per-joint accessors beyond the IBMs
  - A single small binary buffer for the IBM data only

The pose is encoded directly in the node translation/rotation fields.
Godot's glTF importer reads these as the bone rest transforms and will
produce a Skeleton3D with the correct bone positions for the current frame.
"""

import json
import struct
import base64
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any


# ---------------------------------------------------------------------------
# Public data types
# ---------------------------------------------------------------------------

@dataclass
class Transform:
    """Global-space transform for a single joint."""
    position: Tuple[float, float, float]            # (x, y, z)  meters
    rotation: Tuple[float, float, float, float]     # (x, y, z, w) quaternion


# A single pose frame: joint name → Transform
PoseFrame = Dict[str, Transform]


# ---------------------------------------------------------------------------
# Skeleton hierarchy
# ---------------------------------------------------------------------------

#: Maps each bone name to its parent bone name, or None for the single root.
PARENT_MAP: Dict[str, Optional[str]] = {
    # Root
    "pelvis":      None,
    # Pelvis children
    "pelvis.R":    "pelvis",
    "pelvis.L":    "pelvis",
    "spine":       "pelvis",
    # Spine chain
    "spine.001":   "spine",
    "neck":        "spine.001",
    "face":        "neck",
    # Arms
    "shoulder.R":  "spine.001",
    "shoulder.L":  "spine.001",
    "upper_arm.R": "shoulder.R",
    "upper_arm.L": "shoulder.L",
    "forearm.R":   "upper_arm.R",
    "forearm.L":   "upper_arm.L",
    "hand.R":      "forearm.R",
    "hand.L":      "forearm.L",
    # Legs
    "thigh.R":     "pelvis.R",
    "thigh.L":     "pelvis.L",
    "shin.R":      "thigh.R",
    "shin.L":      "thigh.L",
    "foot.R":      "shin.R",
    "foot.L":      "shin.L",
    "heel.02.R":   "shin.R",
    "heel.02.L":   "shin.L",
}

BONE_NAMES: List[str] = list(PARENT_MAP.keys())


# ---------------------------------------------------------------------------
# Quaternion / vector math  (stdlib only, no numpy)
# ---------------------------------------------------------------------------

def _quat_conjugate(q: Tuple) -> Tuple:
    x, y, z, w = q
    return (-x, -y, -z, w)


def _quat_mul(a: Tuple, b: Tuple) -> Tuple:
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (
        aw*bx + ax*bw + ay*bz - az*by,
        aw*by - ax*bz + ay*bw + az*bx,
        aw*bz + ax*by - ay*bx + az*bw,
        aw*bw - ax*bx - ay*by - az*bz,
    )


def _quat_rotate_vec(q: Tuple, v: Tuple) -> Tuple:
    vq = (v[0], v[1], v[2], 0.0)
    r = _quat_mul(_quat_mul(q, vq), _quat_conjugate(q))
    return (r[0], r[1], r[2])


def _quat_normalize(q: Tuple) -> Tuple:
    x, y, z, w = q
    mag = math.sqrt(x*x + y*y + z*z + w*w)
    if mag < 1e-10:
        return (0.0, 0.0, 0.0, 1.0)
    return (x/mag, y/mag, z/mag, w/mag)


def _global_to_local(child: Transform,
                     parent: Optional[Transform]) -> Transform:
    """
    Convert child's global transform into parent-space (local) transform.

    local_position = inv(parent_rot) * (child_pos - parent_pos)
    local_rotation = inv(parent_rot) * child_rot
    """
    if parent is None:
        return child

    p_inv = _quat_conjugate(parent.rotation)
    offset = (
        child.position[0] - parent.position[0],
        child.position[1] - parent.position[1],
        child.position[2] - parent.position[2],
    )
    local_pos = _quat_rotate_vec(p_inv, offset)
    local_rot = _quat_normalize(_quat_mul(p_inv, child.rotation))
    return Transform(position=local_pos, rotation=local_rot)


# ---------------------------------------------------------------------------
# glTF builder
# ---------------------------------------------------------------------------

_IDENTITY_MAT4 = [
    1.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 1.0, 0.0,
    0.0, 0.0, 0.0, 1.0,
]


def _build_ibm_buffer(n_bones: int) -> Tuple[str, int]:
    """
    Pack n_bones identity MAT4s into a base64 data-URI buffer.
    Returns (uri_string, byte_length).
    """
    flat = _IDENTITY_MAT4 * n_bones
    raw = struct.pack(f"<{len(flat)}f", *flat)
    b64 = base64.b64encode(raw).decode("ascii")
    uri = f"data:application/octet-stream;base64,{b64}"
    return uri, len(raw)


def pose_to_gltf(
    pose: PoseFrame,
    bone_names: Optional[List[str]] = None,
    parent_map: Optional[Dict[str, Optional[str]]] = None,
    pretty: bool = False,
) -> str:
    """
    Convert a single pose frame to a glTF 2.0 JSON string.

    Parameters
    ----------
    pose        : Dict mapping joint name → Transform (global space, meters).
    bone_names  : Ordered bone list. Defaults to BONE_NAMES.
    parent_map  : Parent hierarchy. Defaults to PARENT_MAP.
    pretty      : Indent JSON for readability. Smaller/faster when False.

    Returns
    -------
    str  — Self-contained glTF 2.0 JSON string.
    """
    bones  = bone_names or BONE_NAMES
    pmap   = parent_map or PARENT_MAP
    n      = len(bones)
    b_idx  = {name: i for i, name in enumerate(bones)}

    # ── Nodes ────────────────────────────────────────────────────────────────
    # Index 0 = Armature wrapper (not a joint)
    # Index 1..n = one node per bone  (bone_index + 1)

    nodes: List[Dict[str, Any]] = []

    # Identify true root bones (no parent) — only these go under Armature.
    root_bones = [b for b in bones if pmap.get(b) is None]
    armature_children = [b_idx[b] + 1 for b in root_bones]

    # Armature wrapper — lists ONLY root bones as direct children.
    # Godot walks the node tree to derive bone parenting; if all bones are
    # direct children of Armature they all appear as roots in Skeleton3D.
    nodes.append({
        "name":     "Armature",
        "children": armature_children,
    })

    identity_tf = Transform(position=(0.0, 0.0, 0.0),
                            rotation=(0.0, 0.0, 0.0, 1.0))

    # Build all bone nodes first (so b_idx lookups are stable).
    for bone in bones:
        global_tf  = pose.get(bone, identity_tf)
        parent_name = pmap.get(bone)
        parent_tf  = pose.get(parent_name) if parent_name else None
        local_tf   = _global_to_local(global_tf, parent_tf)

        node: Dict[str, Any] = {
            "name":        bone,
            "translation": [local_tf.position[0],
                            local_tf.position[1],
                            local_tf.position[2]],
            "rotation":    [local_tf.rotation[0],
                            local_tf.rotation[1],
                            local_tf.rotation[2],
                            local_tf.rotation[3]],
        }
        nodes.append(node)

    # Wire up children in a second pass so all indices are already known.
    for bone in bones:
        children = [
            b_idx[b] + 1
            for b in bones
            if pmap.get(b) == bone
        ]
        if children:
            nodes[b_idx[bone] + 1]["children"] = children

    # ── Skin ─────────────────────────────────────────────────────────────────
    # Required so Godot imports these nodes as a Skeleton3D rather than Node3D.
    ibm_uri, ibm_bytes = _build_ibm_buffer(n)

    gltf: Dict[str, Any] = {
        "asset": {
            "version":   "2.0",
            "generator": "skeleton_to_gltf.py",
        },
        "scene":  0,
        "scenes": [{"name": "Scene", "nodes": [0]}],
        "nodes":  nodes,
        "skins": [{
            "name":                "Skeleton",
            "joints":              list(range(1, n + 1)),
            "inverseBindMatrices": 0,
            "skeleton":            1,
        }],
        "accessors": [{
            "bufferView":    0,
            "byteOffset":    0,
            "componentType": 5126,      # FLOAT
            "type":          "MAT4",
            "count":         n,
            "name":          "inverseBindMatrices",
        }],
        "bufferViews": [{
            "buffer":     0,
            "byteOffset": 0,
            "byteLength": ibm_bytes,
        }],
        "buffers": [{
            "byteLength": ibm_bytes,
            "uri":        ibm_uri,
        }],
    }

    return json.dumps(gltf, indent=2 if pretty else None)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import math

    def _make_test_pose() -> PoseFrame:
        swing = 0.15
        w = math.sqrt(max(0.0, 1.0 - swing ** 2))
        return {
            "pelvis":      Transform(( 0.0,  0.95, 0.0), (0.0, 0.0, 0.0, 1.0)),
            "pelvis.R":    Transform((-0.1,  0.90, 0.0), (0.0, 0.0, 0.0, 1.0)),
            "pelvis.L":    Transform(( 0.1,  0.90, 0.0), (0.0, 0.0, 0.0, 1.0)),
            "spine":       Transform(( 0.0,  1.00, 0.0), (swing, 0.0, 0.0, w)),
            "spine.001":   Transform(( 0.0,  1.30, 0.0), (swing, 0.0, 0.0, w)),
            "neck":        Transform(( 0.0,  1.60, 0.0), (0.0, 0.0, 0.0, 1.0)),
            "face":        Transform(( 0.0,  1.70, 0.0), (0.0, 0.0, 0.0, 1.0)),
            "shoulder.R":  Transform((-0.2,  1.50, 0.0), (0.0, 0.0, 0.0, 1.0)),
            "shoulder.L":  Transform(( 0.2,  1.50, 0.0), (0.0, 0.0, 0.0, 1.0)),
            "upper_arm.R": Transform((-0.4,  1.50, 0.0), (0.0, 0.0, 0.0, 1.0)),
            "upper_arm.L": Transform(( 0.4,  1.50, 0.0), (0.0, 0.0, 0.0, 1.0)),
            "forearm.R":   Transform((-0.7,  1.50, 0.0), (0.0, 0.0, 0.0, 1.0)),
            "forearm.L":   Transform(( 0.7,  1.50, 0.0), (0.0, 0.0, 0.0, 1.0)),
            "hand.R":      Transform((-1.0,  1.50, 0.0), (0.0, 0.0, 0.0, 1.0)),
            "hand.L":      Transform(( 1.0,  1.50, 0.0), (0.0, 0.0, 0.0, 1.0)),
            "thigh.R":     Transform((-0.1,  0.80, 0.0), (0.0, 0.0, 0.0, 1.0)),
            "thigh.L":     Transform(( 0.1,  0.80, 0.0), (0.0, 0.0, 0.0, 1.0)),
            "shin.R":      Transform((-0.1,  0.40, 0.0), (0.0, 0.0, 0.0, 1.0)),
            "shin.L":      Transform(( 0.1,  0.40, 0.0), (0.0, 0.0, 0.0, 1.0)),
            "foot.R":      Transform((-0.1,  0.05, 0.0), (0.0, 0.0, 0.0, 1.0)),
            "foot.L":      Transform(( 0.1,  0.05, 0.0), (0.0, 0.0, 0.0, 1.0)),
            "heel.02.R":   Transform((-0.1,  0.40,-0.05),(0.0, 0.0, 0.0, 1.0)),
            "heel.02.L":   Transform(( 0.1,  0.40,-0.05),(0.0, 0.0, 0.0, 1.0)),
        }

    pose = _make_test_pose()
    result = pose_to_gltf(pose, pretty=True)
    parsed = json.loads(result)

    # Assertions
    assert parsed["asset"]["version"] == "2.0"
    assert "animations" not in parsed,       "should have NO animation block"
    assert len(parsed["nodes"]) == 24,       "23 bones + 1 armature"
    assert len(parsed["skins"]) == 1
    assert len(parsed["accessors"]) == 1,    "IBM only"
    assert len(parsed["bufferViews"]) == 1,  "IBM only"
    assert len(parsed["buffers"]) == 1

    # Confirm local transform conversion: spine.001 should NOT be at world y=1.3
    spine001 = next(n for n in parsed["nodes"] if n["name"] == "spine.001")
    assert abs(spine001["translation"][1] - 1.3) > 0.01, \
        "spine.001 translation should be local, not global"

    size_kb = len(result) / 1024
    print(f"✓ pose glTF built successfully")
    print(f"  Bones    : {len(BONE_NAMES)}")
    print(f"  Nodes    : {len(parsed['nodes'])}  (23 joints + 1 armature)")
    print(f"  Accessors: {len(parsed['accessors'])}  (IBM only)")
    print(f"  Animation: {'YES' if 'animations' in parsed else 'NO'}")
    print(f"  JSON size: {size_kb:.1f} KB")

    with open("/tmp/test_pose.gltf", "w") as f:
        f.write(result)
    print(f"  Saved to : /tmp/test_pose.gltf")