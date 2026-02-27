# AI Generated
"""
skeleton_to_gltf.py
--------------------
Converts a list of per-frame joint dictionaries
    [ { joint_name: Transform, ... }, ... ]

where Transform has:
    .position   – (x, y, z)   global position
    .rotation   – (x, y, z, w) quaternion (global orientation)

into a valid glTF 2.0 JSON payload that contains a skinned skeleton.

The output can be sent over the network, written to a .gltf file, or
consumed by any glTF-aware renderer (Godot, Three.js, Babylon.js, etc.).

Key design decisions
---------------------
* glTF stores joint transforms as LOCAL (parent-space), so every global
  transform is decomposed back to local before packing.
* Animations are stored as a single "Skeleton" animation with one
  translation + rotation sampler pair per joint.
* Accessor data is base64-encoded and embedded in a data-URI buffer so
  the payload is fully self-contained (no external .bin file needed).
* No mesh geometry is emitted – just the skeleton / armature, which is
  all you need for retargeting / streaming.
"""

import json
import struct
import base64
import math
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any

# ---------------------------------------------------------------------------
# Data types you hand in
# ---------------------------------------------------------------------------

@dataclass
class Transform:
    """Global-space transform for a single joint."""
    position: Tuple[float, float, float]           # (x, y, z)
    rotation: Tuple[float, float, float, float]    # (x, y, z, w)  quaternion


# Convenience type aliases
JointFrame  = Dict[str, Transform]          # one instant in time
FrameList   = List[JointFrame]              # the full animation


# ---------------------------------------------------------------------------
# Skeleton hierarchy
# Matches the valid_bone_names list; each entry maps child → parent.
# Root joints have parent = None.
# ---------------------------------------------------------------------------

PARENT_MAP: Dict[str, Optional[str]] = {
    # Spine chain
    "pelvis.R":    None,          # root (right side of pelvis)
    "pelvis.L":    None,          # root (left side of pelvis)
    "spine":       None,          # torso root
    "spine.001":   "spine",
    "neck":        "spine.001",

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
    "heel.02.R":   "foot.R",
    "heel.02.L":   "foot.L",
}

VALID_BONE_NAMES: List[str] = list(PARENT_MAP.keys())


# ---------------------------------------------------------------------------
# Quaternion / vector math  (no external deps)
# ---------------------------------------------------------------------------

def _quat_conjugate(q: Tuple[float,float,float,float]) -> Tuple[float,float,float,float]:
    x, y, z, w = q
    return (-x, -y, -z, w)


def _quat_mul(a: Tuple[float,float,float,float],
              b: Tuple[float,float,float,float]) -> Tuple[float,float,float,float]:
    """Hamilton product  a * b."""
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (
        aw*bx + ax*bw + ay*bz - az*by,
        aw*by - ax*bz + ay*bw + az*bx,
        aw*bz + ax*by - ay*bx + az*bw,
        aw*bw - ax*bx - ay*by - az*bz,
    )


def _quat_rotate_vec(q: Tuple[float,float,float,float],
                     v: Tuple[float,float,float]) -> Tuple[float,float,float]:
    """Rotate vector v by quaternion q."""
    vq = (v[0], v[1], v[2], 0.0)
    res = _quat_mul(_quat_mul(q, vq), _quat_conjugate(q))
    return (res[0], res[1], res[2])


def _vec_sub(a: Tuple[float,float,float],
             b: Tuple[float,float,float]) -> Tuple[float,float,float]:
    return (a[0]-b[0], a[1]-b[1], a[2]-b[2])


def _quat_normalize(q: Tuple[float,float,float,float]) -> Tuple[float,float,float,float]:
    x, y, z, w = q
    mag = math.sqrt(x*x + y*y + z*z + w*w)
    if mag < 1e-10:
        return (0.0, 0.0, 0.0, 1.0)
    return (x/mag, y/mag, z/mag, w/mag)


def _global_to_local(child_global: Transform,
                     parent_global: Optional[Transform]) -> Transform:
    """
    Convert a child's global transform to local (parent-space) transform.

    local_position = inv(parent_rot) * (child_pos - parent_pos)
    local_rotation = inv(parent_rot) * child_rot
    """
    if parent_global is None:
        # Root joint – local == global
        return child_global

    p_rot_inv = _quat_conjugate(parent_global.rotation)

    # Local position: un-rotate the offset vector by parent's inverse rotation
    offset = _vec_sub(child_global.position, parent_global.position)
    local_pos = _quat_rotate_vec(p_rot_inv, offset)

    # Local rotation
    local_rot = _quat_normalize(_quat_mul(p_rot_inv, child_global.rotation))

    return Transform(position=local_pos, rotation=local_rot)


# ---------------------------------------------------------------------------
# Binary buffer helpers
# ---------------------------------------------------------------------------

def _pack_floats(values: List[float]) -> bytes:
    return struct.pack(f"<{len(values)}f", *values)


def _encode_buffer(data: bytes) -> str:
    """Return a glTF data-URI string for the raw bytes."""
    b64 = base64.b64encode(data).decode("ascii")
    return f"data:application/octet-stream;base64,{b64}"


# ---------------------------------------------------------------------------
# Core converter
# ---------------------------------------------------------------------------

class SkeletonGLTFBuilder:
    """
    Builds a self-contained glTF 2.0 JSON dict from a list of JointFrames.

    Usage:
        builder = SkeletonGLTFBuilder(frame_list, fps=30.0)
        gltf_dict = builder.build()
        json_str  = json.dumps(gltf_dict, indent=2)
    """

    def __init__(self, frames: FrameList, fps: float = 30.0,
                 bone_names: Optional[List[str]] = None,
                 parent_map: Optional[Dict[str, Optional[str]]] = None):
        """
        Parameters
        ----------
        frames    : List of dicts, each mapping joint_name → Transform.
                    All frames should contain the same set of joint names.
        fps       : Frames per second (used to compute animation timestamps).
        bone_names: Ordered list of bone names to include. Defaults to
                    VALID_BONE_NAMES if not provided.
        parent_map: Dict mapping each bone name to its parent name (or None
                    for roots). Defaults to PARENT_MAP.
        """
        if not frames:
            raise ValueError("frames list must not be empty")

        self.frames     = frames
        self.fps        = fps
        self.bones      = bone_names or VALID_BONE_NAMES
        self.parent_map = parent_map or PARENT_MAP

        # Derived
        self.n_bones    = len(self.bones)
        self.n_frames   = len(frames)
        self.bone_index = {name: i for i, name in enumerate(self.bones)}

        # glTF building blocks accumulated during build()
        self._accessors: List[Dict]  = []
        self._buffer_views: List[Dict] = []
        self._raw_buffer: bytearray  = bytearray()

    # ------------------------------------------------------------------ build

    def build(self) -> Dict[str, Any]:
        """Return the complete glTF 2.0 dict."""
        self._accessors.clear()
        self._buffer_views.clear()
        self._raw_buffer.clear()

        # ── Nodes (one per joint + one root "Armature" wrapper) ──────────────
        #
        # Index layout:
        #   0              → Armature node  (the scene root)
        #   1 .. n_bones   → one node per joint  (bone_index + 1)
        #
        armature_node_idx = 0
        joint_node_offset = 1   # joints start at index 1

        nodes = []

        # Armature wrapper node
        nodes.append({
            "name":     "Armature",
            "children": list(range(joint_node_offset,
                                   joint_node_offset + self.n_bones)),
        })

        # Determine the T-pose (first frame) local transforms for each joint
        first_frame = self.frames[0]
        for bone in self.bones:
            local_tf = self._local_transform(bone, first_frame)
            node_entry: Dict[str, Any] = {
                "name":        bone,
                "translation": list(local_tf.position),
                "rotation":    list(local_tf.rotation),   # (x, y, z, w)
                "scale":       [1.0, 1.0, 1.0],
            }
            # Wire up children
            children = [self.bone_index[b] + joint_node_offset
                        for b, p in self.parent_map.items()
                        if p == bone and b in self.bone_index]
            if children:
                node_entry["children"] = children
            nodes.append(node_entry)

        # ── Skin ─────────────────────────────────────────────────────────────
        # ibm = Inverse Bind Matrices (one 4×4 per joint).
        # For a skeleton-only payload with no mesh, all IBMs are identity.
        ibm_accessor_idx = self._add_inverse_bind_matrices()

        skin = {
            "name":                "Skeleton",
            "joints":              list(range(joint_node_offset,
                                              joint_node_offset + self.n_bones)),
            "inverseBindMatrices": ibm_accessor_idx,
            "skeleton":            joint_node_offset,   # root joint index
        }

        # ── Animation samplers & channels ────────────────────────────────────
        timestamps    = [i / self.fps for i in range(self.n_frames)]
        time_accessor = self._add_scalar_accessor(timestamps, "TIME")

        samplers: List[Dict] = []
        channels:  List[Dict] = []

        for bone in self.bones:
            node_idx = self.bone_index[bone] + joint_node_offset

            # Collect local translation & rotation for every frame
            translations: List[float] = []
            rotations:    List[float] = []

            for frame in self.frames:
                local_tf = self._local_transform(bone, frame)
                translations.extend(local_tf.position)
                rotations.extend(local_tf.rotation)   # x, y, z, w

            t_acc = self._add_vec3_accessor(translations, f"{bone}_T")
            r_acc = self._add_vec4_accessor(rotations,    f"{bone}_R")

            t_sampler_idx = len(samplers)
            samplers.append({"input": time_accessor, "output": t_acc,
                             "interpolation": "LINEAR"})
            r_sampler_idx = len(samplers)
            samplers.append({"input": time_accessor, "output": r_acc,
                             "interpolation": "LINEAR"})

            channels.append({
                "sampler": t_sampler_idx,
                "target":  {"node": node_idx, "path": "translation"},
            })
            channels.append({
                "sampler": r_sampler_idx,
                "target":  {"node": node_idx, "path": "rotation"},
            })

        animation = {
            "name":     "Skeleton",
            "samplers": samplers,
            "channels": channels,
        }

        # ── Buffer / bufferViews / accessors ─────────────────────────────────
        buffer = {
            "byteLength": len(self._raw_buffer),
            "uri":        _encode_buffer(bytes(self._raw_buffer)),
        }

        # ── Assemble glTF root ────────────────────────────────────────────────
        gltf: Dict[str, Any] = {
            "asset": {
                "version":   "2.0",
                "generator": "skeleton_to_gltf.py",
            },
            "scene":  0,
            "scenes": [{"name": "Scene", "nodes": [armature_node_idx]}],
            "nodes":  nodes,
            "skins":  [skin],
            "animations": [animation],
            "bufferViews": self._buffer_views,
            "accessors":   self._accessors,
            "buffers":     [buffer],
        }

        return gltf

    # ---------------------------------------------------------------- helpers

    def _local_transform(self, bone: str, frame: JointFrame) -> Transform:
        """Return parent-space transform for `bone` in the given frame."""
        global_tf = frame.get(bone)
        if global_tf is None:
            # Fall back to identity if the joint is absent in this frame
            global_tf = Transform(position=(0.0, 0.0, 0.0),
                                  rotation=(0.0, 0.0, 0.0, 1.0))

        parent_name = self.parent_map.get(bone)
        parent_tf: Optional[Transform] = None
        if parent_name and parent_name in frame:
            parent_tf = frame[parent_name]

        return _global_to_local(global_tf, parent_tf)

    # ---- Binary buffer helpers ----

    def _append_raw(self, data: bytes) -> Tuple[int, int]:
        """Append bytes to the shared buffer; return (byte_offset, byte_length)."""
        # glTF requires buffer views to be 4-byte aligned
        pad = (4 - len(self._raw_buffer) % 4) % 4
        self._raw_buffer.extend(b"\x00" * pad)
        offset = len(self._raw_buffer)
        self._raw_buffer.extend(data)
        return offset, len(data)

    def _add_buffer_view(self, data: bytes, target: Optional[int] = None) -> int:
        """Append data to the buffer, register a bufferView, return its index."""
        byte_offset, byte_length = self._append_raw(data)
        bv: Dict[str, Any] = {
            "buffer":     0,
            "byteOffset": byte_offset,
            "byteLength": byte_length,
        }
        if target is not None:
            bv["target"] = target
        self._buffer_views.append(bv)
        return len(self._buffer_views) - 1

    def _add_scalar_accessor(self, values: List[float], name: str = "") -> int:
        data = _pack_floats(values)
        bv_idx = self._add_buffer_view(data)
        acc: Dict[str, Any] = {
            "bufferView":    bv_idx,
            "byteOffset":    0,
            "componentType": 5126,          # FLOAT
            "type":          "SCALAR",
            "count":         len(values),
            "min":           [min(values)],
            "max":           [max(values)],
        }
        if name:
            acc["name"] = name
        self._accessors.append(acc)
        return len(self._accessors) - 1

    def _add_vec3_accessor(self, flat_values: List[float], name: str = "") -> int:
        """flat_values = [x0,y0,z0, x1,y1,z1, ...]"""
        count = len(flat_values) // 3
        data  = _pack_floats(flat_values)
        bv_idx = self._add_buffer_view(data)
        acc: Dict[str, Any] = {
            "bufferView":    bv_idx,
            "byteOffset":    0,
            "componentType": 5126,
            "type":          "VEC3",
            "count":         count,
        }
        if name:
            acc["name"] = name
        self._accessors.append(acc)
        return len(self._accessors) - 1

    def _add_vec4_accessor(self, flat_values: List[float], name: str = "") -> int:
        """flat_values = [x0,y0,z0,w0, x1,y1,z1,w1, ...]"""
        count = len(flat_values) // 4
        data  = _pack_floats(flat_values)
        bv_idx = self._add_buffer_view(data)
        acc: Dict[str, Any] = {
            "bufferView":    bv_idx,
            "byteOffset":    0,
            "componentType": 5126,
            "type":          "VEC4",
            "count":         count,
        }
        if name:
            acc["name"] = name
        self._accessors.append(acc)
        return len(self._accessors) - 1

    def _add_inverse_bind_matrices(self) -> int:
        """
        Pack n_bones identity 4×4 matrices as a MAT4 accessor.
        For a mesh-free skeleton the IBMs can all be identity.
        If you later attach a mesh, replace these with the actual
        inverse of each joint's model-space bind-pose matrix.
        """
        identity_mat4 = [
            1,0,0,0,
            0,1,0,0,
            0,0,1,0,
            0,0,0,1,
        ]
        flat = identity_mat4 * self.n_bones
        data = _pack_floats([float(v) for v in flat])
        bv_idx = self._add_buffer_view(data)
        acc: Dict[str, Any] = {
            "bufferView":    bv_idx,
            "byteOffset":    0,
            "componentType": 5126,
            "type":          "MAT4",
            "count":         self.n_bones,
            "name":          "inverseBindMatrices",
        }
        self._accessors.append(acc)
        return len(self._accessors) - 1


# ---------------------------------------------------------------------------
# Public convenience function
# ---------------------------------------------------------------------------

def skeleton_frames_to_gltf(
    frames: FrameList,
    fps: float = 30.0,
    bone_names: Optional[List[str]] = None,
    parent_map: Optional[Dict[str, Optional[str]]] = None,
    pretty: bool = False,
) -> str:
    """
    Convert a list of joint-transform dicts to a glTF 2.0 JSON string.

    Parameters
    ----------
    frames     : List[Dict[str, Transform]]
                 Each dict maps joint name → Transform(position, rotation).
                 All transforms must be in GLOBAL (world) space.
    fps        : Animation frame rate. Default 30.
    bone_names : Ordered list of bone names. Defaults to VALID_BONE_NAMES.
    parent_map : Parent hierarchy. Defaults to PARENT_MAP.
    pretty     : If True, returns indented JSON (larger but human-readable).

    Returns
    -------
    str  – A self-contained glTF 2.0 JSON string ready to send over the wire.
    """
    builder = SkeletonGLTFBuilder(frames, fps=fps,
                                  bone_names=bone_names,
                                  parent_map=parent_map)
    gltf_dict = builder.build()
    return json.dumps(gltf_dict, indent=2 if pretty else None)


# ---------------------------------------------------------------------------
# Quick self-test  (python skeleton_to_gltf.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import math, random, json

    random.seed(42)

    def _make_test_frame(t: float) -> JointFrame:
        """Generate a simple T-pose with a gentle spine rotation over time."""
        frame: JointFrame = {}

        # pelvis roots – stationary
        frame["pelvis.R"] = Transform(position=(-0.1, 0.9, 0.0),
                                      rotation=(0.0, 0.0, 0.0, 1.0))
        frame["pelvis.L"] = Transform(position=( 0.1, 0.9, 0.0),
                                      rotation=(0.0, 0.0, 0.0, 1.0))

        # spine chain
        swing = math.sin(t * 2 * math.pi) * 0.1
        frame["spine"]     = Transform(position=(0.0, 1.0, 0.0),
                                       rotation=(swing, 0.0, 0.0,
                                                 math.sqrt(max(0, 1 - swing**2))))
        frame["spine.001"] = Transform(position=(0.0, 1.3, 0.0),
                                       rotation=frame["spine"].rotation)
        frame["neck"]      = Transform(position=(0.0, 1.6, 0.0),
                                       rotation=(0.0, 0.0, 0.0, 1.0))

        # shoulders / arms (T-pose)
        for side, sx in (("R", -1), ("L", 1)):
            frame[f"shoulder.{side}"]  = Transform(position=(sx*0.2, 1.5, 0.0),
                                                   rotation=(0.0,0.0,0.0,1.0))
            frame[f"upper_arm.{side}"] = Transform(position=(sx*0.4, 1.5, 0.0),
                                                   rotation=(0.0,0.0,0.0,1.0))
            frame[f"forearm.{side}"]   = Transform(position=(sx*0.7, 1.5, 0.0),
                                                   rotation=(0.0,0.0,0.0,1.0))
            frame[f"hand.{side}"]      = Transform(position=(sx*1.0, 1.5, 0.0),
                                                   rotation=(0.0,0.0,0.0,1.0))
            frame[f"thigh.{side}"]     = Transform(position=(sx*0.1, 0.8, 0.0),
                                                   rotation=(0.0,0.0,0.0,1.0))
            frame[f"shin.{side}"]      = Transform(position=(sx*0.1, 0.4, 0.0),
                                                   rotation=(0.0,0.0,0.0,1.0))
            frame[f"foot.{side}"]      = Transform(position=(sx*0.1, 0.05, 0.0),
                                                   rotation=(0.0,0.0,0.0,1.0))
            frame[f"heel.02.{side}"]   = Transform(position=(sx*0.1, 0.0, -0.05),
                                                   rotation=(0.0,0.0,0.0,1.0))

        return frame

    N_FRAMES = 30
    test_frames = [_make_test_frame(i / N_FRAMES) for i in range(N_FRAMES)]

    gltf_json = skeleton_frames_to_gltf(test_frames, fps=30.0, pretty=True)

    # Validate round-trip
    parsed = json.loads(gltf_json)
    assert parsed["asset"]["version"] == "2.0"
    assert len(parsed["nodes"]) == len(VALID_BONE_NAMES) + 1   # +1 armature
    assert len(parsed["skins"]) == 1
    assert len(parsed["animations"]) == 1
    n_channels = len(parsed["animations"][0]["channels"])
    assert n_channels == len(VALID_BONE_NAMES) * 2, \
        f"expected {len(VALID_BONE_NAMES)*2} channels, got {n_channels}"

    print(f"✓ glTF built successfully")
    print(f"  Bones     : {len(VALID_BONE_NAMES)}")
    print(f"  Frames    : {N_FRAMES}")
    print(f"  Nodes     : {len(parsed['nodes'])}")
    print(f"  Accessors : {len(parsed['accessors'])}")
    print(f"  Anim chan  : {n_channels}")
    print(f"  JSON bytes : {len(gltf_json):,}")

    # Optionally dump to file for inspection
    with open("/tmp/test_skeleton.gltf", "w") as f:
        f.write(gltf_json)
    print("  Saved to   : /tmp/test_skeleton.gltf")