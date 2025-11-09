#!/usr/bin/env python3
import os, math
from typing import Optional, Tuple, Iterable, List
import numpy as np

# --- OCC (STEP reader) ---
from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.IFSelect import IFSelect_RetDone
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_FACE
from OCC.Core.TopoDS import topods_Face
from OCC.Core.BRep import BRep_Tool
from OCC.Core.TopLoc import TopLoc_Location
from OCC.Core.Bnd import Bnd_Box
from OCC.Core.BRepBndLib import brepbndlib_Add

# --- Fallback renderer bits (Matplotlib) ---
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

np.infty = np.inf

_ORIENT_ELEV_AZIM = {
    "iso": (25, 45), "iso_back": (25, -135),
    "front": (0, 90), "back": (0, -90),
    "left": (0, 180), "right": (0, 0),
    "top": (90, 0), "bottom": (-90, 0),
}
_QUALITY_TO_DEFLECTIONS = {
    "draft": (1.00, 0.80),
    "normal": (0.30, 0.50),
    "fine": (0.15, 0.30),
    "ultra": (0.07, 0.20),
}

# ---------- OCC helpers ----------
def _triangulate_shape(shape, deflection: float, angular_deflection: float) -> None:
    BRepMesh_IncrementalMesh(shape, deflection, True, angular_deflection, True)

def _collect_triangles(shape) -> Tuple[np.ndarray, np.ndarray]:
    verts: List[np.ndarray] = []
    norms: List[np.ndarray] = []
    exp = TopExp_Explorer(shape, TopAbs_FACE)
    while exp.More():
        face = topods_Face(exp.Current())
        loc = TopLoc_Location()
        tri = BRep_Tool.Triangulation(face, loc)
        if tri is None:
            exp.Next(); continue
        trsf = loc.Transformation()
        nb_nodes = tri.NbNodes()
        pts = np.empty((nb_nodes, 3), dtype=float)
        if hasattr(tri, "Nodes"):
            nodes_arr = tri.Nodes()
            for i in range(1, nb_nodes + 1):
                p = nodes_arr.Value(i).Transformed(trsf)
                pts[i-1] = (p.X(), p.Y(), p.Z())
        else:
            for i in range(1, nb_nodes + 1):
                p = tri.Node(i).Transformed(trsf)
                pts[i-1] = (p.X(), p.Y(), p.Z())
        nb_tris = tri.NbTriangles()
        for i in range(1, nb_tris + 1):
            if hasattr(tri, "Triangles"):
                a, b, c = tri.Triangles().Value(i).Get()
            else:
                a, b, c = tri.Triangle(i).Get()
            v0, v1, v2 = pts[a-1], pts[b-1], pts[c-1]
            verts.append(np.array([v0, v1, v2]))
            n = np.cross(v1 - v0, v2 - v0)
            nn = np.linalg.norm(n)
            norms.append(n/nn if nn > 1e-12 else np.array([0, 0, 1.0]))
        exp.Next()
    if not verts:
        return np.empty((0,3,3)), np.empty((0,3))
    return np.stack(verts, 0), np.stack(norms, 0)

def _shape_bbox(shape) -> Tuple[np.ndarray, np.ndarray]:
    box = Bnd_Box(); brepbndlib_Add(shape, box)
    xmin, ymin, zmin, xmax, ymax, zmax = box.Get()
    return np.array([xmin, ymin, zmin]), np.array([xmax, ymax, zmax])

def _elev_azim_to_dir(elev_deg: float, azim_deg: float):
    er, ar = math.radians(elev_deg), math.radians(azim_deg)
    return np.array([math.cos(er)*math.cos(ar),
                     math.cos(er)*math.sin(ar),
                     math.sin(er)], float)

def _look_at(eye, target, up=(0,0,1)):
    eye = np.asarray(eye, float); target = np.asarray(target, float); up = np.asarray(up, float)
    z = eye - target; z /= (np.linalg.norm(z) + 1e-12)
    x = np.cross(up, z); x /= (np.linalg.norm(x) + 1e-12)
    y = np.cross(z, x)
    T = np.eye(4); T[:3,:3] = np.column_stack([x,y,z]); T[:3,3] = eye
    return T

# ---------- Main smooth renderer with shadowed PBR ----------
def step_to_png_smooth(
    step_path: str,
    out_path: Optional[str] = None,
    *,
    img_size: Tuple[int, int] = (1600, 1200),
    fov_deg: float = 35.0,
    orient: str = "iso",
    view_elev_azim: Optional[Tuple[float, float]] = None,
    look_dir: Optional[Tuple[float, float, float]] = None,
    background: Tuple[float, float, float] = (1, 1, 1),
    quality: str = "fine",
    mesh_deflection: Optional[float] = None,
    angular_deflection: Optional[float] = None,

    # --- new visibility controls ---
    frame_fill: float = 0.92,                 # 0<fill<=1 (1 = as tight as possible)
    key_from_camera: bool = True,             # light from the camera direction
    exposure: float = 1.2,                    # post multiply (1.0 = no change)
    model_base: Tuple[float, float, float] = (0.75, 0.76, 0.80),  # darker than bg
    metallic: float = 0.08,
    roughness: float = 0.55,
    ambient: float = 0.35,                    # ambient scene light
    key_intensity: float = 10.0,
    fill_intensity: float = 3.0,
    rim_intensity: float = 2.5,
    rim_dir: Tuple[float, float, float] = (-0.4, -0.2, 0.2),
    add_ground: bool = True,
    ground_scale: float = 3.0,
) -> str:
    import numpy as np, math, imageio, trimesh, pyrender
    from OCC.Core.STEPControl import STEPControl_Reader
    from OCC.Core.IFSelect import IFSelect_RetDone

    if out_path is None:
        out_path = os.path.splitext(step_path)[0] + "_smooth.png"

    # --- STEP load + mesh (same as before) ---
    reader = STEPControl_Reader()
    if reader.ReadFile(step_path) != IFSelect_RetDone:
        raise RuntimeError(f"Failed to read STEP: {step_path}")
    if not reader.TransferRoots():
        raise RuntimeError("Failed to transfer STEP contents.")
    shape = reader.OneShape()

    if mesh_deflection is None or angular_deflection is None:
        q_lin, q_ang = _QUALITY_TO_DEFLECTIONS.get(quality, _QUALITY_TO_DEFLECTIONS["fine"])
        mesh_deflection = q_lin if mesh_deflection is None else mesh_deflection
        angular_deflection = q_ang if angular_deflection is None else angular_deflection
    _triangulate_shape(shape, mesh_deflection, angular_deflection)

    V, _ = _collect_triangles(shape)
    if V.size == 0:
        raise RuntimeError("No triangles after meshing.")
    verts_tri = V.reshape(-1, 3)
    rounded = np.round(verts_tri, 9)
    _, idx_first, inv = np.unique(rounded, axis=0, return_index=True, return_inverse=True)
    verts = verts_tri[idx_first]
    faces = inv.reshape(-1, 3)

    bb_min, bb_max = _shape_bbox(shape)
    center = (bb_min + bb_max) / 2
    max_range = float(np.max(bb_max - bb_min) or 1.0)

    # --- camera ---
    if view_elev_azim is not None:
        elev, azim = view_elev_azim
        cam_dir = _elev_azim_to_dir(elev, azim)
    elif look_dir is not None:
        cam_dir = np.asarray(look_dir, float); cam_dir /= (np.linalg.norm(cam_dir) + 1e-12)
    else:
        elev, azim = _ORIENT_ELEV_AZIM.get(orient, _ORIENT_ELEV_AZIM["iso"])
        cam_dir = _elev_azim_to_dir(elev, azim)

    # distance for tight framing
    base = max_range / (2 * math.tan(math.radians(fov_deg) / 2))
    distance = base / max(1e-3, min(1.0, frame_fill))   # smaller distance => larger on screen
    eye = center + cam_dir * distance

    # --- scene ---
    scene = pyrender.Scene(
        bg_color=[background[0], background[1], background[2], 1.0],
        ambient_light=np.array([ambient, ambient, ambient, 1.0])
    )

    mat = pyrender.MetallicRoughnessMaterial(
        baseColorFactor=[model_base[0], model_base[1], model_base[2], 1.0],
        metallicFactor=metallic,
        roughnessFactor=roughness,
        doubleSided=True,
    )
    tri = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    scene.add(pyrender.Mesh.from_trimesh(tri, material=mat, smooth=True))

    if add_ground:
        size = max_range * ground_scale
        plane_z = bb_min[2] - 0.01 * max_range
        plane = trimesh.creation.box(extents=(size, size, max_range * 1e-3))
        plane.apply_translation([center[0], center[1], plane_z - (max_range * 1e-3) / 2])
        ground_mat = pyrender.MetallicRoughnessMaterial(
            baseColorFactor=[1, 1, 1, 1], metallicFactor=0.0, roughnessFactor=0.9, doubleSided=True
        )
        scene.add(pyrender.Mesh.from_trimesh(plane, material=ground_mat, smooth=False))

    cam = pyrender.PerspectiveCamera(yfov=np.radians(fov_deg), aspectRatio=img_size[0] / img_size[1])
    scene.add(cam, pose=_look_at(eye, center))

    # --- lights: key (from camera), fill, rim ---
    key_dir = cam_dir if key_from_camera else np.array([0.35, 0.2, 1.0], float)
    key_dir /= (np.linalg.norm(key_dir) + 1e-12)
    key_pose = _look_at(center - key_dir * (max_range * 3.0), center)
    scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=key_intensity), pose=key_pose)

    fill_dir = -key_dir * np.array([1.0, 0.8, 0.2])
    fill_dir /= (np.linalg.norm(fill_dir) + 1e-12)
    fill_pose = _look_at(center - fill_dir * (max_range * 3.0), center)
    scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=fill_intensity), pose=fill_pose)

    rdir = np.asarray(rim_dir, float); rdir /= (np.linalg.norm(rdir) + 1e-12)
    rim_pose = _look_at(center - rdir * (max_range * 3.0), center)
    scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=rim_intensity), pose=rim_pose)

    # --- render ---
    w, h = int(img_size[0]), int(img_size[1])
    r = pyrender.OffscreenRenderer(viewport_width=w, viewport_height=h)
    flags = pyrender.RenderFlags.SHADOWS_DIRECTIONAL
    color, _ = r.render(scene, flags=flags)
    r.delete()

    # simple exposure bump
    if exposure != 1.0:
        color = np.clip(color.astype(np.float32) * float(exposure), 0, 255).astype(np.uint8)

    import imageio
    imageio.imwrite(out_path, color)
    return out_path


# ----- tiny CLI -----
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="STEP → PNG (smooth, shadows; no Aspose).")
    ap.add_argument("step")
    ap.add_argument("--out", default=None)
    ap.add_argument("--orient", default="iso", choices=list(_ORIENT_ELEV_AZIM.keys()))
    ap.add_argument("--elev", type=float, default=None)
    ap.add_argument("--azim", type=float, default=None)
    ap.add_argument("--quality", default="fine", choices=list(_QUALITY_TO_DEFLECTIONS.keys()))
    ap.add_argument("--size", default="1600x1200")
    args = ap.parse_args()

    w, h = (int(x) for x in args.size.lower().split("x"))
    vea = (args.elev, args.azim) if args.elev is not None and args.azim is not None else None
    out = step_to_png_smooth(args.step, args.out, img_size=(w, h),
                             orient=args.orient, view_elev_azim=vea, quality=args.quality)
    print(out)
