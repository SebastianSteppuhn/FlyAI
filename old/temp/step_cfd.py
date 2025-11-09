#!/usr/bin/env python3
import argparse
import os
import sys
import shutil
import subprocess
from pathlib import Path
import numpy as np
import pandas as pd

# ---------- Gmsh meshing: STEP -> external volume mesh ----------
def build_mesh_from_step(
    step_path: Path,
    msh_path: Path,
    near_ratio=0.02,
    far_ratio=0.5,
    box_upstream=5.0,
    box_downstream=15.0,
    box_side=10.0,
    box_topbottom=10.0,
):
    import gmsh

    gmsh.initialize()
    gmsh.model.add("step_ext_flow")

    # Import geometry
    gmsh.model.occ.importShapes(str(step_path))
    gmsh.model.occ.synchronize()

    # --- Robust overall bounding box (works for any dim) ---
    have_any = False
    xmn, ymn, zmn = +1e300, +1e300, +1e300
    xmx, ymx, zmx = -1e300, -1e300, -1e300
    for dim in (3, 2, 1, 0):
        for _, tag in gmsh.model.getEntities(dim):
            bx = gmsh.model.getBoundingBox(dim, tag)  # <- NOT occ.getBoundingBox
            xmn, ymn, zmn = min(xmn, bx[0]), min(ymn, bx[1]), min(zmn, bx[2])
            xmx, ymx, zmx = max(xmx, bx[3]), max(ymx, bx[4]), max(zmx, bx[5])
            have_any = True
    if not have_any or not np.isfinite([xmn, ymn, zmn, xmx, ymx, zmx]).all():
        gmsh.finalize()
        raise RuntimeError("No valid entities found after STEP import. Ensure the STEP has solids/surfaces.")

    L = max(xmx - xmn, ymx - ymn, zmx - zmn)
    if L <= 0:
        gmsh.finalize()
        raise RuntimeError("Degenerate bounding box from STEP geometry.")

    # --- Build external box ---
    xc = 0.5 * (xmn + xmx)
    x0 = xc - box_upstream * L
    y0 = (ymn - box_side * L)
    z0 = (zmn - box_topbottom * L)
    box = gmsh.model.occ.addBox(
        x0,
        y0,
        z0,
        (box_upstream + box_downstream) * L,
        (box_side * 2) * L,
        (box_topbottom * 2) * L,
    )
    gmsh.model.occ.synchronize()

    # Save box surfaces for 'farfield'
    box_surfs = gmsh.model.getBoundary([(3, box)], oriented=False, combined=False)

    # --- Cut: subtract ONLY the aircraft solids (exclude the box itself) ---
    all_solids = gmsh.model.occ.getEntities(3)
    aircraft_solids = [(d, t) for (d, t) in all_solids if t != box]
    if not aircraft_solids:
        gmsh.finalize()
        raise RuntimeError("No 3D solids in STEP (need watertight solids).")

    fluid, _ = gmsh.model.occ.cut([(3, box)], aircraft_solids, removeTool=False)
    gmsh.model.occ.synchronize()
    if not fluid:
        gmsh.finalize()
        raise RuntimeError("Boolean cut failed (geometry may be non-watertight).")

    fluid_vol = fluid[0]
    fluid_surfs = gmsh.model.getBoundary([fluid_vol], oriented=False, combined=False)

    # 'wall' = fluid boundary that isn't the farfield box
    box_set = set(box_surfs)
    wall_surfs = [s for s in fluid_surfs if s not in box_set]

    # Sizing
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", L * near_ratio)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", L * far_ratio)

    # Mild refinement near the wall
    try:
        gmsh.model.mesh.field.add("Distance", 1)
        gmsh.model.mesh.field.setNumbers(1, "SurfacesList", [s[1] for s in wall_surfs])
        gmsh.model.mesh.field.setNumber(1, "NumPointsPerCurve", 100)

        gmsh.model.mesh.field.add("Threshold", 2)
        gmsh.model.mesh.field.setNumber(2, "InField", 1)
        gmsh.model.mesh.field.setNumber(2, "SizeMin", L * near_ratio)
        gmsh.model.mesh.field.setNumber(2, "SizeMax", L * far_ratio)
        gmsh.model.mesh.field.setNumber(2, "DistMin", 0.1 * L)
        gmsh.model.mesh.field.setNumber(2, "DistMax", 2.0 * L)
        gmsh.model.mesh.field.setAsBackgroundMesh(2)
    except Exception:
        pass

    # Physical groups
    def phys(name, dim, entities):
        tag = gmsh.model.addPhysicalGroup(dim, [e[1] for e in entities])
        gmsh.model.setPhysicalName(dim, tag, name)
        return tag

    phys("farfield", 2, box_surfs)
    phys("wall", 2, wall_surfs)
    phys("fluid", 3, [fluid_vol])

    gmsh.model.mesh.generate(3)
    gmsh.write(str(msh_path))
    gmsh.finalize()
    return L


# ---------- Convert Gmsh -> SU2 ----------
def mshto_su2(msh_path: Path, su2_path: Path):
    import meshio
    mesh = meshio.read(msh_path)
    meshio.write(str(su2_path), mesh)


# ---------- SU2 config ----------
def write_su2_cfg(cfg_path: Path, mesh_su2: Path, mach: float, aoa_deg: float,
                  reynolds: float, ref_len: float, inviscid: bool, iters: int):
    wall_marker_line = "MARKER_EULER= ( wall )" if inviscid else "MARKER_HEATFLUX= ( wall, 0.0 )"
    turb = "NONE" if inviscid else "SA"
    cfg = f"""%
% Auto-generated SU2 config (STEP -> mesh -> SU2)
MATH_PROBLEM= DIRECT
SOLVER= {'EULER' if inviscid else 'RANS'}
KIND_TURB_MODEL= {turb}

NDIM= 3
MESH_FILENAME= {mesh_su2.name}
HISTORY_FILENAME= history.csv
RESTART_FILENAME= restart.dat

MACH_NUMBER= {mach}
AOA= {aoa_deg}
SIDESLIP_ANGLE= 0.0
GAMMA_VALUE= 1.4
FREESTREAM_TEMPERATURE= 288.15
FREESTREAM_PRESSURE= 101325.0
REYNOLDS_NUMBER= {reynolds}
REYNOLDS_LENGTH= {ref_len}

MARKER_FAR= ( farfield )
{wall_marker_line}

NUM_METHOD_GRAD= GREEN_GAUSS
CFL_NUMBER= 5.0
CFL_ADAPT= YES
CFL_ADAPT_PARAM= ( 0.5, 2.0, 1.1, 50.0 )
LINEAR_SOLVER= FGMRES
LINEAR_SOLVER_PREC= ILU
LINEAR_SOLVER_ERROR= 1e-6
LINEAR_SOLVER_ITER= 50

CONV_NUM_METHOD_FLOW= ROE
VENKAT_LIMITER_COEFF= 0.05
MUSCL_FLOW= YES
SLOPE_LIMITER_FLOW= VENKATAKRISHNAN

ITER= {iters}
CONV_RESIDUAL_MINVAL= 1e-10
CONV_STARTITER= 10
SCREEN_OUTPUT= (ITER, RMS_RES, LIFT, DRAG, MACH, AOA)
HISTORY_OUTPUT= (ITER, RMS_RES, CL, CD, CMZ)
"""
    cfg_path.write_text(cfg)


# ---------- Run SU2 ----------
def run_su2(cfg_path: Path, workdir: Path):
    exe = shutil.which("SU2_CFD")
    if not exe:
        raise RuntimeError("SU2_CFD not found in PATH (activate your conda env).")
    res = subprocess.run([exe, cfg_path.name], cwd=workdir, stdout=sys.stdout, stderr=sys.stderr)
    if res.returncode != 0:
        raise RuntimeError("SU2_CFD returned non-zero exit code.")


# ---------- Post-process ----------
def summarize_history(history_csv: Path):
    if not history_csv.exists():
        return "No history.csv produced – SU2 may have failed early."
    try:
        df = pd.read_csv(history_csv)
    except Exception:
        try:
            df = pd.read_csv(history_csv, sep=';')
        except Exception:
            df = pd.read_csv(history_csv, delim_whitespace=True)

    def pick(*cands):
        for c in cands:
            if c in df.columns: return c
        return None

    c_iter = pick("ITER", "Iter", "Iteration")
    c_cl = pick("CL", "Lift_Coefficient", "LIFT")
    c_cd = pick("CD", "Drag_Coefficient", "DRAG")
    c_res = pick("RMS_RES", "RMS_Density", "RMS_RESIDUAL")

    lines = []
    last = df.iloc[-1]

    if c_cl and c_cd:
        CL = float(last[c_cl]); CD = float(last[c_cd])
        lines.append(f"Final coefficients: CL = {CL:.4f}, CD = {CD:.4f}, L/D = {CL/CD:.2f}")
        if len(df) > 20:
            tail = df.tail(20)
            dCL = tail[c_cl].max() - tail[c_cl].min()
            dCD = tail[c_cd].max() - tail[c_cd].min()
            lines.append(f"Stability over last 20 iters: ΔCL = {dCL:.4e}, ΔCD = {dCD:.4e}")
    if c_res:
        res0 = float(df.iloc[0][c_res])
        resF = float(last[c_res])
        drop = np.log10(max(res0, 1e-99)) - np.log10(max(resF, 1e-99))
        lines.append(f"Residual drop: {drop:.1f} orders (final {c_res} = {resF:.2e})")
    if c_iter:
        lines.append(f"Iterations run: {int(last[c_iter])}")

    verdict = "✅ Converged-looking solution" if (c_res and resF < 1e-5) else "⚠️ May need more iters / refinement"
    return verdict + "\n" + "\n".join(lines)


# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser(description="STEP → mesh → SU2 CFD (conda-only).")
    ap.add_argument("--step", required=True, help="Path to STEP geometry (.step/.stp)")
    ap.add_argument("--mach", type=float, default=0.2, help="Freestream Mach number")
    ap.add_argument("--aoa", type=float, default=2.0, help="Angle of attack [deg]")
    ap.add_argument("--re", type=float, default=5e6, help="Reynolds number")
    ap.add_argument("--iters", type=int, default=800, help="SU2 iterations")
    ap.add_argument("--inviscid", action="store_true", help="Run Euler (no viscosity/turbulence)")
    ap.add_argument("--near", type=float, default=0.02, help="Near-wall size ratio (L*near)")
    ap.add_argument("--far", type=float, default=0.5, help="Far-field size ratio (L*far)")
    ap.add_argument("--box-up", type=float, default=5.0, help="Box upstream multiples of L")
    ap.add_argument("--box-down", type=float, default=15.0, help="Box downstream multiples of L")
    ap.add_argument("--box-side", type=float, default=10.0, help="Box lateral multiples of L")
    ap.add_argument("--box-z", type=float, default=10.0, help="Box top/bottom multiples of L")
    ap.add_argument("--outdir", default="run_step", help="Output directory")
    args = ap.parse_args()

    work = Path(args.outdir).resolve()
    work.mkdir(parents=True, exist_ok=True)

    step_path = Path(args.step).resolve()
    msh = work / "mesh.msh"
    su2mesh = work / "mesh.su2"
    cfg = work / "case.cfg"
    hist = work / "history.csv"

    print("=== 1) Meshing STEP with Gmsh ===")
    L = build_mesh_from_step(
        step_path,
        msh,
        near_ratio=args.near,
        far_ratio=args.far,
        box_upstream=args.box_up,
        box_downstream=args.box_down,
        box_side=args.box_side,
        box_topbottom=args.box_z,
    )
    print(f"  wrote: {msh}")

    print("=== 2) Converting to SU2 mesh ===")
    mshto_su2(msh, su2mesh)
    print(f"  wrote: {su2mesh}")

    print("=== 3) Writing SU2 config ===")
    write_su2_cfg(cfg, su2mesh, args.mach, args.aoa, args.re, L, args.inviscid, args.iters)
    print(f"  wrote: {cfg}")

    print("=== 4) Running SU2 ===")
    run_su2(cfg, work)

    print("=== 5) Post-process ===")
    print("\n" + summarize_history(hist))


if __name__ == "__main__":
    main()
