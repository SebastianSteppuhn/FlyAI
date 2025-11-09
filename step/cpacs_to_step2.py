#!/usr/bin/env python3
# cpacs_to_solid_step_debug.py
# CPACS 3.x -> Solid STEP with robust symmetry (x/y/z) + per-part debug export.

# ------------------ USER SETTINGS ------------------
CPACS_FILE = "test.cpacs.xml"   # Input CPACS
STEP_OUT   = "test2_solid.stp"   # Output (all parts)
CONFIG_UID = None                   # e.g. "PromptPlane_UID", or None to auto-pick first
SEW_TOL    = 1e-6                   # Sewing tolerance
FUSE_ALL   = True                  # Fuse solids into one
EXPORT_OPEN_AS_SHELL = True         # Export shells when solid fails
EXPORT_PARTS_DIR = "out_parts"      # Per-part STEP export (set None to disable)
VERBOSE    = True
# ---------------------------------------------------

import os
from pathlib import Path
from lxml import etree

from tixi3 import tixi3wrapper as tixi3
from tigl3 import tigl3wrapper as tigl3
from tigl3.configuration import CCPACSConfigurationManager_get_instance

# pythonOCC
from OCC.Core.BRepBuilderAPI import (
    BRepBuilderAPI_Sewing,
    BRepBuilderAPI_MakeSolid,
    BRepBuilderAPI_Transform,
)
from OCC.Core.BRepCheck import BRepCheck_Analyzer
from OCC.Core.BRep import BRep_Builder
from OCC.Core.gp import gp_Trsf, gp_Pln, gp_Pnt, gp_Dir
from OCC.Core.TopoDS import TopoDS_Shape, topods_Shell, TopoDS_Compound
from OCC.Core.STEPControl import STEPControl_Writer, STEPControl_AsIs
from OCC.Core.IFSelect import IFSelect_RetDone
from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Fuse

def log(msg):
    if VERBOSE:
        print(msg)

# ---------- Config UID detection ----------
def find_config_uids(cpacs_path: str):
    tree = etree.parse(cpacs_path)
    root = tree.getroot()
    uids = []
    for m in root.findall("./vehicles/aircraft/model"):
        uid = m.get("uID")
        if uid: uids.append(uid)
    for m in root.findall("./vehicles/rotorcraft/model"):
        uid = m.get("uID")
        if uid: uids.append(uid)
    return uids

# ---------- Wing symmetry lookup ----------
def map_wing_symmetry(cpacs_path: str, cfg_uid: str):
    """
    Returns {wing_uid: 'x'|'y'|'z'|None} under selected model.
    Looks under both .../aircraftModel/wings and .../wings (fallback).
    """
    tree = etree.parse(cpacs_path)
    root = tree.getroot()

    base = None
    for m in root.findall("./vehicles/aircraft/model"):
        if m.get("uID") == cfg_uid:
            base = m; break
    if base is None:
        for m in root.findall("./vehicles/rotorcraft/model"):
            if m.get("uID") == cfg_uid:
                base = m; break
    if base is None:
        return {}

    wings = base.findall("./aircraftModel/wings/wing")
    if not wings:
        wings = base.findall("./wings/wing")

    sym = {}
    for w in wings:
        uid = w.get("uID")
        sv = None
        se = w.find("symmetry")
        if se is not None and se.text:
            sv = se.text.strip().lower()  # 'x' | 'y' | 'z'
        if uid:
            sym[uid] = sv
    return sym

# ---------- Geometry utils ----------
def shape_is_valid(shape: TopoDS_Shape) -> bool:
    try:
        return BRepCheck_Analyzer(shape, True).IsValid()
    except Exception:
        return False

def sew_to_shell(shape: TopoDS_Shape, tol: float):
    sewer = BRepBuilderAPI_Sewing(tol)
    sewer.Add(shape)
    sewer.Perform()
    sewed = sewer.SewedShape()
    try:
        return topods_Shell(sewed), True
    except Exception:
        return sewed, False

def shell_to_solid(shell_or_shape: TopoDS_Shape):
    try:
        shell = topods_Shell(shell_or_shape)
        mk = BRepBuilderAPI_MakeSolid()
        mk.Add(shell)
        solid = mk.Solid()
        if shape_is_valid(solid):
            return solid
    except Exception:
        pass
    return shell_or_shape

def mirror(shape: TopoDS_Shape, plane: str) -> TopoDS_Shape:
    """
    plane: 'x' -> mirror across YZ (X=0)
           'y' -> mirror across XZ (Y=0)
           'z' -> mirror across XY (Z=0)
    """
    tr = gp_Trsf()
    if plane == "x":
        tr.SetMirror(gp_Pln(gp_Pnt(0,0,0), gp_Dir(1,0,0)))
    elif plane == "y":
        tr.SetMirror(gp_Pln(gp_Pnt(0,0,0), gp_Dir(0,1,0)))
    elif plane == "z":
        tr.SetMirror(gp_Pln(gp_Pnt(0,0,0), gp_Dir(0,0,1)))
    else:
        return shape
    return BRepBuilderAPI_Transform(shape, tr, True).Shape()

def fuse_all_solids(solids):
    if not solids: return None
    res = solids[0]
    for s in solids[1:]:
        try:
            fu = BRepAlgoAPI_Fuse(res, s); fu.Build()
            if fu.IsDone():
                res = fu.Shape()
        except Exception:
            pass
    return res

def make_compound(shapes):
    builder = BRep_Builder()
    comp = TopoDS_Compound()
    builder.MakeCompound(comp)
    for s in shapes:
        try: builder.Add(comp, s)
        except Exception: pass
    return comp

def export_step_shapes(shapes, out_path: str):
    writer = STEPControl_Writer()
    count = 0
    for s in shapes:
        try:
            writer.Transfer(s, STEPControl_AsIs)
            count += 1
        except Exception as e:
            log(f" ! Skip shape (transfer failed): {e}")
    if count == 0:
        raise RuntimeError("No shapes transferred to STEP writer.")
    status = writer.Write(out_path)
    if status != IFSelect_RetDone:
        raise RuntimeError("STEP export failed.")
    return count

def export_single_step(shape, out_path: str):
    writer = STEPControl_Writer()
    writer.Transfer(shape, STEPControl_AsIs)
    status = writer.Write(out_path)
    if status != IFSelect_RetDone:
        raise RuntimeError(f"STEP export failed for {out_path}")

# ---------- TiGL component collection ----------
def collect_components(aircraft):
    """Yield (kind, uid, topo_shape) for wings and fuselages (base shapes only)."""
    # Wings
    try:
        nw = aircraft.get_wing_count()
        log(f"Wings reported by TiGL: {nw}")
        for i in range(1, nw + 1):
            w = aircraft.get_wing(i)
            uid = w.get_uid()
            try:
                shape = w.get_loft().shape()
                yield ("wing", uid, shape)
            except Exception as e:
                log(f" ! Wing {uid} loft failed: {e}")
    except Exception as e:
        log(f" ! Wing enumeration failed: {e}")

    # Fuselages
    try:
        nf = aircraft.get_fuselage_count()
        log(f"Fuselages reported by TiGL: {nf}")
        for i in range(1, nf + 1):
            f = aircraft.get_fuselage(i)
            uid = f.get_uid()
            try:
                shape = f.get_loft().shape()
                yield ("fuselage", uid, shape)
            except Exception as e:
                log(f" ! Fuselage {uid} loft failed: {e}")
    except Exception as e:
        log(f" ! Fuselage enumeration failed: {e}")

def main():
    # --- Open CPACS ---
    log(f"Opening CPACS: {CPACS_FILE}")
    tixi = tixi3.Tixi3()
    tixi.open(CPACS_FILE)

    # --- Pick configuration ---
    uids = find_config_uids(CPACS_FILE)
    if not uids:
        raise RuntimeError("No configurations found at /cpacs/vehicles/(aircraft|rotorcraft)/model[@uID].")
    cfg_uid = CONFIG_UID or uids[0]
    if CONFIG_UID and CONFIG_UID not in uids:
        raise RuntimeError(f'CONFIG_UID="{CONFIG_UID}" not in CPACS. Available: {uids}')
    log(f"Using configuration UID: {cfg_uid}")

    tigl = tigl3.Tigl3()
    tigl.open(tixi, cfg_uid)

    mgr = CCPACSConfigurationManager_get_instance()
    aircraft = mgr.get_configuration(tigl._handle.value)

    # symmetry map for wings
    wing_sym = map_wing_symmetry(CPACS_FILE, cfg_uid)
    if wing_sym:
        log("Wing symmetry map: " + ", ".join([f"{k}:{v}" for k,v in wing_sym.items()]))

    # --- Build shapes ---
    if EXPORT_PARTS_DIR:
        Path(EXPORT_PARTS_DIR).mkdir(parents=True, exist_ok=True)

    solids, shells = [], []
    made = 0
    for kind, uid, base_shape in collect_components(aircraft):
        to_process = [(kind, uid, base_shape)]

        # Add mirrored variant(s) if symmetry declared
        sym = (wing_sym.get(uid) or "").lower() if kind == "wing" else None
        if sym in ("x","y","z"):
            mir = mirror(base_shape, sym)
            to_process.append((f"{kind}_mirror_{sym}", f"{uid}_mirror", mir))

        for k, u, shp in to_process:
            made += 1
            log(f"- {k} {u}: sew (tol={SEW_TOL})")
            sewed, was_shell = sew_to_shell(shp, SEW_TOL)

            solid = shell_to_solid(sewed)
            if shape_is_valid(solid):
                log("  ✓ solid")
                solids.append(solid)
                if EXPORT_PARTS_DIR:
                    export_single_step(solid, os.path.join(EXPORT_PARTS_DIR, f"{k}_{u}_solid.stp"))
            else:
                msg = "  ⚠ solid failed; keeping shell" if was_shell else "  ⚠ solid failed; keeping shape"
                log(msg)
                if EXPORT_OPEN_AS_SHELL:
                    shells.append(sewed)
                    if EXPORT_PARTS_DIR:
                        export_single_step(sewed, os.path.join(EXPORT_PARTS_DIR, f"{k}_{u}_shell.stp"))

    if made == 0:
        raise RuntimeError("No loftable components found (wings/fuselages).")

    # --- Prepare export set ---
    export_shapes = []
    if FUSE_ALL and len(solids) > 1:
        log("Fusing all solids into one (may take time)…")
        fused = fuse_all_solids(solids)
        export_shapes = [fused] if fused else solids[:]
    else:
        export_shapes = solids[:]

    if EXPORT_OPEN_AS_SHELL and shells:
        export_shapes.extend(shells)

    if not export_shapes:
        raise RuntimeError("Nothing to export (no solids and shells disabled).")

    # --- Export combined STEP ---
    log(f"Exporting combined STEP: {STEP_OUT} ({len(export_shapes)} shape(s))")
    n = export_step_shapes(export_shapes, STEP_OUT)
    log(f"✓ Wrote {n} shape(s) to {STEP_OUT}")

    tigl.close(); tixi.close()

if __name__ == "__main__":
    main()
