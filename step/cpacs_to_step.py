from tixi3 import tixi3wrapper
from tigl3 import tigl3wrapper
from tigl3.configuration import CCPACSConfigurationManager_get_instance

# pythonOCC imports
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Sewing, BRepBuilderAPI_MakeSolid
from OCC.Core.TopoDS import topods_Shell
from OCC.Core.STEPControl import STEPControl_Writer, STEPControl_AsIs
from OCC.Core.IFSelect import IFSelect_RetDone

cpacs_file = "test.cpacs.xml"      # your CPACS v3
step_out   = "wing_solid.stp"
wing_uid   = "wing1"             # adjust to your CPACS

# --- Open CPACS and TiGL ---
tixi = tixi3wrapper.Tixi3()
tixi.open(cpacs_file)

tigl = tigl3wrapper.Tigl3()
tigl.open(tixi, "")

mgr = CCPACSConfigurationManager_get_instance()
aircraft = mgr.get_configuration(tigl._handle.value)

# Get the wing object by UID and its lofted geometry
wing = aircraft.get_wing(wing_uid)      # or get_wing(1) if you use indices
named_shape = wing.get_loft()           # CNamedShape
shape = named_shape.shape()             # OCC.TopoDS_Shape

# --- Sew faces into a closed shell ---
sewer = BRepBuilderAPI_Sewing()    # default tolerance ~1e-6
sewer.Add(shape)
sewer.Perform()
sewed_shape = sewer.SewedShape()
shell = topods_Shell(sewed_shape)

# --- Make a solid from the shell ---
solid_maker = BRepBuilderAPI_MakeSolid()
solid_maker.Add(shell)
solid = solid_maker.Solid()

# --- Export the solid to STEP ---
writer = STEPControl_Writer()
writer.Transfer(solid, STEPControl_AsIs)
status = writer.Write(step_out)
if status != IFSelect_RetDone:
    raise RuntimeError("STEP export failed")

# --- Cleanup ---
tigl.close()
tixi.close()

print(f"Wrote solid STEP to {step_out}")
