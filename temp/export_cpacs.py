# export_cpacs.py
# usage: python export_cpacs.py minimal_wing.cpacs.xml model.step model.stl
import sys
from tixi3 import tixi3wrapper
from tigl3 import tigl3wrapper

if len(sys.argv) != 4:
    print("usage: python export_cpacs.py <cpacs.xml> <out.step> <out.stl>")
    raise SystemExit(1)

cpacs, step_out, stl_out = sys.argv[1], sys.argv[2], sys.argv[3]

tixi = tixi3wrapper.Tixi3()
tixi.open(cpacs)

tigl = tigl3wrapper.Tigl3()
tigl.open(tixi, "")  # open first configuration

tigl.exportSTEP(step_out)
tigl.exportMeshedGeometrySTL(stl_out, 1e-3)

print("Exported:", step_out, "and", stl_out)
