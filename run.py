import all.build_wing_domain_fast3
import llm.app2
from all import *
import all.run_su2
import all.plot_wing_drag
import step.cpacs_to_step3
from llm import *

llm.app2.main("llm/simpleAircraft.xml", "llm/plane.cpacs.xml", "Airplane with a dull nose but not inverted")
step.cpacs_to_step3.main()
all.build_wing_domain_fast3.main()
all.run_su2.main()
all.plot_wing_drag.main()