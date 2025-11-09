import app2
import cpacs_to_step3
import cpacs_to_step4
import build_wing_domain_fast3
import run_su2
import plot_wing_drag
import optimize
import visualize

app2.main("simpleAircraft.xml", "plane.cpacs.xml", "Airplane with a circular nose")
cpacs_to_step3.main()

visualize.step_to_png_smooth(
    "plane.stp", "plane1.png",
    view_elev_azim=(10, 160),
    background=(0.08, 0.08, 0.10),   # one simple dark grey
    add_ground=False,                # no floor plane
    model_base=(0.96, 0.96, 0.96),   # light model against dark bg
    key_from_camera=True,
    frame_fill=0.96,
    exposure=1.15,
    quality="ultra",
)

build_wing_domain_fast3.main()
run_su2.run_su2()
plot_wing_drag.main()

prompt = optimize.suggest_change_from_local_image("plane_drag.png")

app2.main("simpleAircraft.xml", "plane2.cpacs.xml", prompt)
cpacs_to_step4.main()

visualize.step_to_png_smooth(
    "plane2.stp", "plane2.png",
    view_elev_azim=(10, 160),
    background=(0.08, 0.08, 0.10),   # one simple dark grey
    add_ground=False,                # no floor plane
    model_base=(0.96, 0.96, 0.96),   # light model against dark bg
    key_from_camera=True,
    frame_fill=0.96,
    exposure=1.15,
    quality="ultra",
)
