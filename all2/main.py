import os
import sys
import subprocess
import gradio as gr
import app2
import cpacs_to_step3
import cpacs_to_step4
import run_su2
import plot_wing_drag
import optimize
import visualize

FIXED_GEN_IMG  = "plane1.png"
FIXED_OPT_IMG1 = "plane_drag.png"
FIXED_OPT_IMG2 = "plane2.png"

def _run_gmsh_domain_in_subprocess():
    """
    Runs build_wing_domain_fast3.main() in a fresh Python process so gmsh.initialize()
    happens in that process's main thread (avoids signal handler error).
    """
    code = "import build_wing_domain_fast3 as m; m.main()"
    subprocess.run([sys.executable, "-c", code], check=True)

def gen_from_prompt(_prompt: str) -> str:
    app2.main("simpleAircraft.xml", "plane.cpacs.xml", _prompt)
    cpacs_to_step3.main()

    visualize.step_to_png_smooth(
        "plane.stp", "plane1.png",
        view_elev_azim=(10, 160),
        background=(0.08, 0.08, 0.10),
        add_ground=False,
        model_base=(0.96, 0.96, 0.96),
        key_from_camera=True,
        frame_fill=0.96,
        exposure=1.15,
        quality="ultra",
    )

    if not os.path.exists(FIXED_GEN_IMG):
        raise gr.Error(f"Datei nicht gefunden: {FIXED_GEN_IMG}")
    return FIXED_GEN_IMG

def enable_opt_button():
    return gr.update(interactive=True)

def run_optimize():
    # <<< run gmsh domain build in a separate process >>>
    _run_gmsh_domain_in_subprocess()

    run_su2.run_su2()
    plot_wing_drag.main()

    prompt = optimize.suggest_change_from_local_image("plane_drag.png")

    app2.main("simpleAircraft.xml", "plane2.cpacs.xml", prompt)
    cpacs_to_step4.main()

    visualize.step_to_png_smooth(
        "plane2.stp", "plane2.png",
        view_elev_azim=(10, 160),
        background=(0.08, 0.08, 0.10),
        add_ground=False,
        model_base=(0.96, 0.96, 0.96),
        key_from_camera=True,
        frame_fill=0.96,
        exposure=1.15,
        quality="ultra",
    )

    for p in (FIXED_OPT_IMG1, FIXED_OPT_IMG2):
        if not os.path.exists(p):
            raise gr.Error(f"Datei nicht gefunden: {p}")

    return (
        gr.update(value=FIXED_OPT_IMG1, visible=True),
        gr.update(value=FIXED_OPT_IMG2, visible=True),
    )

with gr.Blocks(title="FlyAI") as demo:
    gr.Markdown("# FlyAI - efficient design made simple")

    with gr.Tab("Workflow"):
        prompt = gr.Textbox(label="Prompt", placeholder="input prompt")

        with gr.Row():
            btn_gen = gr.Button("Generate", variant="primary")
            btn_opt = gr.Button("Optimize", interactive=False)

        # Smaller images via explicit height (width auto)
        out1 = gr.Image(label="Generate-Resultat (fixed path)", type="filepath", height=320)

        with gr.Row():
            out_opt1 = gr.Image(label="Optimized Bild 1 (fixed)", type="filepath", visible=False, height=320)
            out_opt2 = gr.Image(label="Optimized Bild 2 (fixed)", type="filepath", visible=False, height=320)

        evt = btn_gen.click(gen_from_prompt, inputs=prompt, outputs=out1)
        evt.then(enable_opt_button, outputs=btn_opt)
        btn_opt.click(run_optimize, outputs=[out_opt1, out_opt2])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=8080)
