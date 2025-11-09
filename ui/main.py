import gradio as gr

# --- Replace these with real absolute paths on your machine ---
FIXED_PATH_PAGE1 = "/absolute/path/to/your/fixed_page1.png"
FIXED_PATH_PAGE3 = "/absolute/path/to/your/fixed_page3.png"
# --------------------------------------------------------------

def fixed_from_prompt(_prompt: str) -> str:
    # Ignore the prompt; always return the same file
    return FIXED_PATH_PAGE1

def passthrough_selected(file_obj) -> str | None:
    # Return the uploaded file path as-is (or None if nothing)
    return getattr(file_obj, "name", None) if file_obj else None

def fixed_image_no_input() -> str:
    # Always return the same file
    return FIXED_PATH_PAGE3

with gr.Blocks(title="3-Page Demo") as demo:
    gr.Markdown("# Simple 3-Page Gradio App (filepath mode)")

    with gr.Tabs():
        # Page 1: Prompt -> fixed file path
        with gr.Tab("1) Prompt → Image"):
            prompt = gr.Textbox(label="Prompt", placeholder="Type anything…")
            out1 = gr.Image(label="Image (from file path)", type="filepath")
            btn1 = gr.Button("Show Image")
            btn1.click(fixed_from_prompt, inputs=prompt, outputs=out1)
            prompt.submit(fixed_from_prompt, inputs=prompt, outputs=out1)

        # Page 2: File selector + Run -> show that same selected file
        with gr.Tab("2) File → Run"):
            file_in1 = gr.File(label="Select an image file", file_types=["image"])
            run_btn1 = gr.Button("Run")
            out2 = gr.Image(label="Selected file preview", type="filepath")
            run_btn1.click(passthrough_selected, inputs=file_in1, outputs=out2)

        # Page 3: Button -> fixed file path (no inputs)
        with gr.Tab("3) Button → Image"):
            run_btn2 = gr.Button("Show Fixed Image")
            out3 = gr.Image(label="Fixed Image (from path)", type="filepath")
            run_btn2.click(fixed_image_no_input, outputs=out3)

if __name__ == "__main__":
    demo.launch()
