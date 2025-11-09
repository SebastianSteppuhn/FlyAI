import os
import base64
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


def _guess_mime_type(image_path: str) -> str:
    ext = Path(image_path).suffix.lower()
    if ext in [".jpg", ".jpeg"]:
        return "image/jpeg"
    if ext == ".png":
        return "image/png"
    if ext == ".webp":
        return "image/webp"
    if ext == ".gif":
        return "image/gif"
    # reasonable default
    return "image/png"


def suggest_change_from_local_image(image_path: str) -> str:
    """
    Look at a local image and return exactly one short design suggestion sentence.

    Example return value: "make the nose more pointed"

    Parameters
    ----------
    image_path : str
        Path to a local image file (png/jpg/webp/gif).
    system_prompt : str
        High-level instruction, e.g.:
        "You are an aircraft designer. Suggest improvements to the shape."

    Returns
    -------
    str
        A single short, imperative suggestion sentence (no trailing period).
    """
    load_dotenv()

    system_prompt = "You are an aircraft aerodynamicist critiquing conceptual aircraft designs."

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not found. Put it in a .env file or environment.")

    if not Path(image_path).is_file():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    client = OpenAI(api_key=api_key)

    # Encode image as base64
    with open(image_path, "rb") as image_file:
        b64_image = base64.b64encode(image_file.read()).decode("utf-8")

    mime_type = _guess_mime_type(image_path)

    # Tight instructions so we really get just one short suggestion
    instructions = (
        system_prompt.strip()
        + " Always respond with exactly one short, imperative suggestion sentence "
          "(max 10 words), all lowercase, no trailing period, no explanations."
    )

    user_text = "Suggest one concrete geometric or shape improvement for this object the best would be making the nose more pointed. And nothing complex at all"

    response = client.responses.create(
        model="gpt-4o-mini",  # vision-capable model
        instructions=instructions,
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": user_text},
                    {
                        "type": "input_image",
                        "image_url": f"data:{mime_type};base64,{b64_image}",
                    },
                ],
            }
        ],
    )

    suggestion = (response.output_text or "").strip()

    # If the model somehow returns multiple lines, just keep the first non-empty one
    for line in suggestion.splitlines():
        line = line.strip()
        if line:
            return line

    return suggestion


