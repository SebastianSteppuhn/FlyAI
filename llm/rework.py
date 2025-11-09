import os
import sys
import json
import argparse
import xml.etree.ElementTree as ET
import base64
import mimetypes
from dotenv import load_dotenv
from openai import OpenAI


def build_system_prompt() -> str:
    """
    Instructions for the model to act as a CPACS 'patch generator'.
    It MUST NOT output XML, only a JSON object with edits to apply.
    """
    return (
        "You are a CPACS XML editing assistant.\n"
        "You will be given:\n"
        "1) A natural-language description of desired changes.\n"
        "2) The full current CPACS XML document.\n"
        "3) Optionally, a CFD image of the aircraft defined by the CPACS.\n\n"
        "Your job is NOT to rewrite the XML.\n"
        "Instead, you MUST output ONLY a JSON object describing edits.\n\n"
        "JSON format (no surrounding markdown, no comments):\n"
        "{\n"
        "  \"edits\": [\n"
        "    {\n"
        "      \"action\": \"set_text\",\n"
        "      \"xpath\": \"<an ElementTree-compatible XPath from the document root, e.g. .//rotor[@uID='Propeller']/nominalRotationsPerMinute>\",\n"
        "      \"value\": \"<new text content>\"\n"
        "    },\n"
        "    {\n"
        "      \"action\": \"set_attribute\",\n"
        "      \"xpath\": \"<XPath to element>\",\n"
        "      \"attribute\": \"<attribute_name>\",\n"
        "      \"value\": \"<new attribute value>\"\n"
        "    }\n"
        "  ]\n"
        "}\n\n"
        "Rules:\n"
        "1. NEVER output XML in your response, only the JSON object.\n"
        "2. Use only actions \"set_text\" and \"set_attribute\".\n"
        "3. Use simple XPaths that work with Python's xml.etree.ElementTree, e.g.:\n"
        "   - .//rotor[@uID='Propeller']/nominalRotationsPerMinute\n"
        "   - .//wing[@uID='Wing']/componentSegments/componentSegment[@uID='Wing_CompSeg']/structure/upperShell/skin/material/thickness\n"
        "4. If multiple nodes should be changed the same way, either:\n"
        "   - Use a single XPath that matches all of them, or\n"
        "   - Add multiple edit objects.\n"
        "5. Always include the top-level key \"edits\" (it can be an empty list if nothing should change).\n"
        "6. Do NOT wrap the JSON in ```json or any other markdown.\n"
        "7. Prefer small, local changes that keep the design close to the original, "
        "   instead of large rewrites of the CPACS structure.\n"
    )


def encode_image_to_data_url(image_path: str) -> str:
    """
    Read a local image file and return a data: URL suitable for the Responses API.
    """
    mime_type, _ = mimetypes.guess_type(image_path)
    if mime_type is None:
        # Fallback; most CFD plots are PNG or JPEG, but PNG is a safe default.
        mime_type = "image/png"

    with open(image_path, "rb") as img_file:
        b64 = base64.b64encode(img_file.read()).decode("utf-8")

    return f"data:{mime_type};base64,{b64}"


def call_openai_for_patch(
    cpacs_xml: str,
    user_edit_prompt: str,
    cfd_image_path: str | None = None,
    design_system_prompt: str | None = None,
) -> dict:
    """
    Ask the OpenAI model to produce a JSON patch describing what to edit.
    Optionally uses a CFD image and an extra system-level design prompt.
    Returns the parsed JSON as a Python dict.
    """
    client = OpenAI()

    # Base system prompt: JSON patch format & CPACS rules
    system_prompt = build_system_prompt()

    # Add optional higher-level design guidance (e.g. "make it more aerodynamic")
    if design_system_prompt:
        system_prompt += (
            "\nAdditional high-level design goal (from the user):\n"
            f"{design_system_prompt}\n"
            "Use this goal to guide which small edits you propose.\n"
        )

    # Optional CFD image
    image_data_url = None
    if cfd_image_path:
        try:
            image_data_url = encode_image_to_data_url(cfd_image_path)
        except OSError as e:
            raise RuntimeError(
                f"Failed to read CFD image '{cfd_image_path}': {e}"
            )

    # Build text part of the user message
    user_message_text = ""

    if cfd_image_path:
        user_message_text += (
            "The attached image is a CFD visualization of the aircraft defined "
            "by the CPACS file.\n"
            "Use it as visual context when deciding small, aerodynamically "
            "meaningful changes.\n\n"
        )

    user_message_text += (
        "User requested changes (natural-language description):\n"
        f"{user_edit_prompt}\n\n"
        "Here is the current CPACS XML document:\n"
        f"{cpacs_xml}"
    )

    # Build multimodal content for the user role
    user_content = [
        {"type": "input_text", "text": user_message_text}
    ]

    if image_data_url:
        user_content.append(
            {
                "type": "input_image",
                "image_url": image_data_url,
            }
        )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]

    response = client.responses.create(
        model="gpt-5",  # or another suitable multimodal model
        input=messages,
    )

    # Get raw text from the response
    try:
        raw_text = response.output_text
    except AttributeError:
        # Fallback: manual extraction if .output_text is not available
        parts = []
        for out in response.output:
            for c in out.content:
                if hasattr(c, "text") and c.text is not None:
                    parts.append(c.text)
        raw_text = "".join(parts)

    # Clean and extract JSON (in case the model ignores instructions slightly)
    first_brace = raw_text.find("{")
    last_brace = raw_text.rfind("}")
    if first_brace == -1 or last_brace == -1 or last_brace < first_brace:
        raise RuntimeError(f"Model did not return a JSON object:\n{raw_text}")

    json_str = raw_text[first_brace:last_brace + 1]

    try:
        patch = json.loads(json_str)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Failed to parse JSON from model: {e}\nRaw text:\n{raw_text}")

    if "edits" not in patch or not isinstance(patch["edits"], list):
        raise RuntimeError(f"JSON does not have the expected 'edits' list: {patch}")

    return patch


def apply_patch_to_xml(cpacs_xml: str, patch: dict) -> str:
    """
    Apply the JSON patch to the CPACS XML string using ElementTree.
    Returns the modified XML string.
    """
    # Parse original XML
    try:
        root = ET.fromstring(cpacs_xml)
    except ET.ParseError as e:
        raise RuntimeError(f"Input CPACS is not well-formed XML: {e}")

    # Optional: register xsi namespace to keep prefix
    ET.register_namespace("xsi", "http://www.w3.org/2001/XMLSchema-instance")

    edits = patch.get("edits", [])
    for edit in edits:
        action = edit.get("action")
        xpath = edit.get("xpath")
        value = edit.get("value")

        if not action or not xpath:
            continue  # skip malformed edit entries

        elements = root.findall(xpath)
        if not elements:
            # You might want to log this instead of raising, so one bad xpath doesn't kill everything
            print(f"Warning: XPath did not match any elements: {xpath}", file=sys.stderr)
            continue

        if action == "set_text":
            for el in elements:
                el.text = value
        elif action == "set_attribute":
            attr_name = edit.get("attribute")
            if not attr_name:
                print(f"Warning: 'set_attribute' edit without 'attribute' name: {edit}", file=sys.stderr)
                continue
            for el in elements:
                el.set(attr_name, value)
        else:
            print(f"Warning: Unknown action '{action}' in edit: {edit}", file=sys.stderr)

    # Build new XML string
    tree = ET.ElementTree(root)
    from io import BytesIO
    buf = BytesIO()
    tree.write(buf, encoding="utf-8", xml_declaration=True)
    return buf.getvalue().decode("utf-8")


def main():
    load_dotenv()

    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not found. Put it in a .env file or environment.", file=sys.stderr)
        sys.exit(1)

    parser = argparse.ArgumentParser(
        description="Safely edit CPACS XML using a natural-language prompt and optionally a CFD image via the OpenAI API."
    )
    parser.add_argument(
        "-i", "--input",
        required=True,
        help="Path to input CPACS XML file (e.g. input_cpacs.xml)"
    )
    parser.add_argument(
        "-o", "--output",
        required=True,
        help="Path to output CPACS XML file (e.g. edited_cpacs.xml)"
    )
    parser.add_argument(
        "-p", "--prompt",
        help=(
            "Natural-language description of edits or design intent to apply. "
            "If omitted, you will be prompted interactively."
        )
    )
    parser.add_argument(
        "--image",
        help=(
            "Path to a CFD image (PNG/JPEG) of the CPACS aircraft. "
            "If provided, it will be sent to the model as visual context."
        )
    )
    parser.add_argument(
        "-s", "--system-prompt",
        help=(
            "Additional high-level design system prompt, e.g. "
            "'Improve the design by making it more aerodynamic. Maybe make the nose steeper.'"
        )
    )

    args = parser.parse_args()

    # Get user edit prompt
    if args.prompt:
        edit_prompt = args.prompt
    else:
        print("Describe the changes you want to apply to the CPACS file:")
        edit_prompt = input("> ").strip()
        if not edit_prompt:
            print("No prompt given, aborting.", file=sys.stderr)
            sys.exit(1)

    # Read input CPACS XML
    try:
        with open(args.input, "r", encoding="utf-8") as f:
            cpacs_xml = f.read()
    except OSError as e:
        print(f"Failed to read input file '{args.input}': {e}", file=sys.stderr)
        sys.exit(1)

    # 1) Ask OpenAI for a JSON patch
    try:
        patch = call_openai_for_patch(
            cpacs_xml,
            edit_prompt,
            cfd_image_path=args.image,
            design_system_prompt=args.system_prompt,
        )
    except Exception as e:
        print(f"Error while calling OpenAI or parsing its JSON response:\n{e}", file=sys.stderr)
        sys.exit(1)

    # 2) Apply patch locally, keeping XML structure valid
    try:
        edited_xml = apply_patch_to_xml(cpacs_xml, patch)
    except Exception as e:
        print(f"Error while applying patch to XML:\n{e}", file=sys.stderr)
        sys.exit(1)

    # 3) Final sanity check: ensure result is still well-formed
    try:
        ET.fromstring(edited_xml)
    except ET.ParseError as e:
        print(f"Edited XML is NOT well-formed (this should not happen): {e}", file=sys.stderr)
        sys.exit(1)

    # 4) Write output CPACS
    try:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(edited_xml)
    except OSError as e:
        print(f"Failed to write output file '{args.output}': {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Done. Edited CPACS written to: {args.output}")


if __name__ == "__main__":
    main()
