import os
import sys
import json
import argparse
import xml.etree.ElementTree as ET
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
        "2) The full current CPACS XML document.\n\n"
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
    )


def call_openai_for_patch(cpacs_xml: str, user_edit_prompt: str) -> dict:
    """
    Ask the OpenAI model to produce a JSON patch describing what to edit.
    Returns the parsed JSON as a Python dict.
    """
    client = OpenAI()

    system_prompt = build_system_prompt()

    user_message = (
        "User requested changes:\n"
        f"{user_edit_prompt}\n\n"
        "Here is the current CPACS XML document:\n"
        f"{cpacs_xml}"
    )

    response = client.responses.create(
        model="gpt-5",  # or another suitable model name
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
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


def main(input_path: str, output_path: str, edit_prompt: str) -> str:
    """
    Safely edit a CPACS XML file using a natural-language prompt via the OpenAI API.

    Parameters
    ----------
    input_path : str
        Path to input CPACS XML file.
    output_path : str
        Path where the edited CPACS XML file will be written.
    edit_prompt : str
        Natural-language description of edits to apply.

    Returns
    -------
    str
        The edited CPACS XML content.
    """
    load_dotenv()

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY not found. Put it in a .env file or environment.")

    if not edit_prompt:
        raise ValueError("edit_prompt must be a non-empty string.")

    # Read input CPACS XML
    try:
        with open(input_path, "r", encoding="utf-8") as f:
            cpacs_xml = f.read()
    except OSError as e:
        raise RuntimeError(f"Failed to read input file '{input_path}': {e}") from e

    # 1) Ask OpenAI for a JSON patch
    patch = call_openai_for_patch(cpacs_xml, edit_prompt)

    # 2) Apply patch locally, keeping XML structure valid
    edited_xml = apply_patch_to_xml(cpacs_xml, patch)

    # 3) Final sanity check: ensure result is still well-formed
    try:
        ET.fromstring(edited_xml)
    except ET.ParseError as e:
        raise RuntimeError(f"Edited XML is NOT well-formed (this should not happen): {e}") from e

    # 4) Write output CPACS
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(edited_xml)
    except OSError as e:
        raise RuntimeError(f"Failed to write output file '{output_path}': {e}") from e

    print(f"Done. Edited CPACS written to: {output_path}")
    return edited_xml


if __name__ == "__main__":
    main()
