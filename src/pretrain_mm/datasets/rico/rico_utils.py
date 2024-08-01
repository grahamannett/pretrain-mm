import re


# FOR screenannotation
def parse_ui_elements(text):
    """use like the follwoing:

    # Parse the input text
    parsed_ui = parse_ui_elements(input_string)

    # Convert to JSON
    json_output = json.dumps({"UI": parsed_ui}, indent=2)
    print(json_output)
    """
    # Extract main sections with nested items if present
    pattern = r"([A-Z_]+) ([\d ]+)\((.*?)\),|([A-Z_]+) ([\d ]+),"
    matches = re.finditer(pattern, text, re.DOTALL)
    ui_elements = []

    for match in matches:
        label, other, nested_items, single_label, coords = match.groups()
        if label and nested_items:
            # Extract coordinates for the container
            try:
                container_coords = list(map(int, re.findall(r"\d+", coords)))
            except TypeError:
                container_coords = container_coords = list(map(int, re.findall(r"\d+", other)))
            # Extract nested elements
            nested_pattern = r"([A-Z]+) (.*?) (\d+ \d+ \d+ \d+),"
            nested_matches = re.finditer(nested_pattern, nested_items)
            elements = []
            for nested in nested_matches:
                element_type, desc, bounds = nested.groups()
                x1, y1, x2, y2 = map(int, bounds.split())
                elements.append(
                    {
                        "type": element_type,
                        "label": desc.strip(),
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,
                    }
                )
            # Add container with nested elements
            ui_elements.append(
                {
                    "label": label,
                    "x1": container_coords[0],
                    "y1": container_coords[1],
                    "x2": container_coords[2],
                    "y2": container_coords[3],
                    "elements": elements,
                }
            )
        elif single_label:
            # Extract coordinates for single items
            single_coords = list(map(int, re.findall(r"\d+", coords)))
            ui_elements.append(
                {
                    "type": single_label,
                    "x1": single_coords[0],
                    "y1": single_coords[1],
                    "x2": single_coords[2],
                    "y2": single_coords[3],
                }
            )

    return ui_elements
