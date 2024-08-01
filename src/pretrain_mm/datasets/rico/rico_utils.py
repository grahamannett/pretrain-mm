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


if __name__ == "__main__":
    import json

    # Example usage
    input_str = """
    TOOLBAR 0 999 31 109 (TEXT Inbox 40 158 54 85, PICTOGRAM a white speech bubble with a plus sign on a blue background . 671 789 39 102, PICTOGRAM a magnifying glass icon on a blue background . 787 906 39 101, PICTOGRAM three white circles are lined up on a blue background . 908 999 39 101), BUTTON CHATROOMS 0 333 108 171, NAVIGATION_BAR 0 998 109 171 (BUTTON DIRECT 331 664 109 171, BUTTON CONTACTS 672 998 110 173), PICTOGRAM a green speech bubble with two white circles inside of it is in a yellow circle . 29 166 184 260, TEXT Team Inbox 193 409 194 222, PICTOGRAM three yellow stars are shining on a white background . 412 462 193 223, TEXT You: 199 267 226 251, TEXT Picture Message 273 539 226 252, TEXT now 904 965 202 222, PICTOGRAM check 903 962 228 254, TEXT Inbox is more fun with friends! 258 736 319 345, BUTTON Invite Friends 391 605 346 371
    """

    parsed_ui = parse_ui_elements(input_str)

    # Convert to JSON
    json_output = json.dumps(parsed_ui, indent=2)
    print(json_output)
