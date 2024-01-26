from PIL import Image

screenshot = Image.open("tests/fixtures/screenshot0.png")
input_string = 'Given the following HTML provide the bounding box\\n <button backend_node_id="661"></button>'
input_label = "<box>54, 1066, 102, 1200</box>"
