def _alt_format(previous_actions_text):
    text = f"You are a helpful web assistant. Based on the prior actions and the current browser content, respond with the next action and if necessary action position.\n{previous_actions_text}\nNext Action:\n"
    return text

class PretrainHTML:
    def __init__(self):
        pass
