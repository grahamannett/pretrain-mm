def from_buffer_make_screenshot_sync(filepath, viewport_size=None):
    """
    possibly need this for screenshots from mind2web, but seems like they also are in the raw_dump/task/[annotation_id]/processed folder as b64 encodeds str
    """
    import io
    import numpy as np
    from PIL import Image
    from playwright import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(viewport=viewport_size)
        page = context.new_page()
        page.goto(filepath)
        screenshot_buffer = page.screenshot()
        image = Image.open(io.BytesIO(screenshot_buffer))
        screenshot_arr = np.array(image)
    return screenshot_arr
