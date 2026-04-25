from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page()
    page.goto('http://127.0.0.1:5000')
    
    page.on("console", lambda msg: print(f"Browser console: {msg.type}: {msg.text}"))
    page.on("pageerror", lambda err: print(f"Browser error: {err}"))
    
    page.wait_for_timeout(2000)
    page.screenshot(path="frontend_test.png", full_page=True)
    
    browser.close()
