# from playwright.sync_api import sync_playwright
# import pyautogui
# import json

# def get_ui_description(url):
#     with sync_playwright() as p:
#         browser = p.chromium.launch(headless=False)
#         page = browser.new_page()
#         page.goto(url)
        
#         # Get viewport size
#         viewport_size = page.evaluate("""() => ({
#             width: document.documentElement.clientWidth,
#             height: document.documentElement.clientHeight
#         })""")

#         # Get interactive elements
#         elements = page.query_selector_all("button,a,input,select,textarea,[role=button],[tabindex]")
        
#         ui_description = {
#             "screen_size": [viewport_size["width"], viewport_size["height"]],
#             "elements": []
#         }

#         for element in elements:
#             try:
#                 box = element.bounding_box()
#                 if not box: continue
                
#                 ui_description["elements"].append({
#                     "text": element.inner_text().strip()[:50],
#                     "type": element.get_attribute("type") or element.tag_name,
#                     "position": [box["x"], box["y"], box["x"] + box["width"], box["y"] + box["height"]],
#                     "metadata": {
#                         "id": element.get_attribute("id"),
#                         "role": element.get_attribute("role")
#                     }
#                 })
#             except:
#                 continue

#         browser.close()
#         return ui_description

# def execute_commands(commands):
#     for line in commands.split("\n"):
#         if line.strip() and not line.startswith("#"):
#             parts = line.strip().split()
#             if len(parts) >= 3:
#                 x, y = int(parts[0]), int(parts[1])
#                 action = parts[2]
#                 pyautogui.moveTo(x, y, duration=0.25)
                
#                 if action == "click":
#                     pyautogui.click()
#                 elif action == "scroll":
#                     pyautogui.scroll(int(parts[3]) if len(parts) > 3 else 1)

# # Example workflow
# if __name__ == "__main__":
#     # 1. Get UI description from website
#     ui_desc = get_ui_description("https://www.toolstation.com/")
    
#     # 2. Generate LLM prompt (this would go to your LLM)
#     llm_prompt = f"""Analyze this UI and perform login action:
    
#     UI Description:
#     {json.dumps(ui_desc, indent=2)}
    
#     Output commands in format:
#     X Y ACTION # Comment"""
    
#     print("LLM Prompt:\n", llm_prompt)
    
#     # 3. Sample LLM output (this would come from your LLM)
#     llm_output = """
#     150 82 click   # Username input
#     150 120 click   # Password input
#     200 160 click   # Login button
#     """
    
#     # 4. Execute the commands
#     execute_commands(llm_output)

import requests
import json

response = requests.post(

  url="https://openrouter.ai/api/v1/chat/completions",
  headers={
    "Authorization": "Bearer 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",
    # "HTTP-Referer": "<YOUR_SITE_URL>", # Optional. Site URL for rankings on openrouter.ai.
    # "X-Title": "<YOUR_SITE_NAME>", # Optional. Site title for rankings on openrouter.ai.
  },
  data=json.dumps({
    "model": "deepseek/deepseek-r1", # Optional
    "messages": [
      {
        "role": "user",
        "content": "What is the meaning of life?"
      }
    ]
    
  })
)   

print(response.json())