from ollama import chat
from playwright.sync_api import sync_playwright
import pyautogui
import json

def query_ollama(prompt):
    messages = [
        {
            'role': 'user', 
            'content': prompt
        }
    ]
    
    full_response = ""
    try:
        stream = chat(
            model='deepseek-r1:1.5b',  # Use your installed model
            messages=messages,
            stream=True,
            options={
                'temperature': 0.3,
                'num_ctx': 4096
            }
        )
        
        for chunk in stream:
            content = chunk['message']['content']
            full_response += content
            print(content, end='', flush=True)  # Stream to console
            
        return full_response
        
    except Exception as e:
        print(f"\nOllama error: {e}")
        return None

def get_ui_description(url):
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(url)
        
        # Get all visible text elements with their positions
        elements = page.evaluate('''() => {
            const allElements = Array.from(document.querySelectorAll('*'));
            const textElements = [];
            
            allElements.forEach(element => {
                const text = element.textContent?.trim();
                if (text && element.checkVisibility()) {
                    const rect = element.getBoundingClientRect();
                    textElements.push({
                        text: text,
                        x: rect.left + window.scrollX,
                        y: rect.top + window.scrollY,
                        width: rect.width,
                        height: rect.height
                    });
                }
            });
            
            // Remove duplicates and empty texts
            return textElements.filter((item, index, self) =>
                item.text.length > 0 &&
                self.findIndex(t => 
                    t.text === item.text && 
                    t.x === item.x && 
                    t.y === item.y
                ) === index
            );
        }''')

        # Get viewport size
        viewport_size = page.evaluate("""() => ({
            width: document.documentElement.clientWidth,
            height: document.documentElement.clientHeight
        })""")

        ui_description = {
            "screen_size": [viewport_size["width"], viewport_size["height"]],
            "text_elements": []
        }

        # Process elements with error handling
        seen = set()
        for element in elements:
            try:
                # Create unique identifier for the text element
                element_id = f"{element['text'][:20]}-{element['x']}-{element['y']}"
                
                if element_id not in seen and element['width'] > 0 and element['height'] > 0:
                    seen.add(element_id)
                    ui_description["text_elements"].append({
                        "text": element['text'][:200],  # Limit text length
                        "position": [
                            round(element['x'], 1),
                            round(element['y'], 1),
                            round(element['x'] + element['width'], 1),
                            round(element['y'] + element['height'], 1)
                        ],
                        "area": element['width'] * element['height']
                    })
            except Exception as e:
                print(f"Error processing element: {e}")
                continue

        browser.close()
        
        # Sort elements by visual importance (larger area first)
        ui_description["text_elements"].sort(key=lambda x: -x["area"])
        
        return ui_description

def clean_ui_description(ui_desc, min_area=1000, min_text_length=3):
    """Clean UI description data for LLM consumption."""
    cleaned_elements = []
    seen_texts = set()
    
    for element in ui_desc.get('text_elements', []):
        text = element.get('text', '').strip()
        area = element.get('area', 0)
        position = element.get('position', [])
        
        # Calculate element dimensions
        width = position[2] - position[0] if len(position) >=4 else 0
        height = position[3] - position[1] if len(position) >=4 else 0
        
        # Exclusion criteria
        exclude = (
            not text or
            len(text) < min_text_length or
            area < min_area or
            any(char in text for char in {'{', '}', ';', 'function(', '=>'}) or  # JS/CSS
            text.lower().startswith(('cookie', 'privacy', 'terms')) or
            'policy' in text.lower() or
            width == 0 or height == 0 or
            any(keyword in text.lower() for keyword in {
                'window[', 'iframe', 'display:none', 'visibility:hidden', 'css', 'data-ssrc'
            })
        )
        
        # Normalize text for deduplication
        normalized_text = text.lower().replace(" ", "").translate(str.maketrans('', '', ',.:;!?'))
        
        if not exclude and normalized_text not in seen_texts:
            seen_texts.add(normalized_text)
            cleaned_elements.append({
                "text": text,
                "position": position,
                "area": area
            })
    
    # Sort by area descending (largest elements first)
    cleaned_elements.sort(key=lambda x: -x['area'])
    
    return {
        "screen_size": ui_desc['screen_size'],
        "text_elements": cleaned_elements[:50]  # Keep top 50 elements
    }

def execute_commands(commands):
    for line in commands.split("\n"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
            
        # Extract command part before comment
        command_part = line.split("#")[0].strip()
        parts = command_part.split()
        
        # Validate command structure
        if len(parts) < 3:
            print(f"Skipping invalid command: {line}")
            continue
            
        try:
            # Convert coordinates to integers (round float values)
            x = round(float(parts[0]))
            y = round(float(parts[1]))
            action = parts[2].lower()
            
            # Execute command
            pyautogui.moveTo(x, y, duration=0.25)
            
            if action == "click":
                pyautogui.click()
                print(f"Clicked at ({x}, {y})")
            elif action == "scroll":
                clicks = int(parts[3]) if len(parts) > 3 else 1
                pyautogui.scroll(clicks)
                print(f"Scrolled {clicks} clicks at ({x}, {y})")
            else:
                print(f"Unsupported action: {action}")
                
        except ValueError as e:
            print(f"Invalid number format in command: {line} | Error: {e}")
        except Exception as e:
            print(f"Error executing command: {line} | Error: {e}")
if __name__ == "__main__":
    # 1. Get UI description from website
    url = "https://www.toolstation.com/"
    ui_desc = get_ui_description(url)

    print(ui_desc)

    # 2. Generate LLM prompt
    # llm_prompt = f"""Analyze this UI and perform 'search for screwdriver' action. 
    # You are an AI assistant controlling a computer mouse. Use only mouse clicks and scrolls.
    llm_prompt = f"""Analyze this UI and give the description of site. 
    You are an AI assistant controlling a computer mouse. Use only mouse clicks and scrolls.
    
    UI Description:
    {json.dumps(ui_desc, indent=2)}
    
    # Output commands in this exact format (one per line):
    # X Y ACTION # Comment
    
    # Rules:
    # 1. Choose coordinates within element bounding boxes
    # 2. Click on the search field first
    # 3. Then click on suggested items if available
    # 4. Use scroll if needed
    # 5. Add comments explaining each action"""
    
    # 3. Get commands from local Ollama
    print("Querying Ollama...")
    llm_output = query_ollama(llm_prompt)
    
    if llm_output:
        print("\nGenerated commands:")
        print(llm_output)
        
        # 4. Execute the commands
        print("\nExecuting commands...")
        execute_commands(llm_output)
    else:
        print("Failed to get commands from Ollama")