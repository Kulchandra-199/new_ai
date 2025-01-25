import sys
import pyautogui
import time
import random

# TEST_MODE = True  # Uncomment this to use test input
TEST_MODE = True   # Comment this out when using real LLM input

TEST_COMMANDS = [
    "100 200 click",
    "300 400 right_click",
    "500 600 double_click",
    "700 800 scroll 3",
    "900 1000 move",
    "150 750 scroll -5",
    "invalid command here",
    "123 abc click",
    "800 600 unknown_action",
    "1920 1080 click"  # Assuming 1080p screen
]

def process_command(line):
    try:
        parts = line.strip().split()
        if len(parts) < 3:
            print(f"Invalid command: '{line.strip()}' (not enough parts)")
            return
        
        x = int(parts[0])
        y = int(parts[1])
        action = parts[2].lower()
        
        print(f"Moving to ({x}, {y}) for {action}...")
        pyautogui.moveTo(x, y, duration=0.25)
        
        if action == "click":
            pyautogui.click()
            print("Single click executed")
        elif action == "right_click":
            pyautogui.rightClick()
            print("Right click executed")
        elif action == "double_click":
            pyautogui.doubleClick()
            print("Double click executed")
        elif action == "scroll":
            clicks = int(parts[3]) if len(parts) > 3 else 1
            pyautogui.scroll(clicks)
            print(f"Scrolled {clicks} clicks")
        elif action == "move":
            print("Move completed")
        else:
            print(f"Unknown action: {action}")
            
    except ValueError as ve:
        print(f"Invalid number format: {ve}")
    except Exception as e:
        print(f"Error processing command: {e}")

if __name__ == "__main__":
    if TEST_MODE:
        print("Running in test mode with generated input")
        for cmd in TEST_COMMANDS:
            print(f"\nProcessing test command: {cmd}")
            process_command(cmd)
            time.sleep(1)  # Pause between commands for observation
        print("\nTest sequence completed")
    else:
        print("Mouse controller started. Waiting for commands...")
        try:
            for line in sys.stdin:
                if line.strip():
                    process_command(line)
        except KeyboardInterrupt:
            print("\nExiting...")