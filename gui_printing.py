#!/usr/bin/env python3
import subprocess
import time
import pyperclip

def paste_to_focused_input(text, delay=3):
    """
    Pastes text at the currently focused text input.
    
    Args:
        text: Text to paste
        delay: Seconds to wait before starting
    """
    print(f"Starting in {delay} seconds. Make sure your text input is focused...")
    time.sleep(delay)
    
    # Copy text to clipboard
    pyperclip.copy(text)
    
    # Send paste command to currently focused window
    subprocess.run(["xdotool", "key", "--clearmodifiers", "ctrl+shift+v"])

if __name__ == "__main__":
    # Example text to paste
    text_to_paste = "Hello, this is automated text!"
    
    paste_to_focused_input(text_to_paste)
