from RealtimeSTT import AudioToTextRecorder
import os
import logging
from datetime import date
import torch
import time
import threading
from gui_printing import paste_to_focused_input

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('dictation')

# Set up speech logging
speech_logger = logging.getLogger('speech')
speech_logger.setLevel(logging.INFO)

# Comment out or remove this line to allow GPU usage
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Global variables for tracking activity
last_activity_time = time.time()
INACTIVITY_TIMEOUT = 300  # time out after 5 minutes
stop_recording = False

def setup_daily_log_handler():
    """Configure and return a file handler for the current day's log file"""
    today = date.today().strftime("%Y-%m-%d")
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, f"speech_{today}.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(message)s', datefmt='%H:%M:%S')
    file_handler.setFormatter(formatter)
    
    # Remove any existing handlers to avoid duplicates
    for handler in speech_logger.handlers[:]:
        speech_logger.removeHandler(handler)
    
    speech_logger.addHandler(file_handler)
    return log_file

def process_text(text):
    global last_activity_time
    
    # Print to console for monitoring
    print(f"\nDetected: {text}")
    
    # Log the detected text
    speech_logger.info(text)
    
    # Paste the text at current cursor position with minimal delay
    paste_to_focused_input(text + " ", delay=0.1)
    
    # Update the last activity timestamp
    last_activity_time = time.time()

def check_inactivity():
    global stop_recording
    while not stop_recording:
        if time.time() - last_activity_time > INACTIVITY_TIMEOUT:
            # Add a newline before the message to avoid overlap with spinner
            print(f"\n\nNo speech detected for {INACTIVITY_TIMEOUT} seconds.")
            stop_recording = True
            # Force the recorder to stop immediately instead of continuing to run
            try:
                recorder.abort()  # Attempt to abort the recording immediately
            except Exception as e:
                logging.error(f"Error stopping recorder: {str(e)}")
            break
        time.sleep(1)  # Check every second

if __name__ == '__main__':
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA version: {torch.version.cuda if hasattr(torch.version, 'cuda') else 'Not found'}")
    if torch.cuda.is_available():
        print(f"Current device: {torch.cuda.current_device()}")
        print(f"Device name: {torch.cuda.get_device_name()}")
    
    print("Wait until it says 'speak now'...")
    try:
        # Make recorder globally accessible for the inactivity thread
        global recorder
        
        # Set up the daily log file
        log_file = setup_daily_log_handler()
        print(f"Logging speech to: {log_file}")
        
        recorder = AudioToTextRecorder(
            model="tiny",
            device="cuda" if torch.cuda.is_available() else "cpu",
            compute_type="float16" if torch.cuda.is_available() else "int8",
        )
        
        # Start a background thread to monitor inactivity
        inactivity_thread = threading.Thread(target=check_inactivity)
        inactivity_thread.daemon = True
        inactivity_thread.start()
        
        # Main thread continues with recording
        while not stop_recording:
            recorder.text(process_text)
            
    except Exception as e:
        logging.error(f"Error: {str(e)}", exc_info=True)
    finally:
        stop_recording = True
        try:
            # Ensure recorder is properly shut down
            if 'recorder' in locals():
                # Print a newline and separator for clean shutdown message
                print("\nShutting down recorder...")
                recorder.shutdown()
        except Exception as e:
            logging.error(f"Error during shutdown: {str(e)}")
        print("\nRecording stopped. Goodbye!")
