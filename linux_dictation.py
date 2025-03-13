from RealtimeSTT import AudioToTextRecorder
import os
import logging
import torch
import time
import threading

# Set up logging
logging.basicConfig(level=logging.DEBUG)

# Comment out or remove this line to allow GPU usage
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Global variables for tracking activity
last_activity_time = time.time()
INACTIVITY_TIMEOUT = 5  # seconds
stop_recording = False

def process_text(text):
    global last_activity_time
    # Use a newline before printing detected text for better readability
    print(f"\nDetected: {text}")
    # Update the last activity timestamp whenever speech is detected
    last_activity_time = time.time()

def check_inactivity():
    global stop_recording
    while not stop_recording:
        if time.time() - last_activity_time > INACTIVITY_TIMEOUT:
            # Add a newline before the message to avoid overlap with spinner
            print(f"\n\nNo speech detected for {INACTIVITY_TIMEOUT} seconds. Stopping...")
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
                print("\n----------------------------")
                print("Shutting down recorder...")
                recorder.shutdown()
        except Exception as e:
            logging.error(f"Error during shutdown: {str(e)}")
        print("Recording stopped. Goodbye!")
