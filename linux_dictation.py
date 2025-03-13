from RealtimeSTT import AudioToTextRecorder
import os
import logging
import torch

# Set up logging
logging.basicConfig(level=logging.DEBUG)

# Comment out or remove this line to allow GPU usage
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

def process_text(text):
    print(text)

if __name__ == '__main__':
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA version: {torch.version.cuda if hasattr(torch.version, 'cuda') else 'Not found'}")
    if torch.cuda.is_available():
        print(f"Current device: {torch.cuda.current_device()}")
        print(f"Device name: {torch.cuda.get_device_name()}")
    
    print("Wait until it says 'speak now'...")
    try:
        recorder = AudioToTextRecorder(
            model="tiny",
            device="cuda" if torch.cuda.is_available() else "cpu",
            compute_type="float16" if torch.cuda.is_available() else "int8"
        )
        while True:
            recorder.text(process_text)
    except Exception as e:
        logging.error(f"Error: {str(e)}", exc_info=True)
