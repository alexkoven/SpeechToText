from RealtimeSTT import AudioToTextRecorder
import os
import logging
from datetime import date
import torch
import time
import threading
from gui_printing import paste_to_focused_input
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import gc
import re

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

# Global variables for model and tokenizer
llm_model = None
llm_tokenizer = None

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

def initialize_llm():
    """Initialize the LLM model and tokenizer for text polishing"""
    global llm_model, llm_tokenizer
    
    model_name = "meta-llama/Llama-3.2-3B-Instruct"
    hf_token = os.getenv('HF_TOKEN')
    
    if not hf_token:
        logger.error("HF_TOKEN environment variable not found. Please set it with your Hugging Face token.")
        logger.error("You can get your token from: https://huggingface.co/settings/tokens")
        logger.error("You also need to request access at: https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct")
        raise ValueError("HF_TOKEN environment variable is required")
    
    logger.info(f"Loading LLM model: {model_name}")
    
    try:
        llm_tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
        llm_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            token=hf_token,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            trust_remote_code=True  # Allow model-specific code from the hub
        )
        logger.info("LLM model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load LLM model: {str(e)}")
        logger.error("Please check your HF_TOKEN is valid and you have access to the model")
        raise

def chunk_text(text, max_chunk_length=100):
    """Split text into chunks at sentence boundaries, respecting max length."""
    # Split on sentence endings (., !, ?) followed by spaces or end of string
    sentences = re.split('([.!?]+(?:\s+|$))', text)
    
    # Rejoin sentences with their punctuation
    sentences = [''.join(i) for i in zip(sentences[::2], sentences[1::2] + [''] * (len(sentences[::2]) - len(sentences[1::2])))]
    sentences = [s.strip() for s in sentences if s.strip()]
    
    chunks = []
    current_chunk = []
    current_length = 0
    
    for sentence in sentences:
        sentence_length = len(sentence.split())
        if current_length + sentence_length <= max_chunk_length:
            current_chunk.append(sentence)
            current_length += sentence_length
        else:
            if current_chunk:
                chunks.append(' '.join(current_chunk))
            current_chunk = [sentence]
            current_length = sentence_length
    
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks

def polish_text(text):
    """Polish the transcribed text using LLama model to improve grammar and clarity"""
    if llm_model is None or llm_tokenizer is None:
        logger.warning("LLM not initialized, returning original text")
        return text
    
    try:
        # Simulate a GPU memory error for testing
        if "test memory error" in text.lower():
            raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB. GPU 0 has a total capacity of 8.00 GiB of which 7.50 GiB is free.")
        
        # Break text into manageable chunks
        chunks = chunk_text(text)
        polished_chunks = []
        
        for chunk in chunks:
            prompt = f"""<|system|>You are a helpful AI assistant that polishes transcribed text. You must only fix punctuation and combine/separate sentences based on context. You may also fix grammar and correct words but only if you are absolutely certain that that was the speaker's intent. Otherwise, you should leave the text as is. You must not add any words before or after the transcribed text. If the text is already correct, you should just return the original text.</s>
<|user|>Polish this transcribed text: {chunk}</s>
<|assistant|>"""

            inputs = llm_tokenizer(prompt, return_tensors="pt").to(llm_model.device)
            
            with torch.no_grad():
                outputs = llm_model.generate(
                    **inputs,
                    max_new_tokens=100,
                    temperature=0.3,
                    do_sample=True,
                    top_p=0.9,
                    num_return_sequences=1,
                )
            
            polished_chunk = llm_tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            try:
                polished_chunk = polished_chunk.split("<|assistant|>")[1].strip()
                polished_chunk = polished_chunk.replace("</s>", "").strip()
                polished_chunks.append(polished_chunk)
            except IndexError:
                logger.warning(f"Failed to extract polished text for chunk: {chunk}")
                polished_chunks.append(chunk)
            
            # Clean up GPU memory after each chunk
            del outputs
            torch.cuda.empty_cache()
            gc.collect()
        
        # Combine the polished chunks
        return ' '.join(polished_chunks)
        
    except Exception as e:
        logger.warning(f"Failed to polish text due to error: {str(e)}")
        logger.warning("Using original text instead")
        return text

def process_text(text):
    global last_activity_time
    
    # Print raw text to console for monitoring
    print(f"\nDetected (raw): {text}")
    
    try:
        # Polish the text using LLM
        polished_text = polish_text(text)
        print(f"Polished: {polished_text}")
    except Exception as e:
        # If polishing fails completely, use original text
        logger.error(f"Critical error during text polishing: {str(e)}")
        polished_text = text
        print(f"Using original text due to polishing error: {text}")
    
    # Log both raw and polished text
    speech_logger.info(f"Raw: {text}")
    speech_logger.info(f"Polished: {polished_text}")
    
    # Paste the text at current cursor position with minimal delay
    paste_to_focused_input(polished_text + " ", delay=0.1)
    
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
    
    print("Initializing LLM for text polishing...")
    initialize_llm()
    
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
