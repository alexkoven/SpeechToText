# Linux Dictation

Simple speech-to-text dictation tool hacked together for Linux machine based on [RealtimeSTT library](https://github.com/KoljaB/RealtimeSTT?tab=readme-ov-file#realtimestt). 

## Features

- Real-time speech recognition using Whisper models through RealtimeSTT
- Automatic text pasting to the currently focused input field
- Automatic shutdown on inactivity

## Files

- **linux_dictation.py**: Main script that handles speech recognition and text processing
- **gui_printing.py**: Utility for pasting text to the currently focused input field
- **simple_stt_demo.py**: Simple demonstration of the speech-to-text capabilities

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/linux-dictation.git
   cd linux-dictation
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

3. Install the required packages (check your torch versions before executing):
   ```bash
   pip install torch==2.5.1+cu121 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
   pip install -r requirements.txt
   ```

## Usage

Run the dictation tool with `python linux_dictation.py`


Wait for the "speak now" prompt, then start speaking. If you execute the file for the first time, the code will download the necessary Speech-to-Text models from [Silero VAD](https://github.com/snakers4/silero-vad). Once downloaded, the tool is ready to transcribe your speech and paste it into the currently focused input field.

## Terminal Alias

Add this alias to your `.bashrc` for quick access:

`alias start_dictation="cd ~/path/to/SpeechToText && source ~/path/to/virtual_environment && python linux_dictation.py"`

Replace `/path/to/dir/` with your paths.
