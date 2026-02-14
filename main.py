"""
Offline Speech-to-Text System with RAG-based Action Execution
Author: Vishal Saha
Date: February 2026

Main entry point for the voice command system.
"""

import os
import sys
import json
import time
import threading
import queue
from pathlib import Path

# Audio processing
import sounddevice as sd
import numpy as np

# Speech-to-Text
from faster_whisper import WhisperModel

# RAG components
from sentence_transformers import SentenceTransformer
import faiss

# Action execution
import subprocess
import webbrowser
from datetime import datetime
import psutil


class VoiceActivityDetector:
    """Detect speech in audio stream using energy-based detection"""
    
    def __init__(self, sample_rate=16000, frame_duration=30):
        self.sample_rate = sample_rate
        self.frame_duration = frame_duration
        self.frame_size = int(sample_rate * frame_duration / 1000)
        
    def is_speech(self, audio_frame):
        """Check if audio frame contains speech using energy threshold"""
        if len(audio_frame) == 0:
            return False
        
        # Calculate RMS energy
        energy = np.sqrt(np.mean(audio_frame**2))
        
        # Threshold for speech detection
        # Adjust this value if needed (0.01 = moderate sensitivity)
        # Lower = more sensitive, Higher = less sensitive
        return energy > 0.01


class SpeechToText:
    """Offline speech-to-text using Faster-Whisper"""
    
    def __init__(self, model_size="base", device="cpu"):
        print(f"Loading Whisper model: {model_size}...")
        start_time = time.time()
        
        # Use int8 quantization for faster inference
        self.model = WhisperModel(
            model_size, 
            device=device,
            compute_type="int8" if device == "cpu" else "float16"
        )
        
        load_time = time.time() - start_time
        print(f"Model loaded in {load_time:.2f}s")
        
    def transcribe(self, audio_data, language="en"):
        """Transcribe audio to text"""
        start_time = time.time()
        
        # Whisper expects float32 audio
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)
        
        # Normalize audio
        if np.max(np.abs(audio_data)) > 0:
            audio_data = audio_data / np.max(np.abs(audio_data))
        
        # Transcribe
        segments, info = self.model.transcribe(
            audio_data,
            language=language,
            beam_size=1,  # Faster inference
            best_of=1,
            temperature=0.0,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500)
        )
        
        # Collect all segments
        text = ""
        timestamps = []
        
        for segment in segments:
            text += segment.text + " "
            timestamps.append({
                "start": segment.start,
                "end": segment.end,
                "text": segment.text
            })
        
        latency = time.time() - start_time
        
        return {
            "text": text.strip(),
            "timestamps": timestamps,
            "latency": latency,
            "language": info.language
        }


class ActionDatabase:
    """Manage action definitions and retrieval"""
    
    def __init__(self, actions_file="actions.json"):
        self.actions_file = actions_file
        self.actions = self.load_actions()
        
        # Initialize embedding model
        print("Loading embedding model...")
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Create embeddings for all action descriptions
        self.action_texts = [
            f"{action['intent']} {action['description']} {' '.join(action.get('examples', []))}"
            for action in self.actions
        ]
        
        print("Computing action embeddings...")
        self.action_embeddings = self.embedder.encode(
            self.action_texts,
            show_progress_bar=False
        )
        
        # Build FAISS index
        dimension = self.action_embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(self.action_embeddings.astype('float32'))
        
        print(f"Loaded {len(self.actions)} actions")
    
    def load_actions(self):
        """Load action definitions from JSON"""
        if not os.path.exists(self.actions_file):
            return self.get_default_actions()
        
        with open(self.actions_file, 'r') as f:
            return json.load(f)
    
    def get_default_actions(self):
        """Default action definitions"""
        return [
            {
                "id": "open_browser",
                "intent": "open browser",
                "description": "Open web browser",
                "examples": ["open chrome", "launch browser", "start firefox"],
                "parameters": [],
                "executable": "browser"
            },
            {
                "id": "open_notepad",
                "intent": "open notepad",
                "description": "Open text editor notepad",
                "examples": ["open notepad", "launch notepad", "start text editor"],
                "parameters": [],
                "executable": "notepad"
            },
            {
                "id": "search_web",
                "intent": "search web",
                "description": "Search the internet for information",
                "examples": ["search for", "google", "look up"],
                "parameters": ["query"],
                "executable": "search"
            },
            {
                "id": "create_file",
                "intent": "create file",
                "description": "Create a new file",
                "examples": ["create file", "make file", "new file"],
                "parameters": ["filename"],
                "executable": "create_file"
            },
            {
                "id": "delete_file",
                "intent": "delete file",
                "description": "Delete a file",
                "examples": ["delete file", "remove file"],
                "parameters": ["filename"],
                "executable": "delete_file",
                "requires_confirmation": True
            },
            {
                "id": "get_weather",
                "intent": "get weather",
                "description": "Get current weather information",
                "examples": ["what's the weather", "weather forecast", "how's the weather"],
                "parameters": [],
                "executable": "weather"
            },
            {
                "id": "calculator",
                "intent": "calculate",
                "description": "Perform calculation",
                "examples": ["calculate", "what is", "compute"],
                "parameters": ["expression"],
                "executable": "calculator"
            },
            {
                "id": "set_reminder",
                "intent": "set reminder",
                "description": "Set a reminder",
                "examples": ["remind me", "set reminder", "create reminder"],
                "parameters": ["message", "time"],
                "executable": "reminder"
            },
            {
                "id": "list_files",
                "intent": "list files",
                "description": "List files in directory",
                "examples": ["show files", "list files", "what files are here"],
                "parameters": ["directory"],
                "executable": "list_files"
            },
            {
                "id": "open_folder",
                "intent": "open folder",
                "description": "Open file explorer",
                "examples": ["open folder", "show folder", "open explorer"],
                "parameters": ["path"],
                "executable": "folder"
            },
            {
                "id": "system_info",
                "intent": "system information",
                "description": "Get system information like CPU, memory usage",
                "examples": ["system info", "cpu usage", "memory usage", "system status"],
                "parameters": [],
                "executable": "system_info"
            },
            {
                "id": "close_app",
                "intent": "close application",
                "description": "Close a running application",
                "examples": ["close", "quit", "exit application"],
                "parameters": ["app_name"],
                "executable": "close_app",
                "requires_confirmation": True
            },
            {
                "id": "take_screenshot",
                "intent": "take screenshot",
                "description": "Capture screen screenshot",
                "examples": ["take screenshot", "capture screen", "screenshot"],
                "parameters": [],
                "executable": "screenshot"
            },
            {
                "id": "play_music",
                "intent": "play music",
                "description": "Play music or media",
                "examples": ["play music", "start music", "play song"],
                "parameters": ["query"],
                "executable": "music"
            },
            {
                "id": "shutdown",
                "intent": "shutdown system",
                "description": "Shutdown the computer",
                "examples": ["shutdown", "turn off computer", "power off"],
                "parameters": [],
                "executable": "shutdown",
                "requires_confirmation": True
            }
        ]
    
    def find_action(self, text, k=1):
        """Find most relevant action using RAG"""
        # Encode query
        query_embedding = self.embedder.encode([text])
        
        # Search in FAISS
        distances, indices = self.index.search(
            query_embedding.astype('float32'), 
            k
        )
        
        # Get top action
        action_idx = indices[0][0]
        confidence = 1.0 / (1.0 + distances[0][0])  # Convert distance to confidence
        
        return self.actions[action_idx], confidence


class ActionExecutor:
    """Execute actions based on recognized commands"""
    
    def __init__(self):
        self.action_log = []
    
    def execute(self, action, parameters, confidence):
        """Execute the action"""
        action_type = action['executable']
        action_id = action['id']
        
        print(f"\n🎯 Action: {action['intent']}")
        print(f"📊 Confidence: {confidence:.2%}")
        
        # Check if confirmation needed
        if action.get('requires_confirmation', False):
            print("⚠️  This action requires confirmation")
            confirm = input("Proceed? (yes/no): ").strip().lower()
            if confirm not in ['yes', 'y']:
                print("❌ Action cancelled")
                return {"status": "cancelled"}
        
        # Execute based on type
        result = None
        
        try:
            if action_type == "browser":
                result = self.open_browser()
            elif action_type == "notepad":
                result = self.open_notepad()
            elif action_type == "search":
                result = self.search_web(parameters.get('query', ''))
            elif action_type == "create_file":
                result = self.create_file(parameters.get('filename', 'newfile.txt'))
            elif action_type == "delete_file":
                result = self.delete_file(parameters.get('filename', ''))
            elif action_type == "weather":
                result = self.get_weather()
            elif action_type == "calculator":
                result = self.calculate(parameters.get('expression', ''))
            elif action_type == "reminder":
                result = self.set_reminder(parameters.get('message', ''))
            elif action_type == "list_files":
                result = self.list_files(parameters.get('directory', '.'))
            elif action_type == "folder":
                result = self.open_folder(parameters.get('path', '.'))
            elif action_type == "system_info":
                result = self.get_system_info()
            elif action_type == "close_app":
                result = self.close_app(parameters.get('app_name', ''))
            elif action_type == "screenshot":
                result = self.take_screenshot()
            elif action_type == "music":
                result = self.play_music(parameters.get('query', ''))
            elif action_type == "shutdown":
                result = self.shutdown_system()
            else:
                result = {"status": "unknown", "message": f"Unknown action type: {action_type}"}
            
            # Log action
            self.log_action(action_id, parameters, result)
            
            return result
            
        except Exception as e:
            error_result = {"status": "error", "message": str(e)}
            print(f"❌ Error: {str(e)}")
            return error_result
    
    def open_browser(self):
        """Open web browser"""
        webbrowser.open('https://www.google.com')
        return {"status": "success", "message": "Browser opened"}
    
    def open_notepad(self):
        """Open notepad"""
        if sys.platform == 'win32':
            subprocess.Popen(['notepad.exe'])
        else:
            subprocess.Popen(['gedit'])  # Linux
        return {"status": "success", "message": "Notepad opened"}
    
    def search_web(self, query):
        """Search the web"""
        if not query:
            return {"status": "error", "message": "No search query provided"}
        
        search_url = f"https://www.google.com/search?q={query.replace(' ', '+')}"
        webbrowser.open(search_url)
        return {"status": "success", "message": f"Searching for: {query}"}
    
    def create_file(self, filename):
        """Create a new file"""
        if not filename:
            filename = f"file_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        
        Path(filename).touch()
        return {"status": "success", "message": f"Created: {filename}"}
    
    def delete_file(self, filename):
        """Delete a file"""
        if not filename or not os.path.exists(filename):
            return {"status": "error", "message": "File not found"}
        
        os.remove(filename)
        return {"status": "success", "message": f"Deleted: {filename}"}
    
    def get_weather(self):
        """Get weather (mock - would need API in real implementation)"""
        return {
            "status": "success",
            "message": "Weather feature requires internet connection. This is offline mode."
        }
    
    def calculate(self, expression):
        """Perform calculation"""
        if not expression:
            return {"status": "error", "message": "No expression provided"}
        
        try:
            # Safe eval for basic math
            result = eval(expression, {"__builtins__": {}}, {})
            return {"status": "success", "message": f"{expression} = {result}"}
        except:
            return {"status": "error", "message": "Invalid expression"}
    
    def set_reminder(self, message):
        """Set a reminder (mock implementation)"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        return {
            "status": "success",
            "message": f"Reminder set: {message} at {timestamp}"
        }
    
    def list_files(self, directory):
        """List files in directory"""
        if not os.path.exists(directory):
            directory = "."
        
        files = os.listdir(directory)
        return {
            "status": "success",
            "message": f"Files in {directory}:",
            "files": files[:10]  # Limit to 10
        }
    
    def open_folder(self, path):
        """Open file explorer"""
        if not os.path.exists(path):
            path = "."
        
        if sys.platform == 'win32':
            os.startfile(path)
        else:
            subprocess.Popen(['xdg-open', path])
        
        return {"status": "success", "message": f"Opened: {path}"}
    
    def get_system_info(self):
        """Get system information"""
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        
        info = {
            "cpu_usage": f"{cpu_percent}%",
            "memory_usage": f"{memory.percent}%",
            "memory_available": f"{memory.available / (1024**3):.2f} GB"
        }
        
        return {"status": "success", "message": "System info", "data": info}
    
    def close_app(self, app_name):
        """Close an application"""
        if not app_name:
            return {"status": "error", "message": "No app name provided"}
        
        # Kill process by name (simplified)
        for proc in psutil.process_iter(['name']):
            if app_name.lower() in proc.info['name'].lower():
                proc.kill()
                return {"status": "success", "message": f"Closed: {app_name}"}
        
        return {"status": "error", "message": f"App not found: {app_name}"}
    
    def take_screenshot(self):
        """Take screenshot"""
        try:
            from PIL import ImageGrab
            screenshot = ImageGrab.grab()
            filename = f"screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            screenshot.save(filename)
            return {"status": "success", "message": f"Screenshot saved: {filename}"}
        except:
            return {"status": "error", "message": "Screenshot failed (PIL required)"}
    
    def play_music(self, query):
        """Play music"""
        search_url = f"https://www.youtube.com/results?search_query={query.replace(' ', '+')}"
        webbrowser.open(search_url)
        return {"status": "success", "message": f"Searching music: {query}"}
    
    def shutdown_system(self):
        """Shutdown system"""
        if sys.platform == 'win32':
            os.system('shutdown /s /t 60')  # 60 second delay
        else:
            os.system('shutdown -h +1')  # 1 minute delay
        
        return {"status": "success", "message": "System will shutdown in 1 minute"}
    
    def log_action(self, action_id, parameters, result):
        """Log executed action"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "action_id": action_id,
            "parameters": parameters,
            "result": result
        }
        self.action_log.append(log_entry)
        
        # Save to file
        with open('action_log.json', 'w') as f:
            json.dump(self.action_log, f, indent=2)


class VoiceCommandSystem:
    """Main voice command system"""
    
    def __init__(self, model_size="base"):
        self.stt = SpeechToText(model_size=model_size)
        self.action_db = ActionDatabase()
        self.executor = ActionExecutor()
        self.vad = VoiceActivityDetector()
        
        # Audio settings
        self.sample_rate = 16000
        self.chunk_duration = 0.03  # 30ms chunks
        self.chunk_size = int(self.sample_rate * self.chunk_duration)
        
        # State
        self.is_recording = False
        self.audio_buffer = []
        self.audio_queue = queue.Queue()
    
    def audio_callback(self, indata, frames, time_info, status):
        """Callback for audio stream"""
        if status:
            print(f"Audio status: {status}")
        
        # Add to queue
        self.audio_queue.put(indata.copy())
    
    def process_audio_stream(self):
        """Process audio stream with VAD"""
        speech_frames = []
        silence_frames = 0
        max_silence_frames = 30  # ~1 second of silence
        
        is_speaking = False
        
        while self.is_recording:
            try:
                # Get audio chunk
                chunk = self.audio_queue.get(timeout=0.1)
                
                # Check for speech
                has_speech = self.vad.is_speech(chunk[:, 0])  # Use first channel
                
                if has_speech:
                    is_speaking = True
                    silence_frames = 0
                    speech_frames.append(chunk)
                    print("🎤", end="", flush=True)
                elif is_speaking:
                    silence_frames += 1
                    speech_frames.append(chunk)
                    
                    # End of speech
                    if silence_frames >= max_silence_frames:
                        print("\n⏸️  Processing speech...")
                        
                        # Concatenate audio
                        audio_data = np.concatenate(speech_frames)[:, 0]
                        
                        # Transcribe
                        result = self.stt.transcribe(audio_data)
                        
                        if result['text']:
                            print(f"\n📝 Transcribed: \"{result['text']}\"")
                            print(f"⚡ Latency: {result['latency']:.3f}s")
                            
                            # Find and execute action
                            action, confidence = self.action_db.find_action(result['text'])
                            
                            if confidence > 0.3:  # Confidence threshold
                                # Extract parameters (simplified)
                                parameters = self.extract_parameters(result['text'], action)
                                
                                # Execute
                                exec_result = self.executor.execute(action, parameters, confidence)
                                print(f"✅ {exec_result.get('message', 'Done')}")
                            else:
                                print(f"❓ Low confidence ({confidence:.2%}). Command unclear.")
                        
                        # Reset
                        speech_frames = []
                        silence_frames = 0
                        is_speaking = False
                        print("\n🎧 Listening...")
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error processing audio: {e}")
    
    def extract_parameters(self, text, action):
        """Extract parameters from text (simplified)"""
        parameters = {}
        
        # Basic parameter extraction
        if 'query' in action.get('parameters', []):
            # Extract everything after intent
            intent_words = action['intent'].split()
            for word in intent_words:
                text = text.replace(word, '').strip()
            parameters['query'] = text
        
        if 'filename' in action.get('parameters', []):
            # Look for filename patterns
            words = text.split()
            if len(words) > 2:
                parameters['filename'] = words[-1]
        
        return parameters
    
    def start_listening(self):
        """Start continuous listening mode"""
        print("\n🎧 Voice Command System Started")
        print("=" * 50)
        print("Say a command... (Ctrl+C to stop)\n")
        
        self.is_recording = True
        
        # Start audio stream
        with sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            callback=self.audio_callback,
            blocksize=self.chunk_size
        ):
            # Start processing thread
            process_thread = threading.Thread(target=self.process_audio_stream)
            process_thread.start()
            
            try:
                process_thread.join()
            except KeyboardInterrupt:
                print("\n\n🛑 Stopping...")
                self.is_recording = False
                process_thread.join()
    
    def transcribe_file(self, audio_file):
        """Transcribe an audio file"""
        print(f"\n📁 Loading file: {audio_file}")
        
        # Load audio
        import soundfile as sf
        audio_data, sr = sf.read(audio_file)
        
        # Resample if needed
        if sr != self.sample_rate:
            from scipy import signal
            audio_data = signal.resample(
                audio_data,
                int(len(audio_data) * self.sample_rate / sr)
            )
        
        # Transcribe
        result = self.stt.transcribe(audio_data)
        
        print(f"\n📝 Transcription:")
        print(f"   Text: \"{result['text']}\"")
        print(f"   Language: {result['language']}")
        print(f"   Latency: {result['latency']:.3f}s")
        
        if result['timestamps']:
            print(f"\n⏱️  Timestamps:")
            for segment in result['timestamps']:
                print(f"   [{segment['start']:.2f}s - {segment['end']:.2f}s] {segment['text']}")
        
        return result


def main():
    """Main entry point"""
    print("=" * 60)
    print("  Offline Voice Command System with RAG")
    print("  Author: Vishal Saha")
    print("=" * 60)
    
    # Initialize system
    print("\n🚀 Initializing system...")
    system = VoiceCommandSystem(model_size="base")
    
    # Check for command line arguments
    if len(sys.argv) > 1:
        # File mode
        audio_file = sys.argv[1]
        system.transcribe_file(audio_file)
    else:
        # Live mode
        system.start_listening()


if __name__ == "__main__":
    main()