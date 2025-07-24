import os
import uuid
import threading
import time
import wave
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
import shutil
from pydantic import BaseModel
import logging
from datetime import datetime
import dotenv
import numpy as np

# PyAudio for recording
import pyaudio

# For local Hugging Face model (Whisper)
import torch
from transformers import pipeline
import torchaudio

# For Gemini API
import google.generativeai as genai

# --- Configuration & Model Loading ---
dotenv.load_dotenv()

# Gemini API Key
GOOGLE_GEMINI_API_KEY = os.getenv("GOOGLE_GEMINI_API_KEY")

if not GOOGLE_GEMINI_API_KEY:
    print(
        "WARNING: GOOGLE_GEMINI_API_KEY environment variable not set. Summarization will fail."
    )
else:
    genai.configure(api_key=GOOGLE_GEMINI_API_KEY)

# Setup base directories
BASE_DATA_DIR = Path.home() / "meeting_app_data"
TRANSCRIPTIONS_AUDIO_DIR = BASE_DATA_DIR / "transcriptions_audio"
MEETING_RESUMES_DIR = BASE_DATA_DIR / "meeting_resumes"

TRANSCRIPTIONS_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
MEETING_RESUMES_DIR.mkdir(parents=True, exist_ok=True)

# --- Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Hugging Face Model Setup (Whisper) ---
HF_MODEL_ID = "openai/whisper-base"
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = (
    torch.float16 if torch.cuda.is_available() and device == "cuda:0" else torch.float32
)

hf_asr_pipeline = None


def load_hugging_face_model():
    global hf_asr_pipeline
    if hf_asr_pipeline is None:
        logger.info(
            f"Loading Hugging Face Whisper model: {HF_MODEL_ID} on device: {device}"
        )
        try:
            hf_asr_pipeline = pipeline(
                "automatic-speech-recognition",
                model=HF_MODEL_ID,
                torch_dtype=torch_dtype,
                device=device,
            )
            logger.info(
                f"Hugging Face Whisper model {HF_MODEL_ID} loaded successfully."
            )
        except Exception as e:
            logger.error(f"Error loading Hugging Face Whisper model {HF_MODEL_ID}: {e}")
            hf_asr_pipeline = None


# --- Audio Recording Class ---
class AudioRecorder:
    def __init__(self):
        # Suppress ALSA warnings by redirecting stderr temporarily
        import os
        import sys

        # Store original stderr
        self.original_stderr = sys.stderr

        # Initialize PyAudio with suppressed ALSA warnings
        try:
            # Redirect stderr to suppress ALSA warnings during PyAudio init
            with open(os.devnull, "w") as devnull:
                sys.stderr = devnull
                self.audio = pyaudio.PyAudio()
        finally:
            # Restore stderr
            sys.stderr = self.original_stderr

        self.recording = False
        self.frames = []
        self.stream = None
        self.mic_stream = None
        self.system_stream = None
        self.recording_thread = None

        # Audio settings optimized for your system
        self.format = pyaudio.paInt16
        self.channels = 2
        self.rate = 44100
        self.chunk = 1024
        self.recording_type = "MIC_ONLY"  # or "SCREEN_PLUS_MIC"
        self.selected_device = None

        # Updated device indices based on your current diagnostic results
        self.system_device_indices = [6, 7, 8]  # pipewire(6), pulse(7), default(8)
        self.preferred_mic_indices = [4]  # Your available builtin mic

    def get_audio_devices(self):
        """
        Get list of available audio input devices, filtered and categorized
        Returns: List of device info dictionaries
        """
        devices = []
        try:
            import os
            import sys

            # Suppress ALSA warnings during device enumeration
            with open(os.devnull, "w") as devnull:
                stderr_backup = sys.stderr
                sys.stderr = devnull

                try:
                    for i in range(self.audio.get_device_count()):
                        info = self.audio.get_device_info_by_index(i)
                        if info["maxInputChannels"] > 0:  # Only input devices
                            device_info = {
                                "index": i,
                                "name": info["name"],
                                "channels": info["maxInputChannels"],
                                "default_sample_rate": info["defaultSampleRate"],
                                "type": self._categorize_device(info["name"]),
                            }
                            devices.append(device_info)
                finally:
                    sys.stderr = stderr_backup

            # Sort devices by preference (PipeWire/Pulse first, then hardware)
            devices.sort(
                key=lambda x: (
                    0
                    if x["type"] == "system"
                    else 1
                    if x["type"] == "hardware"
                    else 2
                    if x["type"] == "usb"
                    else 3
                )
            )

        except Exception as e:
            logger.error(f"Error getting audio devices: {e}")
        return devices

    def _categorize_device(self, device_name):
        """Categorize device type for better selection"""
        name_lower = device_name.lower()
        if any(sys_name in name_lower for sys_name in ["pipewire", "pulse", "default"]):
            return "system"
        elif "usb" in name_lower or "webcam" in name_lower:
            return "usb"
        elif any(hw in name_lower for hw in ["hw:", "alc", "hda"]):
            return "hardware"
        else:
            return "other"

    def start_recording(self, recording_type="MIC_ONLY", device_index=None):
        """
        Start audio recording

        Args:
            recording_type: "MIC_ONLY" or "SCREEN_PLUS_MIC"
            device_index: Index of audio input device (None for default)
        """
        if self.recording:
            raise Exception("Recording already in progress")

        # Validate device index
        if device_index is not None:
            devices = self.get_audio_devices()
            if not any(d["index"] == device_index for d in devices):
                available_indices = [d["index"] for d in devices if d["channels"] > 0]
                logger.warning(
                    f"Selected device {device_index} not found. Available devices: {available_indices}"
                )
                if available_indices:
                    device_index = available_indices[0]
                    logger.info(f"Falling back to device {device_index}")
                else:
                    raise Exception("No audio input devices available")

        self.recording_type = recording_type
        self.selected_device = device_index
        self.frames = []
        self.recording = True

        # Start recording in a separate thread
        self.recording_thread = threading.Thread(target=self._record_audio)
        self.recording_thread.daemon = (
            True  # Make thread daemon so it doesn't prevent shutdown
        )
        self.recording_thread.start()

        logger.info(
            f"Started recording with type: {recording_type}, device: {device_index}"
        )

    def _record_audio(self):
        """
        Internal method to handle audio recording in a separate thread
        """
        try:
            if self.recording_type == "MIC_ONLY":
                self._record_microphone_only()
            else:  # SCREEN_PLUS_MIC
                self._record_combined_audio()
        except Exception as e:
            logger.error(f"Error during recording: {e}")
            self.recording = False
            # Don't re-raise here to prevent thread from crashing the app

    def _record_microphone_only(self):
        """Record only microphone audio"""
        try:
            logger.info(f"Opening microphone stream (device: {self.selected_device})")

            # Open microphone stream with error handling
            self.mic_stream = self.audio.open(
                format=self.format,
                channels=1,
                rate=self.rate,
                input=True,
                input_device_index=self.selected_device,
                frames_per_buffer=self.chunk,
            )

            logger.info("🎤 Recording microphone audio...")

            while self.recording:
                try:
                    data = self.mic_stream.read(self.chunk, exception_on_overflow=False)
                    if data:
                        self.frames.append(data)
                except Exception as e:
                    logger.error(f"Error reading microphone data: {e}")
                    break

        except Exception as e:
            logger.error(f"Error setting up microphone recording: {e}")
            # Get available devices for debugging
            devices = self.get_audio_devices()
            logger.error(
                f"Available devices: {[(d['index'], d['name']) for d in devices]}"
            )
            raise
        finally:
            if self.mic_stream:
                try:
                    self.mic_stream.stop_stream()
                    self.mic_stream.close()
                except:
                    pass

    def _record_combined_audio(self):
        """
        Record both system audio and microphone (optimized for monitor sources)
        """
        try:
            devices = self.get_audio_devices()

            # Find system audio device - try multiple methods
            system_device = None
            system_device_name = ""

            # Method 1: Look for monitor devices (best for pavucontrol setup)
            for device in devices:
                device_name = device["name"].lower()
                if "monitor" in device_name and device["channels"] > 0:
                    system_device = device["index"]
                    system_device_name = device["name"]
                    logger.info(
                        f"Found monitor device: {device['name']} (index: {device['index']})"
                    )
                    break

            # Method 2: Try known good indices for your system
            if system_device is None:
                for device_idx in self.system_device_indices:
                    try:
                        test_device_info = None
                        for device in devices:
                            if device["index"] == device_idx:
                                test_device_info = device
                                break

                        if test_device_info and test_device_info["channels"] > 0:
                            system_device = device_idx
                            system_device_name = test_device_info["name"]
                            logger.info(
                                f"Found system audio device by index: {device_idx} - {test_device_info['name']}"
                            )
                            break
                    except Exception as e:
                        logger.debug(f"Device index {device_idx} not accessible: {e}")
                        continue

            # Method 3: Fallback to name matching
            if system_device is None:
                for device in devices:
                    device_name = device["name"].lower()
                    if any(
                        sys_name in device_name
                        for sys_name in ["pipewire", "pulse", "default"]
                    ):
                        system_device = device["index"]
                        system_device_name = device["name"]
                        logger.info(
                            f"Found system audio device by name: {device['name']} (index: {device['index']})"
                        )
                        break

            # Method 4: Try any device with high channel count
            if system_device is None:
                high_channel_devices = [d for d in devices if d["channels"] >= 32]
                if high_channel_devices:
                    system_device = high_channel_devices[0]["index"]
                    system_device_name = high_channel_devices[0]["name"]
                    logger.info(
                        f"Using high-channel device for system audio: {high_channel_devices[0]['name']} (index: {system_device})"
                    )

            # Open microphone stream
            logger.info(f"Opening microphone stream (device: {self.selected_device})")
            try:
                self.mic_stream = self.audio.open(
                    format=self.format,
                    channels=1,
                    rate=self.rate,
                    input=True,
                    input_device_index=self.selected_device,
                    frames_per_buffer=self.chunk,
                )
                logger.info("✅ Microphone stream opened successfully")
            except Exception as e:
                logger.error(f"❌ Failed to open microphone stream: {e}")
                raise

            # Try to open system audio stream if available
            if system_device is not None:
                try:
                    logger.info(
                        f"Attempting to open system audio stream (device: {system_device} - {system_device_name})"
                    )

                    # For monitor devices or PipeWire/Pulse, try different channel configurations
                    for channels in [1, 2]:
                        try:
                            logger.info(
                                f"Trying {channels} channel(s) for system audio..."
                            )
                            self.system_stream = self.audio.open(
                                format=self.format,
                                channels=channels,
                                rate=self.rate,
                                input=True,
                                input_device_index=system_device,
                                frames_per_buffer=self.chunk,
                            )
                            logger.info(
                                f"✅ Successfully opened system audio stream with {channels} channel(s)"
                            )
                            break
                        except Exception as e:
                            logger.warning(
                                f"Failed to open system audio with {channels} channels: {e}"
                            )
                            if self.system_stream:
                                try:
                                    self.system_stream.close()
                                except:
                                    pass
                                self.system_stream = None
                            continue

                    if self.system_stream:
                        logger.info(
                            "🎵 Successfully opened both microphone and system audio streams"
                        )
                    else:
                        logger.warning(
                            "⚠️ Failed to open system audio with any channel configuration"
                        )

                except Exception as e:
                    logger.warning(
                        f"❌ Failed to open system audio device {system_device}: {e}"
                    )
                    logger.info("Falling back to microphone-only recording")
                    self.system_stream = None
            else:
                logger.warning("❌ No suitable system audio device found")
                logger.info("Available devices:")
                for device in devices:
                    logger.info(
                        f"   {device['index']}: {device['name']} ({device['channels']} ch)"
                    )

            recording_mode = (
                "microphone + system audio" if self.system_stream else "microphone only"
            )
            logger.info(f"🎵 Recording {recording_mode}...")

            frame_count = 0
            while self.recording:
                try:
                    # Read microphone data
                    mic_data = self.mic_stream.read(
                        self.chunk, exception_on_overflow=False
                    )

                    if self.system_stream:
                        # Read system audio data and mix
                        try:
                            system_data = self.system_stream.read(
                                self.chunk, exception_on_overflow=False
                            )

                            # Convert to numpy arrays for mixing
                            mic_array = np.frombuffer(mic_data, dtype=np.int16)
                            system_array = np.frombuffer(system_data, dtype=np.int16)

                            # Ensure arrays are same length for mixing
                            min_length = min(len(mic_array), len(system_array))
                            if min_length > 0:
                                mic_array = mic_array[:min_length]
                                system_array = system_array[:min_length]

                                # Mix audio with optimized levels
                                mixed = (mic_array * 0.8 + system_array * 0.6).astype(
                                    np.int16
                                )
                                self.frames.append(mixed.tobytes())

                                # Log audio levels periodically for debugging
                                frame_count += 1
                                if frame_count % 100 == 0:  # Every ~2 seconds
                                    mic_level = np.abs(mic_array).mean()
                                    system_level = np.abs(system_array).mean()
                                    logger.debug(
                                        f"Audio levels - Mic: {mic_level:.1f}, System: {system_level:.1f}"
                                    )
                            else:
                                logger.warning(
                                    "Empty audio arrays, using mic only for this chunk"
                                )
                                self.frames.append(mic_data)

                        except Exception as e:
                            # If system audio read fails, just use microphone
                            logger.debug(
                                f"System audio read failed, using mic only: {e}"
                            )
                            self.frames.append(mic_data)
                    else:
                        self.frames.append(mic_data)

                except Exception as e:
                    logger.error(f"Error reading audio data: {e}")
                    break

            logger.info(f"Recording stopped. Captured {len(self.frames)} audio chunks")

        except Exception as e:
            logger.error(f"Error setting up combined recording: {e}")
            raise
        finally:
            if self.mic_stream:
                try:
                    self.mic_stream.stop_stream()
                    self.mic_stream.close()
                    logger.info("Microphone stream closed")
                except Exception as e:
                    logger.error(f"Error closing microphone stream: {e}")
            if self.system_stream:
                try:
                    self.system_stream.stop_stream()
                    self.system_stream.close()
                    logger.info("System audio stream closed")
                except Exception as e:
                    logger.error(f"Error closing system audio stream: {e}")

    def stop_recording(self):
        """Stop audio recording and return the recorded audio file path"""
        if not self.recording:
            return None

        self.recording = False

        # Wait for recording thread to finish
        if self.recording_thread:
            self.recording_thread.join(timeout=5)

        if not self.frames:
            logger.warning("No audio frames recorded")
            return None

        # Save audio to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"recording_{timestamp}.wav"
        file_path = TRANSCRIPTIONS_AUDIO_DIR / filename

        try:
            with wave.open(str(file_path), "wb") as wf:
                wf.setnchannels(
                    1 if self.recording_type == "MIC_ONLY" else 1
                )  # Mono for simplicity
                wf.setsampwidth(self.audio.get_sample_size(self.format))
                wf.setframerate(self.rate)
                wf.writeframes(b"".join(self.frames))

            logger.info(f"Audio saved to {file_path}")
            return filename

        except Exception as e:
            logger.error(f"Error saving audio file: {e}")
            return None

    def is_recording(self):
        """Check if currently recording"""
        return self.recording

    def cleanup(self):
        """Clean up PyAudio resources"""
        if self.recording:
            self.stop_recording()

        try:
            import os
            import sys

            # Suppress ALSA warnings during cleanup
            with open(os.devnull, "w") as devnull:
                stderr_backup = sys.stderr
                sys.stderr = devnull
                try:
                    self.audio.terminate()
                finally:
                    sys.stderr = stderr_backup
        except Exception as e:
            logger.error(f"Error during audio cleanup: {e}")


# --- Global recorder instance ---
recorder = AudioRecorder()

# --- FastAPI App ---
app = FastAPI(title="Meeting Processing API with Audio Recording")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Pydantic Models ---
class ProcessResponse(BaseModel):
    filename: str
    transcription: str | None = None
    summary: str | None = None
    error: str | None = None


class UploadResponse(BaseModel):
    filename: str
    message: str


class RecordingStatusResponse(BaseModel):
    is_recording: bool
    message: str


class AudioDeviceResponse(BaseModel):
    devices: list


class StartRecordingRequest(BaseModel):
    recording_type: str = "MIC_ONLY"  # "MIC_ONLY" or "SCREEN_PLUS_MIC"
    device_index: int | None = None


# --- Helper Functions ---
def generate_timestamped_filename(
    original_filename: str, project_tag: str | None = "untitled"
):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base, ext = os.path.splitext(original_filename)
    safe_project_tag = "".join(
        c if c.isalnum() or c in (" ", "_", "-") else "_" for c in project_tag
    ).replace(" ", "_")
    return f"{safe_project_tag}_{timestamp}{ext}"


async def transcribe_audio_local_whisper(audio_path: Path) -> str | None:
    global hf_asr_pipeline
    if hf_asr_pipeline is None:
        load_hugging_face_model()
        if hf_asr_pipeline is None:
            raise Exception(
                f"Whisper transcription model ({HF_MODEL_ID}) could not be loaded."
            )

    try:
        logger.info(f"Transcribing audio file with local Whisper: {audio_path}")
        result = hf_asr_pipeline(
            str(audio_path),
            generate_kwargs={"task": "transcribe"},
            return_timestamps=True,
        )
        transcription = result["text"]
        logger.info("Local Whisper transcription successful.")
        return transcription
    except Exception as e:
        logger.error(f"Error during local Whisper transcription: {e}")
        raise


# --- Audio Recording API Endpoints ---
@app.get("/audio-devices/", response_model=AudioDeviceResponse)
async def get_audio_devices():
    """Get list of available audio input devices"""
    try:
        devices = recorder.get_audio_devices()
        return AudioDeviceResponse(devices=devices)
    except Exception as e:
        logger.error(f"Error getting audio devices: {e}")
        raise HTTPException(
            status_code=500, detail=f"Error getting audio devices: {str(e)}"
        )


@app.post("/start-recording/", response_model=RecordingStatusResponse)
async def start_recording(request: StartRecordingRequest):
    """Start audio recording"""
    try:
        if recorder.is_recording():
            raise HTTPException(status_code=400, detail="Recording already in progress")

        recorder.start_recording(
            recording_type=request.recording_type, device_index=request.device_index
        )

        return RecordingStatusResponse(
            is_recording=True,
            message=f"Recording started with type: {request.recording_type}",
        )
    except Exception as e:
        logger.error(f"Error starting recording: {e}")
        raise HTTPException(
            status_code=500, detail=f"Error starting recording: {str(e)}"
        )


@app.post("/stop-recording/", response_model=RecordingStatusResponse)
async def stop_recording():
    """Stop audio recording"""
    try:
        if not recorder.is_recording():
            raise HTTPException(status_code=400, detail="No recording in progress")

        filename = recorder.stop_recording()

        if filename:
            return RecordingStatusResponse(
                is_recording=False,
                message=f"Recording stopped. File saved as: {filename}",
            )
        else:
            return RecordingStatusResponse(
                is_recording=False,
                message="Recording stopped but no audio was captured",
            )
    except Exception as e:
        logger.error(f"Error stopping recording: {e}")
        raise HTTPException(
            status_code=500, detail=f"Error stopping recording: {str(e)}"
        )


@app.get("/recording-status/", response_model=RecordingStatusResponse)
async def get_recording_status():
    """Get current recording status"""
    is_recording = recorder.is_recording()
    message = "Recording in progress" if is_recording else "Not recording"
    return RecordingStatusResponse(is_recording=is_recording, message=message)


# --- Existing API Endpoints (Updated) ---
@app.post("/upload-audio/", response_model=UploadResponse)
async def upload_audio_file(
    project_tag: str = Form("untitled"), audio_file: UploadFile = File(...)
):
    """Upload audio file for processing"""
    if not GOOGLE_GEMINI_API_KEY:
        raise HTTPException(
            status_code=500,
            detail="GOOGLE_GEMINI_API_KEY is not configured on the server for summarization.",
        )

    if not audio_file.content_type.startswith("audio/"):
        logger.error(f"Invalid file type: {audio_file.content_type}")
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Expected audio/*, got {audio_file.content_type}",
        )

    server_filename = generate_timestamped_filename(audio_file.filename, project_tag)
    file_path = TRANSCRIPTIONS_AUDIO_DIR / server_filename

    logger.info(f"Attempting to save uploaded audio to: {file_path}")
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(audio_file.file, buffer)
        logger.info(f"Audio file '{server_filename}' saved successfully.")
    except Exception as e:
        logger.error(f"Error saving audio file '{server_filename}': {e}")
        raise HTTPException(status_code=500, detail=f"Could not save audio file: {e}")
    finally:
        audio_file.file.close()

    return UploadResponse(
        filename=server_filename, message="Audio file uploaded successfully."
    )


@app.post("/process-meeting/", response_model=ProcessResponse)
async def process_meeting_audio(filename: str = Form(...)):
    """Process uploaded audio file for transcription and summarization"""
    logger.info(f"Processing request for audio file: {filename}")
    audio_file_path = TRANSCRIPTIONS_AUDIO_DIR / filename

    if not audio_file_path.exists():
        logger.error(f"Audio file not found: {audio_file_path}")
        raise HTTPException(
            status_code=404, detail=f"Audio file '{filename}' not found on server."
        )

    transcription_text = None
    summary_text = None
    error_message = None

    # 1. Transcription with Local Whisper Model
    try:
        logger.info(f"Starting local transcription for {filename}...")
        transcription_text = await transcribe_audio_local_whisper(audio_file_path)

        if transcription_text is not None:
            logger.info(f"Local Whisper transcription successful for {filename}.")
            transcription_output_filename = f"{Path(filename).stem}_transcription.txt"
            transcription_output_path = (
                MEETING_RESUMES_DIR / transcription_output_filename
            )
            with open(transcription_output_path, "w", encoding="utf-8") as f:
                f.write(transcription_text)
            logger.info(f"Transcription saved to {transcription_output_path}")
        else:
            error_message = "Local Whisper transcription returned no text or failed."

    except Exception as e:
        logger.error(f"Error during local Whisper transcription for {filename}: {e}")
        error_message = f"Local Whisper transcription failed: {str(e)}"

    # 2. Summarization with Gemini
    if transcription_text and not error_message:
        try:
            logger.info(f"Starting summarization for {filename}...")
            if not GOOGLE_GEMINI_API_KEY:
                raise ValueError(
                    "Google Gemini API Key not configured for summarization."
                )

            gemini_model = genai.GenerativeModel("gemini-1.5-flash-latest")
            prompt = f'Por favor, resuma a seguinte transcrição de reunião e liste os pontos chave. Responda em Português do Brasil:\n\nTranscrição:\n"{transcription_text}"'

            response = gemini_model.generate_content(prompt)
            summary_text = response.text
            logger.info(f"Summarization successful for {filename}.")

            summary_output_filename = f"{Path(filename).stem}_summary.txt"
            summary_output_path = MEETING_RESUMES_DIR / summary_output_filename
            with open(summary_output_path, "w", encoding="utf-8") as f:
                f.write(summary_text)
            logger.info(f"Summary saved to {summary_output_path}")

        except Exception as e:
            logger.error(f"Error during summarization for {filename}: {e}")
            current_error = f"Summarization failed: {e}"
            error_message = (
                (error_message + f"; {current_error}")
                if error_message
                else current_error
            )

    return ProcessResponse(
        filename=filename,
        transcription=transcription_text,
        summary=summary_text,
        error=error_message,
    )


# --- Cleanup on shutdown ---
@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources on shutdown"""
    recorder.cleanup()
    logger.info("Application shutdown complete")


if __name__ == "__main__":
    import uvicorn

    logger.info(
        f"Attempting to load Whisper transcription model ({HF_MODEL_ID}) on startup..."
    )
    load_hugging_face_model()
    logger.info(f"Audio files will be saved to/read from: {TRANSCRIPTIONS_AUDIO_DIR}")
    logger.info(f"Outputs will be written to: {MEETING_RESUMES_DIR}")
    logger.info("Ensure GOOGLE_GEMINI_API_KEY is set as an environment variable.")
    logger.info(f"Using device for Whisper model: {device}")

    # Run on port 8462 as requested
    uvicorn.run(app, host="0.0.0.0", port=8462)
