#!/usr/bin/env python3
"""
Safe recording solution - test what actually works for your app
"""

import os
import sys
import requests
import time
import json

# Suppress ALSA warnings
with open(os.devnull, "w") as devnull:
    stderr_backup = sys.stderr
    sys.stderr = devnull
    try:
        import pyaudio

        audio = pyaudio.PyAudio()
    finally:
        sys.stderr = stderr_backup


def get_current_devices():
    """Get current clean device list"""
    print("📱 Current Audio Devices (After Restoration)")
    print("=" * 45)

    devices = []
    for i in range(audio.get_device_count()):
        info = audio.get_device_info_by_index(i)
        if info["maxInputChannels"] > 0:
            name = info["name"]
            channels = info["maxInputChannels"]

            # Categorize for easy identification
            if "webcam" in name.lower() or "usb" in name.lower():
                category = "🎤 USB Microphone"
            elif "monitor" in name.lower():
                category = "🔊 Audio Monitor"
            elif "analog" in name.lower() or "alc" in name.lower():
                category = "🎤 Built-in Microphone"
            elif "pipewire" in name.lower():
                category = "🔧 PipeWire"
            elif "pulse" in name.lower():
                category = "🔧 PulseAudio"
            else:
                category = "❓ Other"

            print(f"   {i:2d}: {name} ({channels} ch) - {category}")
            devices.append(
                {"index": i, "name": name, "channels": channels, "category": category}
            )

    return devices


def test_backend_connectivity():
    """Test if backend is running and responsive"""
    print("\n🔗 Testing Backend Connectivity")
    print("=" * 35)

    try:
        response = requests.get("http://localhost:8462/audio-devices/", timeout=5)
        if response.status_code == 200:
            devices = response.json()
            print("✅ Backend is running and responsive")
            print("📱 Backend sees these devices:")
            for device in devices["devices"]:
                print(
                    f"   {device['index']}: {device['name']} ({device['channels']} ch)"
                )
            return True, devices["devices"]
        else:
            print(f"❌ Backend responded with status: {response.status_code}")
            return False, []
    except requests.exceptions.ConnectionError:
        print("❌ Backend not running or not accessible")
        print("💡 Start backend with: python main.py")
        return False, []
    except Exception as e:
        print(f"❌ Error connecting to backend: {e}")
        return False, []


def test_microphone_recording():
    """Test microphone recording specifically"""
    print("\n🎤 Testing Microphone Recording")
    print("=" * 35)

    # Find microphone devices
    devices = get_current_devices()
    mic_devices = [d for d in devices if "🎤" in d["category"]]

    if not mic_devices:
        print("❌ No microphone devices found")
        return None

    print("Available microphone devices:")
    for device in mic_devices:
        print(f"   Device {device['index']}: {device['name']}")

    # Test the first microphone
    test_device = mic_devices[0]
    print(f"\n🧪 Testing Device {test_device['index']}: {test_device['name']}")
    print("🗣️ Speak into your microphone for 3 seconds...")
    input("Press Enter when ready...")

    try:
        import numpy as np
        import wave

        stream = audio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=44100,
            input=True,
            input_device_index=test_device["index"],
            frames_per_buffer=1024,
        )

        frames = []
        print("🔴 Recording...")
        for i in range(132):  # ~3 seconds
            data = stream.read(1024, exception_on_overflow=False)
            frames.append(data)
            if i % 22 == 0:
                print(".", end="", flush=True)

        print("\n📊 Analyzing...")
        stream.close()

        # Analyze and save
        audio_data = b"".join(frames)
        audio_array = np.frombuffer(audio_data, dtype=np.int16)
        avg_amplitude = np.abs(audio_array).mean()

        filename = f"mic_test_device_{test_device['index']}.wav"
        with wave.open(filename, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
            wf.setframerate(44100)
            wf.writeframes(audio_data)

        print(f"Average amplitude: {avg_amplitude:.1f}")
        print(f"💾 Saved: {filename}")

        if avg_amplitude > 100:
            print("✅ Microphone recording works perfectly!")
            return test_device["index"]
        else:
            print("❌ Microphone recording failed or too quiet")
            return None

    except Exception as e:
        print(f"❌ Error testing microphone: {e}")
        return None


def test_backend_mic_recording(backend_devices, working_mic_device):
    """Test backend recording with working microphone"""
    print("\n🎯 Testing Backend Microphone Recording")
    print("=" * 42)

    if not working_mic_device:
        print("❌ No working microphone device to test")
        return False

    print(f"Testing with device {working_mic_device}")

    try:
        # Start recording via backend API
        response = requests.post(
            "http://localhost:8462/start-recording/",
            json={"recording_type": "MIC_ONLY", "device_index": working_mic_device},
            timeout=10,
        )

        if response.status_code != 200:
            print(f"❌ Failed to start recording: {response.text}")
            return False

        print("✅ Backend recording started")
        print("🗣️ Speak into microphone for 5 seconds...")
        time.sleep(5)

        # Stop recording
        response = requests.post("http://localhost:8462/stop-recording/", timeout=10)

        if response.status_code != 200:
            print(f"❌ Failed to stop recording: {response.text}")
            return False

        result = response.json()
        print("✅ Backend recording stopped")
        print(f"📋 Result: {result['message']}")

        # Check if file was created
        if "File saved as:" in result["message"]:
            print("✅ Backend microphone recording works!")
            return True
        else:
            print("❌ Backend recording failed to create file")
            return False

    except Exception as e:
        print(f"❌ Error testing backend recording: {e}")
        return False


def show_system_audio_solutions():
    """Show safe solutions for system audio capture"""
    print("\n🖥️ SAFE SYSTEM AUDIO SOLUTIONS")
    print("=" * 35)
    print()
    print("Since automatic routing caused issues, here are SAFE alternatives:")
    print()
    print("1️⃣ PAVUCONTROL METHOD (Recommended):")
    print("   ✅ Safe - no permanent changes")
    print("   ✅ Reliable - works every time")
    print("   📋 Steps:")
    print("      1. Start recording with 'Screen + Mic' in your app")
    print("      2. Open pavucontrol")
    print("      3. Go to 'Recording' tab")
    print("      4. Find your app and change source to 'Monitor of...'")
    print("      5. Continue recording - will capture browser audio")
    print()
    print("2️⃣ BROWSER SCREEN SHARING:")
    print("   ✅ Built-in - uses browser's own audio capture")
    print("   ✅ No system changes needed")
    print("   📋 Steps:")
    print("      1. In browser, start screen share of the tab")
    print("      2. Enable 'Share system audio' in the dialog")
    print("      3. Record the screen share instead of system audio")
    print()
    print("3️⃣ EXTERNAL RECORDING:")
    print("   ✅ Professional - use dedicated software")
    print("   📋 Tools:")
    print("      - OBS Studio (free, powerful)")
    print("      - SimpleScreenRecorder")
    print("      - Record audio separately and mix later")
    print()
    print("🛡️ WHY THESE ARE SAFER:")
    print("   - No permanent audio system changes")
    print("   - No risk of audio feedback loops")
    print("   - No risk of breaking system audio")
    print("   - Easy to undo/change")


def recommend_backend_config(working_mic_device, backend_working):
    """Recommend backend configuration"""
    print("\n🔧 BACKEND CONFIGURATION RECOMMENDATIONS")
    print("=" * 45)

    if working_mic_device and backend_working:
        print("✅ MICROPHONE RECORDING IS WORKING!")
        print(f"🎯 Use device {working_mic_device} for microphone")
        print()
        print("📝 Backend Configuration:")
        print(f"   - Microphone device: {working_mic_device}")
        print(f"   - Recording mode: 'MIC_ONLY' (safe and working)")
        print()
        print("🎵 For System Audio:")
        print("   - Use pavucontrol method during recording")
        print("   - Start with 'Screen + Mic', then route in pavucontrol")
        print("   - This avoids all the dangerous automatic routing")

    elif working_mic_device:
        print("🎤 MICROPHONE WORKS, BUT BACKEND HAS ISSUES")
        print(f"✅ Device {working_mic_device} captures microphone correctly")
        print("❌ Backend recording failed")
        print()
        print("🔧 Troubleshooting:")
        print("   1. Check backend logs for specific errors")
        print("   2. Verify device permissions")
        print("   3. Test with different device indices")

    else:
        print("❌ MICROPHONE RECORDING ISSUES")
        print("🔧 Troubleshooting:")
        print("   1. Check microphone permissions")
        print("   2. Test different microphone devices")
        print("   3. Verify audio system is working correctly")


def main():
    print("🎯 Safe Recording Solution")
    print("Let's figure out what actually works for your app")
    print("=" * 50)

    # Step 1: Get current devices
    devices = get_current_devices()

    # Step 2: Test backend connectivity
    backend_running, backend_devices = test_backend_connectivity()

    # Step 3: Test microphone recording
    working_mic_device = test_microphone_recording()

    # Step 4: Test backend with working microphone
    backend_working = False
    if backend_running and working_mic_device:
        backend_working = test_backend_mic_recording(
            backend_devices, working_mic_device
        )

    # Step 5: Show system audio solutions
    show_system_audio_solutions()

    # Step 6: Final recommendations
    recommend_backend_config(working_mic_device, backend_working)

    print("\n" + "=" * 50)
    print("📋 SUMMARY:")
    print(f"🎤 Microphone: {'✅ Working' if working_mic_device else '❌ Issues'}")
    print(f"🔗 Backend: {'✅ Working' if backend_running else '❌ Not running'}")
    print(f"🎵 Backend Recording: {'✅ Working' if backend_working else '❌ Issues'}")
    print()
    print("🎯 RECOMMENDED APPROACH:")
    if working_mic_device and backend_working:
        print("   1. Use 'Mic Only' mode for regular recording")
        print("   2. Use pavucontrol method when you need system audio")
        print("   3. No automatic routing - keep it simple and safe!")
    else:
        print("   1. Fix microphone/backend issues first")
        print("   2. Focus on getting basic recording working")
        print("   3. Add system audio capture later using safe methods")

    audio.terminate()


if __name__ == "__main__":
    main()
