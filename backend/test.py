#!/usr/bin/env python3
"""
Live recording debug - test exactly what the backend is doing during actual recording
"""

import os
import sys
import time
import threading
import wave
import numpy as np
import requests

# Suppress ALSA warnings
with open(os.devnull, "w") as devnull:
    stderr_backup = sys.stderr
    sys.stderr = devnull
    try:
        import pyaudio

        audio = pyaudio.PyAudio()
    finally:
        sys.stderr = stderr_backup


class LiveRecordingTest:
    def __init__(self):
        self.recording = False
        self.frames = []
        self.mic_frames = []
        self.system_frames = []
        self.mic_stream = None
        self.system_stream = None

    def test_live_recording(self, mic_device=4, system_device=6, duration=10):
        """Test live recording exactly like the backend does"""
        print("🎵 Live Recording Test")
        print("=" * 30)
        print(f"Microphone: Device {mic_device}")
        print(f"System Audio: Device {system_device}")
        print(f"Duration: {duration} seconds")
        print()

        # Check current pavucontrol routing
        print("📋 Current PulseAudio routing:")
        os.system("pactl list sources short | grep -E '(monitor|pipewire|pulse)'")
        print()

        print("🔊 IMPORTANT: Before starting:")
        print("1. Play browser audio (YouTube, music, etc.)")
        print("2. Open pavucontrol")
        print("3. Go to Recording tab")
        print("4. Make sure the source is set to 'Monitor of Family 17h' or similar")
        print()
        input("Press Enter when ready...")

        try:
            # Open microphone stream
            print(f"🎤 Opening microphone stream (device {mic_device})...")
            self.mic_stream = audio.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=44100,
                input=True,
                input_device_index=mic_device,
                frames_per_buffer=1024,
            )
            print("✅ Microphone stream opened")

            # Open system audio stream
            print(f"🖥️ Opening system audio stream (device {system_device})...")
            self.system_stream = audio.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=44100,
                input=True,
                input_device_index=system_device,
                frames_per_buffer=1024,
            )
            print("✅ System audio stream opened")

            print()
            print("🔴 RECORDING STARTED!")
            print("- Speak into your microphone")
            print("- Keep browser audio playing")
            print("- Recording will analyze audio levels in real-time")
            print()

            self.recording = True
            start_time = time.time()

            while self.recording and (time.time() - start_time) < duration:
                try:
                    # Read from both streams
                    mic_data = self.mic_stream.read(1024, exception_on_overflow=False)
                    system_data = self.system_stream.read(
                        1024, exception_on_overflow=False
                    )

                    # Store separate for analysis
                    self.mic_frames.append(mic_data)
                    self.system_frames.append(system_data)

                    # Mix for final output
                    mic_array = np.frombuffer(mic_data, dtype=np.int16)
                    system_array = np.frombuffer(system_data, dtype=np.int16)

                    # Mix audio
                    mixed = (mic_array * 0.8 + system_array * 0.6).astype(np.int16)
                    self.frames.append(mixed.tobytes())

                    # Real-time analysis every 2 seconds
                    if len(self.frames) % 88 == 0:  # ~2 seconds
                        mic_level = np.abs(mic_array).mean()
                        system_level = np.abs(system_array).mean()
                        elapsed = time.time() - start_time
                        print(
                            f"[{elapsed:.1f}s] Mic: {mic_level:6.1f} | System: {system_level:6.1f} | {'🎤' if mic_level > 100 else '🔇'} {'🔊' if system_level > 100 else '🔇'}"
                        )

                        # If no system audio after 4 seconds, give hint
                        if elapsed > 4 and system_level < 50:
                            print(
                                "      ⚠️ Low system audio - check pavucontrol Recording tab!"
                            )

                except Exception as e:
                    print(f"❌ Recording error: {e}")
                    break

            print("\n🔴 RECORDING STOPPED!")
            self.recording = False

            # Analyze results
            self.analyze_recording()

        except Exception as e:
            print(f"❌ Failed to start recording: {e}")
        finally:
            self.cleanup()

    def analyze_recording(self):
        """Analyze the recorded audio"""
        print("\n📊 ANALYSIS:")
        print("=" * 20)

        if not self.frames:
            print("❌ No audio recorded")
            return

        # Analyze microphone audio
        if self.mic_frames:
            mic_data = b"".join(self.mic_frames)
            mic_array = np.frombuffer(mic_data, dtype=np.int16)
            mic_avg = np.abs(mic_array).mean()
            mic_max = np.abs(mic_array).max()

            print(f"🎤 Microphone:")
            print(f"   Average: {mic_avg:.1f}")
            print(f"   Max: {mic_max:.1f}")
            print(
                f"   Status: {'✅ Good' if mic_avg > 100 else '⚠️ Low' if mic_avg > 10 else '❌ Silent'}"
            )

            # Save microphone file
            with wave.open("live_test_microphone.wav", "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
                wf.setframerate(44100)
                wf.writeframes(mic_data)
            print(f"   💾 Saved: live_test_microphone.wav")

        # Analyze system audio
        if self.system_frames:
            system_data = b"".join(self.system_frames)
            system_array = np.frombuffer(system_data, dtype=np.int16)
            system_avg = np.abs(system_array).mean()
            system_max = np.abs(system_array).max()

            print(f"🖥️ System Audio:")
            print(f"   Average: {system_avg:.1f}")
            print(f"   Max: {system_max:.1f}")
            print(
                f"   Status: {'✅ Good' if system_avg > 100 else '⚠️ Low' if system_avg > 10 else '❌ Silent'}"
            )

            # Save system audio file
            with wave.open("live_test_system.wav", "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
                wf.setframerate(44100)
                wf.writeframes(system_data)
            print(f"   💾 Saved: live_test_system.wav")

            if system_avg < 50:
                print("\n💡 SYSTEM AUDIO TROUBLESHOOTING:")
                print("   1. Check pavucontrol Recording tab")
                print("   2. Make sure input is 'Monitor of Family 17h' or similar")
                print("   3. Increase system volume")
                print("   4. Try different browser/application")

        # Analyze mixed audio
        mixed_data = b"".join(self.frames)
        mixed_array = np.frombuffer(mixed_data, dtype=np.int16)
        mixed_avg = np.abs(mixed_array).mean()

        print(f"🎵 Mixed Audio:")
        print(f"   Average: {mixed_avg:.1f}")
        print(
            f"   Status: {'✅ Good' if mixed_avg > 100 else '⚠️ Low' if mixed_avg > 10 else '❌ Silent'}"
        )

        # Save mixed file
        with wave.open("live_test_mixed.wav", "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
            wf.setframerate(44100)
            wf.writeframes(mixed_data)
        print(f"   💾 Saved: live_test_mixed.wav")

        print()
        self.give_recommendations()

    def give_recommendations(self):
        """Give specific recommendations based on results"""
        print("💡 RECOMMENDATIONS:")

        # Check if we have the data we need
        if not self.mic_frames or not self.system_frames:
            print("❌ Could not analyze - check if streams opened correctly")
            return

        mic_data = b"".join(self.mic_frames)
        system_data = b"".join(self.system_frames)
        mic_avg = np.abs(np.frombuffer(mic_data, dtype=np.int16)).mean()
        system_avg = np.abs(np.frombuffer(system_data, dtype=np.int16)).mean()

        if mic_avg > 100 and system_avg > 100:
            print("✅ Both audio sources working! Backend should work correctly.")
            print(
                "🔧 If backend still fails, check the exact device indices it's using."
            )
        elif mic_avg > 100 and system_avg < 50:
            print("🎤 Microphone works, but system audio is silent.")
            print("🔧 Solutions:")
            print("   - Open pavucontrol during recording")
            print("   - Switch to 'Monitor of Family 17h' in Recording tab")
            print("   - Try different system audio device (7 or 8 instead of 6)")
        elif mic_avg < 50 and system_avg > 100:
            print("🖥️ System audio works, but microphone is silent.")
            print("🔧 Solutions:")
            print("   - Check microphone permissions")
            print("   - Try device 7 instead of device 4 for microphone")
        else:
            print("❌ Both sources have issues.")
            print("🔧 Check audio system configuration")

    def cleanup(self):
        """Clean up streams"""
        self.recording = False
        if self.mic_stream:
            try:
                self.mic_stream.stop_stream()
                self.mic_stream.close()
            except:
                pass
        if self.system_stream:
            try:
                self.system_stream.stop_stream()
                self.system_stream.close()
            except:
                pass


def test_backend_api():
    """Test if the actual backend API is working"""
    print("\n🔗 Testing Backend API...")
    try:
        # Test if backend is running
        response = requests.get("http://localhost:8462/audio-devices/")
        if response.status_code == 200:
            devices = response.json()
            print("✅ Backend is running")
            print("📱 Backend sees these devices:")
            for device in devices["devices"]:
                print(
                    f"   {device['index']}: {device['name']} ({device['channels']} ch)"
                )
        else:
            print("❌ Backend not responding correctly")
    except Exception as e:
        print(f"❌ Backend not accessible: {e}")
        print("💡 Make sure backend is running: python main.py")


def main():
    print("🎵 Live Recording Debug")
    print("This will test exactly what happens during real recording")
    print()

    # Test backend API first
    test_backend_api()

    # Test live recording
    tester = LiveRecordingTest()

    print("\n" + "=" * 50)
    print("🎯 LIVE RECORDING TEST")
    print("This will record for 10 seconds and analyze audio levels in real-time")

    try:
        tester.test_live_recording(mic_device=4, system_device=6, duration=10)
    except KeyboardInterrupt:
        print("\n🛑 Test interrupted by user")
        tester.recording = False
        tester.cleanup()

    audio.terminate()

    print("\n📋 SUMMARY:")
    print("Check the saved .wav files to verify what was actually recorded:")
    print("- live_test_microphone.wav (microphone only)")
    print("- live_test_system.wav (system audio only)")
    print("- live_test_mixed.wav (combined)")


if __name__ == "__main__":
    main()
