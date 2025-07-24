import React, { useState, useEffect, useCallback } from 'react';

// --- Icons ---
const MicIcon = () => (
  <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" className="w-5 h-5 mr-2">
    <path d="M7 4a3 3 0 016 0v4a3 3 0 11-6 0V4z" />
    <path fillRule="evenodd" d="M5.5 8.5A.5.5 0 016 9v1a4 4 0 004 4h.043c.21 0 .415.025.615.073A3.994 3.994 0 0014 14V9.5a.5.5 0 011 0V14a5 5 0 01-4.688 4.969A4.992 4.992 0 015 14V9a.5.5 0 01.5-.5z" clipRule="evenodd" />
  </svg>
);

const ScreenIcon = () => (
  <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" className="w-5 h-5 mr-2">
    <path d="M2.5 4A1.5 1.5 0 001 5.5v9A1.5 1.5 0 002.5 16h15a1.5 1.5 0 001.5-1.5v-9A1.5 1.5 0 0017.5 4h-15zM2 5.5a.5.5 0 01.5-.5h15a.5.5 0 01.5.5v9a.5.5 0 01-.5.5h-15a.5.5 0 01-.5-.5v-9z" />
    <path d="M6.5 18a.5.5 0 000 1h7a.5.5 0 000-1h-7z" />
  </svg>
);

const SaveIcon = () => (
  <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" className="w-5 h-5 mr-1">
    <path d="M2.5 3A1.5 1.5 0 001 4.5v11A1.5 1.5 0 002.5 17h15a1.5 1.5 0 001.5-1.5v-11A1.5 1.5 0 0017.5 3h-15zM2 4.5a.5.5 0 01.5-.5h15a.5.5 0 01.5.5v11a.5.5 0 01-.5.5h-15a.5.5 0 01-.5-.5v-11z" />
    <path d="M10 7a.75.75 0 01.75.75v2.5h2.5a.75.75 0 010 1.5h-2.5v2.5a.75.75 0 01-1.5 0v-2.5h-2.5a.75.75 0 010-1.5h2.5v-2.5A.75.75 0 0110 7z" />
  </svg>
);

const RefreshIcon = () => (
  <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" className="w-4 h-4">
    <path fillRule="evenodd" d="M4 2a1 1 0 011 1v2.101a7.002 7.002 0 0111.601 2.566 1 1 0 11-1.885.666A5.002 5.002 0 005.999 7H9a1 1 0 010 2H4a1 1 0 01-1-1V3a1 1 0 011-1zm.008 9.057a1 1 0 011.276.61A5.002 5.002 0 0014.001 13H11a1 1 0 110-2h5a1 1 0 011 1v5a1 1 0 11-2 0v-2.101a7.002 7.002 0 01-11.601-2.566 1 1 0 01.61-1.276z" clipRule="evenodd" />
  </svg>
);

// --- Constants ---
const DB_NAME = 'meetingSummariesDB_Backend_API';
const RECORDING_TYPE = {
  MIC_ONLY: 'MIC_ONLY',
  SCREEN_PLUS_MIC: 'SCREEN_PLUS_MIC',
};
const BACKEND_URL = 'http://localhost:8462';

// --- Types ---
interface AudioDevice {
  index: number;
  name: string;
  channels: number;
  default_sample_rate: number;
}

interface Recording {
  id: string;
  dateTime: string;
  projectTag: string;
  serverFilename: string;
  transcription: string;
  summary: string;
}

interface StatusMessage {
  text: string;
  type: 'info' | 'success' | 'warning' | 'error';
}

// --- Main App Component ---
function App() {
  const [isRecording, setIsRecording] = useState(false);
  const [recordingType, setRecordingType] = useState(RECORDING_TYPE.MIC_ONLY);
  const [currentTranscription, setCurrentTranscription] = useState('');
  const [currentSummary, setCurrentSummary] = useState('');
  const [allRecordings, setAllRecordings] = useState<Recording[]>([]);
  const [activeRecordingId, setActiveRecordingId] = useState<string | null>(null);

  const [statusMessage, setStatusMessage] = useState<StatusMessage>({
    text: 'Ready to record.',
    type: 'info'
  });
  const [isLoading, setIsLoading] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);

  const [showProjectTagModal, setShowProjectTagModal] = useState(false);
  const [projectTagInput, setProjectTagInput] = useState('');
  const [currentRecordingFilename, setCurrentRecordingFilename] = useState('');

  const [audioInputDevices, setAudioInputDevices] = useState<AudioDevice[]>([]);
  const [selectedAudioInputDevice, setSelectedAudioInputDevice] = useState<number | null>(null);
  const [backendConnected, setBackendConnected] = useState(false);

  // --- API Functions ---
  const checkBackendConnection = async () => {
    try {
      const response = await fetch(`${BACKEND_URL}/recording-status/`);
      if (response.ok) {
        setBackendConnected(true);
        return true;
      }
      throw new Error('Backend not responding');
    } catch (error) {
      setBackendConnected(false);
      updateStatus('Backend not connected. Please ensure the backend is running on port 8462.', 'error');
      return false;
    }
  };

  const fetchAudioDevices = async () => {
    try {
      const response = await fetch(`${BACKEND_URL}/audio-devices/`);
      if (!response.ok) {
        throw new Error(`Failed to fetch audio devices: ${response.statusText}`);
      }
      const data = await response.json();
      setAudioInputDevices(data.devices);

      if (data.devices.length > 0 && selectedAudioInputDevice === null) {
        setSelectedAudioInputDevice(data.devices[0].index);
      }
    } catch (error) {
      console.error('Error fetching audio devices:', error);
      updateStatus('Failed to fetch audio devices from backend.', 'error');
    }
  };

  const checkRecordingStatus = async () => {
    try {
      const response = await fetch(`${BACKEND_URL}/recording-status/`);
      if (response.ok) {
        const data = await response.json();
        setIsRecording(data.is_recording);
        return data.is_recording;
      }
    } catch (error) {
      console.error('Error checking recording status:', error);
    }
    return false;
  };

  const startRecordingAPI = async () => {
    try {
      const response = await fetch(`${BACKEND_URL}/start-recording/`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          recording_type: recordingType,
          device_index: selectedAudioInputDevice,
        }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to start recording');
      }

      const data = await response.json();
      setIsRecording(true);
      updateStatus(data.message, 'success');
    } catch (error) {
      console.error('Error starting recording:', error);
      updateStatus(`Error starting recording: ${error.message}`, 'error');
    }
  };

  const stopRecordingAPI = async () => {
    try {
      const response = await fetch(`${BACKEND_URL}/stop-recording/`, {
        method: 'POST',
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to stop recording');
      }

      const data = await response.json();
      setIsRecording(false);

      // Extract filename from the message
      const match = data.message.match(/File saved as: (.+)/);
      if (match) {
        setCurrentRecordingFilename(match[1]);
        setShowProjectTagModal(true);
        updateStatus('Recording stopped. Enter project tag to process.', 'success');
      } else {
        updateStatus(data.message, 'warning');
      }
    } catch (error) {
      console.error('Error stopping recording:', error);
      updateStatus(`Error stopping recording: ${error.message}`, 'error');
    }
  };

  const processRecordingWithBackend = async (projectTag: string) => {
    if (!currentRecordingFilename) {
      updateStatus('No recording to process.', 'error');
      return;
    }

    setShowProjectTagModal(false);
    setIsProcessing(true);
    updateStatus(`Processing recording for "${projectTag}"...`, 'info', true);

    try {
      const formData = new FormData();
      formData.append('filename', currentRecordingFilename);

      const processResponse = await fetch(`${BACKEND_URL}/process-meeting/`, {
        method: 'POST',
        body: formData,
      });

      if (!processResponse.ok) {
        const errorData = await processResponse.json();
        throw new Error(`Processing failed: ${processResponse.status} ${errorData.detail || processResponse.statusText}`);
      }

      const processResult = await processResponse.json();

      if (processResult.error) {
        updateStatus(`Error processing: ${processResult.error}`, 'error');
        setCurrentTranscription('');
        setCurrentSummary('');
      } else {
        setCurrentTranscription(processResult.transcription || '');
        setCurrentSummary(processResult.summary || '');
        updateStatus('Processing complete!', 'success');

        const newRecording: Recording = {
          id: `rec_${Date.now()}_${currentRecordingFilename}`,
          dateTime: new Date().toISOString(),
          projectTag: projectTag || 'untitled',
          serverFilename: currentRecordingFilename,
          transcription: processResult.transcription || '',
          summary: processResult.summary || '',
        };

        const updatedRecordings = [newRecording, ...allRecordings];
        setAllRecordings(updatedRecordings);
        localStorage.setItem(DB_NAME, JSON.stringify(updatedRecordings));
        setActiveRecordingId(newRecording.id);
      }
    } catch (error) {
      console.error('Error during backend processing:', error);
      updateStatus(`Error: ${error.message}`, 'error');
    } finally {
      setIsProcessing(false);
      setIsLoading(false);
      setProjectTagInput('');
      setCurrentRecordingFilename('');
    }
  };

  // --- Helper Functions ---
  const updateStatus = useCallback((text: string, type: StatusMessage['type'] = 'info', loading = false) => {
    setStatusMessage({ text, type });
    setIsLoading(loading);
  }, []);

  const loadRecordingForView = useCallback((id: string) => {
    const rec = allRecordings.find(r => r.id === id);
    if (rec) {
      setActiveRecordingId(id);
      setCurrentTranscription(rec.transcription);
      setCurrentSummary(rec.summary || '');
      updateStatus(`Loaded: ${rec.projectTag}`, 'info');
    }
  }, [allRecordings, updateStatus]);

  // --- Event Handlers ---
  const handleStartRecording = async () => {
    if (!backendConnected) {
      updateStatus('Backend not connected. Please check the connection.', 'error');
      return;
    }

    setCurrentTranscription('');
    setCurrentSummary('');
    setActiveRecordingId(null);
    setCurrentRecordingFilename('');

    updateStatus('Starting recording...', 'info', true);
    await startRecordingAPI();
    setIsLoading(false);
  };

  const handleStopRecording = async () => {
    updateStatus('Stopping recording...', 'info', true);
    await stopRecordingAPI();
    setIsLoading(false);
  };

  const refreshDevices = async () => {
    updateStatus('Refreshing audio devices...', 'info', true);
    await fetchAudioDevices();
    setIsLoading(false);
    updateStatus('Audio devices refreshed.', 'success');
  };

  // --- Effects ---
  useEffect(() => {
    // Load saved recordings from localStorage
    const data = localStorage.getItem(DB_NAME);
    if (data) {
      try {
        setAllRecordings(JSON.parse(data));
      } catch (e) {
        console.error("Error parsing recordings from localStorage:", e);
        localStorage.removeItem(DB_NAME);
      }
    }

    // Check backend connection and fetch devices
    const initializeApp = async () => {
      const connected = await checkBackendConnection();
      if (connected) {
        await fetchAudioDevices();
        // Check if recording is already in progress
        await checkRecordingStatus();
      }
    };

    initializeApp();
  }, []);

  // Poll recording status when recording
  useEffect(() => {
    let interval: NodeJS.Timeout;

    if (isRecording) {
      interval = setInterval(async () => {
        const status = await checkRecordingStatus();
        if (!status && isRecording) {
          // Recording stopped externally
          setIsRecording(false);
          updateStatus('Recording stopped.', 'info');
        }
      }, 2000);
    }

    return () => {
      if (interval) clearInterval(interval);
    };
  }, [isRecording]);

  return (
    <div className="flex w-full min-h-screen font-inter bg-gray-800 text-gray-100">
      {/* Sidebar */}
      <aside className="w-72 bg-gray-700 p-6 border-r border-gray-600 h-screen overflow-y-auto flex-shrink-0">
        <h2 className="text-xl font-semibold mb-4">Past Recordings</h2>
        <ul>
          {allRecordings.length === 0 && (
            <li className="text-gray-400 italic">No recordings yet.</li>
          )}
          {allRecordings.map((rec) => (
            <li
              key={rec.id}
              className={`p-3 mb-2 rounded-md cursor-pointer transition-colors ${activeRecordingId === rec.id
                ? 'bg-blue-600 text-white'
                : 'bg-gray-600 hover:bg-blue-500'
                }`}
              onClick={() => loadRecordingForView(rec.id)}
            >
              <span className="font-medium block">{rec.projectTag}</span>
              <span className="text-xs text-gray-300">
                {new Date(rec.dateTime).toLocaleString()}
              </span>
              {rec.serverFilename && (
                <span className="text-xs text-gray-400 block truncate">
                  File: {rec.serverFilename}
                </span>
              )}
            </li>
          ))}
        </ul>
      </aside>

      {/* Main Content */}
      <main className="flex-grow p-6 flex flex-col items-center max-h-screen overflow-y-auto">
        <div className="w-full max-w-3xl bg-gray-700 p-8 rounded-lg shadow-xl">
          <header className="text-center mb-6">
            <h1 className="text-3xl font-bold">Meeting Recorder & Summarizer</h1>
            <p className="text-gray-400 mt-1">
              Backend-powered recording with AI transcription & summarization
            </p>

            {/* Backend Status */}
            <div className={`mt-2 px-3 py-1 rounded-full text-sm inline-block ${backendConnected
              ? 'bg-green-800 text-green-200'
              : 'bg-red-800 text-red-200'
              }`}>
              Backend: {backendConnected ? 'Connected' : 'Disconnected'}
            </div>
          </header>

          {/* Device Selection */}
          <div className="mb-4">
            <div className="flex items-center gap-2 mb-2">
              <label htmlFor="micSelect" className="block text-sm font-medium text-gray-300">
                Select Audio Input Device:
              </label>
              <button
                onClick={refreshDevices}
                disabled={isRecording || isProcessing || !backendConnected}
                className="p-1 rounded hover:bg-gray-600 disabled:opacity-50"
                title="Refresh device list"
              >
                <RefreshIcon />
              </button>
            </div>
            <select
              id="micSelect"
              value={selectedAudioInputDevice || ''}
              onChange={(e) => setSelectedAudioInputDevice(Number(e.target.value))}
              disabled={isRecording || isProcessing || !backendConnected}
              className="w-full p-2.5 rounded-md bg-gray-600 border border-gray-500 text-gray-100 focus:ring-blue-500 focus:border-blue-500 disabled:opacity-50"
            >
              {audioInputDevices.length === 0 && (
                <option value="">No audio devices found</option>
              )}
              {audioInputDevices.map(device => (
                <option key={device.index} value={device.index}>
                  {device.name} ({device.channels} channels)
                </option>
              ))}
            </select>
          </div>

          {/* Recording Type Selection */}
          <div className="mb-6 flex justify-center space-x-4">
            <button
              onClick={() => setRecordingType(RECORDING_TYPE.MIC_ONLY)}
              disabled={isRecording || isProcessing || !backendConnected}
              className={`flex items-center justify-center px-4 py-2 rounded-md transition-colors ${recordingType === RECORDING_TYPE.MIC_ONLY
                ? 'bg-blue-600 text-white'
                : 'bg-gray-600 hover:bg-blue-500'
                } ${(isRecording || isProcessing || !backendConnected) ? 'opacity-50 cursor-not-allowed' : ''}`}
            >
              <MicIcon /> Mic Only
            </button>
            <button
              onClick={() => setRecordingType(RECORDING_TYPE.SCREEN_PLUS_MIC)}
              disabled={isRecording || isProcessing || !backendConnected}
              className={`flex items-center justify-center px-4 py-2 rounded-md transition-colors ${recordingType === RECORDING_TYPE.SCREEN_PLUS_MIC
                ? 'bg-blue-600 text-white'
                : 'bg-gray-600 hover:bg-blue-500'
                } ${(isRecording || isProcessing || !backendConnected) ? 'opacity-50 cursor-not-allowed' : ''}`}
            >
              <ScreenIcon /> Screen + Mic
            </button>
          </div>

          {/* Recording Controls */}
          <div className="flex flex-col sm:flex-row justify-center items-center space-y-3 sm:space-y-0 sm:space-x-3 mb-6">
            {!isRecording ? (
              <button
                onClick={handleStartRecording}
                disabled={isProcessing || !backendConnected || audioInputDevices.length === 0}
                className="btn-primary w-full sm:w-auto px-6 py-3 text-lg disabled:opacity-50"
                title={
                  !backendConnected
                    ? "Backend not connected"
                    : audioInputDevices.length === 0
                      ? "No audio devices available"
                      : ""
                }
              >
                Start Recording
              </button>
            ) : (
              <button
                onClick={handleStopRecording}
                className="btn-danger w-full sm:w-auto px-6 py-3 text-lg"
              >
                Stop Recording
              </button>
            )}
          </div>

          {/* Status Message */}
          {statusMessage.text && (
            <div className={`p-3 mb-4 rounded-md text-center font-medium ${statusMessage.type === 'error' ? 'bg-red-700' :
              statusMessage.type === 'success' ? 'bg-green-700' :
                statusMessage.type === 'warning' ? 'bg-yellow-700' : 'bg-blue-800'
              }`}>
              {statusMessage.text}
              {(isLoading || isProcessing) && (
                <div className="inline-block ml-2 w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
              )}
            </div>
          )}

          {/* Transcription Output */}
          <div className="mb-6">
            <h2 className="text-xl font-semibold mb-2">Transcription</h2>
            <div className="output-area bg-gray-800 border border-gray-600 p-3 rounded-md min-h-[120px]">
              {currentTranscription || "Transcription will appear here after processing..."}
            </div>
          </div>

          {/* Summary Output */}
          <div>
            <h2 className="text-xl font-semibold mb-2">AI Summary</h2>
            <div className="output-area bg-gray-800 border border-gray-600 p-3 rounded-md min-h-[120px]">
              {currentSummary || "AI summary will appear here after processing..."}
            </div>
          </div>

          {/* Footer */}
          <footer className="mt-8 text-center text-sm text-gray-500">
            <p>Audio recording & processing handled by backend server.</p>
            <p>Ensure backend is running on port 8462 with proper permissions.</p>
          </footer>
        </div>
      </main>

      {/* Project Tag Modal */}
      {showProjectTagModal && (
        <div className="fixed inset-0 bg-black bg-opacity-75 flex items-center justify-center z-50">
          <div className="bg-gray-700 p-6 rounded-lg shadow-xl w-full max-w-md">
            <h3 className="text-xl font-semibold mb-4">Process Recording</h3>
            <label htmlFor="projectTagInputModal" className="block mb-1 text-sm font-medium text-gray-300">
              Project Tag (optional):
            </label>
            <input
              type="text"
              id="projectTagInputModal"
              value={projectTagInput}
              onChange={(e) => setProjectTagInput(e.target.value)}
              placeholder="e.g., Weekly Sync, Client X Meeting"
              className="w-full p-2.5 rounded-md bg-gray-800 border border-gray-600 text-gray-100 focus:ring-blue-500 focus:border-blue-500 mb-4"
              onKeyPress={(e) => {
                if (e.key === 'Enter') {
                  processRecordingWithBackend(projectTagInput);
                }
              }}
            />
            <div className="flex justify-end space-x-3">
              <button
                onClick={() => {
                  setShowProjectTagModal(false);
                  setProjectTagInput('');
                  setCurrentRecordingFilename('');
                  updateStatus('Processing cancelled by user.', 'info');
                }}
                className="btn-secondary px-4 py-2"
              >
                Cancel
              </button>
              <button
                onClick={() => processRecordingWithBackend(projectTagInput)}
                className="btn-primary px-4 py-2 flex items-center"
              >
                <SaveIcon /> Process Recording
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default App;
