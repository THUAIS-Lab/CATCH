# Local demo

Run: python -m v_app

Open http://127.0.0.1:8080/web/index.html.

The demo uses a deterministic local worker, so no model API key is required. The microphone button uses the browser SpeechRecognition API and routes the recognized utterance through the same server execution path. Production voice should use a realtime provider adapter while keeping the persistent cache outside the provider session.
