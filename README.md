# Mudra Mirror

Webcam hand-gesture trainer and mirror game for 10 Bharatanatyam hastas (Pataka, Tripataka, Ardhapataka, Kartarimukha, Mayura, Hamsasya, Suchi, Chandrakala, Mushti, Shikhara).

- Runs fully in the browser. Camera frames are processed on-device by MediaPipe Hand Landmarker (Apache-2.0); nothing is uploaded.
- The app code is in this repo. The MediaPipe runtime (jsdelivr, pinned 0.10.14) and the hand model (Google storage) are fetched once from public CDNs on first use. No paid APIs, no accounts, and camera data is never sent anywhere.
- Needs a secure context for the camera: `https://` or `http://localhost`. Static hosting only (e.g. GitHub Pages).
- Run locally: `python3 -m http.server 8000` then open http://localhost:8000
- Rules live in `gestures.js`; UI and game in `app.js`.

Classification is rule-based on finger bends and touches (`gestures.js`), following common textbook descriptions. It is a prototype: not validated on a range of dancers, skin tones, lighting or camera angles.
