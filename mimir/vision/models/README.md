# Face model

`face_detection_yunet_2023mar.onnx` is the YuNet face detector from
[opencv_zoo](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet)
(MIT License, see `LICENSE.yunet`), run through OpenCV's `cv2.FaceDetectorYN`.

* SHA-256 `8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4` (verified on load)
* 232,589 bytes; no extra runtime dependency beyond `opencv-python-headless`

It is MIMIR's production face detector. On a ground-truth benchmark at the production
analysis size (1280x720; small corner facecams over gameplay, several people, turned,
tilted, edge-cut and partly covered faces, plus face-free gameplay/UI frames) it found
94.7% of faces with no false positives at ~21 ms/frame, against 44.2% recall, 39 false
positives and ~365 ms/frame for the OpenCV Haar cascades.

If this file is missing or fails its checksum, MIMIR falls back to the Haar cascades and
records `face_detector_fallback` in the vision stage ledger (`python -m mimir doctor`
shows the active detector). `MIMIR_FACE_MODEL=/path/to/model.onnx` overrides the file.
