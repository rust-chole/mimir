Optional face detector model for Pro Edit tracking.

Place face_detection_yunet_2023mar.onnx (OpenCV Zoo, MIT license) in this folder,
or point MIMIR_PRO_EDIT_FACE_MODEL at it. Source:
https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet

Without it, Pro Edit uses the Haar cascades shipped with opencv-python(-headless).
Nothing is downloaded automatically.
