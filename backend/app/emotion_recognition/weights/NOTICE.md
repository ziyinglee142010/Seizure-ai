# Bundled model weights

Both files come from OpenCV Zoo (https://github.com/opencv/opencv_zoo) and are
bundled so the platform runs without internet access.

| File | Model | Licence | SHA-256 |
|---|---|---|---|
| `face_detection_yunet_2023mar.onnx` | YuNet face detector (box + 5 landmarks) | MIT (`LICENSE-yunet-MIT.txt`) | `8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4` |
| `facial_expression_recognition_mobilefacenet_2022july.onnx` | Progressive Teacher, MobileFaceNet backbone; 7 expressions; 88.27% accuracy on RAF-DB (as reported by OpenCV Zoo) | Apache 2.0 (`LICENSE-fer-Apache-2.0.txt`) | `4f61307602fc089ce20488a31d4e4614e3c9753a7d6c41578c854858b183e1a9` |

Reference for the expression model: Jiang & Deng, "Boosting Facial Expression
Recognition by A Semi-Supervised Progressive Teacher", IEEE Transactions on
Affective Computing, https://ieeexplore.ieee.org/abstract/document/9629313
