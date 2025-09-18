# Modern Real-Time Object Detection 

This repository contains an **enhanced real-time object detection system** built using **YOLOv8**.  
It supports live webcam input, video files, and RTSP streams with options for saving results, showing FPS, and customizing detection thresholds.

---

## ✨ Features
-  Real-time object detection using **YOLOv8**  
-  Optimized for **CPU, CUDA (GPU), and Apple MPS**  
-  Configurable **confidence & IoU thresholds**  
-  Automatic bounding box coloring and labeling  
-  FPS counter with smoothing  
-  Option to save processed video outputs  
-  Supports multiple input sources (webcam, video file, RTSP streams)

---

## 📦 Installation
Clone this repo and install dependencies:

```bash
git clone https://github.com/your-username/modern-object-detection.git
cd modern-object-detection
pip install -r requirements.txt
```

Or manually install:
```bash
pip install ultralytics torch torchvision opencv-python numpy
```

---

## 🔃 Usage

### Run on Webcam
```bash
python object.py --source 0 --model yolov8n.pt --conf 0.5
```

### Run on Video File
```bash
python object.py --source video.mp4 --model yolov8s.pt --conf 0.3
```

### Run on RTSP Stream
```bash
python object.py --source rtsp://camera_ip --model yolov8m.pt
```

---

## ⚙️ Command Line Arguments
| Argument | Description | Default |
|----------|-------------|---------|
| `--source` | Input source (`0` for webcam, video path, RTSP URL) | `0` |
| `--model` | YOLOv8 model (`yolov8n/s/m/l/x.pt`) | `yolov8n.pt` |
| `--conf` | Confidence threshold (0–1) | `0.5` |
| `--iou` | IoU threshold for NMS (0–1) | `0.45` |
| `--device` | Device (`auto/cpu/cuda/mps`) | `auto` |
| `--half` | Use FP16 inference (faster on GPU) | `False` |
| `--save` | Save output video | `False` |
| `--output` | Output video path | `output.mp4` |
| `--show-fps` | Show FPS on screen | `False` |

---

## 📂 Project Structure
```
modern-object-detection/
│── object.py                   # Main script
│── requirements.txt             # Dependencies
│── README.md                    # Project documentation
```

---

## 📝 Example Output
- Real-time bounding boxes with class labels and confidence  
- FPS display (optional)  
- Saved processed video (if `--save` is enabled)  

---

## 🔮 Future Improvements
- 📌 Add multi-camera support  
- 📌 Integration with web dashboard  
- 📌 Support for tracking (e.g., DeepSORT)  

---

## 🤝 Contributions:
- Narendra K Yadav
- Pranav Trivedi
---

## 📜 License
This project is licensed under the MIT License.  
