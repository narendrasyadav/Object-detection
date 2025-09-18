#!/usr/bin/env python3
"""
Modern Real-Time Object Detection using YOLOv8
Enhanced version with better performance, accuracy, and features

Usage:
    python modern_object_detection.py --source 0 --model yolov8n.pt --conf 0.5
    python modern_object_detection.py --source video.mp4 --model yolov8s.pt --conf 0.3
    python modern_object_detection.py --source rtsp://camera_ip --model yolov8m.pt
"""

import argparse
import time
import cv2
import numpy as np
from pathlib import Path
import logging
from dataclasses import dataclass
from typing import Optional, Tuple, List
import threading
from queue import Queue

try:
    from ultralytics import YOLO
    import torch
except ImportError as e:
    print(f"Missing required packages: {e}")
    print("Please install: pip install ultralytics torch torchvision")
    exit(1)
 
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class DetectionConfig:
    """Configuration class for detection parameters"""
    model_path: str = "yolov8n.pt"   
    confidence_threshold: float = 0.5
    iou_threshold: float = 0.45
    max_detections: int = 100
    input_size: Tuple[int, int] = (640, 640)
    device: str = "auto"   
    half_precision: bool = True  
    
class FPSCounter:
    """Enhanced FPS counter with smoothing"""
    def __init__(self, buffer_size: int = 30):
        self.buffer_size = buffer_size
        self.frame_times = []
        self.start_time = None
        self.frame_count = 0
        
    def start(self):
        self.start_time = time.time()
        self.frame_count = 0
        self.frame_times = []
        return self
    
    def update(self):
        current_time = time.time()
        self.frame_count += 1
        
        if len(self.frame_times) > 0:
            self.frame_times.append(current_time - self.frame_times[-1])
        else:
            self.frame_times.append(0.033)  # Default to ~30 FPS
        
        if len(self.frame_times) > self.buffer_size:
            self.frame_times.pop(0)
            
        # Store last frame time
        if len(self.frame_times) == 0:
            self.frame_times.append(current_time)
        else:
            self.frame_times[-1] = current_time
    
    def get_fps(self) -> float:
        if len(self.frame_times) < 2:
            return 0.0
        
        # Calculate average FPS over recent frames
        recent_times = [t for t in self.frame_times if t > 0]
        if len(recent_times) == 0:
            return 0.0
            
        avg_frame_time = sum(recent_times) / len(recent_times)
        return 1.0 / avg_frame_time if avg_frame_time > 0 else 0.0
    
    def elapsed(self) -> float:
        if self.start_time is None:
            return 0.0
        return time.time() - self.start_time

class ModernObjectDetector:
    """Modern object detection class using YOLOv8"""
    
    def __init__(self, config: DetectionConfig):
        self.config = config
        self.model = None
        self.class_names = []
        self.colors = []
        self.device = self._get_device()
        self._initialize_model()
        
    def _get_device(self) -> str:
        """Determine the best available device"""
        if self.config.device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return "mps"  # Apple Silicon
            else:
                return "cpu"
        return self.config.device
    
    def _initialize_model(self):
        """Initialize YOLO model"""
        try:
            logger.info(f"Loading model: {self.config.model_path}")
            logger.info(f"Using device: {self.device}")
            
            # Load model
            self.model = YOLO(self.config.model_path)
            
            # Move model to device and optimize
            if self.device != "cpu":
                self.model.to(self.device)
                
            if self.config.half_precision and self.device == "cuda":
                self.model.half()  # Convert to FP16
            
            # Get class names
            self.class_names = list(self.model.names.values())
            
            # Generate colors for each class
            np.random.seed(42)  # For consistent colors
            self.colors = np.random.randint(0, 255, size=(len(self.class_names), 3), dtype=np.uint8)
            
            logger.info(f"Model loaded successfully. Classes: {len(self.class_names)}")
            
        except Exception as e:
            logger.error(f"Failed to initialize model: {e}")
            raise
    
    def detect(self, frame: np.ndarray) -> List[dict]:
        """Perform object detection on frame"""
        try:
            # Run inference
            results = self.model(
                frame,
                conf=self.config.confidence_threshold,
                iou=self.config.iou_threshold,
                max_det=self.config.max_detections,
                verbose=False
            )
            
            detections = []
            
            # Process results
            if results and len(results) > 0:
                result = results[0]  # Get first result
                
                if result.boxes is not None and len(result.boxes) > 0:
                    boxes = result.boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2
                    scores = result.boxes.conf.cpu().numpy()
                    class_ids = result.boxes.cls.cpu().numpy().astype(int)
                    
                    for i, (box, score, class_id) in enumerate(zip(boxes, scores, class_ids)):
                        x1, y1, x2, y2 = map(int, box)
                        class_name = self.class_names[class_id]
                        color = self.colors[class_id].tolist()
                        
                        detections.append({
                            'bbox': (x1, y1, x2, y2),
                            'confidence': float(score),
                            'class_id': int(class_id),
                            'class_name': class_name,
                            'color': color
                        })
            
            return detections
            
        except Exception as e:
            logger.error(f"Detection failed: {e}")
            return []
    
    def draw_detections(self, frame: np.ndarray, detections: List[dict]) -> np.ndarray:
        """Draw bounding boxes and labels on frame"""
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            confidence = detection['confidence']
            class_name = detection['class_name']
            color = detection['color']
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Prepare label
            label = f"{class_name}: {confidence:.2f}"
            
            # Calculate label size
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            
            # Draw label background
            label_y1 = max(y1 - label_h - 10, 0)
            label_y2 = label_y1 + label_h + 10
            cv2.rectangle(frame, (x1, label_y1), (x1 + label_w, label_y2), color, -1)
            
            # Draw label text
            cv2.putText(frame, label, (x1, label_y1 + label_h + 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        return frame

class VideoSource:
    """Enhanced video source handler"""
    
    def __init__(self, source, buffer_size: int = 2):
        self.source = source
        self.buffer_size = buffer_size
        self.cap = None
        self.frame_queue = Queue(maxsize=buffer_size)
        self.thread = None
        self.running = False
        
    def start(self):
        """Start video capture"""
        try:
            # Initialize capture
            if isinstance(self.source, int) or self.source.isdigit():
                self.cap = cv2.VideoCapture(int(self.source))
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer for real-time
            else:
                self.cap = cv2.VideoCapture(self.source)
            
            if not self.cap.isOpened():
                raise RuntimeError(f"Failed to open video source: {self.source}")
            
            # Set optimal capture properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            
            # Start capture thread
            self.running = True
            self.thread = threading.Thread(target=self._capture_frames)
            self.thread.daemon = True
            self.thread.start()
            
            logger.info(f"Video source started: {self.source}")
            
        except Exception as e:
            logger.error(f"Failed to start video source: {e}")
            raise
    
    def _capture_frames(self):
        """Capture frames in separate thread"""
        while self.running and self.cap is not None and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                # Clear queue if full and add new frame
                if self.frame_queue.full():
                    try:
                        self.frame_queue.get_nowait()
                    except:
                        pass
                
                try:
                    self.frame_queue.put_nowait(frame)
                except:
                    pass
            else:
                break
    
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read frame from queue"""
        try:
            if not self.frame_queue.empty():
                frame = self.frame_queue.get_nowait()
                return True, frame
            return False, None
        except:
            return False, None
    
    def stop(self):
        """Stop video capture"""
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=1.0)
        if self.cap is not None:
            self.cap.release()

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Modern Real-Time Object Detection")
    
    parser.add_argument("--source", default="0", 
                       help="Video source (0 for webcam, path to video file, or RTSP URL)")
    parser.add_argument("--model", default="yolov8n.pt",
                       help="YOLO model path (yolov8n/s/m/l/x.pt)")
    parser.add_argument("--conf", type=float, default=0.5,
                       help="Confidence threshold (0-1)")
    parser.add_argument("--iou", type=float, default=0.45,
                       help="IoU threshold for NMS (0-1)")
    parser.add_argument("--device", default="auto",
                       help="Device to use (auto/cpu/cuda/mps)")
    parser.add_argument("--half", action="store_true",
                       help="Use half precision (FP16)")
    parser.add_argument("--save", action="store_true",
                       help="Save output video")
    parser.add_argument("--output", default="output.mp4",
                       help="Output video path")
    parser.add_argument("--show-fps", action="store_true",
                       help="Show FPS on frame")
    
    return parser.parse_args()

def main():
    """Main function"""
    args = parse_arguments()
    
    # Create configuration
    config = DetectionConfig(
        model_path=args.model,
        confidence_threshold=args.conf,
        iou_threshold=args.iou,
        device=args.device,
        half_precision=args.half
    )
    
    # Initialize detector
    try:
        detector = ModernObjectDetector(config)
    except Exception as e:
        logger.error(f"Failed to initialize detector: {e}")
        return
    
    # Initialize video source
    try:
        video_source = VideoSource(args.source)
        video_source.start()
        time.sleep(2)  # Warm up camera
    except Exception as e:
        logger.error(f"Failed to initialize video source: {e}")
        return
    
    # Initialize video writer if saving
    video_writer = None
    if args.save:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(args.output, fourcc, 20.0, (640, 480))
    
    # Initialize FPS counter
    fps_counter = FPSCounter().start()
    
    logger.info("Starting object detection... Press 'q' to quit")
    
    try:
        while True:
            
            ret, frame = video_source.read()
            if not ret or frame is None:
                continue
             
            frame = cv2.resize(frame, (640, 480))
            
            
            detections = detector.detect(frame)
             
            frame = detector.draw_detections(frame, detections)
             
            if args.show_fps:
                fps_text = f"FPS: {fps_counter.get_fps():.1f}"
                cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                           1, (0, 255, 0), 2)
           
            if video_writer is not None:
                video_writer.write(frame)
             
            cv2.imshow("Modern Object Detection", frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
             
            fps_counter.update()
             
            if detections:
                det_info = ", ".join([f"{d['class_name']}({d['confidence']:.2f})" 
                                    for d in detections])
                logger.info(f"Detected: {det_info}")
    
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Error during detection: {e}")
    
    finally:
        # Cleanup
        logger.info("Cleaning up...")
        video_source.stop()
        if video_writer is not None:
            video_writer.release()
        cv2.destroyAllWindows()
        
        # Print final stats
        logger.info(f"Total elapsed time: {fps_counter.elapsed():.2f}s")
        logger.info(f"Average FPS: {fps_counter.get_fps():.2f}")
        logger.info("Detection completed!")

if __name__ == "__main__":
    main()