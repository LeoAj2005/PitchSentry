import logging
import numpy as np
import torch
from typing import List, Dict, Any
from ultralytics import YOLO
from config.settings import settings, load_yaml_config

logger = logging.getLogger(__name__)

class PitchDetector:
    """
    Production inference wrapper for the trained YOLO model.
    Decouples the rest of the system from the specific YOLO library.
    """
    
    def __init__(self, weights_path: str = "models/pitch_sentry_v1/weights/best.pt"):
        self.weights_path = weights_path
        
        # Hardware fallback: Use config device if CUDA is available, else force CPU
        self.device = settings.DEVICE if torch.cuda.is_available() else "cpu"
        
        # Correctly load YAML config
        yaml_config = load_yaml_config()
        self.conf_threshold = yaml_config.get('models', {}).get('detection_conf_threshold', 0.5)
        
        try:
            logger.info(f"Loading detection model from {self.weights_path} onto {self.device}")
            self.model = YOLO(self.weights_path)
            # Warmup the model with a dummy frame
            dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
            self.model(dummy_img, device=self.device, verbose=False)
            logger.info("Model loaded and warmed up successfully.")
        except Exception as e:
            logger.error(f"Failed to load model weights at {self.weights_path}: {e}")
            logger.warning("Falling back to pre-trained standard YOLO model for testing.")
            self.model = YOLO("yolo11n.pt") # Fallback to standard COCO model if custom weights missing

    def predict(self, frame: np.ndarray) -> Dict[str, List[Dict[str, Any]]]:
        """
        Runs inference on a single frame and standardizes the output.
        
        Args:
            frame: OpenCV image (NumPy array).
            
        Returns:
            Dictionary containing lists of detections separated by class.
            Format: {'players': [{'bbox': [x1,y1,x2,y2], 'conf': 0.85}], 'ball': [...]}
        """
        # Run inference
        results = self.model(frame, device=self.device, conf=self.conf_threshold, verbose=False)[0]
        
        # Standardized output structure
        detections = {
            'players': [],
            'ball': [],
            'goalposts': []
        }
        
        # COCO mapping fallback (if using un-finetuned model)
        # 0: person, 32: sports ball
        
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().tolist()
            conf = float(box.conf[0].cpu().numpy())
            cls_id = int(box.cls[0].cpu().numpy())
            
            det_data = {
                'bbox': [int(x1), int(y1), int(x2), int(y2)],
                'conf': conf
            }
            
            # Map class IDs to our specific vocabulary
            # Assuming custom training mapping: 0=player, 1=ball, 2=goalpost
            if cls_id == 0:
                detections['players'].append(det_data)
            elif cls_id == 1 or cls_id == 32: # 32 is COCO fallback for ball
                detections['ball'].append(det_data)
            elif cls_id == 2:
                detections['goalposts'].append(det_data)
                
        return detections

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    import cv2
    import os
    
    detector = PitchDetector()
    
    # Path to the specific frame you uploaded
    image_path = "datasets/extracted_frames/Messi_Freekick_Liverpool_frame_000108.jpg"
    
    if os.path.exists(image_path):
        logger.info(f"Testing on actual frame: {image_path}")
        test_frame = cv2.imread(image_path)
        
        # Run inference
        results = detector.predict(test_frame)
        logger.info(f"Found {len(results['players'])} players and {len(results['ball'])} balls.")
        
        # Let's draw the bounding boxes to visualize the output!
        # Green boxes for players
        for p in results['players']:
            x1, y1, x2, y2 = p['bbox']
            cv2.rectangle(test_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
        # Red boxes for the ball
        for b in results['ball']:
            x1, y1, x2, y2 = b['bbox']
            cv2.rectangle(test_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            
        # Save the visualization
        output_path = "datasets/extracted_frames/test_detection_output.jpg"
        cv2.imwrite(output_path, test_frame)
        logger.info(f"Saved visual output to {output_path}. Open it to see the detections!")
    else:
        logger.error(f"Image not found at {image_path}. Check your path!")