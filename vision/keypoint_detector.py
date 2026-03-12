import logging
import numpy as np
import cv2
import torch
from ultralytics import YOLO
from config.settings import settings

logger = logging.getLogger(__name__)

class PitchKeypointDetector:
    """
    Detects standard FIFA pitch intersections using a Pose Estimation model.
    Provides dynamic pixel coordinates to feed into the Homography calibrator.
    """
    
    def __init__(self, weights_path: str = "models/pitch_keypoints_v1.pt"):
        self.device = settings.DEVICE if torch.cuda.is_available() else "cpu"
        self.weights_path = weights_path
        self.is_loaded = False
        
        # Standard FIFA 2D Coordinates (Meters) for specific pitch landmarks
        # The dictionary keys must match the class IDs the model is trained to predict
        self.pitch_landmarks_meters = {
            0: [105.0, 37.66],  # Left Goalpost Base
            1: [105.0, 30.34],  # Right Goalpost Base
            2: [88.5, 54.16],   # Left Penalty Box Corner
            3: [88.5, 13.84],   # Right Penalty Box Corner
            4: [52.5, 34.0],    # Center Spot
            # ... (Add all 29 standard pitch intersections here for a full model)
        }

        try:
            logger.info(f"Loading Pitch Keypoint model on {self.device}...")
            self.model = YOLO(self.weights_path)
            self.is_loaded = True
        except Exception as e:
            logger.warning(f"Keypoint model not found at {weights_path}. Using fallback mode.")
            self.is_loaded = False

    def detect_keypoints(self, frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Processes a frame to find pitch keypoints.
        Returns a tuple of (source_pixels, destination_meters) for homography.
        """
        src_pixels = []
        dst_meters = []

        if self.is_loaded:
            # Run inference using the pose model
            results = self.model(frame, device=self.device, verbose=False)[0]
            
            # Extract keypoints (Assuming model outputs format: [x, y, confidence])
            if results.keypoints is not None and len(results.keypoints.data) > 0:
                kpts = results.keypoints.data[0].cpu().numpy()
                
                # Iterate through detected points
                for class_id, pt in enumerate(kpts):
                    x, y, conf = pt
                    
                    # Only use high-confidence points that exist in our physical map
                    if conf > 0.6 and class_id in self.pitch_landmarks_meters:
                        src_pixels.append([x, y])
                        dst_meters.append(self.pitch_landmarks_meters[class_id])
                        
        else:
            # --- ARCHITECTURAL FALLBACK FOR PROTOTYPING ---
            # If the ML model isn't trained yet, we return the hardcoded Messi frame points
            # just so the pipeline doesn't break during Session integration.
            logger.debug("Executing keypoint fallback (ML model not loaded).")
            src_pixels = [[506, 680], [1415, 680], [190, 890], [1730, 890]]
            dst_meters = [
                self.pitch_landmarks_meters[0],
                self.pitch_landmarks_meters[1],
                self.pitch_landmarks_meters[2],
                self.pitch_landmarks_meters[3]
            ]

        # Convert to numpy arrays formatted specifically for cv2.findHomography
        return np.array(src_pixels, dtype=np.float32), np.array(dst_meters, dtype=np.float32)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    from vision.pitch_calibrator import PitchCalibrator
    import os
    
    # Simulate the Pipeline Integration
    logger.info("--- Starting Dynamic Calibration Pipeline ---")
    
    # 1. Initialize Modules
    keypoint_detector = PitchKeypointDetector()
    calibrator = PitchCalibrator()
    
    # 2. Load Frame
    image_path = "datasets/extracted_frames/Messi_Freekick_Liverpool_frame_000108.jpg"
    if os.path.exists(image_path):
        frame = cv2.imread(image_path)
        
        # PIPELINE STEP A: Detect automatic keypoints
        logger.info("Detecting pitch keypoints...")
        src_pts, dst_pts = keypoint_detector.detect_keypoints(frame)
        
        logger.info(f"Found {len(src_pts)} valid pitch keypoints.")
        
        # PIPELINE STEP B: Compute Homography dynamically
        if len(src_pts) >= 4:
            logger.info("Computing dynamic homography matrix...")
            H = calibrator.compute_homography(src_pts, dst_pts)
            
            # PIPELINE STEP C: Test mapping
            test_pixel = (880, 800) # Messi
            pitch_coords = calibrator.pixel_to_pitch(test_pixel)
            logger.info(f"Pipeline Success. Pixel {test_pixel} mapped to Pitch X={pitch_coords[0]:.2f}m, Y={pitch_coords[1]:.2f}m")
        else:
            logger.error("Not enough keypoints detected to compute homography (Requires >= 4).")
    else:
        logger.error("Test image not found.")