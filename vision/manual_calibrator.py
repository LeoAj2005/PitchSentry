import cv2
import numpy as np
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

class InteractiveCalibrator:
    """
    GUI tool for manually clicking pitch keypoints to compute a highly precise
    Homography matrix.
    """
    
    def __init__(self, output_dir: str = "data"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Standard FIFA 2D Coordinates (Meters)
        # Origin (0,0) is bottom-left of the pitch from a top-down view
        self.landmarks = [
            {"name": "Left Goalpost Base", "pos": [105.0, 37.66]},
            {"name": "Right Goalpost Base", "pos": [105.0, 30.34]},
            {"name": "Left Penalty Box Corner (Top)", "pos": [88.5, 54.16]},
            {"name": "Right Penalty Box Corner (Bottom)", "pos": [88.5, 13.84]},
            {"name": "Left 6-Yard Box Corner", "pos": [99.5, 43.16]},
            {"name": "Right 6-Yard Box Corner", "pos": [99.5, 24.84]},
            {"name": "Penalty Spot", "pos": [94.0, 34.0]},
            {"name": "Center Spot", "pos": [52.5, 34.0]}
        ]
        
        self.src_pixels = []
        self.dst_meters = []
        self.current_idx = 0
        self.frame_copy = None

    def _mouse_callback(self, event, x, y, flags, param):
        """Captures mouse clicks to store pixel coordinates."""
        if event == cv2.EVENT_LBUTTONDOWN:
            logger.info(f"Clicked {self.landmarks[self.current_idx]['name']} at Pixel: ({x}, {y})")
            
            # Store the point
            self.src_pixels.append([x, y])
            self.dst_meters.append(self.landmarks[self.current_idx]["pos"])
            
            # Draw a visual confirmation on the image
            cv2.circle(self.frame_copy, (x, y), 5, (0, 255, 0), -1)
            cv2.putText(self.frame_copy, self.landmarks[self.current_idx]["name"], 
                        (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            self.current_idx += 1

    def run_calibration(self, image_path: str, save_name: str = "homography.npy"):
        """Runs the interactive GUI loop."""
        if not os.path.exists(image_path):
            logger.error(f"Image not found at {image_path}")
            return False

        frame = cv2.imread(image_path)
        self.frame_copy = frame.copy()
        
        window_name = "Pitch Calibration (Press 's' to Skip, 'q' to Quit)"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(window_name, self._mouse_callback)

        logger.info("--- Starting Manual Calibration ---")
        logger.info("Click the requested point. If it's not visible, press 's' to skip.")

        while self.current_idx < len(self.landmarks):
            display_frame = self.frame_copy.copy()
            
            # Instruction HUD
            target = self.landmarks[self.current_idx]["name"]
            cv2.rectangle(display_frame, (0, 0), (700, 60), (0, 0, 0), -1)
            cv2.putText(display_frame, f"CLICK: {target}", (20, 35), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 215, 255), 2)
            cv2.putText(display_frame, "Press 's' to Skip if not visible in this camera angle.", 
                        (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            cv2.imshow(window_name, display_frame)
            
            key = cv2.waitKey(10) & 0xFF
            if key == ord('s'):
                logger.info(f"Skipped {target}")
                self.current_idx += 1
            elif key == ord('q'):
                logger.warning("Calibration aborted by user.")
                break

        cv2.destroyAllWindows()

        # Compute Homography if we have enough points
        if len(self.src_pixels) >= 4:
            src_pts = np.array(self.src_pixels, dtype=np.float32)
            dst_pts = np.array(self.dst_meters, dtype=np.float32)
            
            H, status = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            
            save_path = self.output_dir / save_name
            np.save(save_path, H)
            logger.info(f"SUCCESS! Homography matrix saved to {save_path}")
            logger.info(f"Used {len(src_pts)} points for optimization.")
            return True
        else:
            logger.error(f"Failed. You only clicked {len(self.src_pixels)} points. Minimum 4 required.")
            return False

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    import argparse
    import cv2
    import os
    
    parser = argparse.ArgumentParser(description="Interactive Pitch Calibrator")
    parser.add_argument("--video", type=str, help="Path to the raw video clip (e.g., datasets/raw_videos/yamal.mp4)")
    parser.add_argument("--image", type=str, help="Path to an already extracted frame (optional)")
    parser.add_argument("--out", type=str, required=True, help="Output name for the matrix (e.g., H_yamal.npy)")
    parser.add_argument("--frame", type=int, default=0, help="Which frame number to extract from the video (default is 0, the very first frame)")
    
    args = parser.parse_args()
    
    target_image_path = args.image
    
    # --- AUTOMATIC FRAME EXTRACTION LOGIC ---
    if args.video:
        if not os.path.exists(args.video):
            logger.error(f"Video file not found at {args.video}")
            exit(1)
            
        logger.info(f"Automating frame extraction from {args.video} (Frame {args.frame})...")
        cap = cv2.VideoCapture(args.video)
        
        # Seek to the specific frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            logger.error("Failed to extract frame. The video might be corrupted or the frame index is too high.")
            exit(1)
            
        # Save the automatically extracted frame
        os.makedirs("datasets/extracted_frames", exist_ok=True)
        video_filename = os.path.basename(args.video).split('.')[0]
        target_image_path = f"datasets/extracted_frames/auto_calib_{video_filename}.jpg"
        
        cv2.imwrite(target_image_path, frame)
        logger.info(f"Successfully extracted and saved frame to {target_image_path}")
        
    elif not args.image:
        logger.error("You must provide either a --video or an --image path!")
        exit(1)
        
    # --- RUN THE GUI ---
    calibrator = InteractiveCalibrator()
    calibrator.run_calibration(target_image_path, save_name=args.out)