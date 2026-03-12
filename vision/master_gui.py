import cv2
import numpy as np
import logging
import os
import tkinter as tk
from tkinter import filedialog
from pathlib import Path

# Import our AI Engine Modules
from vision.detector import PitchDetector
from vision.pitch_calibrator import PitchCalibrator
from vision.visualizer import PitchVisualizer
from analytics.spatial import SpatialAnalyzer
from analytics.xg_model import ExpectedGoalsModel
from analytics.defensive import DefensiveAnalyzer

logger = logging.getLogger(__name__)

class MasterPipelineGUI:
    """
    Unified application handling file picking, video scrubbing, 
    manual calibration, and automated pipeline execution.
    """
    def __init__(self):
        self.output_dir = Path("datasets/extracted_frames")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir = Path("data")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Load AI Models (Initializing them here so they are ready instantly later)
        logger.info("Initializing AI Engine in the background...")
        self.detector = PitchDetector()
        self.calibrator = PitchCalibrator()
        self.visualizer = PitchVisualizer()
        self.spatial = SpatialAnalyzer()
        self.xg_model = ExpectedGoalsModel()
        self.defensive = DefensiveAnalyzer()
        
        self.landmarks = [
            {"name": "Left Goalpost Base", "pos": [105.0, 37.66]},
            {"name": "Right Goalpost Base", "pos": [105.0, 30.34]},
            {"name": "Left Penalty Box Corner", "pos": [88.5, 54.16]},
            {"name": "Right Penalty Box Corner", "pos": [88.5, 13.84]},
            {"name": "Center Spot", "pos": [52.5, 34.0]}
        ]
        
        self.src_pixels = []
        self.dst_meters = []
        self.current_idx = 0
        self.calib_frame_copy = None

    def pick_video(self) -> str:
        """Opens a native OS GUI window to select a video file."""
        root = tk.Tk()
        root.withdraw() # Hide the main empty tkinter window
        file_path = filedialog.askopenfilename(
            title="Select Football Clip",
            filetypes=[("Video Files", "*.mp4 *.mkv *.avi")]
        )
        return file_path

    def scrub_and_select_frame(self, video_path: str) -> np.ndarray:
        """Video player allowing the user to scrub frames to find a clear view."""
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_idx = 0
        
        window_name = "Video Scrubber (Find a clear frame)"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        
        logger.info("Opened Scrubber. Controls: [n] Next, [p] Prev, [c] Select this Frame, [q] Quit")
        
        selected_frame = None
        
        while True:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if not ret:
                break
                
            display_frame = frame.copy()
            
            # Draw solid background HUD for readable text
            h, w = display_frame.shape[:2]
            cv2.rectangle(display_frame, (0, 0), (w, 80), (0, 0, 0), -1)
            
            # Draw Scrubber Instructions
            instructions = f"Frame: {frame_idx}/{total_frames} | [n]=Next [p]=Prev [c]=CALIBRATE THIS FRAME"
            cv2.putText(display_frame, instructions, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 215, 255), 2)
            
            cv2.imshow(window_name, display_frame)
            
            key = cv2.waitKey(0) & 0xFF # Block and wait for user keypress
            if key == ord('n'):
                frame_idx = min(total_frames - 1, frame_idx + 5) # Jump forward 5 frames
            elif key == ord('p'):
                frame_idx = max(0, frame_idx - 5) # Jump backward 5 frames
            elif key == ord('c'):
                selected_frame = frame.copy()
                break
            elif key == ord('q'):
                break
                
        cap.release()
        cv2.destroyAllWindows()
        return selected_frame

    def _mouse_callback(self, event, x, y, flags, param):
        """Captures mouse clicks for calibration."""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.src_pixels.append([x, y])
            self.dst_meters.append(self.landmarks[self.current_idx]["pos"])
            
            cv2.circle(self.calib_frame_copy, (x, y), 6, (0, 255, 0), -1)
            cv2.putText(self.calib_frame_copy, "OK", (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            self.current_idx += 1

    def calibrate_frame(self, frame: np.ndarray, save_name: str) -> np.ndarray:
        """The improved interactive calibration GUI with highly readable text."""
        self.calib_frame_copy = frame.copy()
        self.src_pixels = []
        self.dst_meters = []
        self.current_idx = 0
        
        window_name = "Calibration (Click requested point)"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(window_name, self._mouse_callback)

        while self.current_idx < len(self.landmarks):
            display_frame = self.calib_frame_copy.copy()
            target = self.landmarks[self.current_idx]["name"]
            
            # --- FIXED TEXT RENDERING ---
            # Draw a massive, solid black bar at the top so text is always 100% visible
            h, w = display_frame.shape[:2]
            cv2.rectangle(display_frame, (0, 0), (w, 100), (0, 0, 0), -1)
            
            cv2.putText(display_frame, f"CLICK: {target}", (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 215, 255), 3)
            cv2.putText(display_frame, "Press 's' to Skip if off-screen. Press 'q' to Quit.", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            cv2.imshow(window_name, display_frame)
            
            key = cv2.waitKey(10) & 0xFF
            if key == ord('s'):
                self.current_idx += 1
            elif key == ord('q'):
                break

        cv2.destroyAllWindows()

        if len(self.src_pixels) >= 4:
            src_pts = np.array(self.src_pixels, dtype=np.float32)
            dst_pts = np.array(self.dst_meters, dtype=np.float32)
            H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            
            np.save(self.data_dir / save_name, H)
            logger.info("Homography saved successfully.")
            return H
        else:
            logger.error("Not enough points clicked for calibration.")
            return None

    def execute_analytics_pipeline(self, frame: np.ndarray, H_matrix: np.ndarray, output_filename: str):
        """Runs the fully automated AI pipeline on the calibrated frame."""
        logger.info("--- Executing Automated AI Pipeline ---")
        self.calibrator.H = H_matrix
        
        # 1. Vision
        logger.info("Detecting entities...")
        detections = self.detector.predict(frame)
        players = detections.get('players', [])
        balls = detections.get('ball', [])

        if not balls or not players:
            logger.error("Could not find ball or players in this frame. Cannot compute analytics.")
            return

        ball_bbox = balls[0]['bbox']
        ball_center = ((ball_bbox[0] + ball_bbox[2]) // 2, (ball_bbox[1] + ball_bbox[3]) // 2)

        # 2. Context Logic
        shooter_pixel = None
        min_dist = float('inf')
        defenders_pixels = []

        for p in players:
            p_bbox = p['bbox']
            p_feet = ((p_bbox[0] + p_bbox[2]) // 2, p_bbox[3]) 
            dist = np.linalg.norm(np.array(p_feet) - np.array(ball_center))
            if dist < min_dist:
                if shooter_pixel is not None:
                    defenders_pixels.append(shooter_pixel)
                shooter_pixel = p_feet
                min_dist = dist
            else:
                defenders_pixels.append(p_feet)

        # 3. Analytics
        logger.info("Computing spatial and defensive metrics...")
        shooter_pitch = self.calibrator.pixel_to_pitch(shooter_pixel)
        defenders_pitch = [self.calibrator.pixel_to_pitch(d) for d in defenders_pixels]

        spatial_data = self.spatial.analyze_shot_situation(shooter_pitch)
        defensive_data = self.defensive.calculate_defensive_pressure(shooter_pitch, defenders_pitch)
        
        xg_features = {**spatial_data, **defensive_data}
        xg_value = self.xg_model.predict_xg(xg_features)

        calculated_metrics = {
            "xg": xg_value,
            "distance": spatial_data["distance_meters"],
            "angle": spatial_data["shot_angle_degrees"],
            "block_prob": defensive_data["max_block_probability"]
        }

        # 4. Rendering
        logger.info("Rendering final broadcast graphics...")
        post_left_pixel = (506, 680) # Fallbacks if reverse mapping isn't implemented yet
        post_right_pixel = (1415, 680)
        
        # If we have the original clicked goalposts, use them for the visual cone
        if len(self.src_pixels) >= 2:
             post_left_pixel = tuple(map(int, self.src_pixels[0]))
             post_right_pixel = tuple(map(int, self.src_pixels[1]))

        frame = self.visualizer.draw_shot_cone(frame, shooter_pixel, post_left_pixel, post_right_pixel)
        frame = self.visualizer.draw_shooter_highlight(frame, shooter_pixel)
        frame = self.visualizer.overlay_analytics_hud(frame, calculated_metrics)

        # Save Output
        out_path = str(self.output_dir / output_filename)
        cv2.imwrite(out_path, frame)
        logger.info(f"SUCCESS! Final render saved to: {out_path}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    app = MasterPipelineGUI()
    
    # 1. User picks video via GUI
    video_path = app.pick_video()
    if not video_path:
        logger.warning("No video selected. Exiting.")
        exit(0)
        
    clip_name = os.path.basename(video_path).split('.')[0]
    
    # 2. Scrubber loop to find the best frame
    selected_frame = app.scrub_and_select_frame(video_path)
    if selected_frame is None:
        logger.warning("Frame selection aborted. Exiting.")
        exit(0)
        
    # 3. Calibrate on the selected frame (with fixed text UI)
    matrix_name = f"H_{clip_name}.npy"
    H = app.calibrate_frame(selected_frame, save_name=matrix_name)
    
    # 4. Automate the rest!
    if H is not None:
        app.execute_analytics_pipeline(selected_frame, H, output_filename=f"Render_{clip_name}.jpg")