import logging
import cv2
import numpy as np
from typing import List, Dict, Any
from ultralytics import YOLO
import torch
from config.settings import settings, load_yaml_config

logger = logging.getLogger(__name__)

class PitchTracker:
    """
    Production wrapper for BoT-SORT tracking.
    Maintains temporal state of players across sequential video frames.
    """
    
    def __init__(self, weights_path: str = "yolo11n.pt"):
        # Hardware fallback logic
        self.device = settings.DEVICE if torch.cuda.is_available() else "cpu"
        
        try:
            logger.info(f"Initializing BoT-SORT Tracker on {self.device}...")
            # We load the model specifically for tracking
            self.model = YOLO(weights_path)
            logger.info("Tracker initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize tracker: {e}")

    def process_video(self, video_path: str, output_path: str):
        """
        Processes a video file, applies BoT-SORT tracking, and saves the output.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Error opening video file: {video_path}")
            return

        # Get video properties for the output writer
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Setup the video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        logger.info(f"Starting tracking on {total_frames} frames. This may take a moment...")

        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Run YOLO tracking using BoT-SORT
            # persist=True is crucial: it tells the model to remember IDs from the previous frame
            results = self.model.track(
                frame, 
                persist=True, 
                tracker="botsort.yaml", 
                device=self.device,
                verbose=False
            )[0]

            # Annotate the frame with bounding boxes and track IDs
            annotated_frame = results.plot()

            out.write(annotated_frame)
            frame_count += 1
            
            if frame_count % 30 == 0:
                logger.info(f"Processed {frame_count}/{total_frames} frames...")

        cap.release()
        out.release()
        logger.info(f"Tracking complete. Output saved to {output_path}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    import os
    
    tracker = PitchTracker()
    
    # Define paths
    input_video = "datasets/raw_videos/Messi_Freekick_Liverpool.mp4"
    output_video = "datasets/raw_videos/Messi_Tracked_Output.mp4"
    
    if os.path.exists(input_video):
        tracker.process_video(input_video, output_video)
    else:
        logger.error(f"Input video not found at {input_video}")