import cv2
import os
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class VideoProcessor:
    """Extracts frames from broadcast videos at a specified interval."""
    
    def __init__(self, output_dir: str = "datasets/extracted_frames"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def extract_frames(self, video_path: str, frames_per_second: int = 2):
        """
        Slices video into frames.
        Args:
            video_path: Path to the raw .mp4 or .mkv file.
            frames_per_second: How many frames to extract per second of video.
        """
        if not os.path.exists(video_path):
            logger.error(f"Video file not found: {video_path}")
            return

        video_name = Path(video_path).stem
        cap = cv2.VideoCapture(video_path)
        
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if video_fps == 0:
            logger.error("Failed to read video FPS. Video might be corrupted.")
            return

        # Calculate how many frames to skip to hit the target extraction FPS
        frame_interval = int(video_fps / frames_per_second)
        logger.info(f"Processing {video_name}: {video_fps} native FPS, total {total_frames} frames.")
        logger.info(f"Extracting 1 frame every {frame_interval} frames.")

        frame_count = 0
        saved_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % frame_interval == 0:
                output_filename = self.output_dir / f"{video_name}_frame_{frame_count:06d}.jpg"
                cv2.imwrite(str(output_filename), frame)
                saved_count += 1

            frame_count += 1

        cap.release()
        logger.info(f"Extraction complete. Saved {saved_count} frames to {self.output_dir}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Initialize the processor
    vp = VideoProcessor()
    
    # Define the path to your specific local video
    local_video_path = "datasets/raw_videos/Messi_Freekick_Liverpool.mp4"
    
    # Extract 2 frames per second (optimal for preventing identical consecutive frames)
    vp.extract_frames(local_video_path, frames_per_second=2)