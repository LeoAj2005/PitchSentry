import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)

class PitchCalibrator:
    """
    Handles camera calibration and perspective transformation.
    Maps 2D broadcast pixels to a 2D top-down tactical pitch in meters.
    """
    
    def __init__(self):
        # Standard FIFA Pitch Dimensions (in meters)
        self.pitch_length = 105.0
        self.pitch_width = 68.0
        self.H = None
        
    def compute_homography(self, src_pts: np.ndarray, dst_pts: np.ndarray) -> np.ndarray:
        """
        Computes the Homography matrix given pixel points and true pitch points.
        Requires at least 4 points.
        """
        if len(src_pts) < 4 or len(dst_pts) < 4:
            logger.error("Homography requires at least 4 point pairs.")
            return None
            
        # RANSAC makes the computation robust against slight point inaccuracies
        H, status = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        self.H = H
        logger.info("Homography matrix computed successfully.")
        return self.H

    def pixel_to_pitch(self, pixel_point: tuple) -> tuple:
        """
        Transforms a (x, y) pixel coordinate into (x, y) pitch coordinates in meters.
        """
        if self.H is None:
            logger.error("Homography matrix not computed. Call compute_homography first.")
            return (0.0, 0.0)
            
        # Format point for cv2.perspectiveTransform: requires shape (1, 1, 2)
        pt = np.array([[[float(pixel_point[0]), float(pixel_point[1])]]], dtype=np.float32)
        
        # Apply the matrix multiplication H * pt
        transformed_pt = cv2.perspectiveTransform(pt, self.H)
        
        pitch_x = transformed_pt[0][0][0]
        pitch_y = transformed_pt[0][0][1]
        
        return (pitch_x, pitch_y)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    calibrator = PitchCalibrator()
    
    # ---------------------------------------------------------
    # CALIBRATION FOR MESSI FREEKICK (FRAME 108)
    # ---------------------------------------------------------
    
    # 1. Source Points (Pixels from the image you uploaded)
    # We select 4 clear anchor points on the pitch geometry.
    # [Left Goalpost Base, Right Goalpost Base, Left Penalty Box Corner, Right Penalty Box Corner]
    # Note: These are approximated from standard 1080p resolution for this broadcast angle.
    src_pixels = np.array([
        [506, 680],   # Left goalpost base
        [1415, 680],  # Right goalpost base
        [190, 890],   # Left penalty box line (approx bottom left)
        [1730, 890]   # Right penalty box line (approx bottom right)
    ], dtype=np.float32)
    
    # 2. Destination Points (Standard FIFA Dimensions in meters)
    # Origin (0,0) is the bottom-left corner of the full pitch.
    # The goal is exactly in the middle of the 68m width (from y=30.34 to y=37.66)
    dst_meters = np.array([
        [105.0, 37.66],  # Left goalpost base (Attacking end)
        [105.0, 30.34],  # Right goalpost base (Attacking end)
        [88.5, 54.16],   # Left penalty box corner
        [88.5, 13.84]    # Right penalty box corner
    ], dtype=np.float32)
    
    # Compute the matrix
    calibrator.compute_homography(src_pixels, dst_meters)
    
    # 3. Let's test it! Where is Messi?
    # Based on your tracking image, Messi (ID 10) is roughly at pixel (880, 800) at his feet.
    messi_pixel = (880, 800)
    messi_pitch_coords = calibrator.pixel_to_pitch(messi_pixel)
    
    # Calculate distance to the center of the goal (105.0, 34.0)
    goal_center = (105.0, 34.0)
    distance = np.sqrt((messi_pitch_coords[0] - goal_center[0])**2 + (messi_pitch_coords[1] - goal_center[1])**2)
    
    logger.info(f"Messi's pixel coordinates: {messi_pixel}")
    logger.info(f"Messi's actual pitch coordinates: X={messi_pitch_coords[0]:.2f}m, Y={messi_pitch_coords[1]:.2f}m")
    logger.info(f"Calculated Shot Distance: {distance:.2f} meters")