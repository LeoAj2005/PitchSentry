import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)

class PitchVisualizer:
    """
    Renders tactical graphics and analytics overlays onto broadcast frames.
    """
    
    def __init__(self):
        # Brand colors for the HUD (BGR format for OpenCV)
        self.color_primary = (0, 215, 255)    # Gold/Yellow
        self.color_danger = (0, 0, 255)       # Red
        self.color_safe = (0, 255, 0)         # Green
        self.color_white = (255, 255, 255)
        self.font = cv2.FONT_HERSHEY_SIMPLEX

    def draw_shot_cone(self, frame: np.ndarray, shooter_pt: tuple, post_left: tuple, post_right: tuple, alpha: float = 0.3):
        """
        Draws a semi-transparent shot cone from the shooter to the goal.
        Points must be in (X, Y) pixel coordinates.
        """
        overlay = frame.copy()
        
        # Define the triangle vertices
        pts = np.array([shooter_pt, post_left, post_right], np.int32)
        pts = pts.reshape((-1, 1, 2))
        
        # Fill the polygon with a red tint to indicate danger/shot threat
        cv2.fillPoly(overlay, [pts], self.color_danger)
        
        # Draw the boundary lines
        cv2.polylines(overlay, [pts], isClosed=True, color=self.color_danger, thickness=2, lineType=cv2.LINE_AA)
        
        # Blend the overlay with the original frame
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        return frame

    def draw_shooter_highlight(self, frame: np.ndarray, shooter_pt: tuple, radius: int = 30):
        """
        Draws a tactical ring under the active player.
        """
        cv2.ellipse(
            frame, 
            center=shooter_pt, 
            axes=(radius, int(radius * 0.4)), # Flattened ellipse to look like it's on the ground
            angle=0, startAngle=0, endAngle=360, 
            color=self.color_primary, thickness=2, lineType=cv2.LINE_AA
        )
        return frame

    def overlay_analytics_hud(self, frame: np.ndarray, metrics: dict):
        """
        Draws a sleek, semi-transparent HUD panel with the calculated metrics.
        """
        overlay = frame.copy()
        h, w = frame.shape[:2]
        
        # Panel dimensions (Top Right Corner)
        panel_w, panel_h = 350, 220
        margin = 30
        x1, y1 = w - panel_w - margin, margin
        x2, y2 = w - margin, margin + panel_h
        
        # Draw dark background
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        # Draw Border
        cv2.rectangle(frame, (x1, y1), (x2, y2), self.color_primary, 2, cv2.LINE_AA)
        
        # Render Text
        text_x = x1 + 20
        text_y = y1 + 40
        line_spacing = 35
        
        # Header
        cv2.putText(frame, "PITCH SENTRY ANALYTICS", (text_x, text_y), self.font, 0.6, self.color_primary, 2, cv2.LINE_AA)
        cv2.line(frame, (text_x, text_y + 10), (x2 - 20, text_y + 10), self.color_primary, 1)
        text_y += line_spacing + 10
        
        # Format the dictionary metrics into display strings
        display_lines = [
            f"xG (Goal Prob): {metrics.get('xg', 0.0) * 100:.1f}%",
            f"Distance: {metrics.get('distance', 0.0):.1f}m",
            f"Shot Angle: {metrics.get('angle', 0.0):.1f} deg",
            f"Block Prob: {metrics.get('block_prob', 0.0) * 100:.1f}%"
        ]
        
        for line in display_lines:
            cv2.putText(frame, line, (text_x, text_y), self.font, 0.55, self.color_white, 1, cv2.LINE_AA)
            text_y += line_spacing
            
        return frame

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    import os
    
    visualizer = PitchVisualizer()
    
    # Path to the actual Messi frame you have on your D: drive
    image_path = "datasets/extracted_frames/Messi_Freekick_Liverpool_frame_000108.jpg"
    
    if not os.path.exists(image_path):
        logger.error(f"Could not find {image_path}. Ensure the file exists.")
    else:
        logger.info("Rendering Analytics Overlay...")
        frame = cv2.imread(image_path)
        
        # 1. We use the pixel coordinates we mapped in Session 5
        shooter_pixel = (880, 800) # Messi's feet
        post_left_pixel = (506, 680)
        post_right_pixel = (1415, 680)
        
        # 2. We inject the exact metrics we calculated in Sessions 6, 7, and 9
        calculated_metrics = {
            "xg": 0.0524,
            "distance": 26.52,
            "angle": 15.35,
            "block_prob": 0.775
        }
        
        # 3. Apply the rendering layers
        frame = visualizer.draw_shot_cone(frame, shooter_pixel, post_left_pixel, post_right_pixel, alpha=0.25)
        frame = visualizer.draw_shooter_highlight(frame, shooter_pixel)
        frame = visualizer.overlay_analytics_hud(frame, calculated_metrics)
        
        # 4. Save the output
        output_path = "datasets/extracted_frames/Messi_Analytics_Render.jpg"
        cv2.imwrite(output_path, frame)
        logger.info(f"Successfully rendered broadcast graphics to {output_path}")