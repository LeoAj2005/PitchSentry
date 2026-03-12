import os
import logging
from SoccerNet.Downloader import SoccerNetDownloader
from config.settings import settings

logger = logging.getLogger(__name__)

class FootballDataDownloader:
    """Handles secure downloading of football datasets like SoccerNet."""
    
    def __init__(self, download_dir: str = "datasets/raw_videos"):
        self.download_dir = os.path.abspath(download_dir)
        os.makedirs(self.download_dir, exist_ok=True)
        # Using a dummy password/local config for SoccerNet
        # Note: In a real scenario, you need to sign up for a SoccerNet password
        self.downloader = SoccerNetDownloader(LocalDirectory=self.download_dir)

    def download_sample_match(self):
        """Downloads a small subset (e.g., tracking data/video) for testing."""
        logger.info(f"Initiating download to {self.download_dir}...")
        try:
            # We specifically target 'tracking' or 'action-spotting' to get video clips
            # We download just one match to save local disk space (D: drive)
            self.downloader.downloadDataTask(task="tracking", split=["train"], tiny=True)
            logger.info("Sample dataset downloaded successfully.")
        except Exception as e:
            logger.error(f"Failed to download dataset: {e}")
            logger.warning("Ensure you have network access and valid SoccerNet credentials if required.")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    dl = FootballDataDownloader()
    dl.download_sample_match()