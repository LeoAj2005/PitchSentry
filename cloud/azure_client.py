import logging
from azure.storage.blob import BlobServiceClient
from azure.core.exceptions import AzureError
from config.settings import settings

logger = logging.getLogger(__name__)

class AzureBlobManager:
    """
    Singleton client for interacting with Azure Blob Storage.
    Handles data ingestion (datasets/video) and artifact registry (model weights).
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(AzureBlobManager, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        try:
            self.blob_service_client = BlobServiceClient.from_connection_string(
                settings.AZURE_STORAGE_CONNECTION_STRING
            )
            self.container_name = settings.AZURE_CONTAINER_NAME
            
            # Ensure container exists
            self.container_client = self.blob_service_client.get_container_client(self.container_name)
            if not self.container_client.exists():
                self.container_client.create_container()
                logger.info(f"Created Azure container: {self.container_name}")
            else:
                logger.info(f"Connected to existing Azure container: {self.container_name}")
                
        except AzureError as e:
            logger.error(f"Failed to initialize Azure Blob Storage: {e}")
            self.blob_service_client = None

    def upload_file(self, local_file_path: str, blob_name: str) -> bool:
        """Uploads a local file to Azure Blob Storage."""
        if not self.blob_service_client:
            logger.error("Azure client not initialized.")
            return False
            
        try:
            blob_client = self.blob_service_client.get_blob_client(
                container=self.container_name, blob=blob_name
            )
            with open(local_file_path, "rb") as data:
                blob_client.upload_blob(data, overwrite=True)
            logger.info(f"Successfully uploaded {local_file_path} to {blob_name}")
            return True
        except Exception as e:
            logger.error(f"Upload failed: {e}")
            return False

    def download_file(self, blob_name: str, download_file_path: str) -> bool:
        """Downloads a blob to a local file path."""
        if not self.blob_service_client:
            logger.error("Azure client not initialized.")
            return False
            
        try:
            blob_client = self.blob_service_client.get_blob_client(
                container=self.container_name, blob=blob_name
            )
            with open(download_file_path, "wb") as download_file:
                download_file.write(blob_client.download_blob().readall())
            logger.info(f"Successfully downloaded {blob_name} to {download_file_path}")
            return True
        except Exception as e:
            logger.error(f"Download failed: {e}")
            return False