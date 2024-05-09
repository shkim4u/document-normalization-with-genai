from typing import Optional
from utils import bedrock


class SingletonBedrockClient:
    _instance = None

    def __init__(self,
                 assumed_role: Optional[str] = None,
                 region: Optional[str] = None,
                 runtime: Optional[bool] = True,
                 max_pool_connections: Optional[int] = 50):
        if SingletonBedrockClient._instance is not None:
            raise Exception("This class is a singleton!")
        else:
            SingletonBedrockClient._instance = self._create_bedrock_client(assumed_role, region, runtime,
                                                                           max_pool_connections)

    @staticmethod
    def get_instance():
        if SingletonBedrockClient._instance is None:
            SingletonBedrockClient()
        return SingletonBedrockClient._instance

    @staticmethod
    def _create_bedrock_client(assumed_role: Optional[str] = None,
                               region: Optional[str] = None,
                               runtime: Optional[bool] = True,
                               max_pool_connections: Optional[int] = 50):
        # The original code from get_bedrock_client function goes here
        bedrock_client = bedrock.get_bedrock_client(assumed_role, region, runtime, max_pool_connections)
        return bedrock_client
