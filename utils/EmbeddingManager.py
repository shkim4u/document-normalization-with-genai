import logging

from langchain.embeddings.bedrock import BedrockEmbeddings
from langchain_community.vectorstores import Chroma

from utils.constants import MODEL_ID_AMAZON_TITAN_EMBEDDING_TEXT_V2

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmbeddingManager:
    def __init__(self, all_sections, persist_directory='db', bedrock_client=None, **kwargs):
        log_level = kwargs.get("log_level", "info")
        logger.setLevel(log_level.upper())
        self.all_sections = all_sections
        self.persist_directory = persist_directory
        self.vectordb = None
        self.bedrock_client = bedrock_client

    # Method to create and persist embeddings
    def create_and_persist_embeddings(self):
        logger.info('Using embeddings from %s' % MODEL_ID_AMAZON_TITAN_EMBEDDING_TEXT_V2)
        embedding = BedrockEmbeddings(model_id=MODEL_ID_AMAZON_TITAN_EMBEDDING_TEXT_V2, client=self.bedrock_client)
        # Creating an instance of Chroma with the sections and the embeddings
        logger.info(f"Creating embeddings of {len(self.all_sections)} sections from files in directory {self.persist_directory}")
        logger.info(f"Please be patient. This may take several minutes depending on the number of sections and the size of the documents.")
        self.vectordb = Chroma.from_documents(documents=self.all_sections, embedding=embedding,
                                              persist_directory=self.persist_directory)
        # Persisting the embeddings
        logger.info('Persisting the embeddings')
        self.vectordb.persist()

    def load_embeddings(self):
        # Loading the persisted embeddings into the vectordb
        self.vectordb = Chroma(persist_directory=self.persist_directory, embedding_function=BedrockEmbeddings(model_id=MODEL_ID_AMAZON_TITAN_EMBEDDING_TEXT_V2, client=self.bedrock_client))

    def delete_all_collections(self):
        """
        :return:
        """
        self.load_embeddings()
        if self.vectordb:
            for collection in self.vectordb._client.list_collections():
                ids = collection.get()['ids']
                logger.info('Deleting %s document(s) from %s collection' % (str(len(ids)), collection.name))
                if len(ids): collection.delete(ids)
