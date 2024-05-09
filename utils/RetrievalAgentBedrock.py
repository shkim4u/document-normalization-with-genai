import logging
import math

import pandas as pd
from flashrank import Ranker
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import FlashrankRerank
from langchain_aws import ChatBedrock
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, ConfigurableField

from utils.TokenUsageCallbackHandler import TokenUsageCallbackHandler
from utils.constants import MODEL_ID_ANTHROPIC_CLAUDE_3_SONNET, RETRIEVAL_PROMPT_SYSTEM

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# pd.set_option('display.max_colwidth', 30)


def pretty_print(df):
    # return display(HTML(df.to_html().replace("\\n", "<br>")))
    # pd.set_option('display.max_colwidth', 30)
    # Refer to: https://pandas.pydata.org/docs/user_guide/style.html
    df.style.set_table_styles([{'selector' : '', 'props' : [('border', '1.3px solid black')]}])
    df.style.set_table_styles({
        'Score': [{'selector': '', 'props': [('color', 'blue')]}],
        'Content': [{'selector': '', 'props': [('width', '400px')]}],
        'Source': [{'selector': '', 'props': [('width', '150px')]}],
        'Metadata': [{'selector': '', 'props': [('width', '300px')]}],
    }, overwrite=False)
    print(df)


def visualize_retrieved_results(results_with_scores):
    result_dicts = []
    for doc, score in results_with_scores:
        result_dict = {"Score": score, "Content": doc.page_content, "Source": doc.metadata["source"], "Metadata": doc.metadata}
        result_dicts.append(result_dict)
    pretty_print(pd.DataFrame(result_dicts))


class RetrievalAgentBedrock():
    """
    This class is a wrapper around the ChatBedrock model that is used for retrieval augmented tasks.
    Refer to:
    - https://python.langchain.com/docs/use_cases/question_answering/chat_history/
    - https://api.python.langchain.com/en/latest/vectorstores/langchain_community.vectorstores.chroma.Chroma.html
    """

    def __init__(self, vectordb, documents_count, model_id=MODEL_ID_ANTHROPIC_CLAUDE_3_SONNET, bedrock_client=None,
                 **kwargs):
        log_level = kwargs.get("log_level", "info")
        logger.setLevel(log_level.upper())

        self.rag_chain = None
        self.vectordb = vectordb
        self.documents_count = documents_count
        self.rag_k_exclusion_percent = kwargs.get("rag_k_exclusion_percent", 0.25)
        self._use_reranker = kwargs.get("use_reranker", True)
        self._log_similarity_search_results = kwargs.get("log_similarity_search_results", True)
        # Claude 3 by default.
        model_kwargs = {
            "temperature": kwargs.get("temperature", 0.1),
            "top_p": kwargs.get("top_p", 0.9),
            "top_k": kwargs.get("top_k", 250),
            "max_tokens": kwargs.get("max_tokens_to_sample", 4096),
            "anthropic_version": "bedrock-2023-05-31",
            "stop_sequences": ["\n\nHuman:"],
        }
        self.llm = ChatBedrock(model_id=model_id, client=bedrock_client, model_kwargs=model_kwargs)
        # self.callback_handler = StdOutCallbackHandler()

        # self.callback_handler = BedrockAnthropicTokenUsageCallbackHandler()
        self.callback_handler = TokenUsageCallbackHandler(**kwargs)
        # self.callback_handler = TokenCounterHandler()

    def setup_bot(self):
        # Calculating k based on the number of documents and the exclusion percent, ceiling it to an integer.
        k = math.ceil(self.documents_count * (1 - self.rag_k_exclusion_percent))
        logger.info(
            f"Setting k to {k} based on the number of documents {self.documents_count} and the exclusion percent {self.rag_k_exclusion_percent}.")
        # Create a retriever from the vector database.
        vectordb_retriever = self.vectordb.as_retriever(search_kwargs={"k": k})
        # Add a configurable field to the retriever for dynamic metadata search on the fly.
        retriever = vectordb_retriever.configurable_fields(
            search_kwargs=ConfigurableField(
                id="search_kwargs",
                name="Search Kwargs",
                description="The search kwargs to use",
            )
        )
        # TODO: Experiment with more options.
        # retriever = self.vectordb.as_retriever(search_type="mmr", search_kwargs={"k": 2, 'lambda_mult': 0.25})
        if self.use_reranker:
            # https://github.com/langchain-ai/langchain/discussions/20904
            # Initialize the FlashrankRerank with a specific model to support multi-lingual including Korean.
            model_name = "ms-marco-MultiBERT-L-12"
            flashrank_client = Ranker(model_name=model_name)
            compressor = FlashrankRerank(client=flashrank_client, top_n=3, model=model_name)
            retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=vectordb_retriever)

        system = RETRIEVAL_PROMPT_SYSTEM
        human = "{question}"

        prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])
        logger.info(f"Prompt: {prompt}")

        def format_docs(docs):
            logger.debug(f"Retrieved context: {docs}")
            return "\n\n".join(doc.page_content for doc in docs)

        self.rag_chain = (
                {"context": retriever | format_docs, "question": RunnablePassthrough()}
                | prompt
                | self.llm
                | StrOutputParser()
        )

    def ask(self, header, question):
        # Count the number of "#" in the header to determine the depth of the header level.
        header_level = header.count("#")
        # Strip the "#" from the header to get the actual header text.
        stripped_header = header.strip("#").strip()
        logger.debug(f"Stripped header: {stripped_header}, Header level in asking: {header_level}")

        # Passing prefilter dynamically.
        # OpenSearch prefilter to search for the header in the documents.
        # pre_filter = {
        #     "match": {
        #         f"Header {header_level}": stripped_header
        #     }
        # }

        # ChromaDB.
        # dynamic_filter = {
        #     '$or': [{f"Header {header_level}": {'$eq': stripped_header}}]
        # }
        dynamic_filter = {f"Header {header_level}": {'$eq': stripped_header}}

        if self._log_similarity_search_results:
            results_with_scores = self.vectordb.similarity_search_with_score(question, filter=dynamic_filter)
            logger.info(f"Visualizing retrieved results:")
            logger.info(f"(Note) Lower scores are better, representing a shorter distance between vectors.")
            visualize_retrieved_results(results_with_scores)

            logger.info(f"\n{'>' * 100}")
            logger.info(f"Query: {question}")
            logger.info(f"Retrieved {len(results_with_scores)} documents with scores.")
            logger.info(f"\n{'=' * 100}")
            for doc, score in results_with_scores:
                logger.info(f"\nRetrieved Content: {doc.page_content}\nMetadata: {doc.metadata}, Score: {score}\n\n")
                logger.info(f"\n{'-' * 100}")
            logger.info(f"\n{'<' * 100}\n")

        k = math.ceil(self.documents_count * (1 - self.rag_k_exclusion_percent))
        config = {
            'callbacks': [self.callback_handler],
            "configurable": {
                "search_kwargs": {
                    # "k": 5,
                    # "pre_filter": pre_filter,
                    # "search_type": "script_scoring"
                    "filter": dynamic_filter,
                    "k": k
                }
            }
        }

        answer = self.rag_chain.invoke(question, config=config)
        if logger.level == logging.DEBUG:
            print(repr(self.callback_handler))
            # self.callback_handler.report()
        return answer

    @property
    def use_reranker(self):
        return self._use_reranker

    @use_reranker.setter
    def use_reranker(self, value):
        self._use_reranker = value

    def report_token_usage(self):
        self.callback_handler.report()
