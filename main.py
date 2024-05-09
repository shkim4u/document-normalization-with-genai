import argparse
import logging
import os

from dotenv import load_dotenv

from utils.DocumentManager import DocumentManager
from utils.SingletonBedrockClient import SingletonBedrockClient

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """
    Main function to normalize documents for welfare benefits policy of Korea Job World.
    TODO: 프로그램의 매개변수가 많아질 경우 "dotenv" 사용을 고려할 것
    :return:
    """
    parser = argparse.ArgumentParser(
        description="Normalize documents for welfare benefits policy of Korea Job World.")

    parser.add_argument("-d", "--directory", default="files",
                        type=str, help="The directory where policy documents are stored.", required=False)
    parser.add_argument("-o", "--output-directory", default="output",
                        type=str, help="The directory where the normalized documents will be stored.", required=False)
    parser.add_argument("-l", "--log-level", default='info',
                        help="Logging level. Example --log-level debug. Default=info")
    parser.add_argument("-w", "--workers", type=int, default=4,
                        help='Number of workers to parallelize the normalization task against headers. Default=4')
    parser.add_argument("-r", "--bedrock-region", default=os.environ.get("BEDROCK_REGION", "us-west-2"),
                        help='Bedrock region. You must request and get granted to the specified model in this region. Default: us-west-2')
    parser.add_argument("-rarn", "--assume-role-arn", default=os.environ.get("BEDROCK_ASSUME_ROLE", None),
                        help='ARN of an AWS IAM role to assume for calling the Bedrock service. If not specified, the current active credentials will be used.')
    parser.add_argument("-m", "--model-id", default='anthropic.claude-3-sonnet-20240229-v1:0',
                        help='Model ID. Default: "anthropic.claude-3-sonnet-20240229-v1:0"')
    parser.add_argument("-t", "--temperature", type=float, default=0.1, help='Temperature. Default: 0.1')
    parser.add_argument("-tp", "--top-p", type=float, default=0.9, help='Top-p. Default: 0.9')
    parser.add_argument("-tk", "--top-k", type=int, default=250, help='Top-k. Default: 250')
    parser.add_argument("-mt", "--max-tokens-to-sample", type=int, default=4096,
                        help='Max tokens to sample. Default: 4096')
    parser.add_argument("-mp", "--max-pool-connections", type=int, default=60,
                        help='Botocore max connection pool. Default: 60')
    parser.add_argument("-re", "--rag-k-exclusion-percent", type=float, default=0.25, help='RAG k exclusion percent. Default: 0.25')
    parser.add_argument("-ur", "--use-reranker", action='store_true', help='Use reranker')
    parser.add_argument("-dr", "--dry-run", action='store_true', help="Dry run. Do not actually annotate findings.")

    args = parser.parse_args()

    # Set logging level.
    logger.setLevel(args.log_level.upper())

    assume_role_arn = args.assume_role_arn
    bedrock_region = args.bedrock_region
    max_pool_connections = args.max_pool_connections

    # Get Bedrock client with singleton.
    bedrock_client = SingletonBedrockClient(assumed_role=assume_role_arn, region=bedrock_region, runtime=True, max_pool_connections=max_pool_connections).get_instance()

    # Instantiate document manager
    # - Loads, splits, and stores embeddings of the documents to the vector database (ChromaDB).
    # - Sets up the retriever and the question-answering chain.
    doc_manager = DocumentManager(bedrock_client, args.directory, return_each_line=False, strip_headers=False, log_level=args.log_level.upper(),
                                  rag_k_exclusion_percent=args.rag_k_exclusion_percent,
                                  use_reranker=args.use_reranker)

    # Generate normalized document structure.
    doc_manager.generate_norm_structure(model_id=args.model_id,
                                        # output_directory=args.output_directory,
                                        temperature=args.temperature,
                                        top_p=args.top_p,
                                        top_k=args.top_k,
                                        max_tokens_to_sample=args.max_tokens_to_sample,
                                        dry_run=args.dry_run)
    # Identify headers from the normalized document structure.
    headers = doc_manager.identify_norm_structure_headers()

    # Normalize documents over the headers.
    doc_manager.normalize_document(document={i: {"header": header, "content": ""} for i, header in enumerate(headers)},
                                   workers=args.workers,
                                   model_id=args.model_id,
                                   temperature=args.temperature,
                                   top_p=args.top_p,
                                   top_k=args.top_k,
                                   max_tokens_to_sample=args.max_tokens_to_sample,
                                   dry_run=args.dry_run)


if __name__ == '__main__':
    main()
