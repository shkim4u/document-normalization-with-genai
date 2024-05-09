import concurrent.futures
import json
import logging
import os
import re
import xml.etree.ElementTree as ET
from typing import List, Dict

from botocore.exceptions import ClientError
from langchain.text_splitter import MarkdownHeaderTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader

from utils.EmbeddingManager import EmbeddingManager
from utils.RetrievalAgentBedrock import RetrievalAgentBedrock
from utils.constants import NORMALIZED_DOCUMENT_STRUCTURE_FILE_NAME, NORMALIZED_DOCUMENT_OUTPUT_FILE_NAME_WO_RERANKER, \
    NORMALIZED_DOCUMENT_OUTPUT_FILE_NAME_W_RERANKER, NORMALIZATION_PROMPT_DEFAULT_COMMON_DEPRECATED, \
    NORMALIZATION_PROMPT_DEFAULT_COMMON_COMPREHENSIVE

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NormalizationInfo:
    def __init__(self, files):
        self.files = files


def identify_headers(lines: List[str]) -> List[str]:
    headers = []
    # Assume space is allowed after the hash symbol
    re_hashtag_headers = r"^#+\ .*$"
    re_alternative_header_lvl1 = r"^=+ *$"
    re_alternative_header_lvl2 = r"^-+ *$"

    for i, line in enumerate(lines):
        # identify headers by leading hashtags
        if re.search(re_hashtag_headers, line):
            headers.append(line)

        # identify alternative headers
        elif re.search(re_alternative_header_lvl1, line):
            headers.append('# ' + lines[i - 1])  # unified h1 format
        elif re.search(re_alternative_header_lvl2, line):
            headers.append('## ' + lines[i - 1])  # unified h2 format

    return headers


def add_newline_per_heading_level(markdown_text):
    # Add two newlines before each markdown heading level
    modified_text = re.sub(r"(#)([^#]+?)(?=#|$)", r"\1\2\n\n", markdown_text)
    return modified_text.strip()  # strip() is used to remove leading and trailing newlines


def get_files(directory: str):
    # Get all files in the directory.
    files = []
    for root, _, filenames in os.walk(directory):
        # Sort filenames before processing
        filenames.sort()
        for filename in filenames:
            if filename.endswith('.md'):
                files.append(os.path.join(root, filename))
    return files


def as_xml_bundles(files):
    # Enumerate the files to build the document tags
    xml_bundles = []
    for file in files:
        with open(file, 'r') as f:
            content = f.read()
        doc_id = os.path.splitext(os.path.basename(file))[0]
        xml_bundles.append(f"<{doc_id}>{content}</{doc_id}>")
    return xml_bundles


def get_response_completion(response_body):
    completion = response_body.get("completion")
    content = response_body.get("content")

    if completion:
        response_completion = completion
    elif content:
        if isinstance(content, list):
            # Enumerate all the element of content and join its "text" element.
            response_completion = ' '.join([item.get("text") for item in content])
        else:
            response_completion = content
    else:
        response_completion = None

    return response_completion


def validate(text):
    # Trim the leading and trailing white spaces.
    result = text.strip()
    # Trim the preceding text before the first "<" character, eg. "Here is the translation in Korean:"
    result = result[result.find("<"):]
    # Trim after the last ">" character.
    result = result[:result.rfind(">") + 1]
    # Replace new line with space.
    result = result.replace("\n", " ")
    # Replace double white space with single white space.
    result = re.sub(' +', ' ', result)

    return result


def repair_xml_string(s):
    # Use regex to isolate XML part
    xml_part = re.findall('<.*?>', s)

    # Join the XML parts into a single string
    xml_string = ''.join(xml_part)

    while True:
        try:
            # Try to parse the XML string
            ET.fromstring(xml_string)
            # If no error is thrown, the XML is well-formed
            break
        except ET.ParseError as e:
            # If an error is thrown, find the problematic part
            error_position = e.position
            error_char = xml_string[error_position[1]]

            # Replace or remove the problematic part
            if error_char == '<':
                xml_string = xml_string.replace(error_char, '&lt;', 1)
            elif error_char == '>':
                xml_string = xml_string.replace(error_char, '&gt;', 1)
            else:
                xml_string = xml_string[:error_position[1]] + xml_string[error_position[1] + 1:]

    return xml_string


def invoke_model_for_structure_normalization(bedrock_client, prompt, body, normalization_info, **kwargs):
    model_id = kwargs.get('model_id', 'anthropic.claude-3-sonnet-20240229-v1:0')
    accept = 'application/json'
    content_type = 'application/json'
    try:
        response = bedrock_client.invoke_model(body=body, modelId=model_id, accept=accept, contentType=content_type)
        response_body = json.loads(response.get("body").read())
        response_completion = get_response_completion(response_body)

        validated_completion = validate(f"{response_completion}")
        try:
            try:
                root = ET.fromstring(validated_completion)
            except ET.ParseError as e:
                logger.warning(f"Error parsing XML string: {validated_completion} - {e.text}")
                # Try to repair XML string.
                root = ET.fromstring(repair_xml_string(validated_completion))

            normalized_document = root.find('normalized_document')
            normalized_document_text = normalized_document.text
            return {
                "normalized_document_text": normalized_document_text
            }
        except Exception as e:
            logger.error(f"Error parsing XML string: {validated_completion} - {e}")
            return {
                "normalized_document_text": None
            }
    except ClientError as error:
        if error.response['Error']['Code'] == 'AccessDeniedException':
            logger.error(f"\x1b[41m{error.response['Error']['Message']}\
                            \n해당 이슈를 트러블슈팅하기 위해서는 다음 문서를 참고하세요.\
                             \nhttps://docs.aws.amazon.com/IAM/latest/UserGuide/troubleshoot_access-denied.html\
                             \nhttps://docs.aws.amazon.com/bedrock/latest/userguide/security-iam.html\x1b[0m\n")
        else:
            # logger.error(f"Chunk {chunk_index}, Finding {index}: {error}")
            logger.error(f"{error}")
            raise error


class DocumentManager:
    def __init__(self, bedrock_client, directory_path, glob_pattern="./*.md", output_directory="output",
                 return_each_line=False, strip_headers=True, **kwargs):
        log_level = kwargs.get("log_level", "info")
        logger.setLevel(log_level.upper())

        self.bedrock_client = bedrock_client
        self.directory_path = directory_path
        self.glob_pattern = glob_pattern
        self.files = get_files(directory_path)
        self.documents = []
        self.all_sections = []
        self.output_directory = output_directory
        self.normalized_document_structure_file_path = f"{output_directory}/{NORMALIZED_DOCUMENT_STRUCTURE_FILE_NAME}"
        self.normalized_document_output_file_path_wo_reranker = f"{output_directory}/{NORMALIZED_DOCUMENT_OUTPUT_FILE_NAME_WO_RERANKER}"
        self.normalized_document_output_file_path_w_reranker = f"{output_directory}/{NORMALIZED_DOCUMENT_OUTPUT_FILE_NAME_W_RERANKER}"
        self.load_documents()
        self.split_documents(return_each_line, strip_headers)
        self.embedding_manager = EmbeddingManager(self.all_sections, bedrock_client=bedrock_client, **kwargs)
        self.embedding_manager.delete_all_collections()
        self.embedding_manager.create_and_persist_embeddings()
        self.rag_chain = RetrievalAgentBedrock(self.embedding_manager.vectordb, documents_count=len(self.documents),
                                               bedrock_client=bedrock_client, **kwargs)
        self.rag_chain.setup_bot()

    def load_documents(self):
        loader = DirectoryLoader(self.directory_path, glob=self.glob_pattern, show_progress=True, loader_cls=TextLoader)
        # loader = DirectoryLoader(self.directory_path, glob=self.glob_pattern, show_progress=True, loader_cls=UnstructuredMarkdownLoader)
        self.documents = loader.load()

    def split_documents(self, return_each_line=False, strip_headers=True):
        # 분할할 헤더를 지정.
        headers_to_split_on = [("#", "Header 1"), ("##", "Header 2"), ("###", "Header 3"), ("####", "Header 4")]
        text_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on, return_each_line=return_each_line,
                                                   strip_headers=strip_headers)
        for doc in self.documents:
            sections = text_splitter.split_text(doc.page_content)
            # Enrich metadata of all sections with source for future purpose.
            [section.metadata.update({'source': doc.metadata['source']}) for section in sections]
            self.all_sections.extend(sections)
        self.show_all_sections()

    def show_all_sections(self):
        # f-string to fill 100 characters with "#"
        logger.info(f"{'#' * 100}")
        logger.info(f"Found {len(self.all_sections)} sections in the documents.")
        logger.info(f"{'=' * 100}")
        for index, (page_content, metadata, _) in enumerate(self.all_sections):
            logger.info(f"[{index}] {page_content}, {metadata}")
        logger.info(f"{'#' * 100}")

    def get_norm_structure_prompt(self, **kwargs):
        xml_bundles = as_xml_bundles(self.files)
        backslash = "\n"
        prompt_generators = {
            'default-common': lambda: NORMALIZATION_PROMPT_DEFAULT_COMMON_DEPRECATED.format(backslash.join(xml_bundles)),
            'default-comprehensive': lambda: NORMALIZATION_PROMPT_DEFAULT_COMMON_COMPREHENSIVE.format(backslash.join(xml_bundles))
            # Add more entries here for other model_ids.
        }

        # Get the prompt generator for the given model_id, or use the default one if not found.
        prompt_type = kwargs.get('prompt_type', 'default-comprehensive')
        prompt_generator = prompt_generators[prompt_type]

        # Generate the prompt.
        prompt = prompt_generator()
        return prompt

    def construct_norm_structure_invoke_params(self, **kwargs):
        temperature = kwargs.get('temperature', 0.1)
        top_p = kwargs.get('top_p', 0.9)
        top_k = kwargs.get('top_k', 250)
        max_tokens_to_sample = kwargs.get('max_tokens_to_sample', 4096)

        prompt = self.get_norm_structure_prompt(**kwargs)

        body_generators = {
            'anthropic.claude-v2:1': lambda: json.dumps({
                "prompt": f"\n\nHuman: {prompt}\n\nAssistant:",
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "max_tokens_to_sample": max_tokens_to_sample,
                "stop_sequences": ["\n\nHuman:"],
            }, ensure_ascii=False),
            'default': lambda: json.dumps({
                "messages": [{"role": "user", "content": f"{prompt}"}],
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "max_tokens": max_tokens_to_sample,
                "anthropic_version": "bedrock-2023-05-31",
                "stop_sequences": ["\n\nHuman:"],
            }, ensure_ascii=False),
            # Add more entries here for other model_ids.
        }

        body = body_generators.get(kwargs.get('model_id', 'anthropic.claude-3-sonnet-20240229-v1:0'),
                                   body_generators['default'])()

        # Return prompt, body, and TranslationInfo which holds the source text.
        return prompt, body, NormalizationInfo(self.files)

    def generate_norm_structure(self, **kwargs):
        logger.info("Normalizing the structure of the document.")
        logger.info(f"Found {len(self.files)} files in the directory.")
        logger.info(f"Files: {self.files}")

        prompt, body, normalization_info = self.construct_norm_structure_invoke_params(**kwargs)

        result = invoke_model_for_structure_normalization(self.bedrock_client, prompt, body, normalization_info,
                                                          **kwargs)

        # output_directory = kwargs.get('output_directory', 'output')
        dry_run = kwargs.get('dry_run', False)
        if not dry_run:
            # Save the normalized document to a file in the directory of "directory".
            normalized_document_text = result.get("normalized_document_text")
            if normalized_document_text:
                # Ensure the directory exists
                os.makedirs(self.output_directory, exist_ok=True)
                with open(self.normalized_document_structure_file_path, mode='w', encoding='utf-8') as f:
                    f.write(add_newline_per_heading_level(normalized_document_text))
                    logger.info(f"Normalized document is saved to {f.name}")
            else:
                logger.error("Error in normalizing the structure of documents.")

    def identify_norm_structure_headers(self) -> List[str]:
        headers = []

        with open(self.normalized_document_structure_file_path, "r", encoding="utf-8") as f:
            lines = f.read().split("\n")

        # Assume space is allowed after the hash symbol
        re_hashtag_headers = r"^#+\ .*$"
        re_alternative_header_lvl1 = r"^=+ *$"
        re_alternative_header_lvl2 = r"^-+ *$"

        for i, line in enumerate(lines):
            # identify headers by leading hashtags
            if re.search(re_hashtag_headers, line):
                headers.append(line)

            # identify alternative headers
            elif re.search(re_alternative_header_lvl1, line):
                headers.append('# ' + lines[i - 1])  # unified h1 format
            elif re.search(re_alternative_header_lvl2, line):
                headers.append('## ' + lines[i - 1])  # unified h2 format

        return headers

    # doc_manager.generate_norm_document(documents=norm_doc_dict,
    #                                    workers=args.workers,
    #                                    model_id=args.model_id,
    #                                    temperature=args.temperature,
    #                                    top_p=args.top_p,
    #                                    top_k=args.top_k,
    #                                    max_tokens_to_sample=args.max_tokens_to_sample,
    #                                    dry_run=args.dry_run)

    def normalize_header_content(self, index, header_content, **kwargs):
        """
        Normalize the content of specified header.
        :param index:
        :param header_content:
        :param kwargs:
        :return:
        """
        logger.debug(f"Normalizing: {header_content}")
        header = header_content.get("header")
        question = f"복지 정책 항목 {header}에 규정된 내용을 구체적이고 상세하게 알려주세요."
#         question = f"""아래 질의에 해당하는 복지 정책의 내용을 찾아주세요:
# 질의: {header}
# 중요한 내용은 빠짐없이 모두 포함되어야 함 의 문서에서만 등장하는 불필요한 내용은 포함되지 않아야 함 (다수에 많이 등장하는 내용을 기반으로 함)
#
# # 숫자, 날짜, 일수 등을 포함하는 Markdown Table, Code Block, List, Blockquote 등의 구조를 최대한 유지할 수 있는 답변을 찾아주세요."""
        answer = self.rag_chain.ask(header, question)
        logger.debug(answer)

        # Extract LLM comments between <llm_comments> and </llm_comments> tags for later use.
        llm_comments = re.search(r'<llm_comments>(.*?)</llm_comments>', answer, re.DOTALL)

        # Finally, get pure answer without LLM comments.
        answer = re.sub(r'<llm_comments>(.*?)</llm_comments>', '', answer, flags=re.DOTALL)
        header_content['content'] = answer

    def normalize_document(self, **kwargs):
        # # It's time to create and persist embeddings to retrieve the most similar content corresponding to the headers.
        # self.embedding_manager.delete_all_collections()
        # self.embedding_manager.create_and_persist_embeddings()

        document = kwargs.get('document', {})
        workers = kwargs.get('workers', 4)
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            # Execute thread for each header from document dict.
            futures_dict = {
                executor.submit(self.normalize_header_content, index, header_content, **kwargs): (index, header_content) for
                index, header_content in document.items()}
            # Wait for all futures to complete.
            for future in concurrent.futures.as_completed(futures_dict):
                index, task = futures_dict[future]
                try:
                    future.result()
                except Exception as exc:
                    print(f"Exception: {exc}")
                else:
                    pass
        self.save_document(document=document, dry_run=kwargs.get('dry_run', False))
        self.rag_chain.report_token_usage()

    def save_document(self, document: Dict[int, Dict[str, str]], dry_run: bool = False):
        # Check if document is not none or empty.
        if not dry_run:
            if document:
                os.makedirs(self.output_directory, exist_ok=True)
                file_path = self.normalized_document_output_file_path_w_reranker if self.rag_chain.use_reranker else self.normalized_document_output_file_path_wo_reranker
                with open(file_path, mode='w', encoding='utf-8') as f:
                    for i, header_content in document.items():
                        f.write(f"{header_content['header']}\n")
                        f.write(f"{header_content['content']}\n\n")
                    logger.info(f"Normalized document is saved to {f.name}")
            else:
                logger.warning("Document is empty.")

