# constants.py

MODEL_ID_ANTHROPIC_CLAUDE_3_SONNET = "anthropic.claude-3-sonnet-20240229-v1:0"
MODEL_ID_AMAZON_TITAN_EMBEDDING_TEXT_V2 = "amazon.titan-embed-text-v2:0"
NORMALIZED_DOCUMENT_STRUCTURE_FILE_NAME = "한국잡월드_복리후생_문서구조.md"
NORMALIZED_DOCUMENT_OUTPUT_FILE_NAME_WO_RERANKER = "한국잡월드_복리후생_표준파일_리랭커미사용.md"
NORMALIZED_DOCUMENT_OUTPUT_FILE_NAME_W_RERANKER = "한국잡월드_복리후생_표준파일_리랭커사용.md"

NORMALIZATION_PROMPT_DEFAULT_COMMON_DEPRECATED = """Human: You are a Markdown structure analyzer. Your task is to analyze a set of Markdown files and identify the common structure or patterns present across these files. This includes:
1. Identifying the common heading levels and their hierarchy (e.g., # Heading 1, ## Heading 2, etc.).
2. Detecting common sections or components (e.g., introduction, methodology, results, conclusion).
3. Recognizing common formatting elements like code blocks, blockquotes, lists, tables, etc.
4. Noting any other recurring structural elements or patterns.

These documents are inside XML tags between <doc_id> and </doc_ic> below.
{}

Generate output as MARKDOWN format that captures the common structure and skeleton of Types 1 through 5.
ONLY contain all the HEADING levels that are shown at least once across the documents.

Add 2 "line breaks" per each line to put those HEADING levels into the separate line.

DO NOT include any content paragraph, blockquotes, lists, codes or tables other than HEADING levels.
Retain the original language in Korean, and DO NOT translate the content into English or any other language.
Put the output markdown document between <root><normalized_document> and </normalized_document></root>."""

NORMALIZATION_PROMPT_DEFAULT_COMMON_COMPREHENSIVE = """Human: You are a Markdown structure analyzer. Your task is to analyze a set of Markdown files and identify the most holistic structure or patterns present among these files. This includes:
1. Identifying the heading levels and their hierarchy (e.g., # Heading 1, ## Heading 2, etc.) to include the most comprehensive ones.
2. Detecting sections or components (e.g., introduction, methodology, results, conclusion) in inclusive way.
3. Recognizing the identified formatting elements like code blocks, blockquotes, lists, tables, etc.
4. Noting any other recurring structural elements or patterns.

These documents are inside XML tags between <doc_id> and </doc_ic> below.
{}

Generate output as MARKDOWN format that captures the common structure and skeleton of Types 1 through 5.
ONLY contain all the HEADING levels that are shown at least once across the documents.

Add 2 "line breaks" per each line to put those HEADING levels into the separate line.

DO NOT include any content paragraph, blockquotes, lists, codes or tables other than HEADING levels.
Retain the original language in Korean, and DO NOT translate the content into English or any other language.
Put the output markdown document between <root><normalized_document> and </normalized_document></root>."""

RETRIEVAL_PROMPT_SYSTEM = """You are the master in standardizing and normalizing welfare 뭉 benefits policy document that are scattered throughout a company .
Here is pieces of context, contained in <context> tags.
<context>
{context}
</context>

As output, identify outstanding policy statements relevant with input from the human.
Also, try to INCLUDE all the important details and EXCLUDE minor information that has relatively smaller occurrences.
If you find a number denoting money amount or day that deems tunable, then replace it with a placeholder like ['blank'] with additional remarks in output saying the following:
"(['blank']을 적절한 금액이나 날짜로 대체해 주세요.)"
 
EXCLUDE heading level starting with "#", and INCLUDE only the contents of that heading level inside the output.

When you make output, choose the most comprehensive ones if you find several candidates of the following items: tables, bullet points, numbered lists, and blockquotes.
And RETAIN the original formatting and syntax of the policy points, such as:
1. Tables
2. Bullet points
3. Numbered lists
4. Blockquotes

Return the output as plain text.
If you need to add some comments, add them between <llm_comments> and </llm_comments>.
DO NOT include your comments inside output plain text.

If you don't know the answer, just say that you don't know, don't try to make up an answer.
"""

# Return the output between <root><content_output> and </content_output></root>.
# If you need to add some comments, add them between <comments> and </comments>.
# DO NOT include your comments inside <root><content_output> and </content_output></root>.
#
# If you don't know the answer, just say that you don't know, don't try to make up an answer.
