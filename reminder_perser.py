### structured_parser.py ###
from langchain.output_parsers import StructuredOutputParser , ResponseSchema
import logging
import json
from conversation_chain import llm

logger = logging.getLogger(__name__)

# 各質問に対応する情報を抽出するためのスキーマ定義
response_schemas = [
    ResponseSchema(
        name="Goodthing_remind", 
        description="今回のレポートの内容で「ポジティブな事柄」を最大100文字程度出力してください"
    ),
    ResponseSchema(
        name="Badthing_remind", 
        description="今回のレポートの内容で「ネガティブな事柄」を最大100文字程度で抽出して出力してください"
    )
]

# StructuredOutputParserの初期化
parser = StructuredOutputParser.from_response_schemas(response_schemas)
output_format = parser.get_format_instructions()

def extract_structured_data_reminder(chat_history: str) -> dict:
    """
    レポートの内容で「ポジティブな事柄」と「ネガティブな事柄」を最大100文字程度で抽出して出力します。
    
    """
    prompt = (
        "以下のレポートから、ユーザーが気にしているであろうことを抽出してください"
        "出力は次のJSON形式に従ってください。\n\n"
        f"{output_format}\n\n"
        f"チャット履歴:\n{chat_history}"
    )

    try:
        # LLMからの出力を取得
        llm_output = llm.invoke(prompt)
        # もし戻り値が AIMessage オブジェクトであれば、.content を取り出す
        if hasattr(llm_output,"content"):
            llm_output=llm_output.content
        logger.debug(f"LLM output:{llm_output}")
        # StructuredOutputParserでパース
        structured_data = parser.parse(llm_output)
    except Exception as e:
        logger.error(f"Error during parsing structured data: {e}")
        structured_data = {}

    return structured_data