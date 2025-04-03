### structured_parser.py ###
from langchain.output_parsers import StructuredOutputParser , ResponseSchema
import logging
import json
from conversation_chain import llm

logger = logging.getLogger(__name__)

# 各質問に対応する情報を抽出するためのスキーマ定義
response_schemas = [
    ResponseSchema(
        name="satisfaction_score", 
        description="今日の満足度を10点満点中で数値のみ抽出してください。"
    ),
    ResponseSchema(
        name="satisfaction_context", 
        description="今日の満足度に影響した具体的な状況や背景を説明してください。"
    ),
    ResponseSchema(
        name="positive_events", 
        description="パートナーとの関係で嬉しかった出来事の概要を記述してください。"
    ),
    ResponseSchema(
        name="hidden_thoughts", 
        description="パートナーに伝えにくかった本音や、言えなかったことを記述してください。"
    ),
    ResponseSchema(
        name="next_week_improvement", 
        description="来週の満足度を上げるための改善アイデアを記述してください。"
    ),
    ResponseSchema(
        name="theme", 
        description="最近話すべきだと感じたテーマや気がかりなことを記述してください。"
    )
]

# StructuredOutputParserの初期化
parser = StructuredOutputParser.from_response_schemas(response_schemas)
output_format = parser.get_format_instructions()

def extract_structured_data(chat_history: str) -> dict:
    """
    チャット履歴から各質問に対するユーザーからの回答内容を抽出し、構造化データ（json）として返します。
    
    """
    prompt = (
        "以下のチャット履歴から、各質問に対するユーザーからの回答内容を抽出してください。"
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