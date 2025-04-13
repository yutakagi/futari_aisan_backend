## emotion_analysis.py
import logging
from google.cloud import language_v1
from langchain_core.output_parsers import JsonOutputParser
from langchain.prompts import PromptTemplate
from conversation_chain import llm

logging.basicConfig(level=logging.INFO)

# 1. ユーザー発言のみ抽出（JsonOutputParser使用）
def extract_partner_mentions_llm(chat_history: str, partner_name: str = None) -> list:
    parser = JsonOutputParser()
    if partner_name:
        instruction = (
            f"以下の会話履歴から、ユーザーによるパートナー（{partner_name}）に関する発言のみを、"
            "原文そのままで抽出してください。"
        )
    else:
        instruction = (
            "以下の会話履歴から、ユーザーがパートナーに言及した発言だけを、"
            "原文のままで抽出してください。"
        )
    prompt_template = PromptTemplate.from_template(
        instruction +
        "\n抽出対象は**ユーザーの発言のみ**で、コーチの発言は含めないでください。"
        "\n出力は**文字列のJSONリスト**として、余計な説明や装飾を含めず返してください。"
        "\n{format_instructions}\n\n会話履歴:\n{chat_history}"
    )
    prompt = prompt_template.format(
        chat_history=chat_history,
        format_instructions=parser.get_format_instructions()
    )
    logging.info("パートナーへの言及部分をLLMで抽出中")
    try:
        result = llm.invoke(prompt)
        if hasattr(result, "content"):
            result = result.content
        logging.info(f"LLM raw output:\n{result}")
        mentions = parser.parse(result)
        if not isinstance(mentions, list):
            raise ValueError("抽出結果がリスト形式ではありません")
        return mentions
    except Exception as e:
        logging.error("LLM抽出中にエラーが発生しました: " + str(e))
        return []

# 2. 感情分析（Google NLP）
def analyze_sentiment(text: str):
    client = language_v1.LanguageServiceClient()
    document = language_v1.Document(content=text, type_=language_v1.Document.Type.PLAIN_TEXT)
    response = client.analyze_sentiment(document=document)
    sentiment = response.document_sentiment
    return sentiment.score, sentiment.magnitude

# 3. 発言全体の感情を集約して5段階に分類する
def classify_partner_emotion(mentions: list) -> dict:
    """
    複数の発言に対して感情分析を実施し、各発言のスコアと強度の重み付き平均（avg_score）と、
    全発言中の最大感情強度（max_magnitude）を求めた上で、パートナーの感情状態を5段階に分類する。
    戻り値は、集約した平均スコア、最大感情強度、およびラベル、絵文字、定型アラートメッセージを含む dict。
    """
    total_weight = 0.0
    weighted_score_sum = 0.0
    max_magnitude = 0.0
    for mention in mentions:
        score, magnitude = analyze_sentiment(mention)
        weighted_score_sum += score * magnitude
        total_weight += magnitude
        if magnitude > max_magnitude:
            max_magnitude = magnitude
    avg_score = weighted_score_sum / total_weight if total_weight else 0.0

    # 5段階分類の条件
    if avg_score < -0.6 and max_magnitude > 2.0:
        alert = {
            "label": "激おこ",
            "emoji": "😡",
            "message": "相手がブチギレ寸前です。慎重に話しかけてください。"
        }
    elif avg_score < -0.5:
        alert = {
            "label": "情緒が乱れている",
            "emoji": "😠",
            "message": "不満や苛立ち、悲しみなど強めの感情があるかもしれません。丁寧なコミュニケーションを。"
        }
    elif avg_score < -0.4:
        alert = {
            "label": "モヤモヤ・落ち込み気味",
            "emoji": "😟",
            "message": "怒りというより、少し気持ちが落ちているかもしれません。静かに寄り添ってみましょう。"
        }
    elif avg_score < -0.2:
        alert = {
            "label": "小さな違和感",
            "emoji": "😐",
            "message": "少しだけ引っかかることがあるかも。気にかけてあげるといいかも。"
        }
    else:
        alert = {
            "label": "安定・ご機嫌",
            "emoji": "😊",
            "message": "相手はご機嫌な様子。気持ちよくコミュニケーションが取れそうです！"
        }
    return {
        "average_score": avg_score,
        "max_magnitude": max_magnitude,
        **alert
    }