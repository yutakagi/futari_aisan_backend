### summarizer.py ###
# ユーザーから送信された回答をGPTのモデルを使用して要約するための処理を実装
import openai
from dotenv import load_dotenv # type: ignore
import os
import asyncio
from dotenv import load_dotenv

# 環境変数の読み込み
load_dotenv() 
# OpenAIクライアントのインスタンスを作成
openai.api_key = os.getenv("OPENAI_API_KEY")

# GPT-4o-miniのAPI呼び出し処理を実装する
report_content = """
このGPTは、夫婦関係コーチングを専門とするコーチとして機能し、ユーザーの1週間の振り返り内容をもとに、パートナーへ向けた簡潔な報告を第三者の視点で作成します。ユーザー自身ではなく、あくまでコーチとして客観的に状況を共有する形で表現します。
チャット開始時に「パートナーに伝える内容を記入してください」と促し、ユーザーの入力をもとに、第三者目線で内容を整理・変換します。
以下は、個人で1週間の振り返りを行ったチャットログです。この内容をもとに、第三者の目線で、夫婦専門のコーチとしてパートナーに送る要約レポートを作成してください。。
"""
async def gpt4o_mini_call(prompt: str) -> str:
    try:
        response = await openai.ChatCompletion.acreate(
            model="gpt-4o-mini",
            messages=[
                {"role": "system","content":report_content},
                {"role": "user","content":  prompt}
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Effor:{e}"

# 複数のドキュメントを結合して要約する
async def summarize_multiple_docs(doc_texts:list[str]) -> str:
    """複数のドキュメントテキストを結合してまとめて要約する簡易関数"""
    combined_text = "\n\n".join(doc_texts)
    prompt = f"以下の複数テキストをまとめて要約してください:\n{combined_text}"
    return await gpt4o_mini_call(prompt)

# 受け取った回答テキストに対して要約プロンプトを生成し、APIを呼び出して要約結果を返す
# 複数のデータに分割してDBに格納する
async def summarize_answer(answer_text: str) -> str:
    prompt = f"次の答えを要約してください: {answer_text}"
    summary = await gpt4o_mini_call(prompt)
    return summary

# 要約をもとにレポートとアドバイスを生成する
async def generate_report(combined_summary: str) -> (str, str):
    prompt = f"以下の要約に基づいて包括的なレポートと実用的なアドバイスを生成します: {combined_summary}"
    result = await gpt4o_mini_call(prompt)
    # ダミーのため、同じ結果をレポートとアドバイスに分割して返す
    report = f"Report: {result}"
    advice = f"Advice: {result}"
    return report, advice

# 非同期関数の実行例
async def main():
    answer = "夫が家事を全くやってくれず、とてもイラつきました。これから一緒に暮らしていくのはとても難しいと思いました"
    summary = await summarize_answer(answer)
    report, advice = await generate_report(summary)
    print("Summary", summary)
    print("Report:", report)
    print("Advice:",advice)

if __name__ == "__main__":
    asyncio.run(main())