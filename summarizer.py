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

async def gpt4o_mini_call(prompt: str, system_prompt:str=None) -> str:
    try:
        response = await openai.ChatCompletion.acreate(
            model="gpt-4o-mini",
            messages=[
                {"role": "system","content":system_prompt},
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
    system_prompt = """
    このGPTは、夫婦関係コーチングを専門とするコーチとして機能し、ユーザーの1週間の振り返り内容をもとに、パートナーへ向けた簡潔な報告を第三者の視点で作成します。ユーザー自身ではなく、あくまでコーチとして客観的に状況を共有する形で表現します。
    チャット開始時に「パートナーに伝える内容を記入してください」と促し、ユーザーの入力をもとに、第三者目線で内容を整理・変換します。
    以下は、個人で1週間の振り返りを行ったチャットログです。この内容をもとに、第三者の目線で、夫婦専門のコーチとしてパートナーに送る要約レポートを作成してください。
    要約レポートの文字数は100文字程度にまとめてください.
    また、日本語の文章として読みやすいように適宜改行を入れてください
    """
    return await gpt4o_mini_call(prompt,system_prompt)

# 最新レポートを受け取り、対話アドバイスを生成
async def generate_couple_conversation_advice(
        user_summary_blocks:list[dict],
        partner_summary_blocks:list[dict],
        user_name: str,
        partner_name: str,
        user_mbti:str,
        partner_mbti:str) -> str:
     """
    user_summary_blocks: [{query_key, summay_text}]
    partner_summary_blocks: [{query_key, summay_text}]
    を受け取り、夫婦の対話に関するコーチ的アドバイスを生成する
    """
     # 最新の3件だけに絞る
     user_summary_blocks = user_summary_blocks[-3:]
     partner_summary_blocks = partner_summary_blocks[-3:]
     
     def format_block(blocks,name):
         return "\n\n".join(
             f"{name}の「{b['query_key']}」に関する要約:\n{b['summay_text']}"
             for b in blocks
         )
     user_text = format_block(user_summary_blocks,"ユーザー")
     partner_text = format_block(partner_summary_blocks,"パートナー")

     prompt = (
         "あなたはMBTI（性格タイプ）にも精通した夫婦向けコーチです。\n"
         "以下は、ある夫婦の個別振り返り要約と、それぞれの性格タイプ（MBTI）です。\n\n"
         f"【ユーザーのMBTI】{user_mbti}\n"
         f"【パートナーのMBTI】{partner_mbti}\n\n"
         f"【ユーザー名】{user_name} さん\n"
         f"【パートナー名】{partner_name} さん\n"
         "それぞれのタイプに合った配慮を踏まえて、以下の観点で前向きなアドバイスをお願いします：\n"
         "1. 今、話し合うと良いテーマの箇条書き（理由も）\n"
         "2. 対話で気をつけるべきこと（性格の違いからくる誤解や注意点）\n"
         "3. 相手との認識ギャップがある場合の橋渡し方法\n\n"
         "出力の際は改行はなくし、基本的には一段落を1つのまとまりとして出力してください"
         "1~3の観点の記載は他の文字より2段階大きくなるようなMarkdownの記述で出力してください"
         "=== 要約データ ===\n"
         f"{user_text}\n\n---\n\n{partner_text}\n\n"
         "=== 以上 ==="
     )
     system_prompt ="""
     このGPTは、夫婦間の建設的な対話をサポートする専門家として振る舞う。夫婦それぞれの性格（例：MBTI）や今週の感情的・生活的状況、話したいアジェンダをもとに、どのような会話をどの順序で行うべきか、優先度をつけてアドバイスする。感情的なトーンや背景事情に配慮しながら、丁寧かつ共感的な口調で会話を導き、対立を避け、相互理解と関係性の改善を促すアプローチをとる。
     パートナー双方に配慮した言葉選びを心がけ、一方的な判断は避ける。ユーザーから得られた情報をもとに、具体的な対話例や質問例を提示することもある。アドバイスは、実際の感情や状況に寄り添った現実的なものであることを重視する。
     アドバイスは夫婦どちらに対しても平等にアドバイスをし、どちらに対してのアドバイスかわかるように主語を明確に名前で呼ぶ
     日本語で対応し、丁寧で安心感のあるトーンを保つ。必要に応じて、感情的負担が軽減されるようなリフレーミングや気持ちの整理のサポートも行う。
     """
     return await gpt4o_mini_call(prompt, system_prompt)

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