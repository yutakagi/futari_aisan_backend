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
以下は、個人で1週間の振り返りを行ったチャットログです。この内容をもとに、第三者の目線で、夫婦専門のコーチとしてパートナーに送る要約レポートを作成してください。レポートは以下の3つのセクションに分け、各セクションは例示された「パートナーに送る内容例」と同様の構成、文章の作り、記述の仕方、文字量（各セクション100～150文字程度、若干の前後は許容）で作成してください。なお、例示に沿いつつも、必要に応じて適切な補足情報を加えても構いません。
【必ず以下の見出しを使用して出力してください】
【セクション1:あなたの状況】
個人振り返りの内容（仕事、家事育児、個人の満足度や気づきなど）をもとに、パートナーの1週間の状況を客観的に要約してください。全体の流れや雰囲気を捉え、簡潔にまとめるとともに、入力内の不要な記号や箇条書き形式は統一した文章形式に変換してください。
【セクション2:パートナーに対するコメント】
チャットログ内の「助かった＆嬉しかったこと」と「気になったこと」を踏まえ、パートナーに対する感謝やポジティブな面、及び改善してほしい点を、温かみがありながらも客観的かつやんわりと伝える文言で記述してください。
【セクション3:夫婦で話し合いたいこと】
振り返りに含まれる「夫婦で話したいこと」や「話し合うと良さそうなこと」の内容を、箇条書きまたは簡潔な文章で整理し、今後の夫婦での話し合いのテーマとして提示してください。
出力例（パートナーに送る内容例）を参考に、上記フォーマットに従って変換を行ってください。
###出力例（パートナーに送る内容例）
【今週のパートナーの状況】
仕事の面談で育休復帰後のキャリアについて考える機会がありました。子どもの成長を実感しつつ、美容室でリフレッシュもできました。ランチや散歩を楽しみ、穏やかな時間を過ごせましたが、時間の使い方をもう少し工夫したいと感じています。
【あなたに対するコメント】
家事や育児、また学びの応援をしてくれることに感謝の気持ちを持っています。雑談のときに、もっと話を聞いてくれるといいなーと思っているようですが、全体的に感謝の気持ちが多いです。
【夫婦で話し合いたいこと】
・保育園開始後の家事・育児の役割分担。
・子どもが病気になった際の対応フロー。
・育休中の時間があるうちにやりたいこと・やるべきことのリスト化（家の整理、役所の手続き、自己学習など）。
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