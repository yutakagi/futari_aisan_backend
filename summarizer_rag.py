### summarizer_rag.py ###
# RAGを用いて蓄積された回答の要約情報からレポートとアドバイスを生成する処理
# DBから取得した各回答の要約をDocumentに変換し、FAISSを利用してベクトル化、検索できるようにしている（ベクトルストアを作成）
# LangchainのRetrivalQAチェーンを使ってレポート＋アドバイスの生成を行う=>ベクトルストアとチェーンを組み合わせることでより関連性の高い情報を参照しながら生成する仕組み
import asyncio
from typing import Tuple
from langchain.docstore.document import Document
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from gpt4omini_llm import GPT4oMiniLLM

async def generate_report_with_rag(answers: list) -> Tuple[str, str, str]:
    """
    answers: DBから取得したAnswerオブジェクトのリスト。各オブジェクトは .summary を持つとする。
    """
    # DBから取得した各回答の要約をDocumentリストへ変換
    docs = [Document(page_content=ans.summary) for ans in answers]
    
    # 埋め込みモデルの初期化
    embeddings = OpenAIEmbeddings()  
    
    # FAISSベクトルストアの構築
    vector_store = FAISS.from_documents(docs, embeddings)
    retriever = vector_store.as_retriever(search_kwargs={"k": 6})
    
    # クエリ文を設定（プロンプト内でレポートとアドバイスを生成する指示を出す）
    query = ("このGPTは、夫婦関係コーチングを専門とするコーチとして機能し、ユーザーの1週間の振り返り内容をもとに、パートナーへ向けた簡潔な報告を第三者の視点で作成します。ユーザー自身ではなく、あくまでコーチとして客観的に状況を共有する形で表現します。"
             "チャット開始時に「パートナーに伝える内容を記入してください」と促し、ユーザーの入力をもとに、第三者目線で内容を整理・変換します。"
             "以下は、個人で1週間の振り返りを行ったチャットログです。この内容をもとに、第三者の目線で、夫婦専門のコーチとしてパートナーに送る要約レポートを作成してください。レポートは以下の3つのセクションに分け、各セクションは例示された「パートナーに送る内容例」と同様の構成、文章の作り、記述の仕方、文字量（各セクション100～150文字程度、若干の前後は許容）で作成してください。なお、例示に沿いつつも、必要に応じて適切な補足情報を加えても構いません。"
             "【必ず以下の見出しを使用して出力してください】"
             "【セクション1:あなたの状況】"
             "個人振り返りの内容（仕事、家事育児、個人の満足度や気づきなど）をもとに、パートナーの1週間の状況を客観的に要約してください。全体の流れや雰囲気を捉え、簡潔にまとめるとともに、入力内の不要な記号や箇条書き形式は統一した文章形式に変換してください。"
             "【セクション2:パートナーに対するコメント】"
             "チャットログ内の「助かった＆嬉しかったこと」と「気になったこと」を踏まえ、パートナーに対する感謝やポジティブな面、及び改善してほしい点を、温かみがありながらも客観的かつやんわりと伝える文言で記述してください。"
             "【セクション3:夫婦で話し合いたいこと】"
             "振り返りに含まれる「夫婦で話したいこと」や「話し合うと良さそうなこと」の内容を、箇条書きまたは簡潔な文章で整理し、今後の夫婦での話し合いのテーマとして提示してください。"
             "回答の形式は下記のようにお願いします:\n"
             "あなたの状況: <状況におけるレポート>\n:パートナーに対するコメント:<コメントのレポート>\n:夫婦で話し合いたいこと:<夫婦で話し合いたいことに対するレポート>")
    
    # カスタムLLM（GPT-4o-mini）を利用してRetrievalQAチェーンを作成
    llm = GPT4oMiniLLM()
    chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
    
    # 非同期でチェーンを実行
    result = await chain.arun(query)
    
    # 生成された結果を「Report」と「Advice」に分割（プロンプトでのフォーマットに依存）
    if "【セクション2:パートナーに対するコメント】  " in result and "【セクション3:夫婦で話し合いたいこと】" in result:
        #「コメント」の位置で一度分割し、前半部分に「あなたの状況」、後半部分に「コメント」と「話し合いたいこと」が残る
        first_part, rest = result.split("【セクション2:パートナーに対するコメント】 ",1)
        # restを分割して「コメント」と「話し合いたいこと」に分ける
        second_part, third_part = rest.split("【セクション3:夫婦で話し合いたいこと】",1)
        first = first_part.replace("【セクション1:あなたの状況】 ","").strip()
        second = second_part.strip()
        third = third_part.strip()
    else:
        first = result
        second = ""
        third = ""
    return first, second, third
