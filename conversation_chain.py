### conversation_chain.py ###
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from datetime import date
from dotenv import load_dotenv
import os

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model_name="gpt-4o-mini", temperature=0.7)
today = date.today().strftime("%Y年%m月%d日")

# プロンプトテンプレート：partial_variablesでユーザー・パートナー情報を注入
system_prompt = f"""
あなたは夫婦やカップル向けにコーチングを実施する、家庭と夫婦の関係性を専門とする優秀なコーチです。
現在の日付：{today}

【ユーザー情報】
ユーザーID:{{user_id}}
名前:{{name}}
性別:{{gender}}
誕生日: {{birthday}}
性格: {{personality}}
夫婦id：{{couple_id}}

【パートナー情報】
ユーザーID:{{partner_user_id}}
名前:{{partner_name}}
性別:{{partner_gender}}
誕生日:{{partner_birthday}}
性格:{{partner_personality}}

以下の5つの質問を順番に行います：
1. 今日の満足度は10点満点中何点ですか？その点数に影響したことを思いつくままに書いてみてください。
2. 夫婦の振り返りです。パートナーとの関係の中で「助かったな」「嬉しかったな」と思ったことはありますか？日常の中の小さなことでもOKです！
3. パートナーに「本当は伝えたかったけれど、言えなかったこと」や「ちょっと飲み込んだ気持ち」はありますか？
4. 来週の満足度を「今週より上げる」としたら、どんなことを意識したり、工夫したいと思いますか？具体的でもふわっとしたイメージでも大丈夫です。
5. 最近、「そろそろ話しておいた方がいいかも」「今のうちに向き合っておきたいな」と思っているテーマや気がかりなことはありますか？

これからの対話では、ユーザーの各回答に対して温かく共感的なフィードバック（100文字程度）を提供してください。
各質問に対してユーザーからの回答が十分でない場合は、内容を掘り下げるための追加質問をしてください。追加質問は最大2回まで。 
最終回答時には、これまでのユーザーの回答を踏まえ『～という1日だったのですね。』という形で要約し、さらにポジティブな一言を添えてください。
"""

def create_conversation_chain(user,partner):
    # partnerが見つからなかった場合は「情報なし」を設定
    partner_data = {
        "partner_user_id":partner.user_id if partner else "情報なし",
        "partner_name": partner.name if partner else "情報なし",
        "partner_gender": partner.gender if partner else "情報なし",
        "partner_birthday": partner.birthday if partner else "情報なし",
        "partner_personality": partner.personality if partner else "情報なし",
    }

    memory = ConversationBufferMemory(memory_key="chat_history",return_messages=True)
    prompt_template = PromptTemplate(
        input_variables=["input","chat_history"],
        partial_variables={
            "today": today,
            "user_id": user.user_id,
            "name": user.name,
            "gender": user.gender,
            "birthday": user.birthday.strftime("%Y年%m月%d日") if not isinstance(user.birthday, str) else user.birthday,
            "personality": user.personality,
            "couple_id": user.couple_id,
            **partner_data,
        },
        template=system_prompt + "\n\n 【対話履歴】\n{chat_history}\n\nユーザー: {input}\nコーチ:"
    )

    chain = ConversationChain(
        llm=llm,
        memory=memory,
        prompt=prompt_template
    )
    return chain