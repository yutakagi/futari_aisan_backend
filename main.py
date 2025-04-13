### main.py ###
import os
# GCP環境変数を指定
# os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "gcp-credentials.json"
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/home/site/wwwroot/gcp-credentials.json"

# 各種エンドポイントを定義
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uuid
import json
import logging
import asyncio
from pydantic import BaseModel
from datetime import datetime, timedelta
from db import SessionLocal, engine, Base
from models import User, UserAnswer, ConversationHistory, StructuredAnswer,GenderEnum ,EmotionAlert,DialogueAdvice
from summarizer import summarize_answer
from summarizer import generate_couple_conversation_advice
from summarizer import summarize_multiple_docs
from summarizer_rag import generate_report_with_rag
from conversation_chain import create_conversation_chain
from structured_parser import extract_structured_data
from structured_vector import (
    build_structured_vector_store,
    search_all_predefined_queries,
    PREDEFINED_QUERIES
)
from models import VectorSummary
from emotion_analysis import (extract_partner_mentions_llm, 
                              classify_partner_emotion,
                              analyze_sentiment,
)


app = FastAPI()

Base.metadata.create_all(bind=engine)

# CORSミドルウェアの設定
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# In-memoryセッション管理（プロトタイプ用）
sessions = {}

# --- エンドポイント用のPydanticスキーマ ---

class AnswerInput(BaseModel):
    user_id: int
    answer_text: str

class ReportResponse(BaseModel):
    first: str
    second: str
    third : str

class ChatRequest(BaseModel):
    user_id: int
    session_id: str = None
    answer: str = None

class ChatResponse(BaseModel):
    session_id: str
    feedback: str = None
    round: int
    message: str = None

    # User用のPydanticスキーマ（models.pyのUser定義を参照）
class UserCreate(BaseModel):
    user_id: int
    name: str
    gender: GenderEnum
    birthday: datetime  # フロントエンドからはYYYY-MM-DD形式で送信される想定
    personality: str
    couple_id: str

# 感情分析確認用の入力スキーマ
class SentimentTestInput(BaseModel):
    text: str

# --- エンドポイント ---

# 一問一答機能：会話セッションの開始または継続の処理
@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    db = SessionLocal()
    try:
        if not request.session_id:
            # 新規セッション開始時にユーザー情報・パートナー情報を取得
            user = db.query(User).filter(User.user_id == request.user_id).first()
            if not user:
                raise HTTPException(status_code=404, detail="ユーザー情報が見つかりません。")
            # 自分以外で同じcouple_idのユーザーを取得
            partner = db.query(User).filter(
                User.couple_id == user.couple_id,
                User.user_id != user.user_id
            ).first()  

            # パートナーが見つからない場合はNone（create_conversation_chain内で「情報なし」に置換）
            session_id = str(uuid.uuid4())
            chain = create_conversation_chain(user,partner)
            sessions[session_id] = chain
            initial_input = "セッション開始"
            response = chain.predict(input=initial_input)
            return ChatResponse(
                session_id=session_id,
                feedback=response,
                round=1,
                message="セッションを開始しました。"
            )
        else:
            # 既存セッションの場合
            session_id = request.session_id
            if session_id not in sessions:
                raise HTTPException(status_code=400, detail="セッションが存在しません。")
            chain = sessions[session_id]
            if not request.answer:
                raise HTTPException(status_code=400, detail="回答が入力されていません。")
            # ラウンド番号を会話履歴から計算（例：単純にメッセージ数から算出）
            round_number = len(chain.memory.chat_memory.messages) // 2 + 1
            user_answer = UserAnswer(
                user_id=request.user_id,
                session_id=session_id,
                round_number=round_number,
                user_question=request.answer
            )
            db.add(user_answer)
            db.commit()
            response = chain.predict(input=request.answer)
            return ChatResponse(
                session_id=session_id,
                feedback=response,
                round=round_number,
                message=""
            )
    except Exception as e:
        logger.exception("Error in chat endpoint")
        raise HTTPException(status_code=500, detail="チャット処理中にエラーが発生しました。")
    finally:
        db.close()

@app.post("/save_conversation")
async def save_conversation(session_id: str, user_id: int):
    db = SessionLocal()
    try:
        # セッションの存在確認
        if session_id not in sessions:
            raise HTTPException(status_code=400, detail="セッションが存在しません。")
        chain = sessions[session_id]

        # 会話履歴の連結
        chat_history = "\n".join([msg.content for msg in chain.memory.chat_memory.messages])
        # ConversationHistory に保存
        conv_history = ConversationHistory(
            user_id=user_id,
            session_id=session_id,
            chat_history=chat_history
        )
        db.add(conv_history)
        db.commit()
        db.refresh(conv_history)

        # 既存の構造化データ保存処理
        structured_data = extract_structured_data(chat_history)
        structured_answer = StructuredAnswer(
            conversation_history_id=conv_history.id,
            user_id=user_id, 
            answer_summary=json.dumps(structured_data, ensure_ascii=False)
        )
        db.add(structured_answer)
        db.commit()

        # 感情分析処理
        # ユーザー発言のみを抽出
        mentions = extract_partner_mentions_llm(chat_history, partner_name="パートナー")
        # すべての発言を集約して5段階に分類
        emotion_alert = classify_partner_emotion(mentions)
        logging.info(f"[集約感情判定結果] {emotion_alert}")
        analysis_result = emotion_alert

        # 感情アラートをDBに保存（パートナーのuser_idを取得）
        user = db.query(User).filter(User.user_id == user_id).first()

        partner = db.query(User).filter(
            User.couple_id == user.couple_id,
            User.user_id != user_id
        ).first()

        if partner:
            alert_record = EmotionAlert(
                user_id=partner.user_id,
                conversation_history_id=conv_history.id,
                most_negative_mention=emotion_alert.get("most_negative_mention", ""),  # なければ空
                score=emotion_alert["average_score"],
                magnitude=emotion_alert["max_magnitude"],
                label=emotion_alert["label"],
                emoji=emotion_alert["emoji"],
                message=emotion_alert["message"]
            )
            db.add(alert_record)
            db.commit()
            logging.info(f"[感情アラート保存済] partner_id={partner.user_id}, label={alert_record.label}")

        return {
            "message": "会話履歴と構造化データの保存に成功しました。",
            "emotion_analysis": analysis_result
        }    
    
    except Exception as e:
        logger.exception("エラー内容:")
        db.rollback()
        raise HTTPException(status_code=500, detail="保存中にエラーが発生しました")
    finally:
        db.close()

@app.post("/register")
async def register_user(user: UserCreate):
    db = SessionLocal()
    try:
        # 同じユーザーIDでの重複登録を防ぐためのチェック
        existing_user = db.query(User).filter(User.user_id == user.user_id).first()
        if existing_user:
            raise HTTPException(status_code=400, detail="このユーザーIDは既に登録されています。")
        
        new_user = User(
            user_id=user.user_id,
            name=user.name,
            gender=user.gender,
            birthday=user.birthday,
            personality=user.personality,
            couple_id=user.couple_id
        )
        db.add(new_user)
        db.commit()
        db.refresh(new_user)
        return {
            "message": "ユーザー登録が完了しました",
            "user": {
                "user_id": new_user.user_id,
                "name": new_user.name,
                "gender": new_user.gender.value,
                "birthday": new_user.birthday.isoformat(),
                "personality": new_user.personality,
                "couple_id": new_user.couple_id
            }
        }
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"ユーザー登録中にエラーが発生しました: {str(e)}")
    finally:
        db.close()

@app.get("/structured_vector_search/fixed_all")
async def fixed_structured_vector_search_all(user_id: int):
    """
    直近days日分の構造化データを抽出し、PREDEFINED_QUERIESに定義されたクエリを
    すべてベクトル検索。クエリごとの検索結果をまとめて返す。
    """
    db = SessionLocal()
    try:
        # -- ユーザー名を取得して"さん"付けする --
        user = db.query(User).filter(User.user_id == user_id).first()
        if not user:
            raise HTTPException(status_code=404, detail="該当するユーザーが見つかりません。")

        user_name_with_suffix = f"{user.name}さん"

        # 1) 指定日数分遡るため、days引いた日時を計算
        cutoff_date = datetime.utcnow() - timedelta(days=4)

        async def process_user_data(target_user_id: int):
            answers = (db.query(StructuredAnswer)
                       .filter(StructuredAnswer.user_id == target_user_id)
                       .filter(StructuredAnswer.created_at >= cutoff_date)
                       .all())
            if not answers:
                return None

            # 3) JSON文字列をPythonの辞書として読み込む
            structured_data_list = [json.loads(ans.answer_summary) for ans in answers]

            # 4) ベクトルストアを構築
            vector_store = build_structured_vector_store(structured_data_list)

            # 5) 全クエリ一括検索
            all_results = search_all_predefined_queries(vector_store, k=3)

            saved_summaries =[]
            for query_key, doc_texts in all_results.items():
                #doc_textsは要約前のベクトルストアから検索したn件のテキスト
                #これを1つの文字列にまとめる
                merged_text = "\n\n".join(doc_texts)
                
                #LLMに要約
                summary_text = await summarize_multiple_docs([merged_text])

                #DBに保存
                new_summary = VectorSummary(
                    user_id=user_id,
                    query_key = query_key,
                    summary_text=summary_text
                )
                db.add(new_summary)
                db.commit()
                db.refresh(new_summary)
                saved_summaries.append({
                    "query_key": query_key,
                    "merged_documents": merged_text,
                    "summay_text": summary_text
                })
            return saved_summaries
        # 自分の処理
        user_summaries = await process_user_data(user_id)

        # パートナーが存在する場合の処理
        partner = db.query(User).filter(
            User.couple_id == user.couple_id,
            User.user_id != user.user_id
        ).first()

        partner_summaries = None
        partner_name_with_suffix = None
        if partner:
            partner_name_with_suffix = f"{partner.name}さん"
            partner_summaries = await process_user_data(partner.user_id)

        return{
            "user_id": user_id,
            "user_name": user_name_with_suffix,
            "partner_user_id": partner.user_id if partner else None,
            "partner_name": partner_name_with_suffix,
            "user_summaries": user_summaries,
            "partner_summaries": partner_summaries
        }
    finally:
        db.close()

@app.get("/emotion_alert/latest")
async def get_latest_emotion_alert(user_id: int):
    db = SessionLocal()
    try:
        alert = (db.query(EmotionAlert)
                   .filter(EmotionAlert.user_id == user_id)
                   .order_by(EmotionAlert.created_at.desc())
                   .first())
        if not alert:
            raise HTTPException(status_code=404, detail="最新の感情アラートは見つかりません。")
        return {
            "label": alert.label,
            "emoji": alert.emoji,
            "message": alert.message,
            "score": alert.score,
            "magnitude": alert.magnitude,
            "created_at": alert.created_at.isoformat()
        }
    finally:
        db.close()

# 感情分析確認用エンドポイント
@app.post("/test_emotion", summary="GCP感情分析APIの動作確認")
async def test_emotion_endpoint(input_data: SentimentTestInput):
    try:
        # 入力テキストに対して感情分析を実行
        score, magnitude = analyze_sentiment(input_data.text)
        return {
            "score": score,
            "magnitude": magnitude,
            "message": "感情分析APIは正常に動作しています。"
        }
    except Exception as e:
        logger.exception("GCP感情分析APIの呼び出し中にエラーが発生しました")
        raise HTTPException(status_code=500, detail=f"GCP感情分析APIエラー: {str(e)}")
    
@app.get("/dialogue_advice")
async def get_dialogue_advice(user_id: int):
    db = SessionLocal()
    try:
        # 最新の要約データを取得
        cutoff_date = datetime.utcnow()-timedelta(days=4)
        def fetch_latest_summaries(uid):
            return(db.query(VectorSummary)
                    .filter(VectorSummary.user_id == uid)
                    .filter(VectorSummary.created_at >= cutoff_date)
                    .all())
        user_summaries = fetch_latest_summaries(user_id)

        user = db.query(User).filter(User.user_id == user_id).first()
        if not user:
            raise HTTPException(status_code=404, detail="ユーザーが見つかりません")
        
        partner = db.query(User).filter(
            User.couple_id == user.couple_id,
            User.user_id != user.user_id
        ).first()

        partner_summaries = fetch_latest_summaries(partner.user_id) if partner else []
        # ユーザーとパートナーのMBTIを取得
        user_mbti = user.personality
        partner_mbti = partner.personality if partner else "不明"
        #ユーザーとパートナーの名前を取得
        user_name = user.name
        partner_name = partner.name if partner else "不明"

        # 対話アドバイスを生成
        advice_text = await generate_couple_conversation_advice(
            user_summary_blocks=[
                {"query_key":s.query_key, "summay_text":s.summary_text}
                for s in user_summaries
            ],
            partner_summary_blocks=[
                {"query_key": s.query_key, "summay_text": s.summary_text}
                for s in partner_summaries
            ],
            user_mbti=user_mbti,
            partner_mbti=partner_mbti,
            user_name=user.name,
            partner_name=partner.name
        )
        # 対話アドバイスをDBに保存
        advice_record = DialogueAdvice(
            couple_id = user.couple_id,
            user_id = user.user_id,
            advice_text=advice_text
        )
        db.add(advice_record)
        db.commit()

        return {"advice":advice_text}

    except Exception as e:
            db.rollback()
            raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()