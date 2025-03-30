# 各種エンドポイントを定義
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from db import SessionLocal, engine, Base, Answer
from summarizer import summarize_answer
from summarizer_rag import generate_report_with_rag
import asyncio
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
Base.metadata.create_all(bind=engine)

class AnswerInput(BaseModel):
    user_id: int
    answer_text: str

class ReportResponse(BaseModel):
    first: str
    second: str
    third : str

# CORSミドルウェアの設定
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ユーザーの回答時にsumarizer.pyの要約処理を呼び出して要約した回答をDBに保存する処理
@app.post("/answers/")
async def post_answer(answer_input: AnswerInput):
    # GPT-4o-miniを用いて回答内容の要約を生成
    summary = await summarize_answer(answer_input.answer_text)
    db = SessionLocal()
    try:
        new_answer = Answer(
            user_id=answer_input.user_id,
            raw_text=answer_input.answer_text,
            summary=summary
        )
        db.add(new_answer)
        db.commit()
        db.refresh(new_answer)
    finally:
        db.close()
    return {"message": "回答を要約して保存しました", "summary": summary}

# レポート生成のリクエストを受けた際にsummarizar_rag.pyのRAG処理を呼び出して生成結果を返す処理
@app.get("/report/", response_model=ReportResponse)
async def get_report_rag(user_id: int):
    db = SessionLocal()
    try:
        answers = db.query(Answer).filter(Answer.user_id == user_id).all()
        if not answers:
            raise HTTPException(status_code=404, detail="No answers found for user")
        first,second, third = await generate_report_with_rag(answers)
    finally:
        db.close()
    return ReportResponse(first=first, second=second, third=third)
