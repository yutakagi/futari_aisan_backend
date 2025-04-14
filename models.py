### models.py ###
import enum
from sqlalchemy import Column, String, Date, Integer, DateTime, Text, Float, Enum, ForeignKey
from datetime import datetime
from db import Base
from sqlalchemy.orm import Mapped, mapped_column

# GenderEnum定義
class GenderEnum(enum.Enum):
    男 = "男"
    女 = "女"
    その他 = "その他"

class User(Base):
    __tablename__ = "users"
    user_id = Column(Integer,primary_key=True)   # ユーザーID
    name = Column(String(50))  # 名前
    gender = Column(Enum(GenderEnum), nullable=False)  # 性別：男, 女, その他
    birthday = Column(Date)   # 誕生日
    personality = Column(String(10))      # 性格(MBTI)
    couple_id = Column(String(50))    # 夫婦id


class UserAnswer(Base):
    __tablename__ = "user_answers"
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, nullable=False)        # 発言したユーザーID（UserテーブルのIDとの関連）
    session_id = Column(String(50), nullable=False)       # 会話 session id
    round_number = Column(Integer, nullable=False)        # ラウンド番号
    user_question = Column(Text, nullable=True)           # ユーザーの質問または回答


class ConversationHistory(Base):
    __tablename__ = "conversation_history"
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, nullable=False)
    session_id = Column(String(50), nullable=False)
    chat_history = Column(Text, nullable=False)  ## 会話の全履歴（LLM のメモリにある内容を結合するなど）
    created_at = Column(DateTime, default=datetime.utcnow)


class StructuredAnswer(Base):
    __tablename__ = "structured_answers"
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.user_id"), nullable=False)  # 追加
    conversation_history_id = Column(Integer, ForeignKey("conversation_history.id"), nullable=False)
    answer_summary = Column(Text, nullable=False)  # JSON形式などでまとめた構造化データを保存
    created_at = Column(DateTime, default=datetime.utcnow)

class VectorSummary(Base):
    """
    ベクトル検索結果を要約したものを保存するテーブルの例
    """
    __tablename__ = "vector_summaries"
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.user_id"), nullable=False)
    query_key = Column(Text, nullable=False)
    summary_text = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

class EmotionAlert(Base):
    __tablename__ = "emotion_alerts"
    id = Column(Integer, primary_key=True, autoincrement=True)
    # このアラートがどの会話履歴に紐付くか
    conversation_history_id = Column(Integer, ForeignKey("conversation_history.id"), nullable=False)
    # アラートを受信するパートナーのユーザーID
    user_id = Column(Integer, ForeignKey("users.user_id"), nullable=False)
    # アラート対象の最もネガティブな発言
    most_negative_mention = Column(Text, nullable=False)
    # 感情スコア (-1.0 ~ +1.0)
    score = Column(Float, nullable=False)
    # 感情の強度（0以上の値）
    magnitude = Column(Float, nullable=False)
    # 固定の定型テキストで決まった感情ラベル（例："激おこ", "情緒が乱れている" など）
    label = Column(String(50), nullable=False)
    # 絵文字で表す、あるいは短いラベル（例："😡"など）
    emoji = Column(String(10), nullable=False)
    # 定型のアラート文
    message = Column(Text, nullable=False)
    # 生成日時（何日前の感情かを示すため）
    created_at = Column(DateTime, default=datetime.utcnow)

class UserReflections(Base):
    __tablename__ = 'reflections'
    
    reflection_id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id: Mapped[str] = mapped_column(Text, comment="メールアドレスなどユーザーID")
    future_plans: Mapped[str] = mapped_column(Text, comment="これからやろうと思うこと")
    want_to_discuss: Mapped[str] = mapped_column(Text, comment="まだ話足りないこと")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)

class DialogueAdvice(Base):
    __tablename__ = "dialogue_advice"
    id = Column(Integer, primary_key=True, autoincrement=True)
    couple_id = Column(String(50), nullable=False)
    user_id = Column(Integer, nullable=False)  # 誰がこのアドバイスを見たか
    advice_text = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
