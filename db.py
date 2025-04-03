### db.py ###
# 役割：データベースの接続設定やテーブル定義を行う
# SQL Alchemyを利用してDBとの接続やモデルを定義している（今回はユーザーからの生の解答テキストと、それを要約したテキストを保存するテーブル）
from sqlalchemy import Column, Integer, String, Text
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
import os
from pathlib import Path
from dotenv import load_dotenv

# 環境変数の読み込み
load_dotenv()
base_path = Path(__file__).parent
env_path = base_path / '.env'


# データベース接続情報
DB_USER = os.getenv('DB_USER')
DB_PASSWORD = os.getenv('DB_PASSWORD')
DB_HOST = os.getenv('DB_HOST')
DB_PORT = os.getenv('DB_PORT')
DB_NAME = os.getenv('DB_NAME')


# SSL証明書のパス（Azure上のKuduにアップしたパスに修正）
ssl_cert = '/home/site/wwwroot/certs/ca-cert.pem'

SQLALCHEMY_DATABASE_URL = f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}" 

# # ローカルで動かすとき ##
# engine = create_engine(
#     SQLALCHEMY_DATABASE_URL, 
#     echo=True
# )

# Azure上にデプロイするとき ##
engine = create_engine(
    SQLALCHEMY_DATABASE_URL, 
    echo=True,
    connect_args={
        "ssl":{
            "ca":ssl_cert
            }
        }
    )
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class Answer(Base):
    __tablename__ = "answers"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, index=True)
    raw_text = Column(Text)
    summary = Column(Text)
