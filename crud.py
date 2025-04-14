from sqlalchemy import create_engine, insert, delete, update, select
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import SQLAlchemyError
import json
from datetime import datetime
from typing import Any, Dict, List

from db import engine
from models import UserReflections, User  # Userモデルも追加でインポート

def myinsert(mymodel: Any, values: Dict[str, Any]) -> str:
    """データ挿入関数（変更なし）"""
    Session = sessionmaker(bind=engine)
    session = Session()
    
    try:
        query = insert(mymodel).values(values)
        session.execute(query)
        session.commit()
        return "inserted"
    except SQLAlchemyError as e:
        session.rollback()
        print(f"Insert error: {str(e)}")
        return f"insert failed: {str(e)}"
    finally:
        session.close()

def myselect(mymodel: Any, user_id: str) -> str:
    """ユーザーIDに基づく振り返りデータ取得関数（変更なし）"""
    Session = sessionmaker(bind=engine)
    session = Session()
    
    try:
        query = session.query(mymodel).filter(mymodel.user_id == user_id)
        results = query.all()
        
        result_list = []
        for reflection in results:
            result_list.append({
                "reflection_id": reflection.reflection_id,
                "user_id": reflection.user_id,
                "future_plans": reflection.future_plans,
                "want_to_discuss": reflection.want_to_discuss,
                "created_at": reflection.created_at.isoformat() if reflection.created_at else None
            })
        
        return json.dumps(result_list, ensure_ascii=False)
    
    except SQLAlchemyError as e:
        print(f"Database error: {str(e)}")
        return json.dumps({"error": str(e)}, ensure_ascii=False)
    
    finally:
        session.close()

def get_user_by_id(user_id: str) -> User:
    """ユーザーIDからユーザー情報を取得する"""
    Session = sessionmaker(bind=engine)
    session = Session()
    
    try:
        user = session.query(User).filter(User.user_id == user_id).first()
        return user
    except SQLAlchemyError as e:
        print(f"Error in get_user_by_id: {str(e)}")
        return None
    finally:
        session.close()

def get_partner(couple_id: str, user_id: str) -> User:
    """同じcouple_idを持つパートナーを取得する"""
    Session = sessionmaker(bind=engine)
    session = Session()
    
    try:
        partner = session.query(User).filter(
            User.couple_id == couple_id,
            User.user_id != user_id
        ).first()
        return partner
    except SQLAlchemyError as e:
        print(f"Error in get_partner: {str(e)}")
        return None
    finally:
        session.close()
