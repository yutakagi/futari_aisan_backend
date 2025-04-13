# ベクトルストアの生成とクエリのベクトルストアへの検索を定義
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS
import json

# 下記のクエリでベクトルストアを検索する
PREDEFINED_QUERIES = {
    "今週の状況": "直近の満足度やその理由についてどのような心境なのか",
    "あなたに対するコメント": "直近のパートナーに対して思うこと",
    "夫婦で話し合いたいこと": "夫婦で話し合いたいと思っていること",
}

# JSONの各フィールドを別々のDocumentにして格納
def build_structured_vector_store(structured_data_list:list[dict]):
    """
    構造化データのリストを受け取り、FAISSベクトルストアを生成して返す。
    """
    documents = [
        Document(page_content=json.dumps(item, ensure_ascii=False))
        for item in structured_data_list
    ]

    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_documents(documents,embeddings)
    return vector_store



# 3つのクエリを別々に検索する関数
def search_all_predefined_queries(vector_store, k=3):
    """
    PREDEFINED_QUERIESに定義された3つのクエリをすべて検索し、
    結果を辞書でまとめて返す。
    """
    results = {}
    for query_key, query_text in PREDEFINED_QUERIES.items():
        docs = vector_store.similarity_search(query_text, k=k)
        # page_contentだけ取り出しておく
        results[query_key] = [doc.page_content for doc in docs]
    return results
    