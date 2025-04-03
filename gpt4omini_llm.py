### gpt4omini_llm.py ###
# Langchainで利用するためのカスタムLLMラッパーを定義
# GPTを非同期的に呼び出す為のラッパーとしてLangChainのLLMのインターフェーズを実装
import asyncio
from typing import Optional, List, Dict
from langchain.llms.base import LLM
from summarizer import gpt4o_mini_call

# _acallや_callのメソッドでGPTへのリクエストをラップし、Langchainのチェーン内で利用できるようにする
class GPT4oMiniLLM(LLM):
    async def _acall(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        return await gpt4o_mini_call(prompt)
    
    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        # 同期的な呼び出しは、内部的に非同期呼び出しをラップする
        return asyncio.run(self._acall(prompt, stop=stop))
    
    @property
    def _identifying_params(self) -> Dict:
        return {"model": "gpt-4o-mini"}
    
    @property
    def _llm_type(self) -> str:
        return "custom_gpt4o_mini"
