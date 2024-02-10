import os
from llama_index import ServiceContext, LLMPredictor, OpenAIEmbedding, PromptHelper
from llama_index.llms import OpenAI
from llama_index.text_splitter import TokenTextSplitter
from llama_index.node_parser import SentenceSplitter
from llama_index import VectorStoreIndex, SimpleDirectoryReader
from llama_index import set_global_service_context
import tiktoken
from fastapi import FastAPI
from dotenv import load_dotenv
load_dotenv()
from pydantic import BaseModel
api_key = os.getenv("openai_api_key")
app = FastAPI()
class RagInputs(BaseModel):
  paths: list
  query: str

@app.post("/qa_rag")
async def main(rag: RagInputs):
  documents = SimpleDirectoryReader(input_files=rag.paths).load_data()
  # print(documents)
  text_splitter = TokenTextSplitter(
  separator=" ",
  chunk_size=1024,
  chunk_overlap=20,
  backup_separators=["\n"],
  tokenizer=tiktoken.encoding_for_model("gpt-3.5-turbo").encode)
  # node_parser = SentenceSplitter.from_defaults(text_splitter=text_splitter)
  llm = OpenAI(model='gpt-3.5-turbo', api_key=api_key, temperature=0, max_tokens=256)
  embed_model = OpenAIEmbedding()
  prompt_helper = PromptHelper(
    context_window=4096,
    num_output=256, 
    chunk_overlap_ratio=0.1, 
    chunk_size_limit=None
  )
  service_context = ServiceContext.from_defaults(
  llm=llm,
  embed_model=embed_model,
  node_parser=text_splitter,
  prompt_helper=prompt_helper
  )
  index = VectorStoreIndex.from_documents(documents, service_context=service_context)
  query_engine = index.as_query_engine(service_context=service_context)
  response = query_engine.query(rag.query)
  return response
