import os
import openai
from llama_index.readers import BeautifulSoupWebReader
from llama_index import GPTSimpleVectorIndex, SimpleDirectoryReader
from langchain.chat_models import ChatOpenAI
from llama_index import GPTSimpleVectorIndex
from transformers import AutoTokenizer, AutoModel

os.environ["OPENAI_API_KEY"] = "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"
openai.organization = "org-rAaBGJuffPllVwZB502H1GH0"

# 商品情報をトークン化してベクトル化する
# GPTSimpleVectorIndexを初期化して、商品情報を追加する
# index = GPTSimpleVectorIndex.load_from_disk("products.json")

# document = SimpleDirectoryReader(input_files=["./documents/brand.txt"]).load_data()[0]
# document.doc_id = "brand.txt"
documents = SimpleDirectoryReader("./documents").load_data()
index = GPTSimpleVectorIndex.from_documents(documents)
# index.insert(document)
index.save_to_disk("products.json")
print(index.query("この商品を売り込む気持ちで23APT-S010についての商品説明を作ってください。"))


# documents = BeautifulSoupWebReaderj().load_data(urls=["https://ja.wikipedia.org/wiki/%E5%A4%A7%E8%B0%B7%E7%BF%94%E5%B9%B3"])
# index = GPTSimpleVectorIndex.load_from_disk("./index.json")
# print(index.query("大谷選手について簡潔に教えてください"))
# index.save_to_disk("index.json")


# openai.api_key = "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"

# # プロンプトの設定
# prompt = "空の色を教えてください。"

# # APIリクエストの設定
# response = openai.Completion.create(
#     model="text-davinci-002",  # GPTのエンジン名を指定します
#     prompt=prompt,
#     max_tokens=100,  # 生成するトークンの最大数
#     n=5,  # 生成するレスポンスの数
#     stop=None,  # 停止トークンの設定
#     temperature=0.7,  # 生成時のランダム性の制御
#     top_p=1,  # トークン選択時の確率閾値
# )

# # 生成されたテキストの取得
# for i, choice in enumerate(response.choices):
#     print(f"\nresult {i}:")
#     print(choice.text.strip())