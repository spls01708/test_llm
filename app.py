from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
import faiss
import numpy as np
import json

# โหลดข้อมูล
with open("data.json", "r", encoding="utf-8") as f:
    documents = json.load(f)

# โหลด Sentence-BERT สำหรับการสร้าง Embedding
embedding_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# สร้าง Embedding และ FAISS Index
contents = [doc["content"] for doc in documents]
embeddings = embedding_model.encode(contents)
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))

# โหลด OpenThaiGPT
tokenizer = AutoTokenizer.from_pretrained("openthaigpt/openthaigpt1.5-7b-instruct", cache_dir="../models/openthaigpt")
model = AutoModelForCausalLM.from_pretrained(
    "openthaigpt/openthaigpt1.5-7b-instruct", cache_dir="../models/openthaigpt"
).to("cuda")  # โหลดโมเดลและย้ายไป GPU (ถ้ามี GPU)

# สร้าง Flask App
app = Flask(__name__)

@app.route("/query", methods=["POST"])
def query():
    try:
        # รับคำถามจากผู้ใช้
        data = request.json
        question = data["question"]

        # สร้าง Embedding สำหรับคำถาม
        question_embedding = embedding_model.encode([question])

        # ค้นหาข้อมูลที่เกี่ยวข้องใน FAISS
        _, indices = index.search(np.array(question_embedding), 1)
        retrieved_content = documents[indices[0][0]]["content"]

        # สร้าง Prompt และคำตอบ
        prompt = f"คำถาม: {question}\nข้อมูล: {retrieved_content}\nคำตอบ: "
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to("cuda")  # ย้าย input ไป GPU
        outputs = model.generate(inputs["input_ids"], max_length=180, temperature=0.7, top_p=0.9)
        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # ส่งผลลัพธ์กลับ (ภาษาไทย)
        return app.response_class(
            response=json.dumps({"answer": answer}, ensure_ascii=False),
            mimetype="application/json"
        )

    except Exception as e:
        # จัดการข้อผิดพลาด
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)