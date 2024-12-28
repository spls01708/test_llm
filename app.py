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

# สร้าง Embedding และ FAISS Index โดยใช้เฉพาะ content
contents = [doc["content"] for doc in documents]
articles = [doc["article"] for doc in documents]  # ดึงข้อมูลมาตรา
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

@app.route("/query", methods=["GET", "POST"])
def query():
    try:
        # รับคำถามจากผู้ใช้
        # data = request.json
        # question = data["question"]
        question = request.args.get("question") if request.method == "GET" else request.json.get("question")

        # สร้าง Embedding สำหรับคำถาม
        question_embedding = embedding_model.encode([question])

        # ค้นหาข้อมูลที่เกี่ยวข้องใน FAISS (ดึง Top-K = 3)
        k = 3
        distances, indices = index.search(np.array(question_embedding), k)
        retrieved_contents = [contents[i] for i in indices[0]]
        retrieved_articles = [articles[i] for i in indices[0]]  # ดึงมาตรา

        # กรองข้อมูลเพิ่มเติม (ถ้าไม่มีข้อมูลที่เกี่ยวข้อง)
        filtered_results = [
            {"content": content, "article": article}
            for content, article in zip(retrieved_contents, retrieved_articles)
            if len(content.strip()) > 0
        ]

        # รวมข้อมูลที่เกี่ยวข้อง (ถ้าไม่มีข้อมูลให้ตอบข้อความเริ่มต้น)
        if not filtered_results:
            retrieved_content = "ไม่พบข้อมูลที่เกี่ยวข้อง"
            retrieved_sources = []
        else:
            retrieved_content = "\n\n".join([result["content"] for result in filtered_results[:2]])  # จำกัด 2 ข้อมูลที่เกี่ยวข้อง
            retrieved_sources = [result["article"] for result in filtered_results[:2]]  # แสดงเฉพาะมาตรา

        # สร้าง Prompt
        prompt = f"คำถาม: {question}\nข้อมูลที่เกี่ยวข้อง:\n{retrieved_content}\nตอบคำถามโดยใช้ข้อมูลที่เกี่ยวข้องท่านั้น\nคำตอบ: "

        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to("cuda")

        # Generate คำตอบ
        outputs = model.generate(inputs["input_ids"], max_length=1200, temperature=0.5, top_p=0.9)

        # Decode คำตอบ
        answer = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

        # ตัดส่วนของ Prompt ออก
        if "คำตอบ: " in answer:
            answer = answer.split("คำตอบ: ")[-1].strip()

        # ส่งผลลัพธ์กลับ (ภาษาไทย) พร้อมแหล่งที่มา
        return app.response_class(
            response=json.dumps({"answer": answer, "sources": retrieved_sources}, ensure_ascii=False),
            mimetype="application/json"
        )

    except Exception as e:
        # จัดการข้อผิดพลาด
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, threaded=True)


