import os
import time
import json
import fitz  # PyMuPDF
import numpy as np
import re
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
INPUT_DIR = os.getenv("INPUT_DIR", "input")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "output")
OUTPUT_FILENAME = "challenge1b_output.json"
TOP_K_SECTIONS = 5
BOILERPLATE_TITLES = {
    "introduction", "table of contents", "summary", "overview",
    "conclusion", "references", "history", "acknowledgements"
}

# ----------------------------------------------------------------------
# Heading Detection Logic
# ----------------------------------------------------------------------
def get_body_font_size(page: fitz.Page) -> float:
    sizes = {}
    for block in page.get_text("dict", sort=True)["blocks"]:
        if "lines" not in block:
            continue
        for line in block["lines"]:
            for span in line["spans"]:
                size = round(span["size"], 1)
                sizes[size] = sizes.get(size, 0) + 1
    return max(sizes, key=sizes.get, default=12.0)

def is_heading(line: Dict, body_size: float) -> bool:
    if not line.get("spans"):
        return False
    span = line["spans"][0]
    text = "".join(s["text"] for s in line["spans"]).strip()

    score = 0
    if round(span["size"]) > body_size: score += 3
    if "bold" in span["font"].lower(): score += 2
    if len(text.split()) <= 10: score += 1
    if text.endswith("."): score -= 2
    if re.match(r"^\s*(\*|\-|\d+\.)", text): score -= 3
    return score >= 3

# ----------------------------------------------------------------------
# PDF Parsing
# ----------------------------------------------------------------------
def parse_documents(pdf_paths: List[str], input_dir: str) -> List[Dict[str, Any]]:
    structured = []
    for filename in pdf_paths:
        try:
            doc = fitz.open(os.path.join(input_dir, filename))
            current_heading = "Introduction"
            current_text = ""
            current_page = 1
            for page_num, page in enumerate(doc, 1):
                body_size = get_body_font_size(page)
                blocks = page.get_text("dict", sort=True)["blocks"]
                for block in blocks:
                    if "lines" not in block:
                        continue
                    block_text = " ".join("".join(s["text"] for s in l["spans"]) for l in block["lines"]).strip()
                    if not block_text:
                        continue
                    is_heading_block = is_heading(block["lines"][0], body_size)
                    if is_heading_block:
                        if current_text.strip():
                            structured.append({
                                "doc_name": filename,
                                "page_num": current_page,
                                "section_title": current_heading.strip(),
                                "content": current_text.strip()
                            })
                        current_heading = block_text
                        current_text = ""
                        current_page = page_num
                    else:
                        current_text += " " + block_text
            if current_text.strip():
                structured.append({
                    "doc_name": filename,
                    "page_num": current_page,
                    "section_title": current_heading.strip(),
                    "content": current_text.strip()
                })
        except Exception as e:
            print(f"❌ Error parsing {filename}: {e}")
    return structured

# ----------------------------------------------------------------------
# Relevance Engine
# ----------------------------------------------------------------------
class RelevanceEngine:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)

    def embed(self, texts: List[str]) -> np.ndarray:
        return self.model.encode(texts, convert_to_tensor=False, show_progress_bar=False)

    def similarity(self, query_vec: np.ndarray, content_vecs: np.ndarray) -> np.ndarray:
        return cosine_similarity(query_vec.reshape(1, -1), content_vecs)[0]

# ----------------------------------------------------------------------
# Utilities
# ----------------------------------------------------------------------
def clean_title(title: str) -> str:
    return title.lower().strip().replace(":", "").replace(".", "")

def deduplicate(sections: List[Dict]) -> List[Dict]:
    seen_titles = set()
    deduped = []
    for s in sections:
        key = (s['doc_name'], clean_title(s['section_title']))
        if key not in seen_titles:
            deduped.append(s)
            seen_titles.add(key)
    return deduped

def split_paragraphs(text: str) -> List[str]:
    parts = re.split(r'\n{2,}|•|\n\s*[-–]\s*', text)
    return [p.strip() for p in parts if len(p.strip()) > 30]

def boost_section_title_score(text: str, score: float) -> float:
    boost_keywords = ["guide", "experience", "tips", "coastal", "beach", "hotel", "itinerary", "food", "culture", "activities", "nightlife"]
    for word in boost_keywords:
        if word in text.lower():
            score *= 1.1
    return score

def penalize_duplicate_documents(para_map: List[Dict]) -> List[Dict]:
    seen = {}
    for p in para_map:
        doc = p["doc_name"]
        seen[doc] = seen.get(doc, 0) + 1
        penalty = 0.9 ** (seen[doc] - 1)
        p["score"] *= penalty
    return para_map

# ----------------------------------------------------------------------
# Main Pipeline
# ----------------------------------------------------------------------
def run_pipeline():
    start = time.time()
    with open(os.path.join(INPUT_DIR, "challenge1b_input.json")) as f:
        meta = json.load(f)

    persona = meta["persona"]["role"]
    task = meta["job_to_be_done"]["task"]
    documents = [d["filename"] for d in meta["documents"]]

    print(f"🧠 Persona: {persona}")
    print(f"🎯 Task: {task}")

    sections = parse_documents(documents, INPUT_DIR)
    sections = [s for s in sections if clean_title(s["section_title"]) not in BOILERPLATE_TITLES]
    sections = deduplicate(sections)

    if not sections:
        print("❌ No valid sections found.")
        return

    para_map = []
    texts = []
    for s in sections:
        paras = split_paragraphs(s["content"])
        for p in paras:
            text = f"{s['section_title']}: {p}"
            texts.append(text)
            para_map.append({
                "doc_name": s["doc_name"],
                "page_num": s["page_num"],
                "section_title": s["section_title"],
                "content": p
            })

    engine = RelevanceEngine()
    query = f"You are a {persona}. Your goal is to {task}. Extract the most useful sections with relevant information, steps, or context."
    q_vec = engine.embed([query])[0]
    para_vecs = engine.embed(texts)
    scores = engine.similarity(q_vec, para_vecs)

    for i, score in enumerate(scores):
        boosted = boost_section_title_score(para_map[i]["section_title"], score)
        para_map[i]["score"] = float(boosted)

    para_map = penalize_duplicate_documents(para_map)
    para_map.sort(key=lambda x: x["score"], reverse=True)

    seen_docs = set()
    top_paras = []
    for p in para_map:
        if p["doc_name"] not in seen_docs:
            top_paras.append(p)
            seen_docs.add(p["doc_name"])
        if len(top_paras) >= TOP_K_SECTIONS:
            break

    extracted_sections = [{
        "document": p["doc_name"],
        "page_number": p["page_num"],
        "section_title": p["section_title"],
        "importance_rank": i + 1
    } for i, p in enumerate(top_paras)]

    subsection_analysis = [{
        "document": p["doc_name"],
        "page_number": p["page_num"],
        "refined_text": p["content"]
    } for p in top_paras]

    output = {
        "metadata": {
            "input_documents": documents,
            "persona": persona,
            "job_to_be_done": task,
            "processing_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        },
        "extracted_sections": extracted_sections,
        "subsection_analysis": subsection_analysis
    }

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, OUTPUT_FILENAME), "w") as f:
        json.dump(output, f, indent=4)

    print(f"✅ Done in {time.time() - start:.2f}s. Output: {os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)}")

if __name__ == "__main__":
    run_pipeline()
