from fastapi import FastAPI, File, UploadFile, Header, HTTPException, Form, Depends
from fastapi.responses import FileResponse
import easyocr
import re
import os
import cv2
import json
import tempfile
from typing import Optional, Dict, Any
from starlette.background import BackgroundTasks

app = FastAPI(title="Aadhaar OCR & Masking API")

# ---------------- OCR ----------------
reader = easyocr.Reader(['en'], gpu=False)

# ---------------- API KEY ----------------
API_KEYS = ["mysecretkey123"]

def verify_api_key(x_api_key: str = Header(None)):
    if not x_api_key or x_api_key not in API_KEYS:
        raise HTTPException(status_code=401, detail="Invalid API Key")
    return x_api_key

# ---------------- CLEANUP FUNCTION ----------------
def remove_file(path: str):
    if os.path.exists(path):
        os.remove(path)

# ---------------- HELPERS ----------------
def clean_name(value):
    if not value: return value
    value = re.sub(r'[^A-Za-z\s]', '', value)
    words = [w for w in value.split() if len(w) > 1]
    return " ".join(words)

# ---------------- MAIN API ----------------
@app.post("/v1/ocr/extract-and-mask")
async def extract_and_mask(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    x_api_key: str = Depends(verify_api_key)
):
    if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
        raise HTTPException(400, "Only image files allowed")

    # Create temporary path
    suffix = os.path.splitext(file.filename)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_input:
        tmp_input.write(await file.read())
        input_path = tmp_input.name

    try:
        # 1. OCR Processing
        ocr_results = reader.readtext(input_path, detail=1)
        lines = [res[1] for res in ocr_results]
        full_text = " ".join(lines)

        # 2. Field Extraction (Name, DOB, Number)
        extracted = {
            "aadhaar_number": None,
            "dob": None,
            "name": None
        }

        # Regex for Aadhaar and DOB
        num_match = re.search(r'\b\d{4}\s?\d{4}\s?\d{4}\b', full_text)
        if num_match: extracted["aadhaar_number"] = num_match.group(0)
        
        dob_match = re.search(r'\d{2}/\d{2}/\d{4}', full_text)
        if dob_match: extracted["dob"] = dob_match.group(0)

        # Simple Name fallback (logic from your previous version)
        for i, line in enumerate(lines):
            if "GOVERNMENT" in line.upper() or "INDIA" in line.upper():
                for j in range(1, 4):
                    if i + j < len(lines):
                        candidate = lines[i + j]
                        if len(candidate.split()) >= 2 and not any(c.isdigit() for c in candidate):
                            extracted["name"] = clean_name(candidate)
                            break
                if extracted["name"]: break

        # 3. Masking the Image
        img = cv2.imread(input_path)
        for (bbox, text, prob) in ocr_results:
            clean_text = text.replace(" ", "")
            if re.fullmatch(r'\d{12}', clean_text) or re.search(r'\b\d{4}\s?\d{4}\s?\d{4}\b', text):
                (tl, tr, br, bl) = bbox
                x_min, y_min = int(tl[0]), int(tl[1])
                x_max, y_max = int(br[0]), int(br[1])
                # Mask first 8 digits
                mask_end_x = x_min + int((x_max - x_min) * 0.70)
                cv2.rectangle(img, (x_min, y_min), (mask_end_x, y_max), (0, 0, 0), -1)

        # Save masked image
        output_path = input_path.replace(suffix, f"_masked{suffix}")
        cv2.imwrite(output_path, img)

        # 4. Return image + OCR data in headers
        # We JSON-encode the extracted data so it can fit in a header string
        headers = {
            "X-OCR-Data": json.dumps(extracted),
            "Access-Control-Expose-Headers": "X-OCR-Data" # Allows frontend to read it
        }

        # Clean up files after sending
        background_tasks.add_task(remove_file, input_path)
        background_tasks.add_task(remove_file, output_path)

        return FileResponse(
            path=output_path,
            media_type=file.content_type,
            filename=f"masked_{file.filename}",
            headers=headers
        )

    except Exception as e:
        remove_file(input_path)
        raise HTTPException(500, str(e))

@app.get("/")
def home():
    return {"status": "OCR & Masking API Running 🚀"}