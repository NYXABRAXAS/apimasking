from fastapi import FastAPI, File, UploadFile, Header, HTTPException, Form, Depends
from fastapi.responses import FileResponse
import easyocr
import re
import os
import cv2
import json
import torch
import tempfile
from typing import Optional, Dict, Any
from starlette.background import BackgroundTasks

app = FastAPI(title="Aadhaar OCR & Masking API")

# ---------------- OCR INITIALIZATION ----------------
# Initializing outside the request to keep it in memory
# gpu=False is required for most Render/Standard cloud tiers
reader = easyocr.Reader(['en'], gpu=False)

# ---------------- API KEY ----------------
API_KEYS = ["mysecretkey123"]

def verify_api_key(x_api_key: str = Header(None)):
    if not x_api_key or x_api_key not in API_KEYS:
        raise HTTPException(status_code=401, detail="Invalid API Key")
    return x_api_key

# ---------------- HELPERS ----------------
def remove_file(path: str):
    if os.path.exists(path):
        try:
            os.remove(path)
        except Exception:
            pass

def clean_name(value):
    if not value: return value
    # Remove non-alphabetic characters and short noise words
    value = re.sub(r'[^A-Za-z\s]', '', value)
    words = [w for w in value.split() if len(w) > 1]
    return " ".join(words)

# ---------------- CORE LOGIC ----------------
@app.post("/v1/ocr/extract-and-mask")
@torch.no_grad() # Reduces memory usage on Render
async def extract_and_mask(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    x_api_key: str = Depends(verify_api_key)
):
    if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
        raise HTTPException(400, "Only image files allowed (JPG/PNG)")

    # Save uploaded file to a temporary location
    suffix = os.path.splitext(file.filename)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_input:
        tmp_input.write(await file.read())
        input_path = tmp_input.name

    try:
        # 1. OCR Processing
        # detail=1 is required to get bounding boxes for masking
        ocr_results = reader.readtext(input_path, detail=1)
        lines = [res[1] for res in ocr_results]
        full_text = " ".join(lines)

        # 2. Field Extraction
        extracted = {
            "aadhaar_number": None,
            "dob": None,
            "name": None
        }

        # Regex for Aadhaar (4-4-4 format) and DOB
        num_match = re.search(r'\b\d{4}\s?\d{4}\s?\d{4}\b', full_text)
        if num_match: extracted["aadhaar_number"] = num_match.group(0)
        
        dob_match = re.search(r'\d{2}/\d{2}/\d{4}', full_text)
        if dob_match: extracted["dob"] = dob_match.group(0)

        # Name Detection Logic (Looking for lines after Govt of India)
        for i, line in enumerate(lines):
            upper_line = line.upper()
            if "GOVERNMENT" in upper_line or "INDIA" in upper_line or "UNIQUE" in upper_line:
                for j in range(1, 4):
                    if i + j < len(lines):
                        candidate = lines[i + j]
                        if len(candidate.split()) >= 2 and not any(c.isdigit() for c in candidate):
                            extracted["name"] = clean_name(candidate)
                            break
                if extracted["name"]: break

        # 3. Masking Logic
        img = cv2.imread(input_path)
        if img is None:
            raise ValueError("Failed to load image for masking")

        masked_count = 0
        for (bbox, text, prob) in ocr_results:
            clean_text = text.replace(" ", "")
            # Check if this box contains a 12-digit sequence
            if re.fullmatch(r'\d{12}', clean_text) or re.search(r'\b\d{4}\s?\d{4}\s?\d{4}\b', text):
                (tl, tr, br, bl) = bbox
                x_min, y_min = int(tl[0]), int(tl[1])
                x_max, y_max = int(br[0]), int(br[1])
                
                # Mask first 8 digits (approx 70% of the box width)
                mask_end_x = x_min + int((x_max - x_min) * 0.70)
                cv2.rectangle(img, (x_min, y_min), (mask_end_x, y_max), (0, 0, 0), -1)
                masked_count += 1

        # 4. Save and Return
        output_path = input_path.replace(suffix, f"_masked{suffix}")
        cv2.imwrite(output_path, img)

        # Encode extraction data into headers for Swagger visibility
        headers = {
            "X-OCR-Data": json.dumps(extracted),
            "X-Masked-Count": str(masked_count),
            "Access-Control-Expose-Headers": "X-OCR-Data, X-Masked-Count"
        }

        # Schedule file deletion after response is sent
        background_tasks.add_task(remove_file, input_path)
        background_tasks.add_task(remove_file, output_path)

        return FileResponse(
            path=output_path,
            media_type="image/jpeg",
            filename=f"masked_{file.filename}",
            headers=headers
        )

    except Exception as e:
        remove_file(input_path)
        raise HTTPException(500, f"Processing Error: {str(e)}")

@app.get("/")
def health():
    return {"status": "Aadhaar OCR & Masking API is Running"}
