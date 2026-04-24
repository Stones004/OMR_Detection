import os
import base64
from mistralai.client import Mistral
from pydantic import BaseModel

# 1. Initialize Client
client = Mistral(api_key="GLFEZ3H2GhVJ5xLvGGQSD9l57zSCzesu")

# 2. Define your Schema (How you want the output to look)
class OMRResult(BaseModel):
    question_number: str
    status: str # "shaded" or "empty"
    handwritten_value: str

class ScriptData(BaseModel):
    answers: list[OMRResult]

def extract_structured_omr(image_path):
    # Encode Image
    with open(image_path, "rb") as f:
        encoded_image = base64.b64encode(f.read()).decode('utf-8')
    
    # Step 1: Run OCR to get the raw content
    ocr_response = client.ocr.process(
        model="mistral-ocr-latest",
        document={
            "type": "image_url",
            "image_url": f"data:image/jpeg;base64,{encoded_image}"
        }
    )
    raw_markdown = ocr_response.pages[0].markdown

    # Step 2: Use Chat Completion to force the JSON Structure
    # This fixes the "boxes vs bubbles" inconsistency
    chat_response = client.chat.complete(
        model="pixtral-large-latest", # Use a vision-capable model
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Extract the OMR status and handwritten values from this script into JSON. Use 'shaded' or 'empty' for status."},
                    {"type": "image_url", "image_url": f"data:image/jpeg;base64,{encoded_image}"}
                ]
            }
        ],
        response_format={
            "type": "json_object"
        }
    )
    
    return chat_response.choices[0].message.content

# Run it
print(extract_structured_omr(r"G:\OMR_Detection\output_final\img_000006_result.png"))