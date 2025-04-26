import traceback
import os
import sys
import time
import torch
import csv
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from torchvision.transforms.functional import to_pil_image

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

# import models and configs
from backend.models.generator import Generator
from configs.config import latent_dim
from backend.utils.device import get_device

import gspread
from google.oauth2.service_account import Credentials

CREDENTIALS_PATH = os.path.join(project_root, "backend", "credentials", "cse-dsci-498-project-1442f8631adf.json") 

SPREADSHEET_ID = "1Tg55z42eFY3Szx9ZpNfTakDzTopMCWy-MVhaYj_BBdU" 

def get_sheet():
    scopes = ["https://www.googleapis.com/auth/spreadsheets"]
    credentials = Credentials.from_service_account_file(CREDENTIALS_PATH, scopes=scopes)
    gc = gspread.authorize(credentials)
    sh = gc.open_by_key(SPREADSHEET_ID)
    worksheet = sh.sheet1 
    return worksheet

# Flask app
app = Flask(__name__, static_folder=os.path.join(project_root, "static"))
CORS(app)

# save generated images and feedback
os.makedirs(os.path.join(project_root, "static", "generated"), exist_ok=True)
os.makedirs(os.path.join(project_root, "feedback"), exist_ok=True)

# generated images classes
n_classes = 4

# Load generator
device = get_device()
generator = Generator(latent_dim, n_classes=n_classes, img_size=16)
generator.to(device)

weight_path = os.path.join(project_root, "checkpoints", "generator_epoch_460.pth")
if os.path.exists(weight_path):
    print(f"Load the generator weights：{weight_path}")
    generator.load_state_dict(torch.load(weight_path, map_location=device, weights_only=True))
    generator.eval()
else:
    print("Warning: Generator weights not found. In checkpoints folder.")

# One-hot
def one_hot(index, num_classes):
    vec = torch.zeros(1, num_classes, device=device)
    vec[0, index] = 1.0
    return vec

# Generate image
def generate_image(seed, char_type_str):
    mapping = {"monster": 0, "human": 1, "item": 2, "equipment": 3}
    char_index = mapping.get(char_type_str.lower(), 0)
    condition = one_hot(char_index, n_classes)

    if seed and seed.strip() and seed.lower() != "random":
        torch.manual_seed(int(seed))
        torch.cuda.manual_seed_all(int(seed)) 

    with torch.no_grad():
        noise = torch.randn((1, latent_dim), device=device)
        img_tensor = generator(noise, condition)
    return img_tensor


# routes
@app.route("/", methods=["GET"])
def index():
    frontend_dir = os.path.join(project_root, "frontend")
    return send_from_directory(frontend_dir, "index.html")

@app.route("/frontend/<path:filename>", methods=["GET"])
def serve_frontend_static(filename):
    frontend_dir = os.path.join(project_root, "frontend")
    return send_from_directory(frontend_dir, filename)

@app.route("/frontend/js/<path:filename>", methods=["GET"])
def serve_js(filename):
    js_dir = os.path.join(project_root, "frontend", "js")
    return send_from_directory(js_dir, filename)

@app.route("/frontend/static/<path:filename>", methods=["GET"])
def serve_css(filename):
    static_dir = os.path.join(project_root, "frontend", "static")
    return send_from_directory(static_dir, filename)

@app.route("/api/generate-character", methods=["POST"])
def api_generate_character():
    try:
        data = request.get_json()
        char_type_str = data.get("character_type", "monster").lower()
        seed = data.get("seed", "random")

        img_tensor = generate_image(seed, char_type_str)
        pil_img = to_pil_image(img_tensor.squeeze(0).cpu())

        timestamp = int(time.time())
        filename = f"{char_type_str}_{timestamp}.png"

        generated_dir = os.path.join(project_root, "static", "generated")
        file_path = os.path.join(generated_dir, filename)
        pil_img.save(file_path, "PNG")

        image_url = f"/static/generated/{filename}"
        print("[API] Generated image:", file_path)
        return jsonify({"image_url": image_url}), 200
    except Exception as e:
        print("Error in /api/generate-character:", e)
        return jsonify({"error": str(e)}), 500

from datetime import datetime

@app.route("/api/submit-feedback", methods=["POST"])
def api_submit_feedback():
    try:
        data = request.get_json()
        print("Get:", data) 

        seed = data.get("seed", "")
        like = data.get("like", "")
        comment = data.get("comment", "")
        image_url = data.get("image_url", "")

        worksheet = get_sheet()

        if image_url:
            image_formula = f'=IMAGE("{image_url}")'
        else:
            image_formula = ""

        new_row = [
            datetime.now().isoformat(),  
            seed,
            like,
            comment,
            image_formula
        ]

        worksheet.append_row(new_row)

        print("Wrote into Google Sheet：\nGoogle it to look our feedbacks'https://docs.google.com/spreadsheets/d/1Tg55z42eFY3Szx9ZpNfTakDzTopMCWy-MVhaYj_BBdU/edit?gid=0#gid=0'", new_row)

        return jsonify({"message": "Feedback saved successfully to Google Sheets."}), 200
    except Exception as e:
        print("Error in /api/submit-feedback:", e)
        return jsonify({"error": str(e)}), 500





if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
