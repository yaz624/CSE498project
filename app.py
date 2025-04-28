import streamlit as st
import torch
from PIL import Image
from torchvision.transforms.functional import to_pil_image
import time
import os
import gspread
from google.oauth2.service_account import Credentials
from backend.models.generator import Generator
from configs.config import latent_dim
from backend.utils.device import get_device

# === Google Sheets Config ===
CREDENTIALS_PATH = "backend/credentials/cse-dsci-498-project-1442f8631adf.json"
SPREADSHEET_ID = "1Tg55z42eFY3Szx9ZpNfTakDzTopMCWy-MVhaYj_BBdU"

def get_sheet():
    scopes = ["https://www.googleapis.com/auth/spreadsheets"]
    creds = Credentials.from_service_account_file(CREDENTIALS_PATH, scopes=scopes)
    client = gspread.authorize(creds)
    sheet = client.open_by_key(SPREADSHEET_ID).sheet1
    return sheet

# === Load Model ===
device = get_device()
generator = Generator(latent_dim, n_classes=4, img_size=16).to(device)
weight_path = "checkpoints/generator_epoch_460.pth"
if os.path.exists(weight_path):
    state = torch.load(weight_path, map_location=device, weights_only=True)
    generator.load_state_dict(state)
    generator.eval()
else:
    st.error("Generator weights not found!")

# === Streamlit UI ===
st.set_page_config(page_title="Pixel Character Generator", page_icon="🎮", layout="centered")
st.title("🎮 Pixel-style Character Generator")

# --- Use Session State ---
if "feedback_submitted" not in st.session_state:
    st.session_state.feedback_submitted = False
if "last_feedback" not in st.session_state:
    st.session_state.last_feedback = {}

# --- User Inputs ---
character_type = st.selectbox("Select Character Type:", ["monster", "human", "item", "equipment"])
seed = st.text_input("Random Seed (optional):", value="random")

generate_clicked = st.button("🎲 Generate Character")

if generate_clicked:
    mapping = {"monster": 0, "human": 1, "item": 2, "equipment": 3}
    idx = mapping.get(character_type, 0)
    condition = torch.zeros(1, 4, device=device)
    condition[0, idx] = 1.0

    if seed.strip() and seed.lower() != "random":
        val = int(seed)
        torch.manual_seed(val)
        torch.cuda.manual_seed_all(val)

    with torch.no_grad():
        noise = torch.randn((1, latent_dim), device=device)
        img_tensor = generator(noise, condition)

    pil_img = to_pil_image(img_tensor.squeeze(0).cpu())
    scaled = pil_img.resize((pil_img.width * 5, pil_img.height * 5), resample=Image.NEAREST)

    st.image(scaled, caption="Generated Character", width=scaled.width)

    tmp_path = f"gen_{int(time.time())}.png"
    scaled.save(tmp_path)
    with open(tmp_path, "rb") as f:
        st.download_button(label="📥 Download Character", data=f, file_name=os.path.basename(tmp_path), mime="image/png")

    # === Feedback Form ===
    with st.form(key="feedback_form"):
        st.write("### 📝 Submit Feedback")
        feedback_text = st.text_area("Your feedback:")
        submit_feedback = st.form_submit_button("✅ Submit")

        if submit_feedback:
            try:
                sheet = get_sheet()
                now = time.strftime("%Y-%m-%d %H:%M:%S")
                row = [now, seed, character_type, feedback_text]
                sheet.append_row(row)

                # Save into session state
                st.session_state.feedback_submitted = True
                st.session_state.last_feedback = {
                    "time": now,
                    "seed": seed,
                    "type": character_type,
                    "comment": feedback_text
                }
                # Force rerun
                st.experimental_rerun()

            except Exception as e:
                st.error(f"❌ Failed to submit feedback: {e}")

    if os.path.exists(tmp_path):
        os.remove(tmp_path)

# === After Rerun: Show success
if st.session_state.feedback_submitted:
    st.success("✅ Feedback submitted successfully!")
    st.write("### 📋 Your Last Feedback:")
    st.json(st.session_state.last_feedback)

    # Reset feedback flag after showing
    st.session_state.feedback_submitted = False
