import os
import sys
import time
import torch
from flask import Flask, request, jsonify
from flask_cors import CORS
from torchvision.transforms.functional import to_pil_image

# 将项目根目录加入 sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from models.generator import Generator
from configs.config import latent_dim
from backend.utils.device import get_device

# 设置静态文件目录为项目根目录下的 static 文件夹
app = Flask(__name__, static_folder=os.path.join(project_root, "static"))
CORS(app)

# 角色条件数 = 3
n_classes = 3

# 实例化生成器（条件生成器）
# 输出尺寸为 3*16*16
output_size = 3 * 16 * 16
generator = Generator(latent_dim, n_classes=n_classes, output_size=output_size)
device = get_device()
generator.to(device)

# 加载训练好的权重（请确保你在 checkpoints 文件夹保存了条件GAN权重）
weight_path = os.path.join(project_root, "checkpoints", "generator_epoch_50.pth")
print("尝试加载权重路径：", weight_path)
print("权重文件存在吗？", os.path.exists(weight_path))
if os.path.exists(weight_path):
    print("加载生成器权重：", weight_path)
    generator.load_state_dict(torch.load(weight_path, map_location=device))
    generator.eval()
else:
    print("警告：未找到生成器权重，使用随机初始化生成器！")

def one_hot(index, num_classes):
    vec = torch.zeros(1, num_classes, device=device)
    vec[0, index] = 1.0
    return vec

def generate_image(seed, char_type_str):
    # 将传入的角色类型字符串映射为整数
    mapping = {"monster": 0, "human": 1, "prop": 2}
    char_index = mapping.get(char_type_str.lower(), 0)
    condition = one_hot(char_index, n_classes)
    
    if seed is not None and seed.strip() != "" and seed.lower() != "random":
        torch.manual_seed(int(seed))
    with torch.no_grad():
        noise = torch.randn((1, latent_dim), device=device)
        img_tensor = generator(noise, condition)
    return img_tensor
@app.route("/", methods=["GET"])
def index():
    html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Conditional Character Generator</title>
  <style>
    body { font-family: sans-serif; background: #f7f7f7; padding: 20px; }
    .container { max-width: 600px; margin: auto; background: #fff; padding: 20px; }
    label { display: inline-block; width: 120px; font-weight: bold; }
    input, select, button { padding: 5px; margin: 5px 0; }
    #resultImg { max-width: 100%; border: 1px solid #ccc; margin-top: 20px; display: none; }
    #loadingMessage { display: none; margin-top: 10px; color: #555; }
  </style>
</head>
<body>
  <div class="container">
    <h1>Conditional Character Generator</h1>
    <div>
      <label for="charType">角色类型:</label>
      <select id="charType">
        <option value="monster">monster</option>
        <option value="human">human</option>
        <option value="prop">prop</option>
      </select>
    </div>
    <div>
      <label for="seedInput">随机种子:</label>
      <input type="text" id="seedInput" placeholder="例如 123 或 'random'">
    </div>
    <div>
      <button id="generateBtn">生成角色</button>
    </div>
    <div id="loadingMessage">生成中，请稍候…</div>
    <img id="resultImg" alt="生成的角色">
  </div>
  <script>
    const generateBtn = document.getElementById("generateBtn");
    const charTypeSelect = document.getElementById("charType");
    const seedInput = document.getElementById("seedInput");
    const loadingMessage = document.getElementById("loadingMessage");
    const resultImg = document.getElementById("resultImg");

    generateBtn.addEventListener("click", function(){
      loadingMessage.style.display = "block";
      resultImg.style.display = "none";
      let payload = {
        "character_type": charTypeSelect.value,
        "seed": seedInput.value
      };
      fetch("/api/generate-character", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      })
      .then(resp => resp.json())
      .then(data => {
        loadingMessage.style.display = "none";
        if(data.image_url) {
          resultImg.src = data.image_url;
          resultImg.style.display = "block";
        } else {
          alert("生成失败: " + JSON.stringify(data));
        }
      })
      .catch(err => {
        loadingMessage.style.display = "none";
        alert("请求错误: " + err);
      });
    });
  </script>
</body>
</html>
    """
    return html_content

@app.route("/api/generate-character", methods=["POST"])

def api_generate_character():
    try:
        data = request.get_json()
        char_type_str = data.get("character_type", "monster").lower()
        seed = data.get("seed", "random")
        print(f"[API] Received => seed={seed}, type={char_type_str}")
        
        # 生成图像张量
        img_tensor = generate_image(seed, char_type_str)
        from torchvision.transforms.functional import to_pil_image
        pil_img = to_pil_image(img_tensor.squeeze(0).cpu())
        
        # 生成文件名
        timestamp = int(time.time())
        filename = f"{char_type_str}_{timestamp}.png"
        
        # 保存图片到项目根目录下的 static/generated/
        generated_dir = os.path.join(project_root, "static", "generated")
        os.makedirs(generated_dir, exist_ok=True)
        file_path = os.path.join(generated_dir, filename)
        pil_img.save(file_path, "PNG")
        
        image_url = f"http://127.0.0.1:5000/static/generated/{filename}"
        print("[API] Generated image:", file_path, " => ", image_url)
        return jsonify({"image_url": image_url}), 200
    except Exception as e:
        print("Error in /api/generate-character:", e)
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
