// frontend/js/app.js

document.addEventListener("DOMContentLoaded", function() {
    const generateButton = document.getElementById("generateButton");
    const characterTypeSelect = document.getElementById("characterType");
    const colorThemeSelect = document.getElementById("colorTheme");
    const seedInput = document.getElementById("seedInput");
    const loadingMessage = document.getElementById("loadingMessage");
    const resultContainer = document.getElementById("resultContainer");
    const regenerateButton = document.getElementById("regenerateButton");
    const submitFeedbackButton = document.getElementById("submitFeedback");
    const feedbackText = document.getElementById("feedbackText");
  
    generateButton.addEventListener("click", sendGenerateRequest);
    regenerateButton.addEventListener("click", sendGenerateRequest);
    
    // 示例反馈事件，可自行扩展
    submitFeedbackButton.addEventListener("click", function() {
      alert("感谢你的反馈！");
    });
    
    function sendGenerateRequest() {
      loadingMessage.style.display = "block";
      resultContainer.innerHTML = "";  // 清空之前的结果
      regenerateButton.style.display = "none";
      
      // 收集用户输入
      const payload = {
        character_type: characterTypeSelect.value,
        color_theme: colorThemeSelect.value,
        seed: seedInput.value
      };
  
      // 调用后端 API, 请确保后端服务已启动且跨域设置正确
      fetch("http://127.0.0.1:5000/api/generate-character", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      })
      .then(response => response.json())
      .then(data => {
        loadingMessage.style.display = "none";
        if (data.image_url) {
          // 在结果区域创建一个 img 标签显示生成的图片
          const img = document.createElement("img");
          img.src = data.image_url;
          resultContainer.appendChild(img);
          regenerateButton.style.display = "block";
        } else {
          alert("生成失败：" + JSON.stringify(data));
        }
      })
      .catch(err => {
        loadingMessage.style.display = "none";
        alert("请求错误: " + err);
      });
    }
  });
  