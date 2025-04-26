document.addEventListener("DOMContentLoaded", function() {
  const generateButton = document.getElementById("generateButton");
  const characterTypeSelect = document.getElementById("characterType");
  const seedInput = document.getElementById("seedInput");
  const loadingMessage = document.getElementById("loadingMessage");
  const resultContainer = document.getElementById("resultContainer");
  const regenerateButton = document.getElementById("regenerateButton");
  const submitFeedbackButton = document.getElementById("submitFeedback");
  const feedbackText = document.getElementById("feedbackText");

  let currentSeed = null;
  let currentImageUrl = null;

  generateButton.addEventListener("click", sendGenerateRequest);
  regenerateButton.addEventListener("click", sendGenerateRequest);
  submitFeedbackButton.addEventListener("click", submitFeedback);

  function sendGenerateRequest() {
    loadingMessage.style.display = "block";
    resultContainer.innerHTML = "";
    regenerateButton.style.display = "none";

    const payload = {
      character_type: characterTypeSelect.value,
      seed: seedInput.value
    };

    currentSeed = payload.seed;

    fetch("/api/generate-character", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    })
    .then(response => response.json())
    .then(data => {
      loadingMessage.style.display = "none";
      if (data.image_url) {
        currentImageUrl = data.image_url;
        const img = document.createElement("img");
        img.src = data.image_url;
        img.style.maxWidth = "64px";
        img.style.width = "70%";
        img.style.height = "auto";
        resultContainer.appendChild(img);
        regenerateButton.style.display = "block";
      } else {
        alert("Generation failed: " + JSON.stringify(data));
      }
    })
    .catch(err => {
      loadingMessage.style.display = "none";
      alert("Request error: " + err);
    });
  }

  function submitFeedback() {
    if (!currentSeed || !currentImageUrl) {
      alert("Please generate a character first before submitting feedback.");
      return;
    }
    const feedbackPayload = {
      seed: currentSeed,
      like: true,
      comment: feedbackText.value,
      image_url: currentImageUrl
    };

    fetch("/api/submit-feedback", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(feedbackPayload)
    })
    .then(response => response.json())
    .then(data => {
      alert("Feedback submitted successfully!");
      feedbackText.value = "";
    })
    .catch(err => {
      alert("Failed to submit feedback: " + err);
    });
  }
});
