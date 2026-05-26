const form = document.getElementById("detectorForm");
const apiKeyInput = document.getElementById("apiKey");
const languageInput = document.getElementById("language");
const fileInput = document.getElementById("audioFile");
const fileDrop = document.getElementById("fileDrop");
const fileName = document.getElementById("fileName");
const submitBtn = document.getElementById("submitBtn");

const resultStatus = document.getElementById("resultStatus");
const resultBody = document.getElementById("resultBody");
const errorBox = document.getElementById("errorBox");
const classificationText = document.getElementById("classificationText");
const confidenceText = document.getElementById("confidenceText");
const languageText = document.getElementById("languageText");
const explanationText = document.getElementById("explanationText");

let confidenceChart;
let classChart;

function resetStatus() {
  resultStatus.className = "status status-idle";
  resultStatus.textContent = "Idle";
}

function setStatusSuccess() {
  resultStatus.className = "status status-success";
  resultStatus.textContent = "Success";
}

function setStatusError() {
  resultStatus.className = "status status-error";
  resultStatus.textContent = "Error";
}

function showError(message) {
  errorBox.textContent = message;
  errorBox.classList.remove("hidden");
  resultBody.classList.add("hidden");
  setStatusError();
}

function hideError() {
  errorBox.classList.add("hidden");
  errorBox.textContent = "";
}

function renderCharts(confidence, classification) {
  const humanProb = classification === "HUMAN" ? confidence : 1 - confidence;
  const aiProb = classification === "AI_GENERATED" ? confidence : 1 - confidence;

  if (confidenceChart) {
    confidenceChart.destroy();
  }

  confidenceChart = new Chart(document.getElementById("confidenceChart"), {
    type: "doughnut",
    data: {
      labels: ["Predicted Confidence", "Remaining"],
      datasets: [{
        data: [confidence * 100, (1 - confidence) * 100],
        backgroundColor: ["#3dd3b2", "#224555"],
        borderWidth: 1,
        borderColor: "#0f2733"
      }]
    },
    options: {
      plugins: {
        legend: {
          labels: { color: "#eaf7ff" }
        }
      }
    }
  });

  if (classChart) {
    classChart.destroy();
  }

  classChart = new Chart(document.getElementById("classChart"), {
    type: "bar",
    data: {
      labels: ["HUMAN", "AI_GENERATED"],
      datasets: [{
        label: "Probability (%)",
        data: [humanProb * 100, aiProb * 100],
        backgroundColor: ["#ffd166", "#3dd3b2"],
        borderRadius: 8
      }]
    },
    options: {
      scales: {
        x: {
          ticks: { color: "#eaf7ff" },
          grid: { color: "rgba(255,255,255,0.08)" }
        },
        y: {
          beginAtZero: true,
          max: 100,
          ticks: { color: "#eaf7ff" },
          grid: { color: "rgba(255,255,255,0.08)" }
        }
      },
      plugins: {
        legend: {
          labels: { color: "#eaf7ff" }
        }
      }
    }
  });
}

function getAudioFormat(file) {
  const fromName = file.name.split(".").pop()?.toLowerCase();
  if (fromName) {
    return fromName;
  }
  const fromType = file.type.split("/").pop();
  return fromType || "mp3";
}

function fileToBase64(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();

    reader.onload = () => {
      const dataUrl = reader.result;
      const base64 = dataUrl.split(",")[1];
      resolve(base64);
    };

    reader.onerror = () => reject(new Error("Could not read audio file"));
    reader.readAsDataURL(file);
  });
}

fileDrop.addEventListener("click", () => fileInput.click());
fileInput.addEventListener("change", () => {
  const file = fileInput.files[0];
  fileName.textContent = file ? file.name : "No file selected";
});

fileDrop.addEventListener("dragover", (event) => {
  event.preventDefault();
  fileDrop.style.borderColor = "#3dd3b2";
});

fileDrop.addEventListener("dragleave", () => {
  fileDrop.style.borderColor = "rgba(61, 211, 178, 0.55)";
});

fileDrop.addEventListener("drop", (event) => {
  event.preventDefault();
  fileDrop.style.borderColor = "rgba(61, 211, 178, 0.55)";

  if (!event.dataTransfer.files.length) {
    return;
  }

  fileInput.files = event.dataTransfer.files;
  fileName.textContent = fileInput.files[0].name;
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  hideError();

  const apiKey = apiKeyInput.value.trim();
  const language = languageInput.value.trim();
  const audioFile = fileInput.files[0];

  if (!apiKey || !language || !audioFile) {
    showError("Please provide API key, language, and audio file.");
    return;
  }

  submitBtn.disabled = true;
  resultStatus.textContent = "Processing...";

  try {
    const audioBase64 = await fileToBase64(audioFile);

    const response = await fetch("/api/voice-detection", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "x-api-key": apiKey
      },
      body: JSON.stringify({
        language,
        audioFormat: getAudioFormat(audioFile),
        audioBase64
      })
    });

    const data = await response.json();

    if (!response.ok || data.status !== "success") {
      throw new Error(data.detail || data.message || "Request failed");
    }

    classificationText.textContent = data.classification;
    confidenceText.textContent = `${(data.confidenceScore * 100).toFixed(2)}%`;
    languageText.textContent = data.language;
    explanationText.textContent = data.explanation;

    resultBody.classList.remove("hidden");
    setStatusSuccess();
    renderCharts(Number(data.confidenceScore), data.classification);
  } catch (error) {
    showError(error.message || "Unexpected error occurred.");
  } finally {
    submitBtn.disabled = false;
  }
});

resetStatus();
