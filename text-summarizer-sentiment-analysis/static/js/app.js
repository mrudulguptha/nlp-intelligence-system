const textInput = document.getElementById("textInput");
const analyzeBtn = document.getElementById("analyzeBtn");
const sampleBtn = document.getElementById("sampleBtn");
const errorText = document.getElementById("errorText");
const loadingBox = document.getElementById("loadingBox");

const sentimentResult = document.getElementById("sentimentResult");
const sentimentModel = document.getElementById("sentimentModel");
const spamResult = document.getElementById("spamResult");
const summaryResult = document.getElementById("summaryResult");

function setLoading(isLoading) {
    analyzeBtn.disabled = isLoading;
    sampleBtn.disabled = isLoading;
    loadingBox.classList.toggle("hidden", !isLoading);
}

function setError(message) {
    errorText.textContent = message || "";
}

function renderResult(data) {
    sentimentResult.textContent = data.sentiment;
    sentimentModel.textContent = `Model: ${data.sentiment_model}`;
    spamResult.textContent = data.spam;
    summaryResult.textContent = data.summary;
}

async function analyzeText() {
    const text = textInput.value.trim();

    if (!text) {
        setError("Please enter some text before analysis.");
        return;
    }

    setError("");
    setLoading(true);

    try {
        const response = await fetch("/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ text }),
        });

        const payload = await response.json();
        if (!response.ok) {
            throw new Error(payload.error || "Unable to process your request.");
        }

        renderResult(payload);
    } catch (error) {
        setError(error.message || "Unexpected error. Please try again.");
    } finally {
        setLoading(false);
    }
}

function loadSampleText() {
    textInput.value =
        "The institute launched a new NLP lab this year. Students appreciated the teaching quality and project mentorship. However, email phishing messages are still common, so awareness training is needed.";
    setError("");
    textInput.focus();
}

analyzeBtn.addEventListener("click", analyzeText);
sampleBtn.addEventListener("click", loadSampleText);
textInput.addEventListener("keydown", (event) => {
    if ((event.ctrlKey || event.metaKey) && event.key === "Enter") {
        analyzeText();
    }
});
