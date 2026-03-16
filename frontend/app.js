/* AI Research Intelligence – Frontend Application */

const API_BASE = window.location.origin;

// Load topic chips on page load
document.addEventListener("DOMContentLoaded", async () => {
    try {
        const res = await fetch(`${API_BASE}/topics`);
        if (res.ok) {
            const topics = await res.json();
            renderTopicChips(topics);
        }
    } catch (e) {
        console.warn("Could not load topics:", e);
    }

    // Enter key sends query
    document.getElementById("query-input").addEventListener("keydown", (e) => {
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            sendQuery();
        }
    });
});

function renderTopicChips(topics) {
    const container = document.getElementById("topic-chips");
    container.innerHTML = "";
    topics.forEach((t) => {
        const chip = document.createElement("span");
        chip.className = "chip";
        chip.textContent = t.topic;
        chip.title = t.translation_hu;
        chip.addEventListener("click", () => {
            const input = document.getElementById("query-input");
            input.value = `What are the recent trends in ${t.topic}?`;
            input.focus();
        });
        container.appendChild(chip);
    });
}

async function sendQuery() {
    const input = document.getElementById("query-input");
    const query = input.value.trim();
    if (!query) return;

    const btn = document.getElementById("send-btn");
    const loading = document.getElementById("loading");
    const answerSection = document.getElementById("answer-section");
    const sourcesSection = document.getElementById("sources-section");

    // UI state: loading
    btn.disabled = true;
    loading.classList.remove("hidden");
    answerSection.classList.add("hidden");
    sourcesSection.classList.add("hidden");

    try {
        const res = await fetch(`${API_BASE}/query`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ query, top_k: 5 }),
        });

        if (!res.ok) {
            throw new Error(`API error: ${res.status}`);
        }

        const data = await res.json();
        renderAnswer(data);
    } catch (e) {
        document.getElementById("answer-text").textContent =
            `Error: ${e.message}. Please check that the API is running.`;
        answerSection.classList.remove("hidden");
    } finally {
        btn.disabled = false;
        loading.classList.add("hidden");
    }
}

function renderAnswer(data) {
    const answerSection = document.getElementById("answer-section");
    const answerText = document.getElementById("answer-text");
    const metaInfo = document.getElementById("meta-info");
    const sourcesSection = document.getElementById("sources-section");
    const sourcesList = document.getElementById("sources-list");

    // Answer
    answerText.textContent = data.answer;
    metaInfo.textContent = `${data.model} · ${Math.round(data.latency_ms)}ms · ${data.retrieval_count} retrieved → ${data.reranked_count} reranked`;
    answerSection.classList.remove("hidden");

    // Sources
    if (data.sources && data.sources.length > 0) {
        sourcesList.innerHTML = "";
        data.sources.forEach((s) => {
            const card = document.createElement("div");
            card.className = "source-card";

            const left = document.createElement("div");
            if (s.url) {
                const link = document.createElement("a");
                link.href = s.url;
                link.target = "_blank";
                link.rel = "noopener noreferrer";
                link.textContent = s.title;
                left.appendChild(link);
            } else {
                left.textContent = s.title;
            }

            const meta = document.createElement("span");
            meta.className = "source-meta";
            meta.textContent = ` · ${s.primary_topic} · ${s.published_date || ""}`;
            left.appendChild(meta);

            const badge = document.createElement("span");
            badge.className = "relevance-badge";
            badge.textContent = `${(s.relevance_score * 100).toFixed(0)}%`;

            card.appendChild(left);
            card.appendChild(badge);
            sourcesList.appendChild(card);
        });
        sourcesSection.classList.remove("hidden");
    }
}
