/*
 * Interface de RAG-Source : un client de l'API, rien de plus.
 *
 * Aucun framework ni étape de compilation : la page est lisible telle quelle, et
 * toute la logique métier reste dans l'API. Trois choix guident ce fichier :
 *
 * 1. la réponse arrive en flux (SSE) — sur CPU, attendre dix secondes devant un
 *    écran figé est la pire expérience possible ;
 * 2. les sources s'affichent avant la réponse, dès que la recherche a conclu ;
 * 3. le mode de souveraineté est visible en permanence, jamais enfoui dans un menu.
 *
 * L'authentification est portée par le reverse proxy (cf. Caddyfile) : le jeton
 * n'est donc jamais exposé au navigateur.
 */

const MAX_HISTORY = 5;
const EXCERPT_LIMIT = 320;

const form = document.getElementById("ask-form");
const questionField = document.getElementById("question");
const sourceField = document.getElementById("source");
const modeField = document.getElementById("mode");
const submitButton = document.getElementById("submit");
const resetButton = document.getElementById("reset");
const thread = document.getElementById("thread");
const errorBox = document.getElementById("error");
const template = document.getElementById("exchange-template");

/** Historique conversationnel envoyé à l'API (questions et réponses précédentes). */
let history = [];

init();

async function init() {
  await Promise.all([loadHealth(), loadDocuments()]);
  form.addEventListener("submit", onSubmit);
  resetButton.addEventListener("click", () => {
    history = [];
    thread.replaceChildren();
    questionField.focus();
  });
  questionField.addEventListener("keydown", (event) => {
    // Entrée envoie, Maj+Entrée passe à la ligne : le réflexe d'une messagerie.
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      form.requestSubmit();
    }
  });
  questionField.focus();
}

async function loadHealth() {
  const badge = document.getElementById("sovereignty");
  const corpus = document.getElementById("corpus");
  try {
    const health = await getJSON("/health");
    const external = health.sovereignty === "external";
    badge.textContent = external ? "⚠ LLM externe" : "100 % local";
    badge.classList.toggle("external", external);
    badge.title = external
      ? "Des extraits de vos documents sont envoyés à un service tiers."
      : "Aucune donnée ne quitte cette machine.";
    const chunks = health.indexed_chunks ?? 0;
    const degraded = health.status !== "ok";
    corpus.textContent = degraded
      ? `${chunks} extraits · service dégradé`
      : `${chunks} extraits indexés`;
  } catch (error) {
    badge.textContent = "API injoignable";
    badge.classList.add("down");
    corpus.textContent = String(error.message || error);
  }
}

async function loadDocuments() {
  try {
    const { documents } = await getJSON("/v1/documents");
    for (const document_ of documents) {
      const option = new Option(
        `${document_.source} (${document_.chunks})`,
        document_.source,
      );
      sourceField.add(option);
    }
  } catch {
    // Le filtre par document est un confort : son absence ne bloque pas l'usage.
  }
}

async function onSubmit(event) {
  event.preventDefault();
  const question = questionField.value.trim();
  if (!question) return;

  hideError();
  setBusy(true);
  const view = createExchange(question);
  questionField.value = "";

  try {
    await streamAnswer(question, view);
  } catch (error) {
    showError(error.message || String(error));
    view.status.textContent = "Réponse interrompue.";
    view.status.classList.add("warn");
  } finally {
    setBusy(false);
    questionField.focus();
  }
}

async function streamAnswer(question, view) {
  const response = await fetch("/v1/ask/stream", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      question,
      history: history.slice(-MAX_HISTORY).map(([q, a]) => ({ question: q, answer: a })),
      source: sourceField.value || null,
      mode: modeField.value,
    }),
  });
  if (!response.ok) {
    throw new Error(await describeFailure(response));
  }

  let answer = "";
  let passages = [];
  for await (const event of readEvents(response)) {
    if (event.name === "passages") {
      passages = event.data.passages;
      renderPassages(view, passages);
      view.status.textContent = passages.length
        ? "Rédaction de la réponse…"
        : "Aucun passage pertinent trouvé.";
    } else if (event.name === "token") {
      answer += event.data.text;
      renderAnswer(view, answer, []);
    } else if (event.name === "error") {
      throw new Error(event.data.detail);
    } else if (event.name === "done") {
      renderAnswer(view, answer, event.data.invalid_citations || []);
      finish(view, event.data, passages, answer);
    }
  }
  if (answer.trim()) {
    history.push([question, answer.trim()]);
  }
}

function finish(view, done, passages, answer) {
  const citations = done.citations || [];
  const invalid = done.invalid_citations || [];
  const parts = [];
  if (!passages.length) {
    parts.push("Aucun passage pertinent : le modèle n'a pas été sollicité.");
  } else if (citations.length) {
    parts.push(`${citations.length} source(s) citée(s) sur ${passages.length} fournie(s).`);
  } else if (answer.trim()) {
    parts.push("Réponse sans citation : à vérifier dans les sources ci-dessous.");
  }
  if (invalid.length) {
    parts.push(`Citations inexistantes signalées : ${invalid.join(", ")}.`);
  }
  view.status.textContent = parts.join(" ");
  view.status.classList.toggle("warn", invalid.length > 0 || !citations.length);
}

/** Découpe le flux SSE en événements { name, data }. */
async function* readEvents(response) {
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let index;
    while ((index = buffer.indexOf("\n\n")) !== -1) {
      const block = buffer.slice(0, index);
      buffer = buffer.slice(index + 2);
      const name = (block.match(/^event: (.*)$/m) || [])[1];
      const raw = (block.match(/^data: (.*)$/m) || [])[1];
      if (name && raw) {
        yield { name, data: JSON.parse(raw) };
      }
    }
  }
}

function createExchange(question) {
  const node = template.content.cloneNode(true);
  const article = node.querySelector(".exchange");
  article.querySelector(".question").textContent = question;
  const view = {
    article,
    answer: article.querySelector(".answer-text"),
    status: article.querySelector(".status-line"),
    sources: article.querySelector(".sources"),
    count: article.querySelector(".count"),
    passages: article.querySelector(".passages"),
  };
  view.status.textContent = "Recherche dans les documents…";
  view.sources.hidden = true;
  thread.prepend(article);
  return view;
}

/** Affiche la réponse en transformant les [n] en renvois cliquables. */
function renderAnswer(view, text, invalid) {
  view.answer.replaceChildren();
  const pattern = /\[(\d{1,2})\]/g;
  let cursor = 0;
  let match;
  while ((match = pattern.exec(text)) !== null) {
    view.answer.append(text.slice(cursor, match.index));
    const number = Number(match[1]);
    const link = document.createElement("a");
    link.className = "cite" + (invalid.includes(number) ? " invalid" : "");
    link.textContent = number;
    link.href = "#";
    link.title = invalid.includes(number)
      ? "Ce numéro ne correspond à aucun passage fourni."
      : "Voir le passage";
    link.addEventListener("click", (event) => {
      event.preventDefault();
      highlightPassage(view, number);
    });
    view.answer.append(link);
    cursor = match.index + match[0].length;
  }
  view.answer.append(text.slice(cursor));
}

function renderPassages(view, passages) {
  view.sources.hidden = passages.length === 0;
  view.count.textContent = `${passages.length} passage(s) utilisé(s)`;
  view.passages.replaceChildren();
  for (const passage of passages) {
    const item = document.createElement("li");

    const location = document.createElement("div");
    location.className = "location";
    location.textContent = `${passage.location} · score ${passage.score}`;
    item.append(location);

    const excerpt = document.createElement("p");
    excerpt.className = "excerpt";
    excerpt.textContent = passage.text;
    item.append(excerpt);

    if (passage.text.length > EXCERPT_LIMIT) {
      const toggle = document.createElement("button");
      toggle.type = "button";
      toggle.className = "toggle";
      toggle.textContent = "Afficher tout";
      toggle.addEventListener("click", () => {
        const open = item.dataset.open === "true";
        item.dataset.open = String(!open);
        toggle.textContent = open ? "Afficher tout" : "Réduire";
      });
      item.append(toggle);
    }
    view.passages.append(item);
  }
}

function highlightPassage(view, number) {
  const item = view.passages.children[number - 1];
  if (!item) return;
  view.sources.open = true;
  item.dataset.open = "true";
  item.scrollIntoView({ behavior: "smooth", block: "center" });
  item.animate(
    [{ background: "var(--accent-soft)" }, { background: "var(--surface)" }],
    { duration: 1200 },
  );
}

async function getJSON(path) {
  const response = await fetch(path);
  if (!response.ok) throw new Error(await describeFailure(response));
  return response.json();
}

async function describeFailure(response) {
  try {
    const body = await response.json();
    return body.detail || `Erreur HTTP ${response.status}`;
  } catch {
    return `Erreur HTTP ${response.status}`;
  }
}

function setBusy(busy) {
  submitButton.disabled = busy;
  submitButton.textContent = busy ? "…" : "Demander";
}

function showError(message) {
  errorBox.textContent = message;
  errorBox.hidden = false;
}

function hideError() {
  errorBox.hidden = true;
}
