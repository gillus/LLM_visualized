// [Aa] simplify-vs-detail toggle — two audiences, one build (see CLAUDE.md).
// Sets a `detail` class on <body>; CSS does all the showing/hiding:
//   .aa-detail  → only visible in detail mode (technical asides, raw numbers)
//   .aa-simple  → only visible in simple mode
// The choice persists in localStorage and is shared by both pages.
export function initAaToggle() {
  const KEY = "llmviz-aa";
  const btns = document.querySelectorAll(".aa-btn");
  if (!btns.length) return;

  const store = {
    get() { try { return localStorage.getItem(KEY); } catch { return null; } },
    set(v) { try { localStorage.setItem(KEY, v); } catch { /* private mode */ } },
  };

  const apply = (mode) => {
    document.body.classList.toggle("detail", mode === "detail");
    btns.forEach((b) => {
      const on = b.dataset.aaMode === mode;
      b.classList.toggle("active", on);
      b.setAttribute("aria-pressed", on);
    });
  };

  apply(store.get() === "detail" ? "detail" : "simple");
  btns.forEach((b) => (b.onclick = () => {
    store.set(b.dataset.aaMode);
    apply(b.dataset.aaMode);
    // detail asides change step heights — re-measure the scroll triggers
    window.__scroller?.resize();
  }));
}

// Friendly failure when a page's JSON can't be fetched — most often a teacher
// opening index.html straight from a folder (file://), where the browser
// blocks fetch. Without this the page is just silently blank.
export function showLoadError(err) {
  console.error(err);
  const box = document.createElement("div");
  box.className = "load-error";
  box.innerHTML =
    `<strong>Non è stato possibile caricare i dati di questa pagina.</strong> ` +
    `Se hai aperto il file direttamente da una cartella, il browser non gli ` +
    `permette di leggere i suoi file di dati. Fai invece girare il sito da un ` +
    `server: esegui <code>python3 -m http.server</code> dentro la cartella ` +
    `<code>web/</code>, poi apri <code>http://localhost:8000</code> — oppure ` +
    `usa il link online.`;
  document.body.prepend(box);
}
