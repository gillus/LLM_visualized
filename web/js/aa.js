// The site is simple-mode only: content marked .aa-detail (technical asides,
// raw numbers) stays in the markup but is permanently hidden by CSS.

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
