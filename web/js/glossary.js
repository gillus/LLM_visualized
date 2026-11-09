// Shared glossary — one source of truth for every page. Definitions are
// written for learners (simple mode); the optional `detail` sentence carries
// the technical extra and is gated by the [Aa] toggle (.aa-detail), like the
// asides in the story pages. mountGlossary() appends a collapsible <details>
// block to <main>; the teacher's guide mounts it pre-opened so it prints.
export const TERMS = [
  { term: "Prompt / contesto",
    def: "Il testo che il modello ha letto finora. È l'unica cosa su cui si " +
         "basa la sua scommessa per la parola successiva — nessuna memoria " +
         "di altre conversazioni, nessuna consultazione esterna." },
  { term: "Token",
    def: "Un piccolo pezzo di testo — una parola intera o un frammento di " +
         "parola — che il modello legge e scrive. Tutto ciò che dice è " +
         "costruito incollando insieme dei token.",
    detail: "Prodotto da un tokenizzatore byte-pair-encoding (BPE); ogni " +
            "token ha un id numerico. Il vocabolario di questo modello ne " +
            "contiene 49.152." },
  { term: "Vocabolario",
    def: "L'elenco completo dei token che un modello conosce. Le parole " +
         "rare non hanno una voce propria — vengono costruite a pezzi " +
         "(osserva “hippopotamus” venire spezzata)." },
  { term: "Embedding",
    def: "L'elenco di numeri che assegna a un token il suo posto sulla " +
         "mappa dei significati. Le parole usate in modo simile ricevono " +
         "numeri simili, così finiscono vicine tra loro.",
    detail: "Qui ogni id di token indicizza un vettore a 576 dimensioni; " +
            "le mappe 2D che vedi sono proiezioni UMAP di quei vettori." },
  { term: "Mappa dei significati",
    def: "La nostra immagine piatta degli embedding. Essere vicini ha un " +
         "significato; le direzioni della bussola no." },
  { term: "Strato (layer)",
    def: "Un passo di elaborazione. Questo modello ne impila 30, e " +
         "ciascuno affina un po' i numeri di ogni token — da “che parola " +
         "è questa?” verso “cosa viene dopo?”.",
    detail: "Ogni strato è composto da attenzione più una piccola rete " +
            "feed-forward, con connessioni residue attorno a entrambe." },
  { term: "Stato nascosto (hidden state)",
    def: "I numeri di lavoro di un token tra uno strato e l'altro — il " +
         "suo “pensiero fin qui”. La fase Strati ne osserva uno spostarsi " +
         "sulla mappa." },
  { term: "Attenzione (attention)",
    def: "Il modo in cui un token guarda indietro ai token precedenti per " +
         "prendere ciò di cui ha bisogno — come “she” che controlla chi è " +
         "stato menzionato prima." },
  { term: "Testa di attenzione",
    def: "Uno dei diversi riflettori di attenzione che lavorano fianco a " +
         "fianco (9 per strato in questo modello). Ognuno ha la propria " +
         "abitudine: uno osserva la parola precedente, un altro va a " +
         "caccia di nomi, e così via." },
  { term: "Pozzo di attenzione (attention sink)",
    def: "L'abitudine del modello di parcheggiare l'attenzione in eccesso " +
         "sul primissimo token. Una linea spessa verso la prima parola di " +
         "solito significa “niente da guardare”, non un interesse reale." },
  { term: "Logit",
    def: "Il punteggio grezzo che il modello assegna a ogni possibile " +
         "token successivo, prima che i punteggi vengano trasformati in " +
         "percentuali." },
  { term: "Softmax",
    def: "La procedura che trasforma i punteggi grezzi in probabilità " +
         "che sommano al 100%." },
  { term: "Distribuzione di probabilità",
    def: "La tabella completa di “quanto è probabile ciascuna prossima " +
         "parola” — tutte le 49.152 insieme. Le barre nella fase " +
         "Previsione ne mostrano le prime, e “tutto il resto” rappresenta " +
         "il resto della tabella." },
  { term: "Temperatura",
    def: "La manopola del rischio usata per scegliere una parola. " +
         "Fredda: si gioca sul sicuro, quasi sempre la scelta migliore. " +
         "Calda: le scommesse si distribuiscono su parole improbabili — " +
         "più sorprendente, più insensato. A temperatura 0 vince sempre " +
         "la parola più probabile: la stessa frase, ogni volta.",
    detail: "Si realizza dividendo ogni logit per T prima del softmax." },
  { term: "Campionamento (sampling)",
    def: "Pescare la parola successiva a sorte, ma rispettando le " +
         "probabilità: le parole probabili escono spesso, quelle rare ogni " +
         "tanto. È il motivo per cui, a temperatura alta, la stessa frase " +
         "può continuare in modi diversi ogni volta." },
  { term: "Logit lens",
    def: "Il trucco che ci permette di sbirciare la migliore ipotesi del " +
         "modello dopo ogni strato, non solo dopo l'ultimo.",
    detail: "Riutilizza la normalizzazione finale e la testa di output " +
            "del modello a ogni strato; l'interruttore “Affinata” " +
            "aggiunge un piccolo traduttore addestrato che ripulisce gli " +
            "strati intermedi." },
  { term: "Neurone",
    def: "Il mattoncino di una rete neurale: pesa i numeri che riceve, li " +
         "somma e “schiaccia” il risultato tra 0 e 1 con una curva a S. " +
         "Da solo sa fare una sola curva a S; tanti neuroni a strati " +
         "imparano legami complicati. Provalo nella scheda “Rete neurale”.",
    detail: "Qui: attivazione sigmoide; nei modelli linguistici moderni si " +
            "usano curve simili (es. SiLU) nella parte feed-forward." },
  { term: "Dati di verifica (test)",
    def: "Esempi tenuti da parte che il modello non vede mai mentre impara. " +
         "Servono a controllare se ha capito davvero o se ha solo " +
         "imparato a memoria — come un compito con esercizi nuovi." },
  { term: "Pesi (weights)",
    def: "I numeri dentro il modello che l'addestramento modifica — 135 " +
         "milioni in questo modello piccolo. Tutto ciò che “sa” è " +
         "memorizzato in essi, non in un database." },
  { term: "Pre-addestramento (pretraining)",
    def: "La prima scuola: leggere una montagna di testo e giocare a " +
         "indovina-il-prossimo-token, miliardi di volte. Il risultato è " +
         "un autocompletamento di livello mondiale — non ancora un " +
         "assistente." },
  { term: "Fine-tuning (SFT)",
    def: "La seconda scuola: studiare conversazioni di esempio scritte " +
         "da persone, e copiarne la forma — quando viene fatta una " +
         "domanda, rispondere.",
    detail: "Fine-tuning supervisionato su coppie istruzione → risposta; " +
            "lo stesso addestramento a prevedere il prossimo token, ma " +
            "con materiale nuovo." },
  { term: "Apprendimento per rinforzo (RL)",
    def: "La terza scuola: scrivere risposte, ricevere un voto, e " +
         "spostarsi verso il tipo di risposte che ottiene punteggi " +
         "migliori. Il voto può venire da persone o da un correttore che " +
         "verifica la risposta.",
    detail: "RLHF quando la ricompensa è la preferenza umana; RLVR " +
            "quando è un controllo verificabile come “il numero è " +
            "giusto?”." },
  { term: "Modello base vs Instruct",
    def: "Stesso cervello, scuola diversa. Il modello base si limita a " +
         "continuare il testo; il modello Instruct è stato addestrato a " +
         "rispondere. La scheda “Come si addestra” li mostra entrambi, " +
         "per davvero." },
];

export function mountGlossary({ open = false } = {}) {
  const host = document.querySelector("main");
  if (!host || document.querySelector(".glossary")) return;
  const d = document.createElement("details");
  d.className = "glossary";
  if (open) d.open = true;
  d.innerHTML =
    `<summary>Elenco dei termini — le parole di questo sito, in linguaggio semplice</summary>` +
    `<dl class="gloss-list">` +
    TERMS.map((t) =>
      `<div class="gloss-item"><dt>${t.term}</dt><dd>${t.def}` +
      (t.detail ? ` <span class="aa-detail gloss-extra">${t.detail}</span>` : "") +
      `</dd></div>`).join("") +
    `</dl>`;
  host.appendChild(d);
}
