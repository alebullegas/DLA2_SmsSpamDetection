# 🛡️ DLA2: SMS SPAM DETECTION 2024/25 - UNICA
<p align="center">
  <a href="https://www.apache.org/licenses/LICENSE-2.0" target="_blank">
    <img src="https://img.shields.io/badge/License-Apache_2.0-4285F4?style=for-the-badge&logo=none&logoColor=white" alt="Apache License 2.0"/>
  </a>
  <a href="https://lmstudio.ai/" target="_blank">
    <img src="https://img.shields.io/badge/Inference-LM_Studio-5A29E4?style=for-the-badge&logo=openai&logoColor=white" alt="LM Studio"/>
  </a>
  <a href="https://unsloth.ai/" target="_blank">
    <img src="https://img.shields.io/badge/Training-Unsloth_AI-000000?style=for-the-badge&logo=huggingface&logoColor=white" alt="Unsloth"/>
  </a>
  <a href="https://python.langchain.com/" target="_blank">
    <img src="https://img.shields.io/badge/Orchestration-LangChain-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white" alt="LangChain"/>
  </a>
</p>

---
> ## 📑 Indice
> 01. [🧑🏻‍🎓 Studente](#studente)  
> 02. [📌 Descrizione Progetto](#descrizione)  
> 03. [🗃️Descrizione Dataset](#dataset)
> 04. [📄 Panoramica File](#panoramica-file)
> 05. [📁 Struttura del Progetto](#struttura-progetto)  
> 06. [🛠️ Stack Tecnologico](#stack-tecnologico)  
> 07. [🚀 Installazione](#installazione)  
> 08. [🧪 Processo di Fine-Tuning](#fine-tuning)  
> 09. [📊 Benchmark e Confronto](#benchmark)  
> 10. [📈 Metriche e Risultati](#metriche)  
> 11. [🖥️ Hardware e Limitazioni](#hardware)  
> 12. [📝 Licenze](#licenze)

---

## 1. 🧑🏻‍🎓 Studente <a name="studente"></a>
> **Alessandro Bullegas**
> - **Matricola:** 60/73/65307
> - **Email:** alebullegas31@gmail.com
>
> - ---


## 2. 📌 Descrizione Progetto <a name="descrizione"></a>

Questo progetto nasce come studio sperimentale per analizzare i trade-off tra **dimensione del modello**, **capacità di ragionamento** e **specializzazione del dominio** nel contesto della Spam Detection.

I Large Language Models (LLM) classici sono strumenti molto potenti ma, spesso, peccano di velocità e stretta aderenza alle richieste dei task che vengono affidati, come ad esempio nel nostro caso filtrare SMS malevoli.

Utilizzando il dataset pubblico **SMS Spam Collection**, il progetto mette a confronto tre filosofie diverse:

1.  **Zero-Shot Generalist:** `Llama 3.2 Instruct`. Un modello leggero e generico, testato sulla sua capacità di riconoscere lo spam senza addestramento specifico
2.  **Reasoning Model:** `DeepSeek-R1-Distill`. Un modello progettato per "pensare" prima di rispondere. Testiamo se il ragionamento logico aiuta a scovare tentativi di phishing più sottili o se aggiunge solo latenza inutile
3.  **Domain Specialist:** `Llama 3.2 Fine-Tuned`. La versione custom, addestrata specificamente per questo task

### 🎯 Obiettivo
Dimostrare che un **modello piccolo ma specializzato** può superare modelli più complessi o "ragionanti" in task verticali, offrendo:
* ✅ **Latenza Minore**
* ✅ **Accuratezza Superiore**

## 3. 📂 Descrizione Dataset <a name="dataset"></a>

Il progetto utilizza l'**SMS Spam Collection**, un dataset pubblico disponibile presso l'UCI Machine Learning Repository.
Si tratta di un set di **messaggi SMS** reali, etichettati come legittimi o indesiderati.

| Etichetta | Percentuale | Descrizione |
| :--- | :--- | :--- |
| **HAM** 🟢 | **76.6%** | Messaggi normali, conversazioni personali, notifiche legittime. |
| **SPAM** 🔴 | **23.4%** | Phishing, truffe, pubblicità aggressiva, vincite false. |

> **Nota Tecnica:** Durante la fase di preparazione (`split_dataset.py`), vengono rinominate le colonne originali (`v1`, `v2`) in `label` e `text` per chiarezza e rimosso eventuali colonne vuote sporche presenti nel CSV originale.

### 📝 Esempi dal Dataset

Ecco come appaiono i dati grezzi che il modello deve imparare a distinguere:

| Tipo | Esempio di Testo|
| :--- | :--- |
| **SPAM** | *"Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry question..."* |
| **SPAM** | *"URGENT! You have won a 1 week FREE membership in our 100,000 Prize Jackpot! Txt the word: CLAIM to No: 81010..."* |
| **HAM** | *"Go until jurong point, crazy.. Available only in bugis n great world la e buffet..."* |
| **HAM** | *"Ok lar... Joking wif u oni..."* |

---

## 4. 📄 Panoramica File <a name="panoramica-file"></a>

| File | Tipo | Descrizione |
| :--- | :--- | :--- |
| `split_dataset.py` | 🐍 Script | Lo script si occupa di pulire il dataset raw (`spam.csv`), mescolarlo e dividerlo rigorosamente in Training Set (80%) e Test Set (20%) per evitare *Overfitting*. |
| `model_evaluation.py` | 🐍 Script | Lo script che interroga LM Studio, misura la latenza e calcola le metriche sui modelli. |
| `train_unsloth.jsonl` | 📄 Dati | Il file JSONL formattato contenente solo gli esempi per l'addestramento da utilizzare su Colab. |
| `test_benchmark.csv` | 📄 Dati | La parte restante del dataset usato solo per la valutazione finale. |
| `Finetuning_Spam.ipynb` | 📓 Notebook | Il notebook Colab che esegue l'addestramento QLoRA e l'esportazione GGUF. |


## 5. 📁 Struttura del Progetto <a name="struttura-progetto"></a>

```plaintext
├── 📁 data/                      # Contiene i dataset (Raw, Train, Test)
│   ├── spam.csv                  # Dataset originale
│   ├── train_unsloth.jsonl       # Dataset formattato per il training
│   └── test_benchmark.csv        # Dataset riservato per il test
│
├── 📁 models/                    # Cartella per i modelli GGUF
│   └── Llama-3.2-3B-Instruct.Q4_K_M.gguf # Il modello Fine-Tunato
│
├── 📁 notebooks/                  # Codice per il fine tuning
│   └── Finetuning_Spam.ipynb      # Notebook Google Colab
│
├── 📁 src/                       # Codice sorgente Python
│   ├── model_evaluation.py       # Script di validazione
│   └── split_dataset.py          # Script di preparazione dati
│
├── 📁 results/                   # Output dei test
│   └── risultati_benchmark.csv   # Risultati grezzi per ogni SMS
│
└── README.md                     # Documentazione
```

## 6. 🛠️ Stack Tecnologico <a name="stack-tecnologico"></a>
### 🟣 LM Studio
**LM Studio** non viene usato come semplice interfaccia grafica, ma come vero e proprio **Server Locale**.
* **Ruolo Architetturale:** LM Studio carica i modelli e sfrutta la GPU/CPU del pc per eseguire i calcoli.
* **Integrazione API:** La funzionalità chiave utile per il progetto è il suo **Local Server** compatibile con le specifiche OpenAI (`http://localhost:1234/v1`). Questo ci permette di disaccoppiare il modello dallo script Python: possiamo sostituire il "motore" ( passando da un modello ad un altro) in tempo reale senza modificare il codice.

### 🦜🔗 LangChain
**LangChain** funge come livello di astrazione logica tra il nostro codice Python e il modello linguistico.
* **Prompt Templating:** Gestisce la costruzione dinamica dei messaggi, inserendo il `System Prompt` (le regole) e lo `User Prompt` (l'SMS da analizzare) nel formato corretto atteso dal modello.
* **Output Parsing:** Utilizzando `StrOutputParser`, LangChain intercetta la risposta grezza dell'LLM e la pulisce da eventuali meta-tag o spazi bianchi, garantendo che il dato salvato nel CSV sia pulito e pronto per l'analisi.

### 🦥 Unsloth AI
Per la fase di Fine-Tuning su Google Colab, viene utilizzata la libreria **Unsloth**, che rappresenta uno strumento fondamentale per migliorare l'efficienza nell'addestramento degli LLM.
* **Perché è essenziale:** Il Fine-Tuning tradizionale di Llama 3 richiederebbe GPU potentissime (A100, 40GB VRAM).
* **Innovazione Tecnica:** Unsloth implementa kernel PyTorch riscritti per l'ottimizzazione e utilizza la tecnica **QLoRA** (Quantized Low-Rank Adaptation).
* **Risultato:** Questo stack ci ha permesso di addestrare un modello da 3 miliardi di parametri su una GPU Tesla T4 gratuita (16GB VRAM), riducendo i tempi di training di 2x e l'occupazione di memoria del 60%.

## 7. 🚀 Installazione <a name="installazione"></a>
 
La procedura è divisa in due parti: configurazione del **Codice Python** e configurazione di **LM Studio**.

---

### Parte A: Configurazione Python

#### 1. Prerequisiti
Assicurati di avere installato **Python 3.10** (o superiore) e **Git**.  
Puoi verificarlo aprendo il terminale e digitando:

    python --version

---

#### 2. Clona il Repository
Scarica il progetto sul tuo computer:

    git clone https://github.com/alebullegas/DLA2_SmsSpamDetection
    cd DLA2-Spam-Detection

---

#### 3. Crea l'Ambiente Virtuale

**Su Windows:**

    python -m venv venv
    .\venv\Scripts\activate

**Su Mac/Linux:**

    python3 -m venv venv
    source venv/bin/activate

---

#### 4. Installa le Dipendenze
Esegui il comando per installarle tutte le dipendenze:

    pip install -r requirements.txt

---

### Parte B: Configurazione LM Studio

Per far funzionare gli script, **LM Studio** deve agire come un server locale.

#### 1. Scarica e Installa
Scarica LM Studio da **https://lmstudio.ai** e installalo.

---

#### 2. Carica il Modello
- Apri **LM Studio**
- Clicca sulla **Lente d'ingrandimento (Search)**
- Cerca il modello che vuoi testare
- Scarica la versione quantizzata **Q4_K_M** (consigliata)

**Nota:** Per usare il modello Fine-Tuned del progetto (file `.gguf`), trascinalo semplicemente nella cartella dove vengono installati gli altri modelli di LM Studio

---

#### 3. Avvia il Server
- Clicca sull'icona Developer nella barra laterale sinistra
- Assicurati che l'opzione **Cross-Origin-Resource-Sharing (CORS)** sia attiva
- Clicca sul pulsante verde **"Start Server"**
- Seleziona il modello che ti interessa nella barra **Select a model to load**

---

✅ **Fatto!**  
Ora il tuo computer è pronto ad eseguire gli script e risponde all'indirizzo:

    http://localhost:1234

**Nota:** Esegui prima lo script `split_dataset.py` per organizzare i dati e solo dopo lo script `model_evaluation.py` per testare il modello.

## 8. 🧪 Processo di Fine-Tuning <a name="fine-tuning"></a>

Il processo di addestramento viene eseguito su **Google Colab** sfruttando una GPU **NVIDIA Tesla T4 (16GB VRAM)**.
Dato che il Fine-Tuning completo di un modello da 3 Miliardi di parametri richiederebbe risorse hardware proibitive, è stata adottata la tecnica **QLoRA** (Quantized Low-Rank Adaptation) tramite la libreria **Unsloth**.

### 📋 Workflow di Addestramento
Il notebook `Finetuning_Spam.ipynb` esegue automaticamente i seguenti passaggi:

---

#### 1. Setup dell'Ambiente
Installazione di **Unsloth**, **Xformers** e **TRL** (Transformer Reinforcement Learning).

---

#### 2. Caricamento Dati
Il notebook carica il file `train_unsloth.jsonl` (generato nel `split_dataset.py`) e applica il **Chat Template** standard di Llama 3:

    <|start_header_id|>system<|end_header_id|>
    You are a Cybersecurity AI...
    <|start_header_id|>user<|end_header_id|>
    [SMS TEXT]
    <|start_header_id|>assistant<|end_header_id|>
    SPAM

---

#### 3. Configurazione Iperparametri
Sono stati impostati i seguenti parametri per massimizzare la stabilità su questo tipo di dataset:

- **Learning Rate:** `2e-4` (standard per QLoRA)
- **Max Steps:** `60` 
- **Batch Size:** `2` 
- **Data Collator:** `train_on_responses_only`  
  Il modello impara a generare solo la risposta, ignorando il prompt utente nel calcolo della loss.

---

#### 4. Conversione ed Esportazione (GGUF)
Al termine del training, gli adattatori **LoRA** sono stati fusi con il modello base.  
Il risultato finale è stato convertito nel formato **GGUF** con quantizzazione **Q4_K_M (4-bit Medium)**.

**Perché proprio Q4_K_M?**  
Rappresenta il compromesso ideale: riduce il peso del modello a ~2GB con una perdita di precisione tranquillamente trascurabile, rendendolo eseguibile su qualsiasi laptop o computer non molto potente.

## 9. 📊 Benchmark e Confronto <a name="benchmark"></a>

Per valutare oggettivamente le performance, non bastava "chattare" manualmente con i modelli quindi è stato creato uno **script di valutazione automatizzato** (`src/model_evaluation.py`) che garantisce che ogni modello venga testato esattamente nelle stesse condizioni.

---

### ⚙️ Metodologia di Test

1. **Stesso Dataset:**  
   Tutti i modelli vengono valutati sul file `test_benchmark.csv` (il 20% dei dati, mai visti durante il training).

2. **Stesso Prompt:**  
   Utilizziamo lo stesso identico *System Prompt* per tutti i modelli, per valutare chi obbedisce meglio alle istruzioni.

---

### 🐍 Analisi del Codice (`model_evaluation.py`)

Lo script orchestra l’intero processo di valutazione.

---

#### 1. Connessione al Server Locale

Connessione a **LM Studio** come se fosse una API remota:

    llm = ChatOpenAI(
        base_url="http://localhost:1234/v1",  # Server locale di LM Studio
        api_key="lm-studio",                  # Placeholder (non serve una vera API key)
        temperature=0.0                       # Determinismo assoluto
    )

Questo approccio permette di **cambiare modello direttamente da LM Studio senza modificare una sola riga di codice Python.

---

#### 2. La Chain (LangChain Expression Language)

Utilizziamo la sintassi **LCEL** (pipe syntax `|`) per definire il flusso di elaborazione in modo chiaro e leggibile:

    # Definizione della struttura: Prompt -> Modello -> Parser
    chain = prompt | llm | StrOutputParser()

- **Prompt:** Inserisce l’SMS nel template di sicurezza  
- **LLM:** Invia la richiesta al server locale  
- **Parser:** Pulisce l’output (spazi, newline, caratteri indesiderati), restituendo una stringa finale pulita

---

#### 3. Loop di Valutazione e Misurazione della Latenza

Il cuore dello script è un ciclo `for` che itera su ogni riga del dataset di test.  
Per ogni SMS, misuriamo il tempo di risposta del modello:

    start_time = time.time()          # ⏱️ Avvio cronometro
    response = chain.invoke(...)      # Chiamata al modello
    end_time = time.time()            # 🛑 Stop cronometro

    latency = end_time - start_time   # Calcolo della latenza

---

#### 4. Parsing "Strict" (Controllo Rigoroso)

Poiché l’automazione richiede risposte precise, il codice verifica che l’output contenga **esattamente** le parole chiave attese:

    cleaned_resp = response.lower().strip()

    if "spam" in cleaned_resp:
        predicted = "spam"
    elif "ham" in cleaned_resp:
        predicted = "ham"
    else:
        # Se il modello divaga o non risponde correttamente,
        # viene segnato come errore o gestito con logica di fallback.
        predicted = "error"

Questo garantisce una valutazione **robusta e imparziale**, penalizzando risposte ambigue o fuori specifica.

---

### 🕹️ Come Eseguire il Benchmark

Per replicare i nostri risultati, segui questa procedura per **ogni modello** che vuoi testare.

#### 1. Prepara LM Studio
- Carica il modello desiderato (es. **Llama 3.2 Fine-Tuned**)  
- Avvia il **Local Server** cliccando sul pulsante verde **"Start Server"**

#### 2. Esegui lo Script
Apri il terminale nella cartella del progetto ed esegui:

    python src/benchmark_runner.py

#### 3. Leggi i Risultati
Lo script stamperà a video le metriche in tempo reale e salverà un file CSV dettagliato:

    🏆 RISULTATI DEL BENCHMARK
    ============================================================
    🎯 ACCURACY:       99.10%
    ⚡ VELOCITÀ MEDIA: 0.2105 secondi/messaggio

#### 4. Cambia e Ripeti
Torna su **LM Studio**, ferma il server, carica un altro modello (es. *DeepSeek R1*), riavvia il server ed esegui nuovamente lo script.

## 9. 📈 Metriche e Risultati <a name="metriche"></a>

In questa sezione presentiamo i risultati quantitativi ottenuti sul **Test Set (1.115 messaggi, pari al 20% del totale)**. Questi dati sono stati isolati prima del training e non sono mai stati visti dal modello durante la fase di apprendimento.

Il benchmark ha confrontato tre architetture distinte per valutare l'impatto del Fine-Tuning rispetto alle capacità native dei modelli.

### 9.1 Tabella Riepilogativa delle Performance

| Modello | Ruolo | Accuracy | Precision (Spam) | Recall (Spam) | Latenza (avg) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Llama-3.2-3B Fine-Tuned** | **Specialist** | **93.09%** | **0.68** | **0.98** | **0.23s** |
| **DeepSeek-R1-Distill (8B)** | Reasoning | 30.04% | 0.14 | 0.76 | 7.41s |
| **Llama-3.2-3B Base** | Baseline | 19.28% | 0.15 | 1.00 | 0.26s |

> **Nota Metodologica:**
> * **Accuracy:** Percentuale di risposte corrette sul totale.
> * **Recall (Spam):** Capacità di intercettare le minacce reali (evitare falsi negativi).
> * **Latenza:** Tempo medio di risposta per singolo messaggio (Hardware: Consumer GPU).

---

### 9.2 Analisi Critica dei Risultati

Di seguito analizziamo il comportamento di ogni modello, spiegando le cause tecniche delle performance.

#### 🏆 1. Llama-3.2-3B Instruct (Fine-Tuned) - Il Vincitore
Il modello addestrato con tecnica **QLoRA** ha raggiunto l'obiettivo del progetto, dimostrando un equilibrio eccellente tra sicurezza e usabilità.
* **Sicurezza Massima (Recall 98%):** Su 157 messaggi di spam reali nel test set, il modello ne ha bloccati correttamente **154**. Ne ha lasciati passare solo 3. Questo lo rende un filtro estremamente robusto.
* **Usabilità (Accuracy 93%):** A differenza degli altri modelli, il Fine-Tuned ha imparato a riconoscere i messaggi legittimi (HAM), riducendo drasticamente i falsi allarmi.
* **Efficienza:** Con **0.23 secondi** di risposta, è idoneo per applicazioni real-time.

#### ⚠️ 2. DeepSeek-R1-Distill-Llama (8B) - Il Paradosso del "Reasoning"
Nonostante sia un modello più grande (8B) e dotato di capacità di ragionamento (*Chain of Thought*), ha performato male (Accuracy 30%).
* **Il problema dell'Over-Thinking:** Il modello tende a "sovra-analizzare" i messaggi. Il processo di ragionamento lo porta a vedere potenziali inganni anche in messaggi innocui, generando un numero elevatissimo di **Falsi Positivi (743)**.
* **Latenza Inaccettabile:** Richiede in media **7.41 secondi** per messaggio, rendendolo 32 volte più lento del modello specializzato.
* **Conclusione:** I modelli *Reasoning* non sono adatti a task di classificazione rapida/binaria senza un prompt engineering estremo.

#### ❌ 3. Llama-3.2-3B Instruct (Base) - La Baseline
Il modello base (senza fine-tuning) mostra perché l'addestramento era necessario.
* **Mode Collapse (Paranoia):** Il modello ha classificato quasi l'intero dataset come SPAM.
* **Il falso mito della Recall 100%:** Sebbene abbia intercettato tutti gli spam (Recall 1.00), lo ha fatto bloccando anche **900 messaggi legittimi su 958**. Questo rende il modello inutilizzabile in uno scenario reale, poiché l'utente non riceverebbe quasi più messaggi.

---

### 9.3 Allegato Tecnico: Report Dettagliati
Per trasparenza scientifica, riportiamo i raw data generati dallo script di benchmark (`benchmark_runner.py`).

**A. Modello Fine-Tuned (Specialist)**
Si nota la diagonale principale della matrice molto popolata (risposte corrette).
    
    Matrice di Confusione:
    [[154 (TP)    3 (FN)]   <-- Ottima intercettazione Spam
     [ 74 (FP)  884 (TN)]]  <-- Buona distinzione Ham

                  precision    recall  f1-score   support
            spam       0.68      0.98      0.80       157
             ham       1.00      0.92      0.96       958
        accuracy                           0.93      1115

**B. Modello DeepSeek R1 (Reasoning)**
Alto numero di Falsi Positivi (743) che distrugge l'Accuracy.

    Matrice di Confusione:
    [[ 120 (TP)   37 (FN)]
     [ 743 (FP)  215 (TN)]] <-- Troppi messaggi buoni bloccati

                  precision    recall  f1-score   support
            spam       0.14      0.76      0.24       157
             ham       0.85      0.22      0.35       958
        accuracy                           0.30      1115

**C. Modello Base (Zero-Shot)**
Il modello ha predetto quasi solo "SPAM", ignorando la classe "HAM".

    Matrice di Confusione:
    [[157 (TP)    0 (FN)]
     [900 (FP)   58 (TN)]] <-- Disastroso sui messaggi legittimi

                  precision    recall  f1-score   support
            spam       0.15      1.00      0.26       157
             ham       1.00      0.06      0.11       958
        accuracy                           0.19      1115


## 10. 🖥️ Hardware e Limitazioni <a name="hardware"></a>

> [!NOTE]
> 🧪 Tutto il processo di training e valutazione è stato condotto con risorse accessibili per dimostrare la scalabilità della soluzione.

| Fase | Hardware Utilizzato | Note Tecniche |
| :--- | :--- | :--- |
| **Training** (Cloud) | **NVIDIA Tesla T4** (16GB VRAM) | Ambiente **Google Colab Free Tier**. Grazie alla quantizzazione 4-bit, il picco di memoria è rimasto sotto i 6GB. |
| **Inference** (Locale) | **Laptop Consumer** (CPU/GPU Integrata) | Modello eseguito via **LM Studio**. La latenza media di ~0.2s dimostra che non serve hardware enterprise per l'inferenza. |

> [!WARNING]
> **Limitazioni del Modello:**
> - Il modello Fine-Tuned è specializzato verticalmente sullo SPAM. Se interrogato su argomenti generali (es. "Qual è la capitale della Francia?"), potrebbe cercare di classificare anche quella domanda come HAM/SPAM o rispondere in modo breve. È un *Specialist*, non un *Generalist*.
> - Il dataset contiene principalmente SMS in lingua inglese. Le performance su SMS in italiano non sono garantite senza un ulteriore fine-tuning multilingua.

---

## 11. 📝 Licenze <a name="licenze"></a>
> [!NOTE]
> **Code**: Il codice sorgente di questo repository è rilasciato sotto licenza [Apache License 2.0](./LICENSE).
>
> **Models**: Il modello base `Llama-3.2` è soggetto alla **Meta Llama Community License**. `DeepSeek-R1` è soggetto alla licenza open source rispettiva.
>
> **Dataset**: L'SMS Spam Collection è di pubblico dominio (UCI Machine Learning Repository).

---

