# 🛡️ SMS SPAM DETECTION: LLM BENCHMARK STUDY
### Comparative Analysis: Base vs Reasoning vs Fine-Tuned Models

<p align="center">
  <img src="https://img.shields.io/badge/Language-Python_3.12-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/Framework-LangChain-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white" alt="LangChain"/>
  <img src="https://img.shields.io/badge/Inference-LM_Studio-5A29E4?style=for-the-badge&logo=openai&logoColor=white" alt="LM Studio"/>
  <img src="https://img.shields.io/badge/Training-Unsloth_AI-000000?style=for-the-badge&logo=huggingface&logoColor=white" alt="Unsloth"/>
</p>

---

## 1. 📖 Project Overview <a name="overview"></a>

Questo progetto universitario si propone di analizzare e confrontare le performance di diverse architetture di **Large Language Models (LLM)** applicate a un task specifico di Cybersecurity: **il rilevamento di SMS Spam**.

L'obiettivo non è solo classificare messaggi, ma condurre uno studio comparativo (**Benchmark**) tra tre approcci fondamentali nell'AI moderna:
1.  **Prompt Engineering** su modelli generalisti.
2.  **Reasoning Models** (modelli che "pensano" prima di rispondere).
3.  **Fine-Tuning** specifico su modelli di piccole dimensioni.

Il progetto dimostra come un modello piccolo (3B) ma specializzato possa superare in efficienza e accuratezza modelli generalisti più complessi in task verticali.

---

## 2. 🛠️ Tech Stack & Tools <a name="tech-stack"></a>

Per realizzare questo ecosistema locale e scalabile, sono stati utilizzati strumenti all'avanguardia:

### 🧠 LM Studio (The Engine)
Utilizziamo **LM Studio** come motore di inferenza locale.
* **Perché:** Permette di eseguire modelli quantizzati (GGUF) su hardware consumer (Laptop con 16GB RAM) senza dipendere dal cloud.
* **Funzione Chiave:** Espone un server locale compatibile con le API di OpenAI (`http://localhost:1234/v1`). Questo ci permette di cambiare modello "al volo" senza modificare una riga di codice Python.

### 🦜🔗 LangChain (The Orchestrator)
Il framework **LangChain** gestisce la logica applicativa.
* **Perché:** Astrazione necessaria per gestire i template dei prompt e le catene di esecuzione.
* **Funzione Chiave:** Ci permette di strutturare i prompt in modo modulare (`SystemMessage` + `UserMessage`) e di parsare le risposte dei modelli in modo programmatico, garantendo che l'output sia pulito per l'analisi dati.

### 🦥 Unsloth AI (The Trainer)
Utilizzato su **Google Colab** per il processo di Fine-Tuning.
* **Perché:** Unsloth ottimizza il backpropagation rendendo l'addestramento 2x più veloce e riducendo il consumo di VRAM del 60%.
* **Risultato:** Ci ha permesso di addestrare un modello Llama 3.2 su GPU T4 (gratuita) in meno di 20 minuti.

---

## 3. 🤖 The Challengers: Selected Models <a name="models"></a>

Il benchmark mette a confronto tre "filosofie" di AI diverse:

| Modello | Tipo | Ruolo nel Test | Descrizione |
| :--- | :--- | :--- | :--- |
| **Llama 3.2 3B Instruct** | *Generalist* | **Baseline** | Il modello base di Meta. Viene testato per vedere quanto bene riesce a seguire le istruzioni "Zero-Shot" (senza esempi) usando solo il Prompt Engineering. |
| **DeepSeek R1 (Distill)** | *Reasoning* | **The Thinker** | Un modello avanzato che utilizza una catena di pensiero (Chain-of-Thought) prima di rispondere. Testato per vedere se la logica superiore aiuta a identificare spam sottili. |
| **Llama 3.2 3B Fine-Tuned** | *Specialist* | **The Expert** | La nostra versione custom, addestrata specificamente sul dataset SMS Spam. Testato per dimostrare la superiorità della specializzazione sulla generalizzazione. |

---

## 4. 🎓 The Fine-Tuning Process (Deep Dive) <a name="finetuning"></a>

Il cuore del progetto è la creazione del modello "Specialista". Ecco come abbiamo trasformato un modello generico in un esperto di Cybersecurity.

### Phase 1: Data Preparation 🧹
Abbiamo utilizzato il dataset pubblico **SMS Spam Collection**.
1.  **Cleaning:** Rimozione di caratteri non UTF-8 e normalizzazione.
2.  **Formatting:** Conversione dal formato CSV raw al formato **ShareGPT (JSONL)**, essenziale per i modelli Chat.
    * *Esempio Struttura:*
        ```json
        {"conversations": [
          {"role": "system", "content": "You are a Cybersecurity AI..."},
          {"role": "user", "content": "URGENT! You won a FREE iPhone..."},
          {"role": "assistant", "content": "SPAM"}
        ]}
        ```

### Phase 2: Training on Google Colab ☁️
Non avendo una GPU potente in locale, abbiamo sfruttato il cloud.
* **Environment:** Google Colab (T4 GPU).
* **Method:** **QLoRA** (Quantized Low-Rank Adaptation). Invece di riaddestrare tutto il modello (che richiederebbe cluster enormi), abbiamo addestrato solo piccoli "adattatori" (l'1% dei parametri) sopra il modello base a 4-bit.
* **Configurazione:**
    * `max_seq_length`: 2048
    * `learning_rate`: 2e-4
    * `steps`: 60
    * `optimizer`: adamw_8bit

### Phase 3: Export & Quantization 📦
Dopo l'addestramento, il modello è stato convertito in formato **GGUF** con quantizzazione **q4_k_m**.
* **Risultato:** Un file `.gguf` di circa 2GB, capace di girare velocemente sul laptop locale mantenendo il 99% dell'intelligenza acquisita.

---

## 5. 📊 Benchmark Methodology <a name="benchmark"></a>

Per valutare i modelli, abbiamo creato uno script di test automatizzato che misura:

1.  **Accuracy (Accuratezza):** La percentuale di SMS classificati correttamente rispetto alle etichette reali.
2.  **Format Adherence (Obbedienza):** La capacità del modello di rispondere ESATTAMENTE con una sola parola ("SPAM" o "HAM") senza aggiungere preamboli inutili.
3.  **Latency (Velocità):** Il tempo impiegato per generare la risposta.

### Risultati Preliminari
> *In questa sezione inserirai i grafici o le tabelle finali del tuo test, es:*
> - **Fine-Tuned:** 99% Accuracy, 100% Format Adherence, 0.2s Latency.
> - **Base Model:** 94% Accuracy, 80% Format Adherence.
> - **DeepSeek:** 96% Accuracy, ma molto lento (3s Latency).

---

## 6. 🚀 How to Run <a name="run"></a>

### Prerequisiti
* Python 3.10+
* LM Studio installato

### Setup
1.  Clona il repository:
    ```bash
    git clone [https://github.com/tuo-username/llm-spam-benchmark.git](https://github.com/tuo-username/llm-spam-benchmark.git)
    ```
2.  Installa le dipendenze:
    ```bash
    pip install -r requirements.txt
    ```
3.  Avvia LM Studio e carica il modello desiderato (il file `.gguf` nella cartella `models/`).
4.  Avvia il server locale di LM Studio.
5.  Esegui lo script di benchmark:
    ```bash
    python src/benchmark.py
    ```

---

## 7. 📝 License <a name="license"></a>

Questo progetto è distribuito sotto licenza **Apache 2.0**.
Il dataset SMS Spam Collection è pubblico e appartiene ai rispettivi autori (UCI Machine Learning Repository).

---
<p align="center">Made with ❤️ and a lot of GPU hours.</p>
