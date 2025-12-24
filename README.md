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

<p align="center">
  Progetto di <b>Binary Classification</b> su SMS Spam confrontando diversi modelli.
</p>

---

> ## 📑 Indice
> 01. [🧑🏻‍🎓 Studente](#studente)  
> 02. [📌 Descrizione](#descrizione)  
> 03. [📄 Panoramica File](#panoramica-file)  
> 04. [📁 Struttura del Progetto](#struttura-progetto)  
> 05. [🛠️ Stack Tecnologico](#stack-tecnologico)  
> 06. [🚀 Installazione](#installazione)  
> 07. [🧪 Run: Processo di Fine-Tuning](#fine-tuning)  
> 08. [📊 Run: Benchmark e Confronto](#benchmark)  
> 09. [📈 Metriche e Risultati](#metriche)  
> 10. [🖥️ Hardware e Limitazioni](#hardware)  
> 11. [📝 Licenze](#licenze)  
> 12. [❓ Come Citare](#citare)

---

## 1. 🧑🏻‍🎓 Studente <a name="studente"></a>
> **Alessandro Bullegas**
> - **Matricola:** 60/73/65307
> - **Email:** alebullegas31@gmail.com
>
> - ---


## 2. 📌 Descrizione <a name="descrizione"></a>

Questo progetto nasce come studio sperimentale per analizzare i trade-off tra **dimensione del modello**, **capacità di ragionamento** (Reasoning) e **specializzazione del dominio** nel contesto della Spam Detection.

I Large Language Models (LLM) classici sono strumenti molto potenti ma, spesso, peccano di velocità e stretta aderenza alle richieste dei task che vengono affidati, come ad esempio filtrare SMS malevoli.

Utilizzando il dataset pubblico **SMS Spam Collection**, il progetto mette a confronto tre filosofie diverse:

1.  **Zero-Shot Generalist:** `Llama 3.2 3B Instruct`. Un modello leggero e generico, testato sulla sua capacità di riconoscere lo spam senza addestramento specifico.
2.  **Chain-of-Thought Reasoning:** `DeepSeek R1`. Un modello progettato per "pensare" prima di rispondere. Testiamo se il ragionamento logico aiuta a scovare tentativi di phishing più sottili o se aggiunge solo latenza inutile.
3.  **Domain Specialist:** `Llama 3.2 3B Fine-Tuned`. La versione custom, addestrata specificamente.

### 🎯 Obiettivo
Dimostrare che un **modello piccolo ma specializzato (Fine-Tuned)** può superare modelli più complessi o "ragionanti" in task verticali, offrendo:
* ✅ **Latenza Minore**
* ✅ **Accuratezza Superiore**


## 3. 📄 Panoramica File <a name="panoramica-file"></a>

| File | Tipo | Descrizione |
| :--- | :--- | :--- |
| `split_dataset.py` | 🐍 Script | Lo script che si occupa di pulire il dataset raw (`spam.csv`), mescolarlo e dividerlo rigorosamente in Training Set (80%) e Test Set (20%) per evitare *Overfitting*. |
| `model_evaluation.py` | 🐍 Script | Lo script che interroga LM Studio, misura la latenza e calcola le metriche (Accuracy, Precision, Recall) sui modelli. |
| `train_unsloth.jsonl` | 📄 Dati | Il file JSONL formattato contenente solo gli esempi per l'addestramento da utilizzare su Colab. |
| `test_benchmark.csv` | 📄 Dati | Il dataset "invisibile" usato solo per la valutazione finale. |
| `Finetuning_Spam.ipynb` | 📓 Notebook | Il notebook Colab che esegue l'addestramento QLoRA e l'esportazione GGUF. |


## 4. 📁 Struttura del Progetto <a name="struttura-progetto"></a>

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

## 5. 🛠️ Stack Tecnologico <a name="stack-tecnologico"></a>
### 🟣 LM Studio
**LM Studio** non viene usato come semplice interfaccia grafica, ma come vero e proprio **Server Locale**.
* **Ruolo Architetturale:** LM Studio carica i modelli e sfrutta la GPU/CPU del pc per eseguire i calcoli.
* **Integrazione API:** La funzionalità chiave utile per il progetto è il suo **Local Server** compatibile con le specifiche OpenAI (`http://localhost:1234/v1`). Questo ci permette di disaccoppiare il modello dallo script Python: possiamo sostituire il "motore" (es. passando da un modello ad un altro) in tempo reale senza modificare il codice.

### 🦜🔗 LangChain
**LangChain** funge come livello di astrazione logica tra il nostro codice Python e il modello linguistico.
* **Prompt Templating:** Gestisce la costruzione dinamica dei messaggi, inserendo il `System Prompt` (le regole di sicurezza) e lo `User Prompt` (l'SMS da analizzare) nel formato corretto atteso dal modello.
* **Output Parsing:** Utilizzando `StrOutputParser`, LangChain intercetta la risposta grezza dell'LLM e la pulisce da eventuali meta-tag o spazi bianchi, garantendo che il dato salvato nel CSV sia pulito e pronto per l'analisi.

### 🦥 Unsloth AI (Optimization Library)
Per la fase di Fine-Tuning su Google Colab, viene utilizzata la libreria **Unsloth**, che rappresenta uno strumento fondamentale per migliorare l'efficienza nell'addestramento degli LLM.
* **Perché è essenziale:** Il Fine-Tuning tradizionale di Llama 3 richiederebbe GPU potentissime (A100, 40GB VRAM).
* **Innovazione Tecnica:** Unsloth implementa kernel PyTorch riscritti per l'ottimizzazione e utilizza la tecnica **QLoRA** (Quantized Low-Rank Adaptation).
* **Risultato:** Questo stack ci ha permesso di addestrare un modello da 3 miliardi di parametri su una GPU Tesla T4 gratuita (16GB VRAM), riducendo i tempi di training di 2x e l'occupazione di memoria del 60%.




