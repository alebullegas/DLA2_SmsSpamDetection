import pandas as pd
import time
import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from tqdm import tqdm

#CONFIGURAZIONE
TEST_DATASET_PATH = os.path.join("data", "test_benchmark.csv")
OUTPUT_DIR = "results"
OUTPUT_FILENAME = "risultati_benchmark.csv"
OUTPUT_PATH = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME) # Crea il percorso completo

#Configurazione LM Studio
LM_STUDIO_URL = "http://localhost:1234/v1"
API_KEY = "lm-studio"
LIMIT_SAMPLES = None 

#SETUP MODELLO
print(f"\n***AVVIO...***")
try:
    llm = ChatOpenAI(
        base_url=LM_STUDIO_URL,
        api_key=API_KEY,
        model="local-model", #LM Studio usa sempre il modello caricato
        temperature=0.0      
    )
    print("* Connessione LM Studio effettuata.")
except Exception as e:
    print(f"X Errore connessione: {e}")
    exit()

#CARICAMENTO DATASET DI TEST
if not os.path.exists(TEST_DATASET_PATH):
    print(f"X ERRORE: Non trovo il file {TEST_DATASET_PATH}!")
    print("   Esegui prima lo script 'split_dataset.py' per generarlo.")
    exit()

try:
    df_test = pd.read_csv(TEST_DATASET_PATH)
    
    # Verifica che ci siano le colonne giuste
    if 'text' not in df_test.columns or 'label' not in df_test.columns:
        raise ValueError(f"Il file deve avere le colonne 'text' e 'label'. Trovate: {df_test.columns.tolist()}")

    if LIMIT_SAMPLES and LIMIT_SAMPLES < len(df_test):
        df_test = df_test.head(LIMIT_SAMPLES)
        print(f"! LIMITAZIONE ATTIVA: Usa solo i primi {LIMIT_SAMPLES} messaggi.")
    else:
        print(f"* Caricati {len(df_test)} messaggi dal Test Set.")

except Exception as e:
    print(f"X Errore lettura file: {e}")
    exit()

#PROMPT
template_spam = """
You are a Cybersecurity AI detecting SMS Spam.

Classify the following text message as 'SPAM' or 'HAM' (Legitimate).

TEXT TO ANALYZE:
"{text_input}"

Reply with ONLY one word: SPAM or HAM.
"""

prompt = ChatPromptTemplate.from_template(template_spam)
chain = prompt | llm | StrOutputParser()

#CICLO DI ANALISI CON MISURAZIONE TEMPO
y_true = []
y_pred = []
latencies = [] #Per misurare la velocità

print(f"\n**Inizio test...")
pbar = tqdm(total=len(df_test))

for idx, row in df_test.iterrows():
    text_sms = str(row['text'])
    real_label = str(row['label']).lower().strip()
    
    start_time = time.time() #Start Timer
    
    try:
        #chiamata modello
        response = chain.invoke({"text_input": text_sms})
        
        #pulizia risposta
        cleaned_resp = response.lower().strip()
        
        #se dice altro oltre spam/ham, si considera errore 
        if "spam" in cleaned_resp:
            predicted = "spam"
        elif "ham" in cleaned_resp:
            predicted = "ham"
        else:
            predicted = "error"
                
    except Exception as e:
        print(f"! Errore API riga {idx}: {e}")
        predicted = "error"

    end_time = time.time() #Stop Timer
    latency = end_time - start_time
    
    y_true.append(real_label)
    y_pred.append(predicted)
    latencies.append(latency)
    
    pbar.update(1)

pbar.close()

#CALCOLO E SALVATAGGIO RISULTATI
#Creazione DataFrame con i risultati
results_df = pd.DataFrame({
    'text': df_test['text'],
    'true_label': y_true,
    'predicted_label': y_pred,
    'latency_seconds': latencies
})

#SALVATAGGIO
if not os.path.exists(OUTPUT_DIR):
    try:
        os.makedirs(OUTPUT_DIR)
        print(f"Creata cartella output: {OUTPUT_DIR}")
    except OSError as e:
        print(f"Errore creazione cartella: {e}")

#Salva il file nel percorso specifico
try:
    results_df.to_csv(OUTPUT_PATH, index=False)
    print(f"\n* Risultati salvati correttamente in: {OUTPUT_PATH}")
except Exception as e:
    print(f"\nX Errore durante il salvataggio del CSV: {e}")

#Calcolo Metriche
acc = accuracy_score(y_true, y_pred)
avg_latency = sum(latencies) / len(latencies)

print("\n" + "="*60)
print(f"***RISULTATI DEL BENCHMARK***")
print("="*60)
print(f"ACCURACY:      {acc:.2%}")
print(f"VELOCITÀ MEDIA: {avg_latency:.4f} secondi/messaggio")
print("="*60)

print("\nMatrice di Confusione:")
print(confusion_matrix(y_true, y_pred, labels=["spam", "ham"]))

print("\nReport Dettagliato:")
print(classification_report(y_true, y_pred, labels=["spam", "ham"]))