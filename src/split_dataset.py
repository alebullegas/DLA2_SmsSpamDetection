import pandas as pd
import json
import numpy as np
import os

#CONFIGURAZIONE
DATA_DIR = "data"

INPUT_FILE = os.path.join(DATA_DIR, "spam.csv")             #dataset originale
TRAIN_OUTPUT = os.path.join(DATA_DIR, "train_unsloth.jsonl") #file per addestramento Colab (80%)
TEST_OUTPUT = os.path.join(DATA_DIR, "test_benchmark.csv")   #file per il Test Finale (20%)
SPLIT_RATIO = 0.8                                            #80% Training, 20% Test

#CARICAMENTO E PULIZIA
print(f"* Lettura {INPUT_FILE}...")
try:
    df = pd.read_csv(INPUT_FILE, encoding='latin-1')
except:
    df = pd.read_csv(INPUT_FILE, encoding='utf-8')

#Rinomina colonne standard
if 'v1' in df.columns: 
    df = df.rename(columns={'v1': 'label', 'v2': 'text'})
elif 'Category' in df.columns: 
    df = df.rename(columns={'Category': 'label', 'Message': 'text'})
else:
    df.columns = ['label', 'text'] + list(df.columns[2:])

#Pulisce label e testo
df['label'] = df['label'].str.strip().str.lower()
df['text'] = df['text'].str.strip()

#SPLIT TRAIN / TEST
#Mescola i dati in modo casuale
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

#Calcola il punto di taglio
split_index = int(len(df) * SPLIT_RATIO)

train_df = df.iloc[:split_index]  
test_df = df.iloc[split_index:]   

print(f"* Totale righe: {len(df)}")
print(f"* Training Set: {len(train_df)} righe")
print(f"* Test Set:     {len(test_df)} righe")

#SALVATAGGIO FILE TRAINING

dataset_data = []
SYSTEM_PROMPT = "You are a Cybersecurity AI specialized in detecting SMS Spam. Analyze the text and classify it as 'SPAM' or 'HAM'."

for index, row in train_df.iterrows():
    entry = {
        "conversations": [
            { "role": "system", "content": SYSTEM_PROMPT },
            { "role": "user", "content": str(row['text']) },
            { "role": "assistant", "content": str(row['label']).upper() }
        ]
    }
    dataset_data.append(entry)

with open(TRAIN_OUTPUT, 'w', encoding='utf-8') as f:
    for entry in dataset_data:
        json.dump(entry, f)
        f.write('\n')

print(f"* Salvato {TRAIN_OUTPUT}, da caricare su colab")

#SALVATAGGIO FILE TEST
#Salva solo text e label per facilitare il successivo confronto con le risposte
test_df[['text', 'label']].to_csv(TEST_OUTPUT, index=False)
print(f"* Salvato {TEST_OUTPUT}")