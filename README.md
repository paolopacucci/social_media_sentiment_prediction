# MLOps per il monitoraggio del sentiment aziendale con RoBERTa

## 1. Panoramica del progetto
Progetto di sentiment analysis per aziende in ambito MLOps basato su RoBERTa, con l’obiettivo di costruire un flusso che comprenda preparazione dei dati, training iniziale, serving tramite API, deploy su Hugging Face, monitoraggio continuo del sentiment e delle performance, e retraining condizionato da soglia.

## Link principali

- **Modello su Hugging Face Hub**: https://huggingface.co/paolopacucci/sentiment-roberta
- **API deployata su Hugging Face Space**: https://huggingface.co/spaces/paolopacucci/sentiment-roberta-api

## 2. Architettura del sistema
Il progetto è organizzato come una pipeline modulare che comprende preparazione dei dati, training del modello, serving tramite API, monitoraggio continuo e retraining condizionato.
Il training iniziale viene eseguito separatamente su Google Colab con GPU, così da ottenere un modello base più solido rispetto a un training CPU locale o in CI.   
**Notebook del training iniziale**: [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1kXN4pmkNMy2hDVmXl-Dkow5aZxMrdE6T?usp=sharing).  
Il modello risultante viene poi pubblicato su Hugging Face Hub e caricato all’avvio dell’applicazione FastAPI.

L’applicazione espone endpoint per health check, inferenza e metriche Prometheus. 
Attorno al servizio principale sono presenti due componenti di monitoraggio distinti:  
- Un generatore di batch di testi non etichettati, usato per simulare un flusso live di messaggi/tweet e il monitoraggio del sentiment   
- Un exporter che utilizza batch etichettati per valutare periodicamente le performance del modello.
Se la metrica monitorata scende sotto soglia per un numero consecutivo di controlli configurato, il sistema può avviare un retraining leggero.

Le metriche prodotte dai servizi vengono raccolte da Prometheus e visualizzate in Grafana tramite dashboard dedicate. 

L’ambiente locale è orchestrato con Docker Compose, mentre test automatici e deploy dell’applicazione su Hugging Face sono gestiti tramite GitHub Actions.


## 3. Dataset e strategia dei dati

### 3.1 Scelta Dataset e caratteristiche
Per il progetto è stato utilizzato il dataset pubblico `Sp1786/multiclass-sentiment-analysis-dataset`, già suddiviso nei tre split originali `train`, `validation` e `test` e contenente alcune colonne aggiuntive non necessarie ai fini del progetto. 

### 3.2 Riutilizzo degli split
Come scelta progettuale, i tre split originali non sono stati riutilizzati semplicemente come train/validation/test, ma come tre sorgenti di dati distinte, in modo da separare con chiarezza i diversi flussi della pipeline, nello specifico sono stati usati nel seguente modo: 

- Lo split `train` per il training iniziale del modello.
Nel preprocessing vengono mantenute le colonne ‘text’ e ‘label’, nel file di training vengono generati gli split finali `train`, `val` e `test`, quest’ultimo viene salvato separatamente come benchmark condiviso per il retraining. 

- Lo split `validation` per il monitoraggio delle performance e al retraining. 
Nel file di monitoraggio avviene il preprocessing dove vengono mantenute le colonne ‘text’ e ‘label’, gli split ’train’ e ‘val’ vengono generati nel file di retraining nel caso in cui esso venga triggerato dal superamento della soglia.

- Lo split `test`, come base per simulare un flusso “live” di testi non etichettati, il preprocessing avviene dentro al file di sentiment monitoring mantenendo solo la colonna ‘text’.  

### 3.3 Limiti strategia dati
Poiché gli split originali del dataset risultano già piuttosto bilanciati, le visualizzazioni finali del sentiment  mostrano distribuzioni più uniformi di quanto sarebbe realistico in uno scenario reale, la scelta è stata comunque ritenuta accettabile nel contesto didattico del progetto. 


## 4. Struttura della repository

La repository è organizzata in modo da separare chiaramente serving, data pipeline, training, monitoring, observability e automazione.

### 4.1 Entry point e serving
  - `src/app/main.py`: avvio dell’applicazione FastAPI ed esposizione degli endpoint principali
  - `src/model_loader.py`: caricamento del modello e del tokenizer da Hugging Face Hub
  - `src/schemas.py`: schemi request/response per l’API
  - `src/metrics.py`: metriche Prometheus esposte dal servizio API

### 4.2 Configurazione
  - `src/config.py`: configurazione centralizzata di path, colonne, parametri di training/retraining, soglie e porte

### 4.3 Data pipeline
  - `src/data/data_utils.py`: utility per caricamento, salvataggio, pulizia e validazione dei file CSV
  - `src/data/prepare_data.py`: preprocessing dei dataset per training e monitoring
  - `src/data/split_data.py`: generazione degli split usati da training iniziale e retraining

### 4.4 Training e retraining
  - `src/training/train_utils.py`: funzioni condivise per training e retraining
  - `src/training/train_roberta_colab.py`: training iniziale eseguito su Colab con GPU
  - `src/training/train_retrain.py`: retraining leggero pensato per CPU
  - `src/training/push_to_hub.py`: pubblicazione degli artifact del modello su Hugging Face Hub
  - `reports/train/final_metrics.json`: metriche finali del training iniziale

### 4.5 Notebook Colab
  - `notebooks/train_roberta_colab.ipynb`: notebook usato per eseguire il training iniziale del modello su Google Colab con GPU

### 4.6 Monitoring services
  - `src/monitoring/sentiment_generator.py`: simulazione di batch di testi non etichettati per il monitoraggio del sentiment
  - `src/monitoring/performance_exporter.py`: valutazione periodica delle performance e trigger del retraining

### 4.7 Observability config
  - `monitoring/prometheus.yml`: configurazione di Prometheus
  - `monitoring/grafana/provisioning/datasources/datasources.yml`: provisioning della datasource Prometheus
  - `monitoring/grafana/provisioning/dashboards/dashboards.yml`: provisioning delle dashboard Grafana
  - `monitoring/grafana/dashboards/sentiment_dashboard.json`: dashboard principale del progetto

### 4.8 Docker runtime e dipendenze
  - `Dockerfile`
  - `docker-compose.yml`
  - `requirements.txt`
  - `requirements-cpu.txt`
  - `requirements-colab.txt`

### 4.9 CI/CD e test
  - `.github/workflows/ci.yml`: lint e test automatici
  - `.github/workflows/cd_hugginface.yml`: deploy dell’applicazione su Hugging Face Space
  - `tests/conftest.py`
  - `tests/test_preprocessing.py`
  - `tests/test_prepare_and_split_integration.py`
  - `tests/test_api.py`


## 5. Esecuzione del progetto

### 5.1 Prerequisiti
Per eseguire il progetto in locale sono necessari Docker e Docker Compose. 
Il runtime applicativo è basato su Python 3.11, FastAPI, Transformers, Prometheus e Grafana.

Il serving carica il modello da Hugging Face Hub tramite MODEL_ID, la gestione delle credenziali Hugging Face necessarie per i workflow automatici è stata effettuata tramite GitHub Secrets, mentre l’esecuzione locale dei servizi avviene tramite Docker Compose.

### 5.2 Avvio dell’ambiente locale

Comandi da terminale:

- Per avviare tutto l’ambiente locale dalla root del progetto:  
`docker compose up --build`

- Riavviare servizi se i container sono già stati costruiti:  
`docker compose up`

- Per fermare l’ambiente:  
`docker compose down`


### 5.3 Servizi disponibili

Dopo l’avvio, i principali servizi disponibili in locale sono:

- **API FastAPI**: http://localhost:8000; per health check, inferenza e metriche  
- **Metriche performance exporter**: http://localhost:8001/metrics;  monitoraggio delle performance del modello e trigger del retraining  
- **Metriche sentiment generator**: http://localhost:8002/metrics; simulazione live stream di testi non etichettati per il monitoraggio del sentiment  
- **Prometheus**: http://localhost:9090; raccolta delle metriche esposte dai servizi  
- **Grafana**: http://localhost:3000;  visualizzazione delle metriche tramite dashboard  

Le credenziali di default di Grafana sono:  
- username: admin
- password: admin

### 5.4 Configurazione del trigger di retraining

Nel repository, il trigger di retraining è configurato con parametri realistici, il modello  deve restare sotto la soglia 0.72 per almeno 3 controlli consecutivi, tuttavia l'esecuzione del retraining (`RUN_RETRAIN`) e il push su Hugging Face (`PUSH_AFTER_RETRAIN`) sono disabilitati.

Se si volesse ossevare più facilmente il comportamento del meccanismo di trigger/retrain, si potrebbero configurare, ad esempio, `WINDOW_CONSECUTIVE = 1`, `F1_TRESHOLD = 0.85` ed eventualmente `RUN_RETRAIN= True` per eseguire il retraining.   


## 6. CI/CD

### 6.1 Continuous Integration
La pipeline di Continuous Integration è definita nel workflow `ci.yml` ed esegue controlli automatici a ogni push e pull_request.
Il workflow è il seguente:
- configura un ambiente Python 3.11 su GitHub Actions;
- installa le dipendenze necessarie;
- esegue controlli di lint con flake8;
- esegue i test con pytest.

### 6.2 Continuous Deployment
La parte di Continuous Deployment è definita nel workflow `cd_hugginface.yml`, è dedicata al deploy automatico dell’applicazione su Hugging Face Space e viene eseguita a ogni push sul main branch.
Il workflow è il seguente:  
- prepara una cartella di deploy contenente i file necessari all’esecuzione della Space;  
- copia il codice applicativo e i file di dipendenze;  
- genera un README.md minimale per la Space;  
- esegue l’upload automatico della Space su Hugging Face tramite huggingface_hub.   
In questo modo l’applicazione FastAPI usata nel progetto viene pubblicata automaticamente come Docker Space.

### 6.3 Gestione dei secret
Le credenziali Hugging Face usate nei workflow automatici vengono gestite tramite GitHub Secrets per rendere il deploy più sicuro.

### 6.4 Scelte progettuali e limiti della pipeline
La pipeline CI/CD non include il training e il push del modello iniziale su Hugging Face, questa decisione è stata presa per ottenere un modello più performante ottenuto con training su GPU tramite Colab.
La logica di push del modello su Hugging Face è integrato nel codice di training.


## 7. Risultati principali

### 7.1 Metriche del modello iniziale
Il training iniziale produce un report finale salvato in `reports/train/final_metrics.json`, i risultati sono:  
- test_accuracy: 0.7618  
- test_macro_f1: 0.7649  

Queste metriche rappresentano il benchmark di riferimento del modello iniziale, usato come base sia per il serving tramite API sia per l’eventuale confronto con i risultati del retraining.

### 7.2 Risultati del sistema nel suo complesso
Il progetto produce un ambiente eseguibile dove poter monitorare e visualizzare alcune tra le metriche più rappresentative per soddisfare le richieste della traccia, la dashboard Grafana mostra pannelli per request rate e latency dell’endpoint /predict, andamento di accuracy e macro-F1, confidence media dell’ultimo batch, numero totale di predizioni e distribuzione delle label predette. 


## 8. Strumenti, librerie e buone pratiche apprese durante il progetto

### 8.1 Strumenti e librerie approfonditi autonomamente
Durante lo sviluppo del progetto ho avuto la possibilità di imparare e approfondire alcune librerie e buone pratiche per mantenere il codice più leggibile, solido, modulare e riusabile, in particolare:  

- uso di pathlib.Path per la gestione dei Path  
- uso di pandas.DataFrame per la manipolazione dei dati  
- inserimento type hints  
- centralizzazione della configurazione nel file config.py  
- creazione di moduli utils per raccogliere funzioni condivise  
- utilizzo di conftest per organizzare meglio i test  
- uso di subprocess separare il trigger del retraining dalla logica principale del monitoring  

### 8.2 Tecnologie necessarie al progetto ma non trattate nel corso
Per lo svolgimento del progetto è stato indispensabile apprendere autonomamente alcuni argomenti come:  
- Transformers e Torch per training e retraining del modello  
- Prometheus per l’esposizione e raccolta delle metriche   
- Costruzione del file per la dashboard di Grafana  
- Strumenti per il deploy del modello e dell’API su Hugging Face  


## 9. Principali difficoltà incontrate

Alcuni punti del progetto mi hanno messo di fronte a problematiche e tecnologie mai affrontate, grazie a ciò ho avuto la possibilità d’imparare nuovi argomenti e costruire soluzioni creative.
I challenge più rilevanti sono stati:  

- Progettazione del file di training con Transformers e Torch  
- Progettazione del file di retraining “light” adatto all’uso con CPU o in CI  
- Progettazione del file per la simulazione del flusso “live” di dati per il monitoraggio del sentiment(`sentiment_generator.py`)  
- Progettazione del file dove integrare la creazione di batch, chiamata all’endpoint /predict, calcolo di accuracy e macro-F1, esposizione delle metriche e attivazione del retrainig in base al superamento della soglia e finestra consecutiva di controlli(`performance_exporter.py`)  
- Configurazione di Prometheus e Grafana  
- Deploy del modello e dell’API su Hugging Face  


## 10. Limiti attuali del progetto

Il progetto implementa una pipeline MLOps coerente con la traccia, ma presenta alcune semplificazioni e limitazioni legate al contesto didattico, a scelte tecniche e di limiti hardware:  

- Come già anticipato il training iniziale non è integrato nella pipeline automatica CI/CD, ma viene eseguito su Google Colab con GPU per ottenere un modello base più solido rispetto a un training eseguito con parametri adatti a lavorare su CPU o GitHub Actions.  
- Il retraining usa parametri più conservativi per consentire l’uso in CI, dunque non rappresenta un processo di aggiornamento paragonabile rispetto al training iniziale.  
- La logica di push del retraining su Hugging Face è implementata ma inibita perché non esiste una promotion rule per verificare se il modello retrainato è effettivamente migliore di quello esistente, il modello viene comunque salvato in artifacts/retrain_model e le metriche in reports/retrain.
Il push è attivabile modificando da `False` a `True` il parametro `PUSH_AFTER_RETRAIN = ` nel file config.py


## 11. Possibili miglioramenti futuri
Possibili miglioramenti del sistema potrebbero essere:  

- Integrazione in CI di training/retraining su GPU, validazione, promozione e push su Hugging Face  
- Collegamento a una sorgente di dati live e non simulata con dataset statico per sentiment  
- Implementazione di dashboard e ulteriori metriche per migliorare l’osservabilità delle prestazioni del modello(es. model drifting) e del sentiment(es. geolocalizzazione, piattaforma)   


## 12. Fonti e riferimenti

### 12.1 Dataset
- https://huggingface.co/datasets/Sp1786/multiclass-sentiment-analysis-dataset  

### 12.2 Training e Retraining RoBERTa

- **Corso “LLM” di Hugging Face:**    
https://huggingface.co/learn/llm-course/chapter1/1  

- **Documentazione RoBERTa:**   
https://huggingface.co/docs/transformers/model_doc/roberta  

- **Documentazione Transformers:**  
**-Tokenizer**    
https://huggingface.co/docs/transformers/model_doc/auto
https://huggingface.co/docs/transformers/main_classes/tokenizer  
**-Fine-Tuning**    
https://huggingface.co/docs/transformers/training  
**-Trainer**    
https://huggingface.co/docs/transformers/main_classes/trainer  
**-Callbacks**   
https://huggingface.co/docs/transformers/trainer_callbacks  
https://huggingface.co/docs/transformers/en/main_classes/callback#transformers.EarlyStoppingCallback  
**-Ottimizzazione retrain su CPU:**  
https://huggingface.co/docs/transformers/perf_train_cpu  

- **Retrain “Light” usando Freeze:**  
https://medium.com/@prabhatzade/freezing-layers-and-fine-tuning-transformer-models-in-pytorch-a-simple-guide-119cad0980c6  
https://github.com/huggingface/transformers/issues/400  
https://discuss.huggingface.co/t/how-to-freeze-layers-while-fine-tuning/155342  

### 12.3 API
- **Model Loader:**  
https://huggingface.co/docs/hub/models-downloading  

- **Schemas:**  
https://fastapi.tiangolo.com/tutorial/response-model/  

### 12.4 Model Deployment su Hugging Face
- https://huggingface.co/docs/huggingface_hub/guides/upload  

### 12.5 API Deploy su Hugging Face Spaces con GitHub Actions  
- https://huggingface.co/docs/hub/spaces-github-actions  

### 12.6 Prometheus/Grafana
- **Exporter:**   
https://prometheus.io/docs/instrumenting/writing_exporters/  

- **File di configurazione Prometheus:**  
https://prometheus.io/docs/prometheus/latest/configuration/configuration/  

- **Metriche:**  
https://prometheus.io/docs/practices/naming/  
https://prometheus.io/docs/instrumenting/writing_clientlibs/  
https://prometheus.github.io/client_python/instrumenting/gauge/  
https://prometheus.github.io/client_python/instrumenting/counter/  
https://prometheus.github.io/client_python/instrumenting/histogram/  

- **Esposizione metriche:**  
https://prometheus.github.io/client_python/exporting/http/

- **Prometheus Data Source e Dashboards:**  
https://grafana.com/tutorials/provision-dashboards-and-data-sources/#introduction  
https://grafana.com/docs/grafana/latest/administration/provisioning/  
https://grafana.com/docs/grafana/latest/visualizations/dashboards/build-dashboards/view-dashboard-json-model/#json-fields  
https://oneuptime.com/blog/post/2026-01-30-grafana-dashboard-json-model/view  

### 12.7 Batch sampling per Sentiment Generator e Performance Monitoring
- https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.sample.html
- https://note.nkmk.me/en/python-pandas-sample/