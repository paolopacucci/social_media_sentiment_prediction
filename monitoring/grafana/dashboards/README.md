Il file sentiment_dashboard.json è la dashboard Grafana contenente i pannelli per visualizzare le metriche raccolte da Prometheus.

Questa consente di visualizzare:
- metriche operative dell’API di serving (`/predict`), come request rate e latenza;
- distribuzione del sentiment predetto nel tempo;
- metriche di performance del modello (`accuracy` e `macro-F1`);
- confidence media dell’ultimo batch;
- numero totale di predizioni;
- metriche legate ai trigger di retraining. 