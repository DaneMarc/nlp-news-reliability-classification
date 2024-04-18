from nlp.models.baselines.lstm_baseline import run_lstm

if __name__ == "__main__":
    enhancements = ["_pca_scaled", "_pca", "_sent_pca_scaled", "_sent_pca", "_sent_tfidfclip", "_sent", "_tfidfclip"]
    for enhancement in enhancements:
        run_lstm(dataset_way=7, num_epochs=1, enhancement=enhancement)