from nlp.models.mlp import run_mlp
from nlp.models.lstm import run_lstm

if __name__ == "__main__":
    # run_mlp()
    for i in range(1, 12):
        run_lstm(dataset_way=i, num_epochs=1)