import pickle
from pathlib import Path


PROJECT_DIRECTORY = Path(__file__).resolve().parent.parent.parent


def pkl_save(_model):
    model_name = str(_model).split("(")[0]
    file_name_format = f'{model_name}_model.pkl'

    file_save_path = Path(PROJECT_DIRECTORY, "pkls", file_name_format)

    with open(file_save_path, 'wb') as file:
        pickle.dump(_model, file)


def get_best_params():
    pkl_name = "best_params.pkl"
    pkl_path = Path(PROJECT_DIRECTORY, "pkls", pkl_name)

    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    return data
