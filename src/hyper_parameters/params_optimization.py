import json

from sklearn.model_selection import train_test_split
from src.utils.top_score_instance import check_the_score
import optuna
from optuna.samplers import TPESampler
from pathlib import Path

BASE = Path(__file__).parent
DATA = None


def set_params_optimization_data(x, y):
    global DATA

    DATA = train_test_split(x, y, test_size=0.1, random_state=42, stratify=y)
    return DATA


def load_schema(schema_path):
    with open(schema_path, "r") as f:
        schema = json.load(f)

    return schema


def xgboost_opt(model: object, schema_name: str):
    """
    model: Model instance
    schema_name: Schema file name
    """

    schema_path = Path(BASE, "schema", f"{schema_name}.json")
    schema = load_schema(schema_path)

    x_train, x_test, y_train, y_test = DATA
    evals = [(x_test, y_test)]

    def objective(trial):
        param = {}
        for parameter_name, parameter_info in schema.items():
            if parameter_info.get("dtype") == "int":
                param[parameter_name] = trial.suggest_int(parameter_name, *parameter_info.get("params"))
                continue
            param[parameter_name] = trial.suggest_float(parameter_name, *parameter_info.get("params"))

        xgbopt = model.set_params(**param, )
        xgbopt.fit(x_train, y_train) # TODO: 추가 인자 적용 하기
        preds = xgbopt.predict(x_test)
        accuracy = check_the_score(y_test, preds)
        return accuracy

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=1, timeout=5000)

    print('Best trial:')
    l_trial = study.best_trial
    print('  Value: ', l_trial.value)
    print('  Params: ')
    for key, value in l_trial.params.items():
        print(f'{key}: {value}')

    return l_trial.params


def lightgbm_opt(model : object, schema_name : str):
    """
    model: Model instance
    schema_name: Schema file name
    """
    schema_path = Path(BASE, "schema", f"{schema_name}.json")
    schema = load_schema(schema_path)

    def objective(trial):
        x_train, x_test, y_train, y_test = DATA
        evals=[(x_test, y_test)]
        param = {}
        for parameter_name, parameter_info in schema.items():
            if parameter_info.get("dtype") == "int":
                param[parameter_name] = trial.suggest_int(parameter_name, *parameter_info.get("params"))
            elif parameter_info.get("dtype") == "var":
                param[parameter_name] = len(set(y_train))
            elif parameter_info.get("dtype") == "float":
                param[parameter_name] = trial.suggest_float(parameter_name, *parameter_info.get("params"))
            else:
                param[parameter_name] = parameter_info.get("params")
        # print(f'params : {param}')
        gbm = model.set_params(**param)
        gbm.fit(x_train, y_train)
        preds = gbm.predict(x_test)
        accuracy = check_the_score(y_test, preds)
        return accuracy

    sampler = TPESampler(seed=1)
    study = optuna.create_study(study_name="lightgbm", direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=1, timeout=2000)

    print('Best trial:')
    l_trial = study.best_trial
    print('  Value: ', l_trial.value)
    print('  Params: ')
    for key, value in l_trial.params.items():
        print(f'{key}: {value}')

    return l_trial.params
