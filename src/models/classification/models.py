from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier,
)
from sklearn.svm import SVC


def not_module_handler(func):
    def wrap(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except ImportError as e:
            print(f"필요한 모듈이 설치되지 않았습니다: {e}")
            return None
    return wrap


class Model:
    """
    models
    """

    @property
    @not_module_handler
    def logistic_regression(self):
        return LogisticRegression(solver="liblinear", random_state=42)

    @property
    @not_module_handler
    def naive_bayes(self):
        return GaussianNB()

    @property
    @not_module_handler
    def decision_tree(self):
        return DecisionTreeClassifier(random_state=42)

    @property
    @not_module_handler
    def random_forest(self):
        return RandomForestClassifier(random_state=42)

    @property
    @not_module_handler
    def svm(self):
        return SVC(kernel="rbf", random_state=42)

    @property
    @not_module_handler
    def ada_boost(self):
        return AdaBoostClassifier(random_state=42)

    @property
    @not_module_handler
    def gradient_boosting(self):
        return GradientBoostingClassifier(random_state=42)

    @property
    @not_module_handler
    def xgboost(self):
        from xgboost import XGBClassifier

        return XGBClassifier(random_state=42)

    @property
    @not_module_handler
    def catboost(self):
        from catboost import CatBoostClassifier

        return CatBoostClassifier(random_seed=42)

    @property
    @not_module_handler
    def mlp_classifier(self) -> MLPClassifier:
        hidden_layer_sizes = (50, 50)
        return MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes, max_iter=1000, random_state=42
        )

    # NOTE: conda install lightgbm으로 설치 하여 사용 하거나, 주석 처리하여 제외 시킬 것
    @property
    @not_module_handler
    def lightgbm(self):
        from lightgbm import LGBMClassifier

        return LGBMClassifier(n_estimators=150, random_state=42)

    @staticmethod
    def get_model_list():
        return [
            instance
            for instance, prop in Model.__dict__.items()
            if isinstance(prop, property)
        ]

    def get_model_instances(self):
        return [getattr(self, _model) for _model in self.get_model_list()]


if __name__ == "__main__":
    model = Model()
    model_instance = getattr(model, "catboost")
    test = model.get_model_instances()
    # print(test)
