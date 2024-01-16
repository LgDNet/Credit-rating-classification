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
from xgboost import XGBClassifier
from catboost import CatBoostClassifier


# import h2o
# from h2o.automl import H2OAutoML


class Model:
    """
    models
    """

    @property
    def logistic_regression(self):
        return LogisticRegression(solver="liblinear", random_state=42)

    @property
    def naive_bayes(self):
        return GaussianNB()

    @property
    def decision_tree(self):
        return DecisionTreeClassifier(random_state=42)

    @property
    def random_forest(self):
        return RandomForestClassifier(random_state=42)

    @property
    def svm(self):
        return SVC(kernel="rbf", random_state=42)

    @property
    def ada_boost(self):
        return AdaBoostClassifier(random_state=42)

    @property
    def gradient_boosting(self):
        return GradientBoostingClassifier(random_state=42)

    @property
    def xgboost(self) -> XGBClassifier:
        return XGBClassifier(random_state=42)

    @property
    def catboost(self) -> CatBoostClassifier:
        return CatBoostClassifier(random_seed=42)

    @property
    def mlp_classifier(self) -> MLPClassifier:
        hidden_layer_sizes = (50, 50)
        # max_iter = 1000
        return MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes, max_iter=1000, random_state=42
        )

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
