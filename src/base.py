from abc import ABCMeta, abstractmethod

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from data.load import Data
# from src.utils.top_score_instance import get_max_score_model_instance
from src.utils.manage_pkl_files import pkl_save
from src.utils.top_score_instance import check_the_score


class BasePiepline(metaclass=ABCMeta):
    def __init__(self):
        self.label_encoder = LabelEncoder()

    def tear_down(self, _model):
        pkl_save(_model)

    @property
    def data(self):
        return Data

    @abstractmethod
    def preprocessing(self, *args):
        pass

    def _common_data_process(self):
        self._train = self.preprocessing(self.data.train)

        X = self._train.drop(['대출등급'], axis=1)
        Y = self._train['대출등급']

        Y = self.label_encoder.fit_transform(Y)

        return X, Y

    def train_score(self, _model):
        X, Y = self._common_data_process()

        x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=42, stratify=Y)
        _model.fit(x_train, y_train)
        predict = _model.predict(x_test)
        check_the_score(predict, y_test)

    def train(self, _model):
        X, Y = self._common_data_process()

        _model.fit(X, Y)
        model_name = str(_model).split("(")[0]

        return {
            "model_name": model_name,
            "model_instance": _model,
        }

    def test(self, _model):
        """ model 추론 """

        # test 전처리
        self._test = self.preprocessing(self.data.test)

        # model 예측
        real = _model.predict(self._test)

        return self.label_encoder.inverse_transform(real)

    def submit_process(self, result):
        _submit = self.data.submission

        # csv 저장
        _submit['대출등급'] = result
        _submit.to_csv("submit.csv", index=False)

    def run(self, _model):
        model_info = self.train(_model)
        print(model_info)

        model = model_info.get("model_instance")
        result = self.test(model)
        self.submit_process(result)
        self.tear_down(model)
