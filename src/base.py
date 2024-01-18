import pickle
from abc import ABCMeta, abstractmethod

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
# from sklearn.model_selection import train_test_split

from data.load import Data
from src.utils.top_score_instance import get_max_score_model_instance
from src.utils.manage_pkl_files import pkl_save


class BasePiepline(metaclass=ABCMeta):
    # def __init__(self, model):
    #     self.model = model

    @property
    def data(self):
        return Data

    @abstractmethod
    def preprocessing(self, *args):
        pass

    def train_process(self, _model):
        self._train = self.preprocessing(self.data.train)

        X = self._train.drop(['대출등급'], axis=1)
        Y = self._train['대출등급']

        self.label_encoder = LabelEncoder()
        Y = self.label_encoder.fit_transform(Y)

        # x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=42, stratify=Y)

        # model 훈련 및 평가
        _model.fit(X, Y)

        y_pred = _model.predict(X)
        score = f1_score(Y, y_pred, average="macro")
        print(f"[Train Score]: {score:.2f}", end=" ")

        # y_pred = _model.predict(x_test)
        # score = f1_score(y_test,y_pred, average="macro")
        # print(f"[Test Score]: {score:.5f}")

        model_name = str(_model).split("(")[0]

        return {
            "model_name": model_name,
            "model_instance": _model,
            "model_score": round(score, 2)
        }

    def test_process(self, _model):
        """ model 추론 """

        # test 전처리
        self._test = self.preprocessing(self.data.test)
        # self._test = self.scaler(self._test) #scaler

        # model 예측
        real = _model.predict(self._test)
        pred = self.label_encoder.inverse_transform(real)
        self.result = pred
        # self.result = np.where(real == 0, 'A',
        #               np.where(real == 1, 'B',
        #               np.where(real == 2, 'C',
        #               np.where(real == 3, 'D',
        #               np.where(real == 4, 'E', 'F')))))
        return _model

    def submit_process(self, _model):
        _submit = self.data.submission

        # csv 저장
        _submit['대출등급'] = self.result
        _submit.to_csv("submit.csv", index=False)

        # 모델 저장
        pkl_save(_model)

    def run(self, _model):
        model_info = self.train_process(_model)
        print(model_info)

        model = model_info.get("model_instance")
        model = self.test_process(model)
        self.submit_process(model)
