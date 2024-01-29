import numpy as np

from abc import ABCMeta
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

from data.load import Data
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

    @property
    def preprocessing_instance(self):
        """전처리 인스턴스"""
        return self._preprocessing_instance

    @preprocessing_instance.setter
    def preprocessing_instance(self, instance):
        """
        외부에서 생성한 전처리 인스턴스를 설정 하여 파이프라인에서 사용
        """
        self._preprocessing_instance = instance

    def _common_data_process(self):
        """
        데이터 전처리 공통 작업
        """
        self._train = self.preprocessing.run()

        pass  # TODO: It's gonna write label encode, drop feature "Y" another logic

    def valid(self, _model):
        """ model 검증 """

        X, Y = self._common_data_process()
        result = {"f1": [], "precision": [], "recall": []}
        stratkfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        # k-fold
        for train_idx, test_idx in tqdm(stratkfold.split(X, Y)):
            x_train, x_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = Y[train_idx], Y[test_idx]

            # 모델 훈련
            _model.fit(x_train, y_train)

            predict = _model.predict(x_test)
            score_result = check_the_score(predict, y_test)
            for name, score in score_result.items():
                result[name].append(score)

        # NOTE: display output
        print('----[K-Fold Validation Score]-----')
        for name, score_list in result.items():
            print(f'{name} score : {np.mean(score_list):.4f} / STD: (+/- {np.std(score_list):.4f})')

    def train(self, _model):
        """ model 훈련 """
        X, Y = self._common_data_process()

        _model.fit(X, Y)  # NOTE: 전체 학습
        model_name = str(_model).split("(")[0]

        return {
            "model_name": model_name,
            "model_instance": _model,
        }

    def test(self, _model):
        """ model 추론 """

        # NOTE: test 전처리
        self.preprocessing.set_up_dataframe(self.data.test, mode=False)  # 전처리 모드 셋업
        self._test = self.preprocessing.run()  # 전처리 실행

        # NOTE: model 예측
        real = _model.predict(self._test)
        return self.label_encoder.inverse_transform(real)

    def submit_process(self, result):
        """
        결과 제출을 위한 테스트 결과 입력
        """
        _submit = self.data.submission

        pass  # TODO: CSV save for submit logic

    def run(self, _model):
        model_info = self.train(_model)
        print(model_info)

        model = model_info.get("model_instance")
        result = self.test(model)
        self.submit_process(result)
        self.tear_down(model)
