from abc import ABCMeta, abstractmethod

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
import numpy as np
from data.load import Data
# from src.utils.top_score_instance import get_max_score_model_instance
from src.utils.manage_pkl_files import pkl_save
from src.utils.top_score_instance import check_the_score
from src.utils.data_preprocessing import DataPreprocessing



class BasePiepline(metaclass=ABCMeta):
    def __init__(self):
        self.label_encoder = LabelEncoder()
        self._preprocessing_instance = DataPreprocessing
        

    def tear_down(self, _model):
        pkl_save(_model)

    @property
    def data(self):
        return Data

    @property
    def preprocessing_instance(self):
        return self._preprocessing_instance

    @preprocessing_instance.setter
    def preprocessing_instance(self, instance):
        self._preprocessing_instance = instance

    # @abstractmethod
    # def preprocessing(self, *args):
    #     pass

    def _common_data_process(self):
        self.preprocessing.set_up_dataframe(self.data.train, mode=True) # 전처리 모드 셋업
        self._train = self.preprocessing.run() # 전처리 실행
        # self._train = self.preprocessing(self.data.train,mode=True)

        X = self._train.drop(['대출등급'], axis=1)
        Y = self._train['대출등급']

        Y = self.label_encoder.fit_transform(Y)
        self.X, self.Y = X, Y
        return X, Y
        
    def valid(self, _model):
        """ model 검증 """
        
        X, Y = self._common_data_process()
        result = {"f1":[], "precision":[], "recall":[]}
        stratkfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        # k-fold
        for train_idx, test_idx in tqdm(stratkfold.split(X,Y)):
            x_train, x_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = Y[train_idx], Y[test_idx]
        
            # 모델 훈련
            _model.fit(x_train, y_train)
            
            predict = _model.predict(x_test)
            score_result = check_the_score(predict, y_test)
            for name, score in score_result.items():
                result[name].append(score)        
        # output
        print('----[K-Fold Validation Score]-----')
        for name, score_list in result.items():
            print(f'{name} score : {np.mean(score_list):.4f} / STD: (+/- {np.std(score_list):.4f})')
        
        
    # def valid(self, _model):
    #     """ model 검증 """
    #     X, Y = self._common_data_process()
    #     x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=42, stratify=Y)
    #     # k-fold code로 변환해야함
    #     _model.fit(x_train, y_train)
    #     predict = _model.predict(x_test)
        
    #     score_result = check_the_score(predict, y_test)
        
    #     print('----[Validation Score]-----')
    #     for name, score in score_result.items():
    #         print(f'{name} score : {score:.4f}')
        # 나중엔 valid도 모델 리턴이 혹시 필요 할 수 있나 해서
        # return {
        #     "model_name": model_name,
        #     "model_instance": _model,
        #         }

    def train(self, _model):
        """ model 훈련 """
        X, Y = self._common_data_process()

        _model.fit(X, Y) #전체 학습
        model_name = str(_model).split("(")[0]

        return {
            "model_name": model_name,
            "model_instance": _model,
        }

    def test(self, _model):
        """ model 추론 """

        # test 전처리
        self.preprocessing.set_up_dataframe(self.data.test, mode=False) # 전처리 모드 셋업
        self._test = self.preprocessing.run() # 전처리 실행
        # self._test = self.preprocessing(self.data.test,mode=False)

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
