import pandas as pd
from sklearn.preprocessing import LabelEncoder


class DataPreprocessing:
    def __init__(self, df):
        self.df = df

    def __call__(self):
        self.set_up()

        step = [func for func in dir(self) if callable(getattr(self, func)) and func.startswith("step_")]
        step_methods_sorted = sorted(step, key=lambda x: self.step_sequence[x.split('_')[1]])

        for step_method in step_methods_sorted:
            getattr(self, step_method)()

        self.tear_down()

        return self.df

    @property
    def step_sequence(self):
        seq = ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
        seq_to_num_list = [{seq[i-1]: i} for i in range(1, len(seq)+1)]
        seq_to_num_dict = {string: intger for i in seq_to_num_list for string, intger in i.items()}

        return seq_to_num_dict

    def set_up(self):
        self.df = self.df.drop(columns=['ID'])

        label_dict = {'< 1 year': 0, '<1 year': 0, '1 year': 1, '1 years': 1, '2 years': 2, '3': 3,
                      '3 years': 3, '4 years': 4, '5 years': 5, '6 years': 6, '7 years': 7, '8 years': 8,
                      '9 years': 9, '10+ years': 10, '10+years': 10, 'Unknown': -1
                      }

        self.df['근로기간'] = self.df['근로기간'].map(label_dict)

    def tear_down(self):
        categorical_features = ['대출기간', '주택소유상태', '대출목적']
        for idx in categorical_features:
            le = LabelEncoder()
            self.df[idx] = le.fit_transform(self.df[idx])

    def step_one_repayment_month(self):
        """강지원 선생님 작품"""

        self.df['대출기간'] = self.df['대출기간'].astype(str).str[:3].astype(int)
        self.df['월원금상환금'] = self.df['대출금액'] / self.df['대출기간']
        self.df['상환개월'] = self.df['총상환원금'] / self.df['월원금상환금']
        self.df.drop(['월원금상환금'], axis=1, inplace=True)
        self.df['상환개월'] = self.df['상환개월'].round().astype(int)

    def step_two_loan_object(self):
        """승범 전처리"""

        self.df[self.df["주택소유상태"] != "ANY"].reset_index(drop=True, inplace=True)
        train_object = self.df['대출목적'].unique()  # NOTE: not used variable
        oject_dict = dict()
        for i, v in enumerate(self.df['대출목적'].unique()):
            oject_dict[v] = i

        self.df['대출목적'] = self.df['대출목적'].apply(lambda x: oject_dict.get(x, '기타'))

    def step_three_remaining_loan_amount(self):
        """건영님 전처리"""

        self.df['연간소득'] = pd.qcut(self.df['연간소득'], q=10, labels=False)
        self.df["총상환금"] = self.df["총상환원금"] + self.df["총상환이자"]
        self.df["남은대출금액"] = self.df["대출금액"] - self.df["총상환금"]
        self.df = self.df.drop(columns=['총상환금'])


if __name__ == "__main__":
    d = DataPreprocessing("a")


    class PreprocessingAdvance(DataPreprocessing):
        def __init__(self, a):
            super().__init__(a)

        def step_four_method(self):
            print("step_four method")

    p = PreprocessingAdvance("a")
    p()
