from dataclasses import dataclass
import pandas as pd
from pathlib import Path


BASE = Path(__file__).resolve().parent


class classproperty:  # noqa
    def __init__(self, method):
        self.func = method

    def __get__(self, instance, cls):
        return self.func(cls)


@dataclass
class Data:
    _train: str = Path(BASE, "train.csv")  # TODO: 파일명 변경
    _test: str = Path(BASE, "test.csv")  # TODO: 파일명 변경
    _submission: str = Path(BASE, "sample_submission.csv")  # TODO: 파일명 변경

    @classproperty
    def train(self):
        return pd.read_csv(self._train)

    @classproperty
    def test(self):
        return pd.read_csv(self._test)

    @classproperty
    def submission(self):
        return pd.read_csv(self._submission)


if __name__ == "__main__":
    print(Data._train)
    print(Data._test)
    print(Data._submission)
