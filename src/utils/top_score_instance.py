from typing import List, Dict, TypeVar, Union
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

MODEL_INSTANCE = TypeVar('MODEL_INSTANCE')


def get_max_score_model_instance(model_info_list: List[Dict[str, Union[str, int, MODEL_INSTANCE]]]) -> MODEL_INSTANCE:
    """ 모델 목록에서 가장 높은 성능의 모델 인스턴스 반환

    Parameters
    ----------
    model_info_list : Models 클래스에 존재하는 모든 모델의 정보 값

    Returns
    ---------
    MODEL_INSTANCE : 가장 높은 성능을 나타낸 모델의 인스턴스 ID 값
    """

    max_score: int = model_info_list[0].get("model_score", 0)
    max_score_model_instance: MODEL_INSTANCE = model_info_list[0].get("model_instance")
    max_score_model_name: str = model_info_list[0].get("model_name")

    # NOTE: 초기 최대값 설정 시 0번째 데이터를 갖고 있기 때문에 1번째 데이터 부터 검사
    for model_info in model_info_list[1:]:
        model_score: int = model_info.get("model_score", 0)
        model_instance: MODEL_INSTANCE = model_info.get("model_instance")
        model_name: str = model_info.get("model_name")

        if model_score > max_score:
            max_score = model_score
            max_score_model_instance = model_instance
            max_score_model_name = model_name

    print(f"MAX SCORE model instance: {max_score_model_name} TEST SCORE: {max_score}")
    return max_score_model_instance


def check_the_score(predict, answer):
    """
    스코어 확인
    """
    result1 = f1_score(predict, answer, average="macro")
    result2 = precision_score(predict, answer, average="macro")
    result3 = recall_score(predict, answer, average="macro")
    # result4 = roc_auc_score(predict, answer, average="macro", multi_class="ovr")

    return {'f1': result1, 'precision': result2, 'recall': result3}


