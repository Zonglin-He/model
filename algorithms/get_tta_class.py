from pre_train_model.pre_train_model import PreTrainModel
from algorithms.accup import ACCUP
from algorithms.eata_accup import EATA
from algorithms.tent_tta import Tent
from algorithms.sar_tta import SAR

def get_algorithm_class(algorithm_name): # 根据给定的算法名称字符串返回对应的算法类
    """Return the algorithm class with the given name."""
    if algorithm_name not in globals():
        raise NotImplementedError("Algorithm not found: {}".format(algorithm_name))
    return globals()[algorithm_name]
