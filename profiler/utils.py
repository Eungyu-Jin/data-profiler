import json
import numpy as np
import pandas as pd

class NumpyEncoder(json.JSONEncoder):
    """ 딕셔너리를 json으로 저장할 때 numpy 형태로 인코딩 
    """
    # json serialize numpy
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, pd.Timestamp):
            return str(obj)
        return super(NumpyEncoder, self).default(obj)


class Converter():
    """ 딕셔너리를 json으로 저장
    """
    def __init__(self):
        pass
    def to_json(self, obj:dict):
        return json.dumps(obj, cls=NumpyEncoder, indent=4)
    
    def save(self, obj:dict, path:str):
        with open(path, 'w') as f:
            json.dump(obj, f, cls=NumpyEncoder, indent=4)