import numpy as np
import pandas as pd
from pandas.api.types import infer_dtype
from scipy import stats
from dateutil.parser import parse

class Generator:
    """## Generator
    iterable loop를 저장, generator를 반복적으로 사용 가능

    ### parameter
        `iterator` : iterable 객체 (함수)
            ex) Generator(lambda: (x for x in range(0, 100))) 
    """
    def __init__(self, iterator):
        self.iterator = iterator

    def __iter__(self):
        return self.iterator()

class Tensor:
    """
    ## Tensor
    data의 variable array를 정의  

    ### args
        data (np.ndarray) : 데이터 array
        name (str) : 데이터 변수명, dataframe의 컬럼명    
    """
    def __init__(self, data:np.ndarray, names=None):
        self.data = data

        if names==None:
            names = [str(i) for i in range(np.size(data, 1))]
        self.names = names

        self._aligned_dtypes = self.dtypes(align=True)
        self._num_idxs = self.dindex('nums', return_names=False)
        self._num_ars = self._generate_ar(dtype='nums', ignore_na=True)

    @property
    def shape(self):
        return self.data.shape
    
    @property
    def ndim(self):
        return self.data.ndim
    
    @property
    def T(self):
        return self.data.T
    
    def astype(self, dtype):
        return self.data.astype(dtype)

    def nrow(self):
        return len(self.data) if self.data.ndim==1 else [len(self.data) for _ in range(np.size(self.data, 1))]
    
    def __len__(self):
        return len(self.data)

    def _iterate_ar(self, dtype, ignore_na=True):
        if dtype == None:
            ar = self.data.copy()
        else: 
            ar = self.dslice(dtype)

        if ar.ndim == 1:
            ar = self._adjust_dims(ar)

        for i in range(np.size(ar, 1)):
            ar_ind = ar[:,i]
            if ignore_na:
                ar_ind = ar_ind[pd.notnull(ar_ind)]
            
            yield ar_ind

    def _generate_ar(self, dtype, ignore_na=True):
        return Generator(lambda: self._iterate_ar(dtype, ignore_na))

    def _adjust_dims(self, ar):
        if ar.ndim==1:
            ar = np.expand_dims(ar, 1)
        return ar
    
    def _align_dtype(self, dtype):
        if dtype in ['string', 'categorical']:
            res = 'str'
        elif dtype in ['floating', 'mixed-integer-float']:
            res = 'float'
        elif dtype == 'integer':
            res = 'int'
        elif 'mixed' in dtype and dtype != 'mixed-integer-float':
            res = 'mixed'
        elif dtype == 'boolean':
            res = 'bool'
        elif dtype in ['datetime64', 'datetime', 'date', 'timedelta64', 'timedelta', 'time', 'period']:
            res = 'dt'
        elif dtype == 'unknown-array':
            res = 'unknown'
        else:
            res = dtype

        return res
    
    def find_datetime(self):
        ars = self._adjust_dims(self.data)

        dts, idxs = [], []
        for i, ad in enumerate(self._aligned_dtypes):
            ar = ars[:, i]
                
            check_ar = ar[~pd.isna(ar)]
            if ad in ['string', 'str']:
                try:
                    check_ar.astype(float)
                except:
                    try:
                        check_ar.astype(np.datetime64)
                        dts.append(self.names[i])
                        idxs.append(i)
                    except:
                        pass
            else:
                continue
        
        return idxs, dts

    def dtypes(self, align=True):
        ars = self._adjust_dims(self.data)

        outputs = []
        for i in range(np.size(ars, 1)):
            ar = ars[:, i]
            res = infer_dtype(ar, skipna=True)

            if align:
                res = self._align_dtype(res)
            
            outputs.append(res)

        return outputs if len(outputs)>1 else outputs[0]
    
    def dindex(self, dtype, return_names =True):
        ar = self._adjust_dims(self.data)
        _dtypes = np.array(self._aligned_dtypes)

        if dtype == 'all':
            idxs = np.arange(0, np.size(ar, 1))
        elif dtype =='nums':
            idxs = []
            for i in ['int', 'float']:
                idxs.append(np.where(_dtypes==i)[0])
            idxs = np.concatenate(idxs)
        else:
            idxs = np.where(_dtypes==dtype)[0]

        if return_names:
            return idxs, np.array(self.names)[idxs]
        else:
            return idxs
    
    def dslice(self, dtype):
        ar = self._adjust_dims(self.data)
        if dtype == 'nums':
            idxs = self._num_idxs
        else:
            idxs = self.dindex(dtype, return_names=False)

        ar = ar[:, idxs]

        dtype_bags = {
            'int': int,
            'float': float,
            'nums': float
        }

        if dtype in dtype_bags.keys():
            ar = ar.astype(dtype_bags[dtype])

        return ar.squeeze()
    
    def size(self, axis = None):
        return np.size(self.data, axis=axis)
    
    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = shape[0]
        return self.data.reshape(shape)

    def transpose(self, *axes):
        if len(axes) == 0:
            axes = None
        elif len(axes) == 1:
            if isinstance(axes[0], (tuple, list)) or axes[0] is None:
                axes = axes[0]
        return self.data.transpose(axes)

    def sum(self, axis=None, keepdims=False):
        ar = self.dslice('nums')
        return ar.sum(axis=axis, keepdims=keepdims)

    def mean(self, axis=None, ignore_na=True, keepdims=False):
        ar = self.dslice('nums')        
        if ignore_na:
            return np.nanmean(ar, axis= axis, keepdims=keepdims)
        else:
            return np.mean(ar, axis= axis, keepdims=keepdims)

    def std(self, axis=None, ignore_na=True, keepdims=False):
        ar = self.dslice('nums')
        if ignore_na:
            return np.nanstd(ar, axis= axis, keepdims=keepdims)
        else:
            return np.std(ar, axis= axis, keepdims=keepdims)

    def max(self, axis=None, ignore_na=True, keepdims=False):
        ar = self.dslice('nums')
        
        if ignore_na:
            return np.nanmax(ar, axis= axis, keepdims=keepdims)
        else:
            return np.max(ar, axis= axis, keepdims=keepdims)

    def min(self, axis=None, ignore_na=True, keepdims=False):
        ar = self.dslice('nums')

        if ignore_na:
            return np.nanmin(ar, axis= axis, keepdims=keepdims)
        else:
            return np.min(ar, axis= axis, keepdims=keepdims)

    def quantile(self, q, axis=None, ignore_na=True, keepdims=False):
        ar = self.dslice('nums')
        if ignore_na:
            return np.nanquantile(ar, q=q, axis= axis, keepdims=keepdims)
        else:
            return np.quantile(ar, q=q, axis= axis, keepdims=keepdims)

    def skew(self, axis=None, ignore_na=True, keepdims=False):
        ar = self.dslice('nums')
        nan_policy = 'omit' if ignore_na else 'propagate'
        return stats.skew(ar, axis=axis, nan_policy=nan_policy, keepdims=keepdims)

    def kurtosis(self, axis=None, ignore_na=True, keepdims=False):
        ar = self.dslice('nums')
        nan_policy = 'omit' if ignore_na else 'propagate' # scipy
        return stats.kurtosis(ar, axis=axis, nan_policy=nan_policy, keepdims=keepdims)

    def mode(self, axis=None, keepdims=False):
        return stats.mode(self.data, axis=axis, nan_policy='omit', keepdims=keepdims).mode

    def isnull(self):
        return pd.isnull(self.data)

    def histogram(self, bins, density=False):
        return [np.histogram(ar, bins=bins, density=density) for ar in self._num_ars]

    def kde(self):
        return [stats.gaussian_kde(ar) for ar in self._num_ars]

    def density(self, bins):
        kdes = self.kde()
        output = []
        for k, a in zip(kdes, self._num_ars):
            cover_range = (a.max()-a.min())/bins
            cover_init= a.min()

            edge = []
            for _ in range(bins+1):
                edge.append(cover_init)
                cover_init += cover_range

            len_ar = len(a)
            pdf = k.integrate_box_1d
            cnt = [len_ar*pdf(edge[v], edge[v+1]) for v in range(bins)]

            output.append((cnt, edge))

        return output

    def bucket(self, bins, smoothing=False):
        _hist = self.density(bins) if smoothing else self.histogram(bins)

        output = []
        for h in _hist:
            cnt, edge = h
            output.append([(edge[i], edge[i+1], cnt[i]) for i in range(len(cnt))])

        return output if len(output)>1 else output[0]

    def unique(self, return_counts=True):
        ar = np.where(pd.notnull(self.data), self.data, 'null')

        if ar.ndim == 1:
            ar = np.expand_dims(ar,1)

        _lbs, _cnt = [], []
        for i in range(np.size(ar, 1)):
            if return_counts:
                lbs, cnt = np.unique(ar[:,i], return_counts=True)
                _lbs.append(lbs.squeeze().tolist())
                _cnt.append(cnt.squeeze().tolist())
            else:
                lbs = np.unique(ar[:,i], return_counts=False)
                _lbs.append(lbs.squeeze().tolist())
    
        if len(_lbs) == 1:
            _lbs, _cnt = _lbs[0], _cnt[0]

        return _lbs, _cnt if return_counts else _cnt

    def _time_attr(self, ar, unit = 's'):
        """시간 데이터의 속성을 계산하는 함수.

        return (delta_ar, forward_idx, inverse_idx)
            delta_ar : 시간 간격의 차이
            foward_idx : 시간이 진행하고 있는 index
            inverse_idx : 시간이 역진행하고 있는 index (순환 데이터인 경우)
        """
        dt_ar = pd.to_datetime(ar).values
        delta_ar = np.diff(dt_ar) / np.timedelta64(1, unit)

        stand_idx = np.isclose(delta_ar, 0.0, rtol=0.0, atol=1e-1)
        nonstand_idx = np.where(stand_idx==False)[0]
        inverse_idx = np.where(delta_ar<0.0)[0]
        forward_idx = np.asarray(list(set(nonstand_idx) - set(inverse_idx)))

        return (delta_ar, forward_idx, inverse_idx)

    def freq(self, unit='s'):
        """### freq
        시간 데이터의 주기성을 추론\n
        주기값으로 추정되는 값들의 평균과 최대빈도값 계산

        ### return
            (mean freq, mode freq)
        """
        ars = self._generate_ar(dtype='dt')
        output=[]
        for a in ars:
            delta_ar, forward_idx, _ = self._time_attr(a, unit=unit)
            forward_ar = delta_ar[forward_idx]
            output.append((np.mean(forward_ar), np.unique(forward_ar).max()))
        
        return output
        
    def cycle(self, unit='s'):
        """### freq
        시간 데이터의 순환성을 추론\n
        순환값으로 추정되는 값들의 평균과 최대빈도값 계산

        ### return
            (mean cycle, mode cycle)
        """
        ars = self._generate_ar(dtype='dt')
        output=[]
        for a in ars:
            _, _, inverse_idx = self._time_attr(a, unit=unit)
            interval_idx = np.diff(inverse_idx)
            if len(interval_idx) > 0:
                output.append((np.mean(interval_idx), np.unique(interval_idx).max()))
            else:
                output.append((None, None))
        return output
    
    def time_edge(self, is_end):
        dt_ars = self._generate_ar('dt')
        edges = np.max if is_end else np.min
        
        return np.array([edges(ar) for ar in dt_ars])

    def __repr__(self):
        if self.data is None:
            return f'Tensor(None),\n name={self.names}'
        p = str(self.data).replace('\n', '\n' + ' ' * 9)
        return f'Tensor({p}), name={self.names})'