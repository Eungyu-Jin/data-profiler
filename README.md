# data-profiler

custom 데이터 검증 모듈, tfdv 데이터 검증 테스트

### tensor
데이터셋 변수 저장, Tensor 정의

### validator
1. DataInfer
  - 데이터 추론
  - 데이터 추론 결과 report -> json
2. Datavalidate
  - 데이터 검증
  - 수치형 : jensen-shannon divergence
  - 범주형 : L-infinite norm
  - TFDV (tensorflow data-validation) 알고리즘 참고
