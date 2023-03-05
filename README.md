</br>

# 다양한 콘텐츠 소비 유도를 위한 웹소설 추천시스템 개발
`python==3.8.15`, `pytorch==1.12,1+cu113`, `프로젝트v.2.0`


### 주요 사용 알고리즘
- `NovelNet` : Li et al.2022[[1](#reference)], `biGRU`, `BahdanauAttention`  
- `GRU4Rec` : Hidashi et al.2016[[3](#reference)] `GRU`  
- `TransformerEncoder` : Vaswani et al.2017[[4](#reference)] `MultiHeadAttention`,  `PositionalEncoding`  
- 프로젝트 개발 알고리즘 
    - `SelfEncoding` : `BahdanauAttention` 에서 아이디어 착안, 하나의 대표 시퀀스를 얻을 수 있도록(정보압축) 간단한 알고리즘 구현
    - `SubGenreEncoding` : `PositionalEncoding` 에서 아이디어 착안, 모델 규모의 변화 없이 추가정보 전달을 위해 장르정보 임베딩 사전학습 후, 시퀀스 데이터에 더하여 정보전달
    - `Attention4NovelRec` : 새로운 어텐션 기반 모델 아키텍쳐 개발


### 주요 수행 내역  
> 서비스 이용 활성화를 위해, 유효소비가 예상되는 다양한 소설들을 추천하는 추천시스템 개발

- 유효소비가 예상되는 콘텐츠들을 추천할 수 있도록, 모델이 학습하게 될 타겟데이터 선별
- 다양한 콘텐츠들의 소비를 유도할 수 있도록, 과거 소비 이력이 있는 소설들에 대한 마스킹 알고리즘 구현(모델 내부)
- `NewNovelRec` : `NovelNet` 에서 제안된 새로운 소설 추천을 위한 알고리즘을 별도의 모델로 구현
- `Attention4Rec` : `Transformer` 에서 제안된 인코더 구조를 기반으로 하는 모델 개발(다중분류 학습)
- `GRU4Rec` : 성능 비교를 위해, 세션 기반 추천 task 에서 좋은 성능을 보이는 `GRU4Rec` 기반 모델 개발
- `SubGenreEncoding` : 모델 규모 변화 없이 성능 향상을 위해 장르정보(임베딩) 사전학습 진행 후 시퀀스에 전달. 하위장르 임베딩에 대해 상위장르 정보를 지닐 수 있도록 설계
- `Attention4NovelRec` : 사전학습된 장르정보를 이용하는 어텐션 메커니즘 기반의 별도의 모델 개발


### Trouble Shooting
- 과적합 발생 이슈
    - 문제 : 기본 실험 모델 구현 및 하이퍼-파라미터 테스트 시, 학습 성능과 검증 성능의 차이가 크게 벌어지는 문제 발생
    - 연구 : 유효소비 데이터 선별 과정에서 데이터 규모는 크게 줄었으나, 기존 논문들에서 제안된 모델 규모가 유지되었던 것이 문제로 추정, 정보 손실을 야기시켜 해결 가능할 것으로 추정
    - 해결 :
        1.  모델규모 및 드랍아웃 비율 조절
        2. `GRU4Rec` 모델에 대해 마지막 히든값을 시퀀스인코딩 결과값으로 이용(i.e. seq2vec)
        3. `SelfEncoding` : 하나의 대표 시퀀스로 정보를 압축하기 위해 `BahdanauAttention` 에서 아이디어 착안, 간단한 알고리즘 구현. `Attention4Rec` 모델에 적용
            - $f_{\text{enc}} \triangleq (H \cdot W)^T \cdot H,\quad H \in \mathbb{R}^{\text{|session|}\times |h|}, \ W \in \mathbb{R}^{|h|}$
    - 결과 : 검증 성능의 소폭 향상 및 학습 성능의 큰 폭의 하락. 다만, 성능 절대치가 낮다는 문제

- 낮은 성능 이슈
    - 문제 : 최고 검증 성능이 `GRU4Rec` 0.11(recall@20) 으로 성능 절대치가 낮다는 문제
    - 연구 : 
        1. `NewNovelNet` 모델의 경우, 반복 소비 추천을 위해 추출된 특징값[[1](#reference)] 이용이 문제로 추정(특징값 임베딩 후 병합하여 이용, 프로젝트 task 에 부합하지 않는 정보 전달이 원인인 것으로 추정)
        2. 다른 모델의 경우 정보 손실 야기에 따른 구조적 한계가 있을 것으로 추정.
    - 해결 :
        1. `NovelNet` 모델에 대해, FeatureSelection 진행(Subset 테스트)
        2. `GRU4Rec` 모델에 대해 히든값 차원을 상향조정(seq2vec 을 이용하는 만큼, 일반화 된 정보를 더 모을 수 있을 것으로 판단)
        3. `SubGenreEncoding` : 모델의 규모 변화 없이, 추가 정보 전달을 위해 장르정보(임베딩) 사전학습 후 시퀀스에 전달. 하위장르 임베딩에 대해 고정된 FF 를 통해 상위장르 다중분류로 학습. 유사한 장르 정보를 얻을 수 있도록 유도.
        4. `Attention4NovelRec` : 드랍아웃 영향을 줄이는 대신, 모델 규모를 축소하기 위한 방향으로 새로운 모델 아키텍쳐 개발. 
            - 핵심 알고리즘 :
                - 소설 시퀀스 임베딩 + `SubGenreEncoding`
                - `PositionalEncoding`
                - `BahdanauAttention`(단, $S_{t-1} = H$)
                - 새로운 소설 추천을 위한 마스킹
                - FC 레이어를 통한 소설별 점수 도출
    - 결과 : 큰 폭의 성능향상 기록.


### 프로젝트 결과
- Li et al.2022[[1](#reference)] 에서 사용된 텐센트 QQ브라우저의 웹소설 이용 데이터 이용
    - 2021.11.11 ~ 2021.11.22 기간 동안의 랜덤샘플링 된 863,000 여개의 인터렉션 데이터
- 평가기준 : Hidashi et al.2016[[3](#reference)] 에서 주 평가지표로 사용된 recall@20 사용

|model|train|valid|test|
|----|----|----|----|
|Attention4Rec|0.17549|0.12877|0.12608|
|NewNovelRec|0.15403|0.13999|0.14156|
|GRU4Rec|0.49632|0.19592|0.19198|
|Attention4NovelRec|0.76474|0.20258|0.20333|
| ㄴ K-Fold@4 `V.2.0`|||0.21044|

- 평가결과
    - `Attention4NovelRec` 모델이 최고 성능을 기록
    - `GRU4Rec`, `Attention4NovelRec` 모델에 대해 과적합 현상 재발생하였으나, 유의미한 검증 성능의 향상도 있었음
    - `Attention4NovelRec` 모델의 경우, 업데이트 필요 주기가 길며 단순한 연산만으로 구현된 모델로, 유지보수 및 예측 비용이 상대적으로 저렴한 효율적인 모델
    - 셀프어텐션 기법으로 콘텐츠 추천에서 중요한 short-term intention 반영 기대(e.g. 소비자의 감정)
    - 유효소비가 예상되는 새로운 소설을 추천하며, 비즈니스 실효성 및 목적 적합성 충족
    - 다양한 지표로 비즈니스 실효성 평가가 가능할 것 
      - e.g. 새로운 소설에 대한 유효소비 주기 변화, 추천된 새로운 소설과 이용자가 탐색한 새로운 소설 간의 유효소비 비율

- 향후 과제
    - [ ] Li et al.2022[[1](#reference)] 연구와 같이 세션 길이 1개 및 등장횟수 5회 미만 소설 데이터 제거 후 성능비교
    - [ ] Hidasi et al.2018[[5](#reference)] 에서 제안된 샘플링 기법 및 다양한 손실함수를 이용하여 성능비교 평가    
   
</br>

## 버전 업데이트 내역
- v.2.0 : k-fold cross validation 도입, Trainer 클래스에서 데이터로더 로드하도록 수정
</br>

## Reference
###### [1]Yuncong Li, et al, 2022, Modeling User Repeat Consumption Behavior for Online Novel Recommendation, RecSys’22 September 18–23
###### [2]김대원, 2019, 추천 알고리즘의 개념과 적용 그리고 발전의 양상, Broadcasting Trend & Insight October 2019 Vol.20, 한국콘텐츠 진흥원
###### [3]Balazs Hidasi, et al, 2016, Session-Based Recommendations with Recurrent Neural Networks, ICLR 2016
###### [4]A.Vaswani, et al, 2017, Attention is all you need, NIPS 2017
###### [5]Balazs Hidasi, et al, 2018, Recurrent Neural Networks woth Top-k Gains for Session-based Recommendations, CIKM'18 October 22-26
