
# [ 버전 업데이트 내역 ]
#   *-- v.2.0
#       *-- trainset_full.csv 이용 : k-fold cross-validation 사용 목적, 기존의 trainset + validset
#       *-- DatasetHandler.callDataLoader, 데이터셋 외부 입력 허용 : split 된 validset 에 대한 로더 호출 목적

import torch
import pandas as pd
import numpy as np

from collections import defaultdict

import os
import json
from tqdm import tqdm
import pickle as pk

class FeedSet(torch.utils.data.Dataset) :
    def __init__(self, feedset:pd.DataFrame, maxSeqLen:int) :
        super(FeedSet, self).__init__()
        self.feedset = feedset
        self.maxSeqLen = maxSeqLen

    def __getitem__(self, idx) :
        return self.feedset.iloc[[idx]]

    def __len__(self) :
        return len(self.feedset)        

class DatasetHandler :
    notUsedFeatures = ['DataType', 'set', 'book_id', 'bookname', 'start_time', 'end_time']
    extractedFeatures = ['recency', 'novelCount', 'repeatConsumGap', 'timeDiff']

    def __init__(self, 
                 mode:str,
                 maxSeqLen:int=49,
                 batch_size:int=2048,
                 batch_shuffle:bool=True,
                 valid_split_ratio:float=0.21, # 전체 데이터셋 기준 비율
                 test_split_ratio:float=0.2, 
                 originDataset:pd.DataFrame=None
                 ) :
        # 유효 입력 확인(모드별 설명은 아래)
        if mode not in ['base', 'preprocssing', 'saved'] :
            print(f"\n[ Error ] input '{mode}' mode is not in ['base', 'preprocessing', 'saved']")
            raise KeyError

        super(DatasetHandler, self).__init__()
        self.maxSeqLen = maxSeqLen # 시퀀스데이터 최대 길이 설정(전처리 후 최대 시퀀스데이터 길이로 설정)
        self.batch_size = batch_size
        self.batch_shuffle = batch_shuffle
        self.valid_split_ratio = valid_split_ratio
        self.test_split_ratio = test_split_ratio
        
        self.originEncMap = dict() # 원본 범주형 데이터의 인코딩 매핑 정보 
        self.validTargetIndices = list() # 유효타겟 데이터들의 원본 데이터셋에서의 인덱스
        self.validUsingIndices = list() # 유효사용 데이터들의 원본 데이터셋의 인덱스
        self.extractedUniques = defaultdict(list) # 추출 특징값 별 고유데이터 정보
        self.validEncMap = dict() # 유효데이터들의 인코딩 매핑 정보
        self.maxValueByFeatures = dict() # 각 특징값 별 최댓값(임베딩 입력으로 사용)
        self.savePath = None # 파일 저장 경로

        # 프로젝트 디렉토리 기준 경로
        if not os.path.isdir('data') :
            os.mkdir('data')
        if not os.path.isdir('data/Handler') :
            os.mkdir('data/Handler')
        self.savePath = os.getcwd() + '/data/Handler/'

        # [ 입력 모드 별 실행 ]
        #   *-- 'base' 모드 : 아무 실행 X
        #   *-- 'preprocessing' 모드 : 자동 전처리 실행(원본 데이터셋:pd.DataFrame 입력 필요)
        #   *-- 'saved' 모드 : 데이터 파일 로드해서 사용
        if mode == 'preprocessing' :
            if type(originDataset) != pd.DataFrame :
                print(f"\n[ Error ] dataset-type is not valid")
                raise TypeError
            self.preprocessing(originDataset)

        elif mode == 'saved' :
            with open(self.savePath+'maxValueByFeatures.p', 'rb') as f :
                self.maxValueByFeatures = pk.load(f)            
        
    def preprocessing(self, originDataset:pd.DataFrame) :
        # [ 학습에 이용할 형태로 데이터셋 전처리 ]
        #   *-- 사전 지정된 경로에 전처리 완료된 데이터셋 저장됨
        attrDataset = self._basicPreprocessing(originDataset)
        targetDataset = self._validTargetExtraction(attrDataset)
        targetDataset = self._adjustSessRatio(targetDataset)
        seqDataset = self._trans2Seq(attrDataset, targetDataset)
        seqDataset = self._validEnc(attrDataset, seqDataset)
        self._splitDataset(seqDataset)

    def callDataLoader(self, mode:str='trainset', dataset=None, batch_size=None, batch_shuffle=None) :
        # [ 데이터로더 호출 ]
        #   *-- 입력 모드로 사용할 데이터셋 설정(default:trainset)
        #   *-- (v.2.0) dataset=None, 외부에서 데이터셋 입력(cross-validation 사용 목적)
        #   *-- collate_fn 함수를 통해 모델 입력(feedset) 형태로 변환(self.maxSeqLen 참고하여 패딩)
        mode_list = ['trainset', 'validset', 'testset', 'trainset_full']
        if mode not in mode_list : 
            print(f"\n===== error message =====")
            print(f"{mode} is not in {mode_list}\n")
            raise KeyError

        dataset = self.loadDataset([mode])[0] if dataset is None else dataset
        feedset = FeedSet(dataset, self.maxSeqLen)

        # [ batch 비교 평가를 위해 외부 옵션 설정 허용 ]
        batch_size = batch_size if batch_size is not None else self.batch_size
        batch_shuffle = batch_shuffle if batch_shuffle is not None else self.batch_shuffle

        maxSeqLen = self.maxSeqLen
        def collate_fn(batchset) :
            featureDict = defaultdict(list)
            for row in batchset :
                for col in row.columns :
                    if col == 'SessionId' : continue
                    if col == 'label' :
                        featureDict[col].append(row[col].to_list())
                        continue
                    def extraction(seq) :                        
                        seq += [0]*(maxSeqLen - len(seq))
                        featureDict[col].append(torch.tensor(seq[:maxSeqLen]))
                    row[col].apply(extraction)
            return featureDict

        dataLoader = torch.utils.data.DataLoader(feedset, batch_size=batch_size, shuffle=batch_shuffle, collate_fn=collate_fn)
        return dataLoader

    def loadDataset(self, setName:list) :
        # [ 버전 업데이트 내용 ]
        #   *-- (v.2.0) trainset_full 추가 : trainset+validset, cross-validation 사용 목적
        setNames = ['attrDataset', 'targetDataset', 'targetDataset_adjusted',
                    'seqDataset', 'seqDataset_encoded', 'trainset', 'validset','testset', 'trainset_full']
        for name in setName :
            if name not in setNames :
                print(f"=======================================")
                print(f"setName error: {name} is not valid")
                print(f"valid setNames: \n{setNames}")
                raise ValueError

        result = list()
        for name in setName :
            tmpSet = pd.read_csv(self.savePath+name+'.csv').copy()

            if 'targetDataset' in name :
                tmpSet.index = tmpSet['Unnamed: 0'].to_list()
                tmpSet.drop('Unnamed: 0', axis=1, inplace=True)
            else :
                if 'Unnamed: 0' in tmpSet.columns :
                    tmpSet.drop('Unnamed: 0', axis=1, inplace=True)

            if setNames.index(name) > 2 :
                for col in tmpSet.columns :
                    if col in ['SessionId', 'Time' ,'label'] : continue
                    else : tmpSet[col] = tmpSet[col].apply(lambda x : list(map(int, x.strip('['']').split(','))))

            result.append(tmpSet)

        return tuple(result)

    def _basicPreprocessing(self, originDataset:pd.DataFrame, auto_save:bool=True) :
        
        print(f"===== basicPreprocessing(about 21 min.) =====")
        
        # [ 원본데이터셋의 Attributes 컬럼의 json 데이터들을 개별 컬럼들로 전환 ]
        attr_list = list(json.loads(originDataset.iloc[0].Attributes).keys())
        attr_dict = defaultdict(list) 
        for row in tqdm(originDataset.iterrows(), total=len(originDataset)) :
            for attr in attr_list :
                attr_dict[attr].append(json.loads(row[1].Attributes)[attr])

        for attr in attr_list :
            originDataset[attr] = attr_dict[attr]

        originDataset.drop('Attributes', axis=1, inplace=True)


        # [ (원본)범주형데이터 인코딩 ]
        for col in ['category', 'subtype', 'author'] :
            uniques = originDataset[col].unique()
            self.originEncMap[col] = {unique:i+1 for i, unique in enumerate(uniques)}
            originDataset[col] = originDataset[col].map(self.originEncMap[col])
        

        # [ 미사용 컬럼 사전 제거 ]
        #   *-- 'DataType', 'set' : NovelNet 저자들이 데이터 구분을 위해 라벨링 해놓은 값(체리피킹의심..)
        #   *-- 'book_id', 'bookname' : 'ItemId' 컬럼과 동일 기능으로 제거
        #   *-- 'start_time', 'end_time' : 
        #           1) 'start_time' 은 'Time' 컬럼과 시차 8 시간 차이뿐 동일 기능으로 제거
        #           2) 'end_time' 은, 'read_duration' 컬럼과 ('end_time'-'start_time') 의 값이 다른 것으로 보아, 서버응답 시간으로 추정하여 제거
        for col in DatasetHandler.notUsedFeatures :
            originDataset.drop(col, axis=1, inplace=True)


        # [ null 값 제거(전체데이터의 3.08%) ]
        originDataset.dropna(axis=0, inplace=True)


        # [ read_duration 단위 처리(min.) ]
        #   *-- NovelNet 저자들은 3600 으로 나누어 처리함 : 원본데이터 sec. 단위로 추정
        #       *-- 저자들은 3600 초과값들을 60 으로 고정, 이하값들은 60 을 나누어 사용함(후처리 +1, 패딩고려)
        #       *-- min. 단위 전환 후 한 시간을 최댓값으로 둔 것으로 보임
        #           *-- 유효치 않은 데이터(e.g. 그냥 켜둔 케이스)의 영향력을 제한하려는 목적으로 보임
        #               *-- 다만, 원래 유효치 않은 데이터가 최댓값을 지니게 되면서 정보 왜곡이 발생할 수 있을 것
        #               *-- 본 프로젝트에서는 범위조사 후 필터링 할 예정
        #   *-- 이후 과정에서 범위 처리 필요
        originDataset.read_duration = (originDataset.read_duration.map(int)//60)+1

        if auto_save :
            originDataset.to_csv(self.savePath+'attrDataset.csv')
            with open(self.savePath+'originEncMap.p', 'wb') as f :
                pk.dump(self.originEncMap, f)

        print(f">> basicPreprocessing Complete!\n")

        # 명시적 처리
        return originDataset

    def _validTargetExtraction(self, attrDataset:pd.DataFrame, auto_save:bool=True) :
        # [ 유효 타겟 데이터 선별 ]
        #   *-- 'read_duration' 에 대한 데이터 범위 조사(IQR) 수행
        #       *-- 'real_read'==1(두 페이지 이상 읽은 경우)데이터 대상
        #           *-- Q3+IQR(3, 보수치) 을 유효범위로 볼 때 세 시간 초과 데이터를 이상치로 간주
        #               *-- 경험에서 벗어나지 않는 범주로 판단하여 유효범위로 사용(95% 데이터가 해당)
        #               *-- 유효범위를 지정함에 따라 실질적이지 않은 데이터(e.g. 그냥 켜둔 케이스) 필터링 가능
        #       *-- 'real_read'==0(두 페이지도 읽지 않은 경우)데이터 조사
        #           *-- 유효데이터 판별을 위한 기준점 조사(두 페이지도 읽지 않은 사용자 대다수보다는 많은 시간을 읽어야 유효데이터로 간주)
        #           *-- Q3+IQR(6, 보수치) 를 유효범위로 봐도 0.4 분
        #               *-- 경험에 부합하지 않음(e.g. 훑어보기 위해 두 페이지를 0.41 분 동안 읽은 이용데이터를 유효데이터로 간주하기는 어려울 것)
        #               *-- 경험에 근거하여 최소 유효데이터 기준점을 4분으로 설정(한 회차 읽는 시간 고려)
        #
        #   *-- 유효 타겟 데이터 별도 선별
        #       *-- 위의 유효 범위를 충족하며, 세션 내 최초로 등장한 소설 데이터들만 유효 타겟 데이터로 선별
        #           *-- 비즈니스 실효성 달성 목적
        #           *-- 세션의 첫번째 소설 데이터는 제외
        #
        print(f"===== validTargetExtraction(about 45 min.) =====")
        validReadDurationCond = ((attrDataset['read_duration']<=180) & (attrDataset['read_duration']>=4))
        targetCand = attrDataset[(attrDataset['real_read']==1) & validReadDurationCond]

        checkSess = None
        for r_idx in tqdm(range(len(targetCand)), total=len(targetCand)) :
            row = targetCand.iloc[[r_idx]]

            df_idx = row.index.to_list()[0]
            sessId = int(row['SessionId'].to_numpy())
            itemId = int(row['ItemId'].to_numpy())

            cond = attrDataset['SessionId']==sessId
            if checkSess is None or sessId != checkSess :
                checkSess = sessId
                if attrDataset[cond].iloc[[0]].index.to_list()[0] == df_idx :
                    continue
                else :
                    checkItem = attrDataset[cond].loc[:df_idx].ItemId.to_list()[:-1]
                    if itemId not in checkItem :
                        self.validTargetIndices.append(df_idx)
            else :
                checkItem = attrDataset[cond].loc[:df_idx].ItemId.to_list()[:-1]
                if itemId not in checkItem :
                    self.validTargetIndices.append(df_idx)
        
        targetDataset = attrDataset.loc[self.validTargetIndices].copy()

        if auto_save :
            targetDataset.to_csv(self.savePath+'targetDataset.csv')
            with open(self.savePath+'validTargetIndices.p', 'wb') as f :
                pk.dump(self.validTargetIndices, f)

        print(f">> validTargetExtraction Complete!\n")

        return targetDataset

    def _adjustSessRatio(self, targetDataset:pd.DataFrame, auto_save:bool=True) :
        # [ 타겟데이터에 대한 세션 비율 조정 ]
        #   *-- 타겟데이터를 기준으로 사용데이터를 선별할 것이기 때문에 타겟데이터에 대한 비율 조정 수행
        #   *-- 세션 길이가 비정상적으로 긴 세션이 있을 경우
        #       *-- 해당 세션 내 (인터렉션)데이터들이 (이후)시퀀스데이터로 변환되면서(세션 내 본인 이전의 데이터들을 누적적으로 지니게 됨)
        #       *-- 동일한 특징을 지니는 데이터들의 비중이 증가
        #           *-- 일부 편향 발생 우려(일반화 성능 저하)
        #           *-- 손실값이 큰 특징을 공유하는 데이터들이 많을 경우 문제가 발생할 수 있을 것
        #       *-- 전체 시퀀스데이터 대상 세션 비중 조사 필요
        #   *-- 유효 타겟데이터셋에서의 세션 범위 조사(IQR) 수행
        #       *-- IQR(3, 보수치) 유효범위에 해당하는 최대 세션 별 시퀀스데이터 개수 : 9
        #           *-- 유효범위를 벗어나는 시퀀스데이터 비율(전체대비) : 4.109%
        #           *-- 유효범위에 포함되는 96% 가량의 데이터로 진행
        
        print(f"===== adjustSessRatio(about 40 sec.) =====")

        checkSessRatio = targetDataset.SessionId.value_counts()
        checkSessIndices = checkSessRatio[checkSessRatio>9].index.to_list()
        for s_idx in tqdm(checkSessIndices, total=len(checkSessIndices)) :
            cond = targetDataset['SessionId'] == s_idx
            targetDataset.drop(targetDataset[cond].iloc[9:].index.to_list(), inplace=True)
        
        self.validTargetIndices = targetDataset.index.to_list()

        if auto_save :
            targetDataset.to_csv(self.savePath+'targetDataset_adjusted.csv')
            with open(self.savePath+'validTargetIndices_adjusted.p', 'wb') as f :
                pk.dump(self.validTargetIndices, f)

        print(f">> adjustSessRatio Complete!\n")     

        # 명시적 처리
        return targetDataset

    def _trans2Seq(self, attrDataset:pd.DataFrame, targetDataset:pd.DataFrame, auto_save:bool=True) :
        # [ 유효 타겟데이터를 라벨로 지니는 시퀀스데이터 생성 ]
        #   *-- 각 유효 타겟데이터 별로 세션 내 본인 이전의 (인터렉션)데이터들을 시퀀스화
        #   *-- 특징값 추출 함께 진행
        #       *-- 'recency' : NovelNet 논문에서의 'time_diff'(target 데이터 'Time' 과의 시간차)
        #           *-- 타겟의 'Time(=start_time, 서버요청시간)' 입력 후 추천을 위해 빠른 연산이 필요할 것이나
        #           *-- 실효성 있는 범위에서의 성능(소요시간)이 나올 수 있다면, 시간에 대한 정보(e.g. 최근성 편향)를 이용할 수 있을 것
        #           *-- 실제 서비스를 위해서는 알고리즘 수정이 필요할 것(Time 입력)
        #       *-- 'novelCount' : NovelNet 논문에서는 반복소비추천을 위해, 소설의 과거 소비 이력(횟수)을 추출하여 사용
        #           *-- 반복소비추천을 위해 정의된 특징값이나, 소비패턴정보를 내포하고 있다는 측면에서 우선 추출한 이후 성능비교
        #       *-- 'repeatConsumGap' : NovelNet 논문에서의 'temporal_gap'(소설 별 소비 간격)
        #           *-- 반복소비추천을 위해 정의된 특징값이나, 소비패턴정보를 내포하고 있다는 측면에서 우선 추출한 이후 성능비교
        #       *-- 'timeDiff' : 본 프로젝트에서 정의한 특징값으로, 이용시간 별 간격을 나타냄
        #           *-- 이용시간 간격으로 흥미의 정도나, 지루함의 정도를 나타낼 수 있을 것으로 기대
        #       *-- 패딩토큰 0 을 감안, 모든 유효데이터에 +1 진행
        print(f"===== trans2seq(about - min.) =====")
        features = defaultdict(list)
        for target_idx in tqdm(self.validTargetIndices, total=len(self.validTargetIndices)) :
            sessId = targetDataset.loc[target_idx].SessionId
            seqIndices = attrDataset[attrDataset['SessionId']==sessId].sort_values(by='Time').loc[:target_idx].index.to_list()[:-1]
            self.validUsingIndices = list(set(self.validUsingIndices + seqIndices))

            seqSet = attrDataset.loc[seqIndices]
            for col in targetDataset.columns :
                if col == 'SessionId' :
                    features[col].append(list(set(seqSet[col].to_list()))[0])
                    continue

                trueValue = 1 if col not in ['ItemId', 'category', 'subtype', 'author'] else 0
                seq = (seqSet[col]+trueValue).to_list()
                features[col].append(seq)
                
                if col == 'Time' :
                    recency = (((targetDataset.loc[target_idx].Time - seqSet[col])//3600).map(int)+1).to_list()
                    features['recency'].append(recency)
                    self.extractedUniques['recency'] = list(set(self.extractedUniques['recency'] + recency))

                    timeDiff = ((seqSet[col].diff().fillna(0)//3600).map(int)+1).to_list()
                    features['timeDiff'].append(timeDiff)
                    self.extractedUniques['timeDiff'] = list(set(self.extractedUniques['timeDiff'] + timeDiff))            
                
                elif col == 'ItemId' :
                    novelCount = defaultdict(lambda : 1)
                    lastConsumTime = dict()
                    tmp_list4Count, tmp_list4Gap = list(), list()
                    for seq_idx, novel in enumerate(seqSet.ItemId) :
                        novelCount[novel] += int(1)
                        tmp_list4Count.append(novelCount[novel])
                        if novel not in lastConsumTime :
                            lastConsumTime[novel] = seqSet.Time.iloc[seq_idx]
                            tmp_list4Gap.append(1)
                        else :
                            gap = int((seqSet.Time.iloc[seq_idx] - lastConsumTime[novel])//3600)+1
                            tmp_list4Gap.append(gap)
                            lastConsumTime[novel] = seqSet.Time.iloc[seq_idx]

                    features['novelCount'].append(tmp_list4Count)
                    features['repeatConsumGap'].append(tmp_list4Gap)
                    self.extractedUniques['novelCount'] = list(set(self.extractedUniques['novelCount'] + tmp_list4Count))
                    self.extractedUniques['repeatConsumGap'] = list(set(self.extractedUniques['repeatConsumGap'] + tmp_list4Gap))

        seqDataset = pd.DataFrame(features)
        seqDataset['label'] = targetDataset.ItemId.to_list()

        if auto_save :
            seqDataset.to_csv(self.savePath+'seqDataset.csv')
            with open(self.savePath+'validUsingIndices.p', 'wb') as f :
                pk.dump(self.validUsingIndices, f)
            with open(self.savePath+'extractedUniques.p', 'wb') as f :
                pk.dump(self.extractedUniques, f)

        print(f">> tran2seq Complete!\n")  

        return seqDataset

    def _validEnc(self, attrDataset:pd.DataFrame, seqDataset:pd.DataFrame, auto_save:bool=True) :
        # [ 유효사용 데이터에 대한 범주형 인코딩 ]
        #   *-- 임베딩 입력 목적의 최댓값 추출 함께 진행
        #   *-- 시퀀스데이터이기 때문에 매핑함수를 구현하여 apply
        for col in ['ItemId', 'category', 'subtype', 'author'] :
            validUsingIndices = self.validUsingIndices if col != 'ItemId' else list(set(self.validUsingIndices + self.validTargetIndices)) 
            uniques = attrDataset[col].loc[validUsingIndices].unique()
            self.validEncMap[col] = {unique:i+1 for i, unique in enumerate(uniques)}
            self.validEncMap[col][0] = 0 # 패딩토큰 추가

            def mapping(seq) : 
                return list(map(lambda x : self.validEncMap[col][x], seq))
            seqDataset[col] = seqDataset[col].apply(mapping)
            self.maxValueByFeatures[col] = max(self.validEncMap[col].values())
            
            if col == 'ItemId' :
                seqDataset['label'] = seqDataset['label'].map(self.validEncMap[col])

        # [ 유효사용 데이터에 대한 연속형(정수) 인코딩 ]
        #   *-- 단순 최댓값 사용 시, 무의미한 임베딩 발생
        #   *-- 'read-duration' 컬럼은 유효범위 초과값에 대해 1로 재설정(basicPreprocessing 참고)
        #   *-- 데이터 범위 조사(최댓값 고정 목적)
        #       *-- 'recency' : IQR(1.5) 유효범위 303(시간) 으로 경험에 부합하지 않음
        #           *-- 경험에 근거하여 최대 일주일(24*7=168시간) 범위내를 유의미한 데이터로 간주
        #       *-- 'repeatCounsumGap' : IQR(1.5) 유효범위 274(시간) 으로 경험에 부합하지 않음
        #           *-- 경험에 근거하여 최대 일주일(24*7=168시간) 범위내를 유의미한 데이터로 간주
        #       *-- 'timeDiff' : IQR(1.5) 유효범위 274(시간) 으로 경험에 부합하지 않음
        #           *-- 경험에 근거하여 최대 일주일(24*7=168시간) 범위내를 유의미한 데이터로 간주        
        for col in ['read_duration', 'recency', 'repeatConsumGap', 'timeDiff', 'novelCount'] :
            if col == 'novelCount' :
                self.maxValueByFeatures[col] = max(self.extractedUniques[col])
                continue
            limit = 180 if col == 'read_duration' else 168
            def cutoff(seq) :
                return list(map(lambda x : x if x <= limit else 1, seq))
            seqDataset[col] = seqDataset[col].apply(cutoff)
            self.maxValueByFeatures[col] = limit

        # [ 미처리 컬럼들에 대한 최댓값 추출 ]
        for col in ['expose', 'click', 'intro', 'read', 'real_read', 'collect'] :
            self.maxValueByFeatures[col] = attrDataset[col].loc[self.validUsingIndices].max()

        # 작업내용 저장
        if auto_save : 
            seqDataset.to_csv(self.savePath+'seqDataset_encoded.csv')
            with open(self.savePath+'validEncMap.p', 'wb') as f :
                pk.dump(self.validEncMap, f)
            with open(self.savePath+'maxValueByFeatures.p', 'wb') as f :
                pk.dump(self.maxValueByFeatures, f)

        return seqDataset
            
    def _splitDataset(self, seqDataset:pd.DataFrame, auto_save:bool=True) :

        # [ 세션 스플릿 ]
        #   *-- 시간 정렬 후 스플릿으로 실제 상황과 유사하도록 설정
        targetDataset = self.loadDataset(['targetDataset_adjusted'])[0]
        targetDataset['idx'] = torch.arange(len(targetDataset))
        sortedIndices = targetDataset.sort_values(by='Time').idx.to_list()

        trainRange = int(len(sortedIndices)*(1-(self.test_split_ratio+self.valid_split_ratio)))
        validRange = trainRange + int(len(sortedIndices)*self.valid_split_ratio)
        trainIndices = sortedIndices[:trainRange]
        validIndices = sortedIndices[trainRange:validRange]
        testIndices = sortedIndices[validRange:]

        seqDataset.drop('Time', axis=1, inplace=True) # 비사용컬럼 제거
        trainset = seqDataset.iloc[trainIndices].drop('SessionId', axis=1)
        validset = seqDataset.iloc[validIndices]
        testset = seqDataset.iloc[testIndices]

        # [ testset/validset 세션 중복 제거 ]
        #   *-- 세션에서 추출된 시퀀스데이터들이기 때문에 정보를 공유하고 있음
        #   *-- 올바른 성능 비교 평가를 위해 중복 세션 제거 
        testset = testset.drop_duplicates(['SessionId']).drop('SessionId', axis=1) # 비사용컬럼 제거
        validset = validset.drop_duplicates(['SessionId']).drop('SessionId', axis=1)

        # 작업내용 저장
        if auto_save :
            trainset.to_csv(self.savePath+'trainset.csv')
            validset.to_csv(self.savePath+'validset.csv')
            testset.to_csv(self.savePath+'testset.csv')
        
