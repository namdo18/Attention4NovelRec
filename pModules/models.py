
# [ 버전 업데이트 내역 ]
#   *-- (v.2.0) datasetHandler 입력 -> 필요 정보 직접 입력(ModelTrainer 에서 매번 데이터셋 로드하는 불필요 감소 목적)

import torch
import numpy as np

# (v.1.0) from pModules.data import DatasetHandler

import os
import pickle as pk


class BahdanauAttention(torch.nn.Module) :

    def __init__(self, hiddenSize:int, mode:str, device:str=None) :
        super(BahdanauAttention, self).__init__()
        self.Wh = torch.nn.Linear(hiddenSize, hiddenSize, bias=False)
        self.Ws = torch.nn.Linear(hiddenSize, hiddenSize, bias=False)
        self.V = torch.nn.Linear(hiddenSize, 1, bias=False)
        self.device = device if device is not None else 'cuda:0' if torch.cuda.is_available() else 'cpu'

        if mode == 'NovelNet' :
            torch.nn.init.xavier_uniform_(self.Wh.weight)
            torch.nn.init.xavier_uniform_(self.Ws.weight)

        self.to(self.device)    

    def forward(self, H, S=None, padMask=None) :
        S = S if S is not None else H
        Wh = self.Wh(H)
        Ws = self.Ws(S)
        aScore = self.V(torch.tanh(Wh + Ws)).transpose(-1, -2) # V 와 연산을 위해 차원 유지 및 transpose
        padMask = padMask.unsqueeze(1) # aScore 에 적용하기 위해 차원 증가

        aScore = aScore.masked_fill(padMask, -1e8)
        aDist = torch.nn.functional.softmax(aScore, dim=-1)
        aValue = torch.bmm(aDist, H)
        
        # 명시적 처리 및 불필요 차원 축소
        rSeqs = aValue.squeeze(1)
        return rSeqs

class NewNovelRec(torch.nn.Module) :
    baseModel = 'NewNovelRec'
    binaryCategoricalFeatures = ['expose', 'click', 'intro', 'read', 'real_read', 'collect']
    
    def __init__(self,
                 # (v.1.0) datasetHandler:DatasetHandler,
                 maxValueByFeatures:dict, # v.2.0
                 mode:str='NovelNet',
                 features:list=['ItemId', 'intro', 'read', 'real_read', 'read_duration', 'recency', 'novelCount', 'repeatConsumGap'],
                 novelEmbeddingSize:int=128, 
                 featureEmbeddingSize:int=32,
                 gruHiddenSize:int=128,
                 dropoutRatio:float=0.5,
                 device:str=None           
                 ) :
        # 유효 입력 확인
        modes = ['NovelNet', 'User']
        if mode not in modes :
            print(f"\n[Error] mode:{mode} is not in {modes}\n")
            raise KeyError      

        super(NewNovelRec, self).__init__()
        # (v.1.0) self.datasetHandler = datasetHandler # preprocessing 완료 이후의 데이터셋 핸들러
        self.mode = mode # 성능 비교 테스트를 위한 모드 구분
        self.features = features # 학습 시, 사용할 feature 리스트
        self.device = device if device is not None else 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.name = NewNovelRec.baseModel + f'+{mode}+nCols{len(features)}+embSize{novelEmbeddingSize}+dropout{dropoutRatio}+{self.device}'
        self.conf = {'mode':mode, 
                     'features':features, 
                     'novelEmbSize':novelEmbeddingSize,
                     'featureEmbSize':featureEmbeddingSize, 
                     'gruHiddenSize':gruHiddenSize,
                     'dropoutRatio':dropoutRatio,
                     'device':self.device}          

        # [ 임베딩 레이어 정의 ]
        self.EmbeddingDict = torch.nn.ModuleDict()
        gruInputSize = 0
        # (v.1.0) maxValueByFeatures = self.datasetHandler.maxValueByFeatures # feature 별 vocab_size 의미
        for col in self.features :
            embeddingSize = featureEmbeddingSize if col != 'ItemId' else novelEmbeddingSize
            if self.mode == 'User' :
                embeddingSize = 3 if col in NewNovelRec.binaryCategoricalFeatures else embeddingSize
            gruInputSize += embeddingSize
            self.EmbeddingDict[col] = torch.nn.Embedding(maxValueByFeatures[col]+1, embeddingSize, padding_idx=0)
        self.EmbDropout = torch.nn.Dropout(dropoutRatio)
        
        # [ 시퀀스인코더(양방향GRU) 정의 ]
        self.GRUencoder = torch.nn.GRU(gruInputSize, int(gruHiddenSize/2), bidirectional=True, batch_first=True)
        if mode == 'NovelNet' : 
            self.GRUencoder.all_weights[0][0] = torch.nn.init.xavier_uniform
            self.GRUencoder.all_weights[0][1] = torch.nn.init.xavier_uniform
        self.SeqDropout = torch.nn.Dropout(dropoutRatio)
        
        # [ 어텐션 레이어 정의 ](별도 구현)
        self.Attention = BahdanauAttention(gruHiddenSize, mode=self.mode, device=self.device)

        self.to(self.device)

    def forward(self, batchedFeatureDict) :
        batch_size = len(batchedFeatureDict['ItemId'])
        maxSeqLen = len(batchedFeatureDict['ItemId'][0])

        # [ 특징값 별 임베딩 후, (인터렉션)데이터 별 통합 ]
        embeddedFeatures = list()
        for feature in self.features :
            embeddedFeatures.append(self.EmbeddingDict[feature](torch.stack(batchedFeatureDict[feature])))
        seqs = torch.cat(embeddedFeatures, dim=-1).to(self.device)
        seqs = self.EmbDropout(seqs)

        # [ 시퀀스인코딩 ]
        #   *-- 양방향 GRU 를 통해 시퀀스 특징 추출(인코딩)
        #   *-- 각 방향 별 마지막 정보를 하나로 통합(batch_size,Hidden)
        #       *-- 시퀀스데이터들(batch_size,maxSeqLen,Hidden)과 차원을 맞춰주기 위해 차원 증가
        hidden = torch.zeros(2, batch_size, self.conf['gruHiddenSize']//2).to(self.device)
        encodedSeqs, bidirectionInfo = self.GRUencoder(seqs, hidden)
        bidirectionInfo = bidirectionInfo.transpose(0, 1).reshape(batch_size, -1).unsqueeze(1)
        encodedSeqs = self.SeqDropout(encodedSeqs)
        bidirectionInfo = self.SeqDropout(bidirectionInfo)

        # [ 어텐션 레이어 ]
        #   *-- 패딩 데이터에 대한 확률 연산을 방지하기 위해 마스크 별도 구현
        #   *-- 고객 이용(시퀀스)데이터에서 중요한 (인터렉션)데이터에 가중치 부여
        #       *-- 고객 이용(시퀀스)의 중요 특징이 표현된 데이터 : representationSeq(rSeq)
        padMask = torch.stack(batchedFeatureDict['ItemId']).eq(0).to(self.device)
        rSeqs = self.Attention(encodedSeqs, bidirectionInfo, padMask=padMask)
        
        # [ 새로운 소설들에 대한 유사도 도출 ]
        #   *-- rSeq 와 임베딩 된 개별 소설들 간의 dot-product 을 통한 내적유사도 도출
        #       *-- torch.nn.functional.cross_entropy 이용으로 별도 확률 변환 X
        #   *-- 새로운 소설 추천들에 대한 확률값만을 계산하기 위해 소비된 소설들 마스킹
        #       *-- 소비된 소설의 정수 인코딩 정보를 인덱스로 활용
        novelScore = torch.matmul(rSeqs, self.EmbeddingDict['ItemId'].weight.T)
        consumMask = torch.zeros_like(novelScore).to(self.device)
        r_indices = np.zeros((batch_size, maxSeqLen), dtype=int) + np.arange(batch_size).reshape(-1,1)
        c_indices = torch.stack(batchedFeatureDict['ItemId']) 
        consumMask[r_indices, c_indices] = 1

        newNovelScore = novelScore.masked_fill(consumMask.bool(), -1e8)
        return newNovelScore

class GRU4Rec(torch.nn.Module) :
    baseModel = 'GRU4Rec'

    def __init__(self, 
                 # (v.1.0) datasetHandler:DatasetHandler,
                 maxValueByFeatures:dict, # v.2.0
                 mode:str='seq2vec',
                 maxSeqLen:int=49, # v.2.0
                 hiddenSize:int=128, 
                 nLayer:int=1, 
                 embFLAG:bool=True, 
                 embSize:int=128,
                 dropoutRatio:float=0.5,
                 device:str=None                
                 ) :  

        # 유효 입력 확인
        modes = ['seq2vec', 'seq2seq']
        if mode not in modes :
            print(f"\n[Error] mode:{mode} is not in {modes}\n")
            raise KeyError                                

        super(GRU4Rec, self).__init__()
        # (v.1.0) self.datasetHandler = datasetHandler
        # (v.1.0) self.vocabSize = datasetHandler.maxValueByFeatures['ItemId']+1
        self.vocabSize = maxValueByFeatures['ItemId']+1 # v.2.0
        self.device = device if device is not None else 'cuda:0' if torch.cuda.is_available() else 'cpu' 
        self.name = GRU4Rec.baseModel + f"+{mode}+emb{embFLAG}{embSize if embFLAG else 0}+nLayer{nLayer}+hiddenSize{hiddenSize}+dropout{dropoutRatio}+{self.device}"
        self.conf = {'mode':mode,
                     'hiddenSize':hiddenSize,
                     'nLayer':nLayer,
                     'embFLAG':embFLAG,
                     'embSize':embSize,
                     'dropoutRatio':dropoutRatio,
                     'device':self.device}          
        
        # [ 임베딩 레이어 정의 ]
        #   *-- 임베딩 미사용시, 원핫인코딩 사용
        #       *-- GRU 입력 길이 상이
        if embFLAG :
            self.EmbeddingLayer = torch.nn.Embedding(self.vocabSize, embSize)
        else :
            self.EmbeddingLayer = self._getOneHotEnc
        self.EmbDropout = torch.nn.Dropout(dropoutRatio)

        # [ GRU 레이어 정의 ]
        inputSize = embSize if embFLAG else self.vocabSize
        self.GRULayer = torch.nn.GRU(inputSize, hiddenSize, nLayer ,batch_first=True)
        self.SeqDropout = torch.nn.Dropout(dropoutRatio)

        # [ 피드포워드 레이어 정의 ]
        #   *-- mode 에 따라 입력 길이 상이
        #       *-- 'seq2vec' 모드는 마지막 hidden 값을 사용 : 일반화 성능 비교 평가 목적
        # (v.1.0) inputSize = hiddenSize*datasetHandler.maxSeqLen if mode != 'seq2vec' else hiddenSize
        inputSize = hiddenSize*maxSeqLen if mode != 'seq2vec' else hiddenSize # v.2.0
        self.FeedForwardLayer = torch.nn.Linear(inputSize, self.vocabSize)            
        
        self.to(self.device)

    def forward(self, batchedFeatureDict) :

        itemSeq = torch.stack(batchedFeatureDict['ItemId']).to(self.device)
        batch_size = itemSeq.shape[0]
        maxSeqLen = itemSeq.shape[1]

        embeddedItemSeqs = self.EmbeddingLayer(itemSeq)
        if self.conf['embSize'] :
            embeddedItemSeqs = embeddedItemSeqs.to(self.device)
        embeddedItemSeqs = self.EmbDropout(embeddedItemSeqs)

        hidden = torch.zeros(self.conf['nLayer'], batch_size, self.conf['hiddenSize']).to(self.device)
        output, hidden = self.GRULayer(embeddedItemSeqs, hidden)
        rSeqs = output.reshape(batch_size, -1) if self.conf['mode'] != 'seq2vec' else hidden[-1]
        rSeqs = self.SeqDropout(rSeqs)

        novelScore = self.FeedForwardLayer(rSeqs)
        
        consumMask = torch.zeros_like(novelScore).to(self.device)
        r_indices = np.zeros((batch_size, maxSeqLen), dtype=int) + np.arange(batch_size).reshape(-1,1)
        c_indices = torch.stack(batchedFeatureDict['ItemId']) 
        consumMask[r_indices, c_indices] = 1
        newNovelScore = novelScore.masked_fill(consumMask.bool(), -1e8)

        return newNovelScore     

    def _getOneHotEnc(self, ItemIdSeqs:torch.tensor) :
        oneHotSeqs = torch.zeros(ItemIdSeqs.shape[0], ItemIdSeqs.shape[1], self.vocabSize)
        oneHotSeqs.scatter_(2, ItemIdSeqs.unsqueeze(2), 1.)
        return oneHotSeqs        

class PosEncoder(torch.nn.Module) :
    def __init__(self, seqLen:int, encDim:int, cycle:float=1e-4, device:str=None) :
        super(PosEncoder, self).__init__()
        self.device = device if device is not None else 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.PosEncoder = self._getPosEncoder(seqLen, encDim, cycle).to(self.device)
        self.to(self.device)
    
    def _getPosEncoder(self, seqLen:int, encDim:int, cycle:float=1e-4) :
        posEncoder = torch.zeros(seqLen, encDim)
        posEncoder.requires_grad = False

        position = torch.arange(0, seqLen).unsqueeze(1)  # seq 에서의 위치 인덱스
        idx = torch.arange(0, encDim, 2) # 인코딩 차원에서의 짝수 인덱스만 추출
        posEncoder[:,0::2] = torch.sin(position*(cycle**torch.div(idx,encDim,rounding_mode='trunc')))
        posEncoder[:,1::2] = torch.cos(position*(cycle**torch.div(idx,encDim,rounding_mode='trunc')))
        return posEncoder

    def forward(self, embeddedSeq) :
        encodedSeq = embeddedSeq + self.PosEncoder
        return encodedSeq

class MultiHeadSelfAttention(torch.nn.Module) :
    # scaledDotProduct 내부 함수로 구현
    def __init__(self, nhead:int, encDim:int, bias:bool=True, device:str=None) :
        super(MultiHeadSelfAttention, self).__init__()
        self.device = device if device is not None else 'cuda:0' if torch.cuda.is_available() else 'cpu'

        if encDim % nhead :
            print(f"encDim%nhead is not 0")
            raise ValueError

        self.nhead = nhead
        self.encDim = encDim
        
        self.Wq = torch.nn.Linear(encDim, encDim, bias=bias)   
        self.Wk = torch.nn.Linear(encDim, encDim, bias=bias)
        self.Wv = torch.nn.Linear(encDim, encDim, bias=bias)
        self.Wo = torch.nn.Linear(encDim, encDim, bias=bias)

        self.to(self.device)

    def forward(self, encodedSeq, padMask=None) :
        batch_size = encodedSeq.shape[0]

        Q, K, V = self.Wq(encodedSeq), self.Wk(encodedSeq), self.Wv(encodedSeq)
        Q, K, V = self._split(Q), self._split(K), self._split(V)
        attentionValue = self._scaledDotProduct(Q, K, V, padMask)
        concatenated_AttentionValue = attentionValue.transpose(1, 2).reshape(batch_size, -1, self.encDim)
        attentionSeq = self.Wo(concatenated_AttentionValue)
        return attentionSeq

    def _split(self, input) :
        # [ encDim 차원으로 추출된 Q, K, V 에서 multihead 에 따라 split 하여 병렬처리 ]
        #   *-- 연산 시, 시퀀스 기준으로 맞춰주기 위해 transpose(torch.permute) 진행
        batch_size = input.shape[0]
        input =  input.reshape(batch_size, -1 , self.nhead, (self.encDim//self.nhead))
        return input.permute(0, 2, 1, 3)

    def _scaledDotProduct(self, Q, K, V, mask=None) :
        attentionScore = torch.einsum('abcd, abde -> abce', Q, K.transpose(-1,-2))
        attentionScore /= self.encDim//self.nhead
        if mask is not None :
            # [ 패딩토큰 해당 -1e8 지정 ]
            #   *-- row_mask, col_mask 각각 적용
            #       *-- 실제값 기준, 모든 패딩토큰(columnn)과의 attention score 연산 방지
            #       *-- masked_fill 적용을 위해 차원 확대
            #           *-- row_mask : (batch, 1, seqLen, 1)
            #           *-- col_mask : (batch, 1, 1, seqLen)
            
            row_mask = mask.unsqueeze(1).unsqueeze(-1)
            col_mask = mask.unsqueeze(1).unsqueeze(2)
            attentionScore = attentionScore.masked_fill(row_mask, -1e8)
            attentionScore = attentionScore.masked_fill(col_mask, -1e8)

        attentionDist = torch.nn.functional.softmax(attentionScore, dim=-1)
        attentionValue = torch.einsum('abcd, abde -> abce', attentionDist, V)
        return attentionValue

class LayerNormWithResidualCC(torch.nn.Module) :
    def __init__(self, encDim:int, device:str=None) :
        super(LayerNormWithResidualCC, self).__init__()
        self.device = device if device is not None else 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.gamma = torch.nn.Parameter(torch.ones(encDim)).to(self.device)
        self.beta = torch.nn.Parameter(torch.zeros(encDim)).to(self.device)
        self.to(self.device)

    def forward(self, encodedSeq, attentionSeq) :
        # Residual Connection
        seq = encodedSeq + attentionSeq
        
        # LayerNormalization
        seqMean = seq.mean(dim=-1, keepdim=True)
        seqStd = seq.std(dim=-1, keepdim=True)
        normalizedSeq = (seq-seqMean)/(seqStd + 1e-8)
        normalizedSeq = self.gamma*normalizedSeq + self.beta

        return normalizedSeq
        
class AttentionLayer(torch.nn.Module) :
    def __init__(self, 
                 encDim:int=128, 
                 nhead:int=4, 
                 hidden:int=128,
                 dropoutRatio:float=0.5, 
                 attentionBias:bool=True, 
                 device:str=None
                 ) :
        super(AttentionLayer, self).__init__()
        self.device = device if device is not None else 'cuda:0' if torch.cuda.is_available() else 'cpu'
        
        self.Attention = MultiHeadSelfAttention(nhead, encDim, attentionBias, device=device)
        self.LayerNorm = LayerNormWithResidualCC(encDim, device=device)
        self.FeedForward = torch.nn.Sequential(
            torch.nn.Linear(encDim, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, encDim)
        )
        self.NormDropout = torch.nn.Dropout(dropoutRatio)
        self.FeedForwardDropout = torch.nn.Dropout(dropoutRatio)
        self.to(self.device)

    def forward(self, encodedSeq, padMask=None) :
        attentionSeq = self.Attention(encodedSeq, padMask)

        normalizedSeq = self.LayerNorm(encodedSeq, attentionSeq)
        normalizedSeq = self.NormDropout(normalizedSeq)

        output = self.FeedForward(normalizedSeq)
        output = self.FeedForwardDropout(normalizedSeq)
        return output
        
class Attention4Rec(torch.nn.Module) :
    baseModel = 'Attention4Rec'

    def __init__(self,
                 # (v.1.0) datasetHandler:DatasetHandler, 
                 maxValueByFeatures:dict, # v.2.0
                 mode:str,
                 method:str,
                 maxSeqLen:int=49, # v.2.0
                 n_EncLayers:int=1, 
                 encDim:int=128, 
                 nhead:int=4, 
                 hidden:int=128, 
                 embDropoutRatio:float=0.5,
                 dropoutRatio:float=0.5,
                 attentionBias:bool=True,
                 device:str=None
                 ) :
        # 유효 입력 확인                 
        validModes = ['Attention', 'novelRec']
        if mode not in validModes : 
            print(f"mode.{mode} is not in {validModes}")
            raise KeyError
        validMethods = ['FF', 'DotSimilarity']
        if method not in validMethods : 
            print(f"method.{method} is not in {validMethods}")
            raise KeyError        

        super(Attention4Rec, self).__init__()
        self.device = device if device is not None else 'cuda:0' if torch.cuda.is_available() else 'cpu'
        # (v.1.0) self.datasetHandler = datasetHandler
        self.mode = mode
        self.method = method
        self.name = Attention4Rec.baseModel + f"+{mode}+{method}+nLayer{n_EncLayers}+encDim{encDim}+nhead{nhead}+hidden{hidden}+dropout{dropoutRatio}+{self.device}"
        self.conf = {'mode':mode,
                     'method':method,
                     'nLayer':n_EncLayers,
                     'encDim':encDim,
                     'nhead':nhead,
                     'hidden':hidden,
                     'dropout':dropoutRatio,
                     'device':self.device}

        # (v.1.0) seqLen = datasetHandler.maxSeqLen
        self.seqLen = maxSeqLen # v.2.0
        # (v.1.0) vocabSize = datasetHandler.maxValueByFeatures['ItemId']+1 
        vocabSize = maxValueByFeatures['ItemId']+1 # v.2.0

        self.Embedding = torch.nn.Embedding(vocabSize, encDim)
        self.EmbDropout = torch.nn.Dropout(embDropoutRatio)

        # [ mode('novelRec') ]
        #   *-- 사전 학습된 하위장르(원본데이터셋 'subtype') 임베딩 값을 고정으로 이용
        #       *-- '임베딩레이어 -> 피드포워드(고정) -> 상위장르분류' 로 임베딩
        #           *-- SubGenrePreEmbedding 클래스로 구현
        #           *-- 같은 상위장르분류에 속하는 하위장르의 특징값(임베딩) 추출
        #       *-- ItemIdSeq 에 장르정보를 더해주기 위해 사용
        #           *-- 대부분의 이용자들이 다양한 장르의 소설이 아닌 특정 선호 장르의 소설들 위주로 읽는다는 점에 착안
        #           *-- positionalEnc 아이디어 응용, 임베딩된 시퀀스에 사전학습된 하위장르 임베딩값을 더해줌 
        if self.mode == 'novelRec' :
            with open(f'data/embeddedSubGenre++{encDim}.tensor', 'rb') as f :
                embeddedSubGenre = pk.load(f).to(self.device)
            self.SubGenreEmb = torch.nn.Embedding(embeddedSubGenre.shape[0], embeddedSubGenre.shape[1])
            self.SubGenreEmb.weight = torch.nn.Parameter(embeddedSubGenre)
            self.SubGenreEmb.weight.requires_grad = False

        # (v.1.0) self.PosEncoder = PosEncoder(seqLen, encDim, device=device)
        self.PosEncoder = PosEncoder(self.seqLen, encDim, device=device) # v.2.0
        self.AttentionLayers = torch.nn.ModuleList(
            [AttentionLayer(encDim, nhead, hidden, dropoutRatio, attentionBias, self.device)
            for _ in range(n_EncLayers)]
        )

        # [ 셀프인코딩레이어 정의 ]
        #   *-- 어텐션레이어(=transformerEncoder) 결과값(batch,seqLen,encDim)을 하나의 시퀀스데이터(batch,encDim)로 인코딩
        #       *-- self.SelfEncLayer 결과값(batch,seqLen,1)을 transpose(-1,-2) 하여 입력값(batch,seqLen,encDim)과 재연산
        #           *-- (batch,1,seqLen,)@(batch,seqLen,encDim) = (batch,encDim)
        #   *-- method('FF') : 피드포워드레이어를 통해 소설별 score 도출
        #   *-- method('DotSimilarity') : 셀프인코딩레이어 결과값을 소설 임베딩값(transpose) 와 연산하여 유사도 도출
        self.SelfEncLayer = torch.nn.Linear(encDim, 1)
        if self.method == 'FF' :
            self.ScoreLayer = torch.nn.Linear(encDim, vocabSize)
        
        self.to(self.device)

    def forward(self, batchedFeatureDict) :
    
        seq = torch.stack(batchedFeatureDict['ItemId']).to(self.device)
        padMask = seq.eq(0).to(self.device)
        
        batch_size = seq.shape[0]
        # (v.1.0) maxSeqLen = self.datasetHandler.maxSeqLen

        embeddedSeq = self.Embedding(seq)
        embeddedSeq = self.EmbDropout(embeddedSeq)
        if self.mode == 'novelRec' :
            subGenreSeq = torch.stack(batchedFeatureDict['subtype']).to(self.device)
            subGenreSeq_embedded = self.SubGenreEmb(subGenreSeq)
            embeddedSeq = embeddedSeq + subGenreSeq_embedded
        encodedSeq = self.PosEncoder(embeddedSeq)
  
        for AttentionLayer in self.AttentionLayers :
            encodedSeq = AttentionLayer(encodedSeq, padMask)

        output = self.SelfEncLayer(encodedSeq).transpose(-1, -2)
        output = torch.bmm(output, encodedSeq).squeeze(1)

        if self.method == 'FF' :
            novelScore = self.ScoreLayer(output)
        elif self.method == 'DotSimilarity' :
            novelScore = torch.matmul(output, self.Embedding.weight.T)
    
        consumMask = torch.zeros_like(novelScore).to(self.device)
        # (v.1.0) r_indices = np.zeros((batch_size, maxSeqLen), dtype=int) + np.arange(batch_size).reshape(-1,1)
        r_indices = np.zeros((batch_size, self.seqLen), dtype=int) + np.arange(batch_size).reshape(-1,1) # v.2.0
        c_indices = torch.stack(batchedFeatureDict['ItemId']) 
        consumMask[r_indices, c_indices] = 1
        newNovelScore = novelScore.masked_fill(consumMask.bool(), -1e8)        
        return newNovelScore

class SubGenrePreEmbedding(torch.nn.Module) :
    def __init__(self, encDim, subGenreVocabSize, GenreVocabSize) :
        super(SubGenrePreEmbedding, self).__init__()
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.encDim = encDim
        self.savePath = 'data/'
        if not os.path.isdir('data') :
            os.mkdir('data')        

        self.Embedding = torch.nn.Embedding(subGenreVocabSize, encDim)
        self.Fixed = torch.nn.Linear(encDim, GenreVocabSize, bias=False)
        self.Fixed.weight.requires_grad = False
        self.to(self.device)

    def forward(self, subGenre) :
        subGenre = subGenre.to(self.device)
        embedded = self.Embedding(subGenre)
        score = self.Fixed(embedded)
        return score

    def fit(self, subGenre=None, genre=None, epochs=5, lr=0.05) :
        if subGenre is None :
            with open(self.savePath+'subGenre.tensor', 'rb') as f :
                subGenre = pk.load(f)
        if genre is None :
            with open(self.savePath+'genre.tensor', 'rb') as f :
                genre = pk.load(f)                

        Optim = torch.optim.Adam(self.parameters(), lr)
        r_indices = torch.randperm(len(subGenre))
        subGenre, genre = subGenre[r_indices], genre[r_indices]
        batch_size = len(subGenre)//4
        for epoch in range(epochs) :
            for iter in range(4) :
                score = self(subGenre[iter*batch_size:(iter+1)*batch_size])
                labels = genre[iter*batch_size:(iter+1)*batch_size].reshape(-1).to(self.device)
                loss = torch.nn.functional.cross_entropy(score, labels, ignore_index=0)
                
                Optim.zero_grad()
                loss.backward()
                Optim.step()

                with torch.no_grad() :
                    preds = score.argmax(dim=-1).reshape(-1)
                    print(f"epoch.{epoch}/iter.{iter}: acc.{sum(preds == labels)/len(labels)} loss.{loss.item()}")
                
    def preEmbSave(self) :
        embedded = self.Embedding.weight.cpu()
        with open(self.savePath + f'embeddedSubGenre++{self.encDim}.tensor', 'wb') as f :
            pk.dump(embedded, f)

class Attention4NovelRec(torch.nn.Module) :
    baseModel = 'Attention4NovelRec'

    def __init__(self,
                 # (v.1.0) datasetHandler:DatasetHandler, 
                 maxValueByFeatures:dict, # v.2.0
                 method:str,
                 maxSeqLen:int=49, # v.2.0
                 encDim:int=128,  
                 dropoutRatio:float=0.5,
                 device:str=None,
                 attentionConfs:dict=None,
                 ) :
        # 유효 입력 확인                 
        validMethods = ['Bahdanau', 'ScaledDot']
        if method not in validMethods : 
            print(f"[Error] method.{method} is not in {validMethods}")
            raise KeyError       
        if method == 'ScaledDot' and attentionConfs is None :
            print(f"[Wanring] attentionConfs <- None, will be set default") 
            attentionConfs = {'nhead':4, 'encDim':128, 'bias':True, 'device':None}

        super(Attention4NovelRec, self).__init__()
        self.device = device if device is not None else 'cuda:0' if torch.cuda.is_available() else 'cpu'
        # (v.1.0) self.datasetHandler = datasetHandler
        self.method = method
        self.name = Attention4NovelRec.baseModel + f"+{method}+encDim{encDim}+dropout{dropoutRatio}+{self.device}"
        self.conf = {'method':method,
                     'encDim':encDim,
                     'dropout':dropoutRatio,
                     'device':self.device}

        # (v.1.0) seqLen = datasetHandler.maxSeqLen
        # (v.1.0) vocabSize = datasetHandler.maxValueByFeatures['ItemId']+1 
        self.maxSeqLen = maxSeqLen # v.2.0
        vocabSize = maxValueByFeatures['ItemId']+1 # v.2.0

        # [ 임베딩/인코딩 레이어 정의 ]
        self.Embedding = torch.nn.Embedding(vocabSize, encDim)
        self.EmbDropout = torch.nn.Dropout(dropoutRatio)
        
        with open(f'data/embeddedSubGenre++{encDim}.tensor', 'rb') as f :
            embeddedSubGenre = pk.load(f).to(self.device)
        self.SubGenreEmb = torch.nn.Embedding(embeddedSubGenre.shape[0], embeddedSubGenre.shape[1])
        self.SubGenreEmb.weight = torch.nn.Parameter(embeddedSubGenre)
        self.SubGenreEmb.weight.requires_grad = False

        # (v.1.0) self.PosEncoder = PosEncoder(seqLen, encDim, device=device)
        self.PosEncoder = PosEncoder(self.maxSeqLen, encDim, device=device) # v.2.0
    
        # [ 어텐션 레이어 정의 ]
        self.AttentionLayer = BahdanauAttention(encDim,'User',self.device) if self.method=='Bahdanau' else MultiHeadSelfAttention(**attentionConfs)
        if self.method == 'ScaledDot' :
            self.SelfEncLayer = torch.nn.Linear(encDim, 1)
        self.ScoreLayer = torch.nn.Linear(encDim, vocabSize)
        
        self.to(self.device)

    def forward(self, batchedFeatureDict) :
        seq = torch.stack(batchedFeatureDict['ItemId']).to(self.device)

        padMask = seq.eq(0).to(self.device)
        batch_size = seq.shape[0]
        # (v.1.0) maxSeqLen = self.datasetHandler.maxSeqLen

        # [ 임베딩 / 인코딩 ]
        embeddedSeq = self.Embedding(seq)
        embeddedSeq = self.EmbDropout(embeddedSeq)
        
        subGenreSeq = torch.stack(batchedFeatureDict['subtype']).to(self.device)
        subGenreSeq_embedded = self.SubGenreEmb(subGenreSeq)
        embeddedSeq = embeddedSeq + subGenreSeq_embedded
        encodedSeq = self.PosEncoder(embeddedSeq)
        
        # [ 어텐션 ]
        encodedSeq = self.AttentionLayer(encodedSeq, padMask=padMask)
        if self.method == 'ScaledDot' :
            output = self.SelfEncLayer(encodedSeq).transpose(-1, -2)
            output = torch.bmm(output, encodedSeq).squeeze(1)

        novelScore = self.ScoreLayer(output if self.method == 'ScaledDot' else encodedSeq)

        # [ 읽은 소설 필터링 ]
        consumMask = torch.zeros_like(novelScore).to(self.device)
        # (v.1.0) r_indices = np.zeros((batch_size, maxSeqLen), dtype=int) + np.arange(batch_size).reshape(-1,1)
        r_indices = np.zeros((batch_size, self.maxSeqLen), dtype=int) + np.arange(batch_size).reshape(-1,1) # v.2.0
        c_indices = torch.stack(batchedFeatureDict['ItemId']) 
        consumMask[r_indices, c_indices] = 1
        newNovelScore = novelScore.masked_fill(consumMask.bool(), -1e8)        
        
        return newNovelScore        