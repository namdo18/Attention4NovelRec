
# [ 버전 업데이트 내역 ]
#   *-- v.2.0
#       *-- k-fold cross-validation 적용
#       *-- 모델 별 DataLoader 호출 -> Trainer 에서 한번 호출
#       *-- 모든 모델에 대해 동일한 배치 적용

from pModules.data import DatasetHandler

import torch
import numpy as np

from collections import defaultdict

import os
import pickle as pk
from tqdm import tqdm
from datetime import datetime
from time import time

class ModelTrainer :
    # evalutaion 기능 내부 함수로 구현
    # 이하 고정 사용(내부에서 선언하여 사용)
    #   *-- loss = torch.nn.functional.cross_entropy 
    #   *-- optim = torch.optim.Adam 
    def __init__(self, 
                 models:list, # [model1, model2, ..]
                 datasetHandler:DatasetHandler, # v.2.0
                 defaultOptim:torch.optim,
                 defaultLr:float=None,
                 maxSeqLen:int=49, # v.2.0
                 confs:dict=None,
                 device:str=None
                 ) :

        self.models = models
        self.datasetHandler = datasetHandler
        self.confs = defaultdict(list)
        self.device = device if device is not None else 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.history = dict()
        self.savePath = None # 모델 학습 로그 저장 경로
        
        # 기본 옵션 설정
        validConfs = ['optim', 'lr', 'batch_size', 'batch_shuffle']
        defaultValues = [defaultOptim, defaultLr, None, None]
        if confs is None :
            for conf_idx, conf in enumerate(validConfs) :
                self.confs[conf] += [defaultValues[conf_idx]]*len(self.models)
        else : # 데이터타입 체크 안함
            if sum(np.in1d(list(confs.keys()), validConfs)) != len(validConfs) :
                print(f"[Error] conf keys are not valid! \n>> validConfs:{validConfs}")
                raise KeyError
            for conf_idx, conf, value_list in enumerate(confs.items()) :
                if len(value_list) != len(self.models) :
                    print(f"[Waring] nConfValue is not correct, missValue/outValue will be default/drop")
                    nlack = len(self.models) - len(value_list)
                    value_list += [defaultValues[conf_idx]]*nlack
                    value_list = value_list[:len(self.models)]
                self.confs[conf] += value_list
        
        # 경로 설정(프로젝트 디렉토리 기준)
        if not os.path.isdir('data') :
            os.mkdir('data')
        if not os.path.isdir('data/Trainer') :
            os.mkdir('data/Trainer')
        self.savePath = os.getcwd() + '/data/Trainer/'        

    def train(self, 
              epochs:int=10, 
              metrics:str='recall@20', 
              k_fold:int=4, # v.2.0
              auto_save:bool=True, 
              verbose:bool=False
              ) :
        metricsNames = [f'recall@{i}' for i in [1, 5, 10, 20]]
        if metrics not in metricsNames :
            print(f"[Error] metrics:{metrics} is not in {metricsNames}")
            raise KeyError  
        
        ########## v.2.0 ##########
        trainLoaders, validLoaders = list(), list()
        if k_fold :
            trainset_full= self.datasetHandler.loadDataset(['trainset_full'])[0]
            valid_bin = int(len(trainset_full)//4)
            for i in range(k_fold) :
                start = i * valid_bin
                end = start + valid_bin

                validset = trainset_full.iloc[start:end]
                validLoader = self.datasetHandler.callDataLoader(dataset=validset) 
                validLoaders.append(validLoader) 

                validindices = validset.index.to_list()
                trainindices = np.in1d(trainset_full.index.to_list(), validindices, invert=True)
                trainset = trainset_full.iloc[trainindices]
                trainLoader = self.datasetHandler.callDataLoader(dataset=trainset)              
                trainLoaders.append(trainLoader)
        else :
            trainLoader = self.datasetHandler.callDataLoader(mode='trainset')
            trainLoaders.append(trainLoader)
            validLoader = self.datasetHandler.callDataLoader(mode='validset')
            validLoaders.append(validLoader)
        ##############################

        for model_idx, model in enumerate(self.models) :
    
            print(f"===================================")
            print(f" training model: {model.name} ")
            print(f"===================================\n")

            optimizer = self.confs['optim'][model_idx](model.parameters(), lr=self.confs['lr'][model_idx])
            # (v.1.0)
            # dataLoader = model.datasetHandler.callDataLoader(
            #     batch_size=self.confs['batch_size'][model_idx], 
            #     batch_shuffle=self.confs['batch_shuffle'][model_idx],
            #     )
            start_time = time()
            foldScore, epochScore = defaultdict(int), defaultdict(int) # v.2.0
            for fold, (trainLoader, validLoader) in enumerate(zip(trainLoaders, validLoaders)) :             
                for epoch in tqdm(range(epochs)) :
                    epoch_loss, epoch_acc, epoch_score = self._train_epoch(model, optimizer, trainLoader, metrics, verbose)
                    inferScore = self.evaluation(evalLoader=validLoader, model_idx=model_idx)

                    print(f"\n=== epoch.{epoch} score ===") 
                    print(f">> loss :: {epoch_loss}\n>> acc({metrics}) :: {epoch_acc}")
                    print(f">> recall@k score ::")
                    for k, score in epoch_score.items() :
                        print(f"    ㄴ{k} : {score}/{inferScore[k]}")
                
                ##### v.2.0 #####
                else :
                    for k, score in inferScore.items() :
                        foldScore[k] += score/k_fold
                        epochScore[k] += epoch_score[k]/k_fold                        

            else :
                ##### v.2.0 #####
                if k_fold :
                    print(f"\n=== fold.{fold} score ===") 
                    print(f">> recall@k score ::")
                    for k, score in foldScore.items() :
                        print(f"    ㄴ{k} : {epochScore[k]}/{score}")
                    print()

                end_time = time()
                timeCost = end_time - start_time
                epoch_score['timeCost'] = timeCost
                
                trainConf = {conf:item[model_idx] for conf, item in self.confs.items()}
                model.name += f'+epoch{epochs}+k{k_fold}'
                
                self.history[model.name] = {'epochScore':epochScore, 'inferScore':foldScore}
                    
                if auto_save :
                    with open(self.savePath + f'{model.name}.model', 'wb') as f :
                        pk.dump(model, f)
                    with open(self.savePath + f'{model.name}_modelConf.dict', 'wb') as f :
                        pk.dump(model.conf, f)                    
                    with open(self.savePath + f'{model.name}_trainConf.dict', 'wb') as f :
                        pk.dump(trainConf, f)                                            
                    with open(self.savePath + f'{model.name}_score.dict', 'wb') as f :
                        pk.dump(epoch_score, f)   
                    with open(self.savePath + f'{model.name}_inferScore.dict', 'wb') as f :
                        pk.dump(inferScore, f)  

        else :
            now_time = datetime.now()
            month, day, hour, minute = now_time.month, now_time.day, now_time.hour, now_time.minute

            best, best_idx = 0, 0
            for m_idx, model in enumerate(self.history.keys()) :
                score = self.history[model]['inferScore'][metrics]
                if score > best : 
                    best = score
                    best_idx = m_idx
            self.history['bestModel'] = self.history[list(self.history.keys())[best_idx]]
            if auto_save :
                with open(self.savePath+f'{month}m{day}d{hour}h{minute}m++ModelTrainHistory.dict', 'wb') as f :
                    pk.dump(self.history, f)

    def evaluation(self, 
                   mode:str='validset', 
                   evalLoader=None, # v.2.0
                   model_idx:int=None,
                   ) :
        
        validModeList = ['validset', 'testset'] 
        if mode not in validModeList :
            print(f"[Error] mode:{mode} is not in validModeList")
            print(f" >> validModeList :: {validModeList}")
            raise KeyError

        ##### v.2.0 #####
        if mode == 'validset' and evalLoader is None :
            print(f"[Error] input-mode'{mode}' needs evalLoader not None")
            raise ValueError

        if mode == 'testset' :
            evalLoader = self.datasetHandler.callDataLoader(mode='testset')

        # 학습 완료된 모델리스트 대상 평가 시, model_idx=None 유지
        start = 0 if model_idx is None else model_idx
        valid = len(self.models) if model_idx is None else model_idx+1
        model_indices = torch.arange(len(self.models)) if model_idx is None else [model_idx] # 코드 동일 사용을 위해 리스트로 선언해 zip 이용

        for m_idx, model in zip(model_indices, self.models[start:valid]) : 
            # (v.1.0)
            # evalDataLoader = model.datasetHandler.callDataLoader(
            #     mode=mode,
            #     batch_size=self.confs['batch_size'][m_idx], 
            #     batch_shuffle=self.confs['batch_shuffle'][m_idx]
            #     )
            model.eval()
            with torch.no_grad() :                
                inferScore = defaultdict(int)
                for i, batchedFeatureDict in enumerate(evalLoader) : # v.2.0 i <- iter
                    newNovelScore = model(batchedFeatureDict)
                    Y = torch.tensor(batchedFeatureDict['label']).reshape(-1).to(self.device)
                    recallScore = self._evaluation(newNovelScore, Y)
                    for k, score in recallScore.items() :
                        inferScore[k] += score

                # (v.1.0) denom = iter + 1
                denom = len(iter(evalLoader))
                for k in inferScore.keys() :
                    inferScore[k] /= denom
            
            if model_idx is None : # train 중 valid 검증에서 출력되지 않도록
                print(f"\n===================================")
                print(f">> model eval({mode}): {model.name} ") 
                for k, score in inferScore.items() :
                    print(f"    ㄴ{k} : {score}")
            else :
                return inferScore     

    def _train_epoch(self, 
                     model, 
                     optimizer, 
                     trainLoader, 
                     metrics, 
                     verbose
                     ) :
        model.train()
        epoch_loss, epoch_acc = 0, 0
        epoch_score = defaultdict(int)
        
        for i, batchedFeatureDict in enumerate(trainLoader) :
            optimizer.zero_grad()

            newNovelScore = model(batchedFeatureDict)
            Y = torch.tensor(batchedFeatureDict['label']).reshape(-1).to(self.device)
            loss = torch.nn.functional.cross_entropy(newNovelScore, Y, ignore_index=0)
        
            loss.backward()
            optimizer.step()
            
            recallScore = self._evaluation(newNovelScore.clone().detach(), Y)
            for k, score in recallScore.items() :
                epoch_score[k] += score
            acc = recallScore[metrics]

            if verbose :
                print(f">>iter.{i} :: loss.{loss.item()}, acc.{acc}")

            epoch_loss += loss.item()
            epoch_acc += acc                
                
        # (v.1.0) denom = iter + 1
        denom = len(iter(trainLoader)) # v.2.0
        for k in epoch_score.keys() :
            epoch_score[k] /= denom
        epoch_loss = epoch_loss/denom
        epoch_acc = epoch_acc/denom   

        return epoch_loss, epoch_acc, epoch_score         

    def _evaluation(self, newNovelScore:torch.tensor, Y:torch.tensor) :
        newNovelScore = newNovelScore.argsort(axis=-1, descending=True)
        recallScore = dict()
        for k in [1, 5, 10, 20] :
            preds = newNovelScore[:,:k]
            score = sum(list(map(lambda y, pred : sum(np.in1d(y, pred)), Y.cpu(), preds.cpu())))
            recallScore[f'recall@{k}'] = score/len(Y)
        return recallScore