#!/usr/bin/env python
# Classifier using HuggingFace Transformer (by Albert)
import importlib
import pandas as pd
import logging
import datetime
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import torch
import os
import sys
import glob
from tqdm import tqdm 

SRC_PATH = './src'
sys.path.insert(0, SRC_PATH)

from transformers import Trainer, TrainingArguments, Pipeline

os.environ["WANDB_DISABLED"] = "true"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# model param => TYPE, Model Class, Tokenizer Class, Tokenizer file, pre trained Model file, Type (pt=pytorch, tf=tensorflow)
model_param_albert_pt = { 'model_name': 'ALBERT', 'model': 'AlbertForSequenceClassification', 'tokenizer' : 'BertTokenizerFast', 'model_file': 'model/base/bert-kor-base', 'tokenizer_file' : 'model/base/bert-kor-base', 'framework': 'pt'}
model_param_bert_pt = { 'model_name': 'BERT', 'model': 'BertForSequenceClassification', 'tokenizer' : 'BertTokenizerFast', 'model_file': 'kykim/bert-kor-base', 'tokenizer_file' : 'kykim/bert-kor-base', 'framework': 'pt'} 
model_param_electa_pt = { 'model_name': 'BERT', 'model': 'ElectraForSequenceClassification', 'tokenizer' : 'ElectraTokenizerFast', 'model_file': 'kykim/electra-kor-base', 'tokenizer_file' : 'kykim/electra-kor-base', 'framework': 'pt'} 

NEWS_OM_MODEL = "model/news_om"

formatter = logging.Formatter(fmt='%(levelname)s: %(name)s: %(asctime)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger('CLF_SVR')
logger.setLevel(logging.INFO)
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)
file_handler = logging.FileHandler('logs/service.log')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

class PytorchDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor([self.labels[idx]])
        return item

    def __len__(self):
        return len(self.labels)

class HuggingClassifier():
    def __init__(self, modelParam, train_mode=False):
        self.model_name = modelParam['model_name']
        tmp = importlib.import_module('transformers')
        self.tokenizerModule = getattr(tmp, modelParam['tokenizer'])
        self.modelModule = getattr(tmp, modelParam['model']) 
        self.tokenizer_file = modelParam['tokenizer_file']
        self.pretrained_model_file = modelParam['model_file']        
        assert(modelParam['framework'] == 'pt' or modelParam['framework'] == 'tf') 
        self.framework = modelParam['framework']
        self.model = None
        self.tokenizer = None
        #self.clf_util = TextUtil()
        self.train_mode = train_mode
        self.labels = None
        self.prediction_model_loaded = False

    def __category_files_to_data_file(self, data_folder, category_level, encoding='utf-8'): 
        # iterate directory (as category name) and text file => data set
        elems = glob.glob(data_folder + '/**/*.txt', recursive=True)
        #data_set = pd.DataFrame(columns=['category', 'text'])
        data_lst = []
        data_file = data_folder+"/clf.dat"

        for elem in elems:
            logger.info("Input File: " + str(elem))
            head_tail = os.path.split(elem)
            file_path = head_tail[0]
            file_name = head_tail[1]
            tmp = file_path.replace(data_folder + '/', '')
            tmp_l = tmp.split('/')
            sub_count = len(tmp_l)
            categories = []
            
            if category_level:
                for i in range(1, sub_count+1):
                    if i > 1:
                        category = "##".join(tmp_l[0:i])                    
                    else:
                        category = tmp_l[0]
                    categories.append(category)
            else:
                category = tmp_l[-1]
                categories.append(category)

            with open(elem, 'r', encoding=encoding) as train_input:
                sentences = train_input.readlines()
                for category in categories:
                    logger.info("Category: " + str(category))
                    for sentence in tqdm(sentences, desc=category):
                        sentence = sentence.strip()
                        if sentence != "":
                            data_lst.append({'category': category, 'text': sentence})
                            
        data_set = pd.DataFrame(data_lst)
        # save data set to file
        logger.info("Writing dataset [" + str(len(data_set)) + "] file : " + str(data_file))
        data_set.to_csv(data_file, index=False, header=None, sep='\t')
        return data_file

    def __read_data_file(self, data_file, columns = ['category', 'text'], header=None, index_col=None, encoding='utf-8'):
        logger.debug("TextUtil::read_data_file() is called...")      
        df = pd.read_csv(data_file, delimiter='\t', names=columns, header=header, index_col=index_col, encoding=encoding)
        logger.debug(df)
        return df

    def get_clf_info(self):
        # get clf info
        info = []
        info.append({'model_name':self.model_name})
        info.append({'model_dir':self.model_dir})
        info.append({'tokenizer_file:': self.tokenizer_file})
        info.append({'pretrained_dir:': self.pretrained_model_file})
        info.append({'deep learning framework': self.framework})
        return info

    def load_prediction_model(self, model_dir, num_categories=-1, labels=None):
        if self.train_mode:
            logger.error("train_mode is not False for prediction model loading")
            return 
        if self.prediction_model_loaded:
            logger.error("clf model is already loaded...")
            return 

        logger.info(">>> Loading clf model file : " + str(model_dir))
        logger.info(">>> Loading Transformer model...")
        self.labels = labels
        self.model_dir = model_dir

        if num_categories == -1 or labels == None: # not setting
            self.labels = self.clf_util.get_labels_from_categories(model_dir+'/tmp.dat')
            num_categories = len(self.labels)
            logger.info(">>> Number of categories : " + str(num_categories))

        self.model = self.modelModule.from_pretrained(self.model_dir, num_labels=num_categories, local_files_only=True)
        self.tokenizer = self.tokenizerModule.from_pretrained(self.model_dir, local_files_only=True)
        self.prediction_model_loaded = True
        logger.info(">>> Prediction model is loaded...")  

    def predict(self, input_text, cut_off=0.7, max_length=100):
        if self.prediction_model_loaded == False:
            logger.error("Prediction Model is not loaded...")
            return None
        input_sentences = []
        #logger.debug(">>> Input: " + str(input_text))
        
        input_sentences.append(input_text)
        encoding_ = self.tokenizer(input_sentences, truncation=True, padding=True, max_length=max_length) 
        result_ = self.model(torch.tensor(encoding_['input_ids']))
        proba = float(torch.nn.functional.softmax(result_['logits'], dim=1).max())
        #logger.debug(">>> Proba : " + str(proba))
        if proba >= float(cut_off):
            pred = self.labels[np.argmax(result_['logits'].tolist())]      
        else:
            pred = None
            logger.debug(">>> Cut off value [" + str(cut_off) + "] is bigger than proba[ " + str(proba) + "]")
        #logger.debug(">>> Result: " + str(pred))
        result_ = {'sentence':input_text, 'pred':pred, 'prob':round(proba,3)}
        logger.info(result_)
        return result_
    
    def prediction(self, input_text, cut_off=0.7, max_length=100):
        if self.prediction_model_loaded == False:
            logger.error("Prediction Model is not loaded...")
            return None
        input_sentences = []
        #logger.debug(">>> Input: " + str(input_text))
        
        input_sentences.append(input_text)
        encoding_ = self.tokenizer(input_sentences, truncation=True, padding=True, max_length=max_length) 
        result_ = self.model(torch.tensor(encoding_['input_ids']))
        proba = float(torch.nn.functional.softmax(result_['logits'], dim=1).max())
        #logger.debug(">>> Proba : " + str(proba))
        if proba >= float(cut_off):
            pred = self.labels[np.argmax(result_['logits'].tolist())]      
        else:
            pred = None
            logger.debug(">>> Cut off value [" + str(cut_off) + "] is bigger than proba[ " + str(proba) + "]")
        #logger.debug(">>> Result: " + str(pred))
        # result_ = {'sentence':input_text, 'pred':pred, 'prob':round(proba,3)}
        result_ = [pred, round(proba,3)]
        # logger.info(result_)
        return result_
   
    def compute_metrics(self, pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        # calculate accuracy using sklearn's function
        acc = accuracy_score(labels, preds)
        return {
            'accuracy': acc,
    }

    def train(self, model_dir, data_file, max_seq_length = 100, learning_rate=1e-4 , test_size=0.1, batch_size=3, epochs=1): 
        train_start = datetime.datetime.now()
        logger.info(">> Training Start ....")
        logger.info("[Step 1] Transformer Classifier Check configuration...")
        logger.info(">>> Data File: " + str(data_file))
        logger.info(">>> Model Name: " + str(self.model_name))
        logger.info(">>> Model Dir: " + str(model_dir))
        logger.info(">>> Max Seq Length: " + str(max_seq_length))
        logger.info(">>> Learning Rate: " + str(learning_rate))
        logger.info(">>> Batch Size: " + str(batch_size))
        logger.info(">>> Epoch : " + str(epochs))
        logger.info(">>> Test Size : " + str(test_size))

        self.model_name = model_dir
        # dataset read
        self.test_size = test_size

        logger.info("[Step 2] Loading Learning Data...")
        df = self.__read_data_file(data_file)
        
        num_categories = len(df['category'].unique())
        logger.info(">>> Number of category : " + str(num_categories))        
        logger.info(">>> Train, Test Data split...")
        # data set to train / test data set
        Y = pd.get_dummies(df['category'], dtype=np.int8) # Category Vector
        x_train, x_test, y_train, y_test = train_test_split(df['text'], Y, test_size=self.test_size, random_state=42)      
        
        # load the model and tokenizer
        logger.info("[Step 3] Loading Transformer model...")
        self.model = self.modelModule.from_pretrained(self.pretrained_model_file, num_labels=num_categories)
        self.tokenizer = self.tokenizerModule.from_pretrained(self.tokenizer_file)

        logger.info("[Step 4] Encoding Text...")
        train_encodings = self.tokenizer(x_train.values.tolist(), truncation=True, padding=True, max_length=max_seq_length)
        test_encodings = self.tokenizer(x_test.values.tolist(), truncation=True, padding=True, max_length=max_seq_length)

        # convert our tokenized data into a torch Dataset
        logger.info("[Step 5] Parameter preparation...")
        y_train_list = np.argmax(y_train.values.tolist(), axis=1)
        y_test_list = np.argmax(y_test.values.tolist(), axis=1)
        train_dataset = PytorchDataset(train_encodings, y_train_list)
        test_dataset = PytorchDataset(test_encodings, y_test_list)

        training_args = TrainingArguments(
            output_dir='tmp', 
            overwrite_output_dir = True,
            num_train_epochs=epochs,              # total number of training epochs
            per_device_train_batch_size=batch_size,  # batch size per device during training
            per_device_eval_batch_size=batch_size,   # batch size for evaluation
            warmup_steps=500,                # number of warmup steps for learning rate scheduler
            weight_decay=0.01,               # strength of weight decay
            logging_dir='./logs',            # directory for storing logs
            load_best_model_at_end=True,     # load the best model when finished training (default metric is loss)
            # but you can specify `metric_for_best_model` argument to change to accuracy or other metric
            logging_steps=500,               # log & save weights each logging_steps
            evaluation_strategy="steps",     # evaluate each `logging_steps`
        )

        logger.info("[Step 6] Learning.... ")
        trainer = Trainer(
            model=self.model,                         # the instantiated Transformers model to be trained
            args=training_args,                  # training arguments, defined above
            train_dataset=train_dataset,         # training dataset
            eval_dataset=test_dataset,          # evaluation dataset
            compute_metrics=self.compute_metrics,     # the callback that computes metrics of interest
        )
        
        # train the model
        trainer.train()

        logger.info("[Step 7] Saving trained model : " + str(model_dir))
        trainer.save_model(model_dir) # classification model
        self.tokenizer.save_pretrained(model_dir) # save tokenizer 
        self.clf_util.copy_input_data_to_model_folder(data_file, model_dir)

        # evaluate the current model after training
        logger.info("[Step 8] Evaluation...")
        y_pred = np.argmax(trainer.predict(test_dataset)[0], axis=1)
        y_labels = y_test.columns
        self.clf_util.evaluation(self.model_name, x_test, y_test.values.argmax(axis=1), y_pred, y_labels, model_dir + '/evaluation.txt')
        
        train_end = datetime.datetime.now()
        logger.critical("Elapsed time: " + str(train_end - train_start))
        logger.info(">>> Training End...")

    def train_folder(self, model_dir, data_dir, kor_postagging=False, category_level=False, 
                     balance_samples = True, max_tokens=-1, min_token_len=-1, encoding='utf-8', 
                     max_seq_length = 100, test_size=0.1, learning_rate=1e-4, 
                     use_cached_data_file=True, batch_size=2, epochs=1):
        logger.info("[PREP] Categorized training data ==> Merged training data")
        
        if self.train_mode == False:
            print("[Error] Plese set train_mode to True !!!")
            return
        data_file = self.__category_files_to_data_file(data_dir, category_level=category_level, encoding=encoding)
        self.train(model_dir, data_file = data_file, max_seq_length=max_seq_length, learning_rate=learning_rate, batch_size=batch_size, test_size=test_size, epochs=epochs)

if __name__ == '__main__':
    def training():
        data_folder = "raw_data/news_om" #"raw_data/naver_movie_om" #om_test" #"raw_data/news/tmp"
        model_dir = "model/news_om_clf2" #'model/kor_bert_news_clf'
        clf = HuggingClassifier(modelParam = model_param_electa_pt, train_mode=True) # BERT
        clf.train_folder(model_dir=model_dir, data_dir=data_folder, batch_size=48, epochs=1)   
        
    def prediction_test():
        #set_debug_log()
        test_sentences=[
            
            "퇴사 1년 만에 '특허 괴물' 돌변…前 임원 공격에 삼성 '발칵'",
            "삼성전자 TV, CES 2022서 최고 제품상 석권",
            "'삼성전자, 반도체 수요 증가 기대'-KB증권",
            "삼성·LG, 사상 최대 매출…인텔·월풀 따라 잡았다",
            "[김대호 박사의 오늘 기업·사람] 씨티그룹·삼성전자·카카오·CATL",
            "삼성전자 신제품 발표로 기대감 주가 상승",
            "거리두기에 1분기 소매경기 '주춤'…부정전망 더 많아",
            "광주지역 소매·유통업 2022년 1분기 경기 '호전' 전망",
            "1분기 소매유통업 경기전망 싸늘…온라인·백화점만 ‘방긋’",
            "회복세 띠던 소매 경기···올 1분기엔 ‘냉랭’",
            "소매경기, 1분기 다시 위축된다…온라인·백화점은 기대감 '솔솔'",
            "‘주식 먹튀’ 논란에…류영준 카카오 대표이사 후보자 자진사퇴",
            "류영준 사퇴에도…카카오 10만원 하회·카뱅 52주 최저가",
            "'먹튀 논란' 류영준, 카카오페이 대표직은 유지…추가 매각 '촉각'",
            "현대차-보스턴 다이내믹스 협업 진행…내년 이후 결과물 보여줄 것"

        ]
        
        clf  = HuggingClassifier(modelParam = model_param_bert_pt, train_mode=False)
        clf.load_prediction_model(model_dir=NEWS_OM_MODEL, num_categories=3, labels=['-1','0','1'])
        
        for sentence in test_sentences:
            ret = clf.predict(sentence)
            print(ret)

    def prediction_test_2():
        #set_debug_log()
        test_sentences=[
            
            "퇴사 1년 만에 '특허 괴물' 돌변…前 임원 공격에 삼성 '발칵'",
            "삼성전자 TV, CES 2022서 최고 제품상 석권",
            "'삼성전자, 반도체 수요 증가 기대'-KB증권",
            "삼성·LG, 사상 최대 매출…인텔·월풀 따라 잡았다",
            "[김대호 박사의 오늘 기업·사람] 씨티그룹·삼성전자·카카오·CATL",
            "삼성전자 신제품 발표로 기대감 주가 상승",
            "거리두기에 1분기 소매경기 '주춤'…부정전망 더 많아",
            "광주지역 소매·유통업 2022년 1분기 경기 '호전' 전망",
            "1분기 소매유통업 경기전망 싸늘…온라인·백화점만 ‘방긋’",
            "회복세 띠던 소매 경기···올 1분기엔 ‘냉랭’",
            "소매경기, 1분기 다시 위축된다…온라인·백화점은 기대감 '솔솔'",
            "‘주식 먹튀’ 논란에…류영준 카카오 대표이사 후보자 자진사퇴",
            "류영준 사퇴에도…카카오 10만원 하회·카뱅 52주 최저가",
            "'먹튀 논란' 류영준, 카카오페이 대표직은 유지…추가 매각 '촉각'",
            "현대차-보스턴 다이내믹스 협업 진행…내년 이후 결과물 보여줄 것"

        ]
        
        clf  = HuggingClassifier(modelParam = model_param_bert_pt, train_mode=False)
        clf.load_prediction_model(model_dir=NEWS_OM_MODEL, num_categories=3, labels=['-1','0','1'])
        
        for sentence in test_sentences:
            ret = clf.prediction(sentence)
            print(ret)
    
    # training()
    prediction_test_2()
