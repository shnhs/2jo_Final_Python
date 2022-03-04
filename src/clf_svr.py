#!/usr/bin/env python
from typing import Optional
from fastapi import FastAPI, HTTPException
import uvicorn
import sys
from os import path
import nest_asyncio
nest_asyncio.apply()

SRC_PATH = './src'
sys.path.insert(0, SRC_PATH)

app = FastAPI()

from hugging_classifier import HuggingClassifier, model_param_bert_pt, NEWS_OM_MODEL, logger
clf = HuggingClassifier(modelParam=model_param_bert_pt, train_mode=False)
clf.load_prediction_model(model_dir=NEWS_OM_MODEL, num_categories=3, labels=['-1','0','1'])

@app.post("/clf_info")
async def clf_info():
    if clf == None: 
        logger.error("Classifier model is not loaded... Please check configuration file...")
        raise HTTPException(status_code = 204, detail=  "Classifier model is not loaded... Please check configuration file...")
    else:
        return clf.get_clf_info()

@app.post("/clf_predict/")
async def predict(text: str, cut_off: Optional[float]=0.7, max_length: Optional[int]=100):
    if clf == None: 
        logger.error("Classifier model is not loaded... Please check configuration file...")
        raise HTTPException(status_code = 204, detail= "Classifier model is not loaded... Please check configuration file...")
    else:
        pred = []
        pred = clf.predict(text, cut_off, max_length)
        return pred

try:
    logger.info("Start News OM Classification API Server....")
    port_num = 9090
    uvicorn.run(app, host='0.0.0.0', port=port_num, log_level='info')
except Exception as ex:
    logger.error("Can't load Server : ", ex)