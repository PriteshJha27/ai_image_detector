import requests
from fastapi import FastAPI, File, UploadFile
import uvicorn
import nest_asyncio
import asyncio
from pydantic import BaseModel
nest_asyncio.apply()
import warnings
warnings.filterwarnings('ignore')
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
import torch.nn.functional as F
from torch import nn
import timm
from torch.nn.parameter import Parameter
import imageio as iio
import re

app = FastAPI()

def Main_func(path):
    
    CKPT = 'fold_0'
    Targets = ['Not AI' 'AI Generated']
    IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

    def read_image(path) :
        # print(9)
        img = cv2.imread(path)
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        return img

    def get_valid_augs() :
        # print(15)
        return  A.Compose([
       A.Resize(height=512, width=512, always_apply=True, p=1),
       A.Normalize(
            mean = IMAGENET_DEFAULT_MEAN,
            std  = IMAGENET_DEFAULT_STD,
            max_pixel_value=255
                ),
       ToTensorV2(),
    ])


    class Backbone(nn.Module) :
        # print(28)
        def __init__(self,name,pretrained) :
            super(Backbone,self).__init__()
            self.net = timm.create_model(name,pretrained=pretrained)
            self.out_features = self.net.get_classifier().in_features
        def forward(self,x) :
            x = self.net.forward_features(x)
            return x



    class CustomModel(nn.Module) :
        # print(40)
        def __init__(self) :
            super(CustomModel,self).__init__()
            self.backbone = Backbone("tf_efficientnetv2_s",False)
            self.pooling = nn.AdaptiveAvgPool2d(1)
            self.head = nn.Linear(self.backbone.out_features,1)
        def forward(self,x) :
            # print(47)
            x = self.backbone(x)
            x = self.pooling(x).squeeze()
            target = self.head(x)
            output = {}
            output['label'] = target
            return output



    def predict_one_image(path) :
        # print(58)
        image = read_image(path)
        image = get_valid_augs()(image=image)['image']
        image = torch.tensor(image,dtype=torch.float)
        # print(62)
        image = image.reshape((1,3,512,512))
        model = CustomModel()
        #loading ckpt
        model.load_state_dict(torch.load(CKPT,map_location=torch.device('cpu')))
        with torch.no_grad() :
            outputs = model(image)
            proba = F.sigmoid(outputs['label']).detach().numpy()[0]
        return {'Not AI' : 1-float(proba),'AI' : float(proba)}#(proba>0.5)*1
    
    result = predict_one_image(path)
    return result


class Text(BaseModel):
    text:str
        
# UploadFile = File(...)
# file: bytes = File()
@app.post("/func1")
async def cb(text: Text):
    #cbot = Main_Class()
    a = str(text)
    my_list = re.findall(r"'([^']*)'", a)
    # file = File(text)
    # img = iio.imread(my_list)
    response = Main_func(my_list[0])
    return response


if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    loop.create_task(uvicorn.run(app, host="localhost",port=8000))
    try:
        loop.run_forever()
    except KeyboardInterrupt():
        # pass
        loop.close()
    # finally:
    #     loop.close()