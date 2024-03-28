import uvicorn
from fastapi import FastAPI
from datetime import datetime
import requests
import zipfile
import os
import base64
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

origins = [
    "http://ai.sondouyin.me/",
    "https://ai.sondouyin.me/",
    "http://localhost",
    "http://localhost:8080",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_key():

    now = datetime.now()

    now = str(now).split(" ")
    key1 = "".join(now[0].split("-"))
    key2 = "".join(("".join(now[1].split(":")).split(".")))
    key = key2 + key1
    return key

def img_to_base64(img_path):
    with open(img_path, "rb") as img_file:
        string_base64 = base64.b64encode(img_file.read())
    return string_base64

def base64_to_img(string_base64):
    imgdata = base64.b64decode(string_base64)
    name_img = "image_server/"+str(get_key())+".jpg"
    with open(name_img,"wb") as f:
        f.write(imgdata)
    return name_img
# @app.get("/detect")
# def detect(img: str,name:str):
#     name_img = imgbase64_to_img(img)
#     list_model =
#     for model in list_model:



@app.get("/train")
def train(url_file: str,name:str):
    #tai file nen ve
    content = requests.get(url_file, stream=True).content
    filename = "raw_dataset/"+get_key()+".zip"
    with open(filename,"wb") as file:
        file.write(content)
    #giai nen file zip
    with zipfile.ZipFile(filename, 'r') as zip_ref:
        zip_ref.extractall("dataset/")
    #check folder dung format chua
    try:
        list_folder = os.listdir("dataset/"+name)
    except:
        return {"msg":"format folder errol","code":"01"}

    if len(list_folder) != 2:
        return {"msg": "format folder errol", "code": "01"}
    if list_folder[0] != "images" or list_folder[1] != "labels":
        return {"msg": "format folder errol", "code": "01"}
    list_img = os.listdir("dataset/" + name + "/images/")
    list_label = os.listdir("dataset/" + name + "/labels/")
    for img in list_img:
        if ".jpg" in img:
            if len(list_img) != len(list_label):
                return {"msg": "labels != images", "code": "03"}
    # if len(list_img) < 100:
    #     return {"msg": "dataset fails (< 100 images)", "code": "02"}
    #chia lai folder
    try:
        os.makedirs("dataset/" + name + "/images/train")
        os.makedirs("dataset/" + name + "/images/val")
        os.makedirs("dataset/" + name + "/labels/train")
        os.makedirs("dataset/" + name + "/labels/val")
        num_train = int(len(list_img)*80/100)
        for i in range(0,num_train):
            os.rename("dataset/" + name + "/images/"+list_img[i],"dataset/" + name + "/images/train/"+list_img[i])
        for i in range(0,num_train):
            os.rename("dataset/" + name + "/labels/"+list_label[i],"dataset/" + name + "/labels/train/"+list_label[i])
        for i in range(num_train,len(list_img)):
            os.rename("dataset/" + name + "/images/"+list_img[i],"dataset/" + name + "/images/val/"+list_img[i])
        for i in range(num_train,len(list_img)):
            os.rename("dataset/" + name + "/labels/"+list_label[i],"dataset/" + name + "/labels/val/"+list_label[i])
    except:
        print("file da co")
    #viet file config
    file_name_config = "config/"+name+".yaml"
    text_list = ["train: dataset/{0}/images/train\n".format(name),"val: dataset/{0}/images/val\n".format(name),"nc: 1\n","names: ['{0}']\n".format(name)]
    with open(file_name_config,"w") as f:
        f.writelines(text_list)
    cmd = "python train.py --img 640 --batch 1 --epochs 1 --data {0} --weights yolov5x.pt --project runs/train\{1}".format(file_name_config,name)
    os.system(cmd)
    return {"msg":"train thanh cong","name":name,"code":"00"}
@app.get('/')
def home():
    return {"msg":"Auto train Detect Object"}
if __name__ == "__main__":
    uvicorn.run(app)