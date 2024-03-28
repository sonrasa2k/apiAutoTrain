from flask import Flask, request, jsonify
from flask_mongoengine import MongoEngine
from photoshoped import Check_Photoshoped
from datetime import datetime
import cv2
import zipfile
import subprocess as sp
from photoshoped import Check_Photoshoped
from check_bill import CheckBill
import os
import requests
from detect_screen import DetectScreen
from get_image import Get_Image
from num_object import NumObject
import json
from fix_txt import FixLabels
import shutil
from rq import Queue
from rq.job import Job
import subprocess
import redis
from train_custom import TrainMODELS
from rq import Worker, Queue, Connection
import time


r = redis.Redis()
q = Queue('train',connection=r)
q1 = Queue('detect',connection=r)
q2 = Queue('pretrain',connection=r)

BATCH = 4
EPOCHS = 400
DEVICE = '0'

TIME_1_EPOCHS = 0.04

ID_TRAIN = 'sondeptrai'
app = Flask(__name__)
app.config['MONGODB_SETTINGS'] = {
    'db': 'admin',
    'host': '192.xxx.xx.xxx',
    'port': 8000,
    'username':'admin',
    'password':'admin'
}
db = MongoEngine()
db.init_app(app)

#hàm random
def get_key():

    now = datetime.now()

    now = str(now).split(" ")
    key1 = "".join(now[0].split("-"))
    key2 = "".join(("".join(now[1].split(":")).split(".")))
    key = key2 + key1
    return key

#cấu hình database

class Train(db.Document):
    productName = db.StringField()
    productId = db.IntField()
    filePath = db.StringField()
    urlCallback = db.StringField()
    time_request = db.DateTimeField(default=datetime.now())
    userId = db.IntField()
class ReturnTrain(db.Document):
    productId = db.IntField()
    productName = db.StringField()
    model_path = db.StringField()
    urlCallback = db.StringField()
    status = db.IntField(default = 1)
class TrainCallBack(db.Document):
    productId = db.IntField()
    status = db.IntField(default=2) #2 la đã nhận đc , 3 là đag train, 4 train that bai, 5 la train thanh cong
    note = db.StringField(default="Đã Nhận Yêu Cầu Train")
    userId = db.IntField()

class Modeled(db.Document):
    product_name = db.StringField()
    productId = db.IntField()
    model_path = db.StringField()
    last_day_train = db.DateTimeField(default=datetime.now())


class RequestDetect(db.Document):
    filePath = db.StringField()
    urlCallback = db.StringField()
    listProduct = db.ListField(db.DictField())
    surveyDataId = db.IntField()
    surveyDataGroupId = db.IntField()
    fileId = db.IntField()
    domain = db.StringField()
    typeAIId = db.ListField(db.IntField())
    projectId = db.IntField()
    surveyId = db.IntField()
    companyId = db.IntField()
    time_request = db.DateTimeField(default=datetime.now())

class DetectMultiObject(db.Document):
    typeAIId = db.IntField(default=2)
    status = db.IntField(default=1)
    note = db.StringField(default='OKE') #1 dme duoc 2 k dem dc
    value = db.ListField(db.DictField(),default=list)

class Detect(db.Document):
    surveyDataId = db.IntField()
    surveyDataGroupId = db.IntField()
    fileId = db.IntField()
    types = db.IntField(default=1)
    note = db.StringField(default="OKE")
    data = db.ListField(db.DictField(),default = list)
    projectId = db.IntField()
    surveyId = db.IntField()
    companyId = db.IntField()
    time_complete_detect = db.DateTimeField(default=datetime.now())

class ListDataDetect(db.Document):
    data = db.ListField(db.DictField(),default = list)
    time_request = db.DateTimeField(default=datetime.now())
class TimePredict(db.Document):
    query_id = db.IntField(default = 1)
    total = db.IntField(default = 0)
    time_start = db.IntField()
    time_predict = db.IntField(default = 0)
    last_save = db.DateTimeField(default=datetime.now())

#api check bill
class RequestDetectBill(db.Document):
    ids = db.IntField()
    zaloId = db.StringField()
    path = db.StringField()
    urlCallback = db.StringField()
    time_request = db.DateTimeField(default=datetime.now())

class CallBackDetectBill(db.Document):
    ids = db.IntField()
    status = db.IntField(default=1) #1 la qua trinh check ok, 0 la qua trinh check loi
    zaloId = db.StringField()
    isBill = db.IntField()
    isFake = db.IntField()
    note = db.StringField()



def data_to_json(database):
    jsons = json.loads(database.to_json())
    del jsons['_id']
    return jsons
#api checkbill ---------------------------------------------------
#ham check background
def check(id,zaloId,path,urlCallback):
    #get image from path
    get_img = Get_Image()
    try:
        img = get_img.get_image_from_link(path)
    except:
        save_callback = CallBackDetectBill(ids=id,zaloId=zaloId,status=0,isBill=999,isFake=999,note="Link anh sai")
        save_callback.save()
        json_callback = data_to_json(save_callback)
        requests.post(urlCallback,json=json_callback)
        return False
    try:
        if img == None:
            save_callback = CallBackDetectBill(ids=id, zaloId=zaloId, status=0, isBill=999, isFake=999, note="Image error")
            save_callback.save()
            json_callback = data_to_json(save_callback)
            requests.post(urlCallback, json=json_callback)
            return False
    except:
        if len(img) < 10:
            save_callback = CallBackDetectBill(ids=id, zaloId=zaloId, status=0, isBill=999, isFake=999,
                                               note="Error")
            save_callback.save()
            json_callback = data_to_json(save_callback)
            requests.post(urlCallback, json=json_callback)
            return False
    name = path.split("/")
    name_img = name[len(name)-1]
    file_img = "public/" + name_img
    cv2.imwrite(file_img,img)
    #check bill co hoac k
    try:
        checker = CheckBill()
        isBill = checker.check_bill_yes_or_no(path_img=file_img,path_modeled="modeled/bill_best.pt",name_object="bill",save_path="runs/detect/")
    except:
        save_callback = CallBackDetectBill(ids=id, zaloId=zaloId, status=0, isBill=999, isFake=999, note="Detect Bill ERROR")
        save_callback.save()
        json_callback = data_to_json(save_callback)
        requests.post(urlCallback, json=json_callback)
        return False

    #check co photoshop hay k
    try:
        checker = Check_Photoshoped()
        isFake = checker.check_photoshoped(img)
    except:
        save_callback = CallBackDetectBill(ids=id, zaloId=zaloId, status=0, isBill=999, isFake=999,
                                           note="Check Fake Bill ERROR")
        save_callback.save()
        json_callback = data_to_json(save_callback)
        requests.post(urlCallback, json=json_callback)
        return False

    #tra ve json hoan thanh
    save_callback = CallBackDetectBill(ids=id, zaloId=zaloId, status=1, isBill=isBill, isFake=isFake,
                                       note="Completed!")
    save_callback.save()
    json_callback = data_to_json(save_callback)
    requests.post(urlCallback, json=json_callback)
    os.remove(file_img)
    return True

@app.route("/check",methods=["POST"])
def check_bill():
    # try:
    request_data = request.get_json()
    ids = request_data["ids"]
    zaloId = request_data["zaloId"]
    path = request_data["path"]
    urlCallback = request_data["urlCallback"]
    save_request = RequestDetectBill(ids=ids,zaloId=zaloId,path=path,urlCallback=urlCallback)
    save_request.save()
    job = q1.enqueue(check,args=(ids,zaloId,path,urlCallback,))
    # except:
    #     return jsonify(code=999,msg="Tham So Sai")
    return jsonify(code=200,id_queue=job.get_id(),msg="Da nhan yeu cau Check Bill")




def detect_one_img_backgroud(request_data):
    filePath = request_data['filePath']
    urlCallback = request_data['urlCallback']
    listProduct = request_data['listProduct']
    surveyDataId = request_data['surveyDataId']
    surveyDataGroupId = request_data['surveyDataGroupId']
    fileId = request_data['fileId']
    domain = request_data['domain']
    typeAIId = request_data['typeAIId']
    projectId = request_data['projectId']
    surveyId = request_data['surveyId']
    companyId = request_data['companyId']
    request_detect = RequestDetect(filePath=filePath, urlCallback=urlCallback, listProduct=listProduct,
                                   surveyDataId=surveyDataId, surveyDataGroupId=surveyDataGroupId,
                                   fileId=fileId, domain=domain, typeAIId=typeAIId, projectId=projectId,
                                   surveyId=surveyId, companyId=companyId)
    request_detect.save()
    link_image = domain + '/' + filePath
    imgs = Get_Image()
    try:
        img = imgs.get_image_from_link(link_image)
    except:
        detects_false = Detect(surveyDataId=surveyDataId, surveyDataGroupId=surveyDataGroupId, types=0, fileId=fileId,
                               note="Khong get dc Image", surveyId
                               =surveyId, projectId=projectId, companyId=companyId).save()
        jsons = data_to_json(detects_false)
        requests.post(urlCallback, json=jsons)
        return False
    name = filePath.split('/')
    filename_img = 'public/' + name[len(name) - 1]
    cv2.imwrite(filename_img, img)
    try:
        if img == None:
            detects = Detect(surveyDataId=surveyDataId, surveyDataGroupId=surveyDataGroupId, types=0, fileId=fileId,
                             note="Path Image Invalid", surveyId=surveyId, projectId=projectId, companyId=companyId)
            detects.save()
            jsons = data_to_json(detects)
            requests.post(urlCallback, json=jsons)
            return jsons
    except:
        if len(img) < 10:
            detects = Detect(surveyDataId=surveyDataId, surveyDataGroupId=surveyDataGroupId, fileId=fileId, types=0,
                             note="Path Image Invalid", surveyId=surveyId, projectId=projectId, companyId=companyId)
            detects.save()
            jsons = data_to_json(detects)
            requests.post(urlCallback, json=jsons)
            return jsons
    data = []
    detects = None
    for ai in typeAIId:
        if ai == 1:
            detect_screen = DetectScreen()
            data.append(detect_screen.check_screen(filename_img))
            pass
        if ai == 2:
            value = []
            num_object = NumObject()
            for product in listProduct:
                # model = Modeled.objects(productId=product['productId']).first()
                model = Modeled.objects(productId=product['productId']).order_by('-last_day_train').first()
                if not model:
                    multi = DetectMultiObject(status=2, note="San Pham Chua Co Tren He Thong")
                    multi.save()
                    value.append(data_to_json(multi))
                    continue
                save_path = 'runs/detect/'
                num_product = num_object.num_object(path_img=filename_img, path_modeled=model.model_path,
                                                    name_object=model.product_name, save_path=save_path)
                json_detect = {
                    "note": "Complete",
                    "product_id": product['productId'],
                    "quantity": num_product['num'],
                    "boxes": num_product['data'],
                    "product_name": num_product['name_object']
                }
                value.append(json_detect)
            multi = DetectMultiObject(value=value)
            multi.save()
            jsons = data_to_json(multi)
            data.append(jsons)
    detects = Detect(surveyDataId=surveyDataId, surveyDataGroupId=surveyDataGroupId, fileId=fileId,
                     data=data, surveyId=surveyId, projectId=projectId, companyId=companyId)
    detects.save()
    result = json.loads(detects.to_json())
    requests.post(urlCallback, json=result)
    os.remove(filename_img)
    return True




# def detect_background(list_request):
#     list_result = []
#     for request_data in list_request:
#         try:
#             filePath = request_data['filePath']
#             urlCallback = request_data['urlCallback']
#             listProduct = request_data['listProduct']
#             surveyDataId = request_data['surveyDataId']
#             surveyDataGroupId = request_data['surveyDataGroupId']
#             fileId = request_data['fileId']
#             domain = request_data['domain']
#             typeAIId = request_data['typeAIId']
#             projectId = request_data['projectId']
#             surveyId = request_data['surveyId']
#             companyId = request_data['companyId']
#         except:
#             return jsonify(code=402, msg="Thiếu Tham Số")
#         request_detect = RequestDetect(filePath=filePath, urlCallback=urlCallback, listProduct=listProduct,
#                                        surveyDataId=surveyDataId, surveyDataGroupId=surveyDataGroupId,
#                                        fileId=fileId, domain=domain, typeAIId=typeAIId,projectId=projectId,surveyId=surveyId,companyId=companyId)
#         request_detect.save()
#         link_image = domain + '/' + filePath
#         imgs = Get_Image()
#         try:
#             img = imgs.get_image_from_link(link_image)
#         except:
#             detects_false = Detect(surveyDataId=surveyDataId, surveyDataGroupId=surveyDataGroupId, types=0, fileId=fileId,
#                    note="Khong get dc Image",surveyId
#                                    =surveyId,projectId=projectId,companyId=companyId).save()
#             jsons = data_to_json(detects_false)
#             requests.post(urlCallback, json=jsons)
#             return False
#         name = filePath.split('/')
#         filename_img = 'public/' + name[len(name) - 1]
#         cv2.imwrite(filename_img, img)
#         try:
#             if img == None:
#                 detects = Detect(surveyDataId=surveyDataId, surveyDataGroupId=surveyDataGroupId, types=0, fileId=fileId,
#                                  note="Path Image Invalid",surveyId=surveyId,projectId=projectId,companyId=companyId)
#                 detects.save()
#                 jsons = data_to_json(detects)
#                 requests.post(urlCallback, json=jsons)
#                 return jsons
#         except:
#             if len(img) < 10:
#                 detects = Detect(surveyDataId=surveyDataId, surveyDataGroupId=surveyDataGroupId, fileId=fileId, types=0,
#                                  note="Path Image Invalid",surveyId=surveyId,projectId=projectId,companyId=companyId)
#                 detects.save()
#                 jsons = data_to_json(detects)
#                 requests.post(urlCallback, json=jsons)
#                 return jsons
#         data = []
#         detects = None
#         for ai in typeAIId:
#             if ai == 1:
#                 detect_screen = DetectScreen()
#                 data.append(detect_screen.check_screen(filename_img))
#                 pass
#             if ai == 2:
#                 value = []
#                 num_object = NumObject()
#                 for product in listProduct:
#                     # model = Modeled.objects(productId=product['productId']).first()
#                     model = Modeled.objects(productId=product['productId']).order_by('-last_day_train').first()
#                     if not model:
#                         multi = DetectMultiObject(status=2, note="San Pham Chua Co Tren He Thong")
#                         multi.save()
#                         value.append(data_to_json(multi))
#                         continue
#                     save_path = 'runs/detect/'
#                     num_product = num_object.num_object(path_img=filename_img, path_modeled=model.model_path,
#                                                         name_object=model.product_name, save_path=save_path)
#                     json_detect = {
#                         "note": "Complete",
#                         "product_id": product['productId'],
#                         "quantity": num_product['num'],
#                         "boxes": num_product['data'],
#                         "product_name": num_product['name_object']
#                     }
#                     value.append(json_detect)
#                 multi = DetectMultiObject(value=value)
#                 multi.save()
#                 jsons = data_to_json(multi)
#                 data.append(jsons)
#         detects = Detect(surveyDataId=surveyDataId, surveyDataGroupId=surveyDataGroupId, fileId=fileId,
#                          data=data,surveyId=surveyId,projectId=projectId,companyId=companyId    )
#         detects.save()
#         result = json.loads(detects.to_json())
#         requests.post(urlCallback, json=result)
#         os.remove(filename_img)
#         list_result.append(result)
#     list_result = json.dumps(list_result)
#     return list_result
@app.route('/detect',methods=['POST'])
def detect():
    try:
        data = request.data
        my_json = data.decode('utf8').replace("'", '"')
        data = json.loads(my_json)
        for request_data in data:
            filePath = request_data['filePath']
            urlCallback = request_data['urlCallback']
            listProduct = request_data['listProduct']
            surveyDataId = request_data['surveyDataId']
            surveyDataGroupId = request_data['surveyDataGroupId']
            fileId = request_data['fileId']
            domain = request_data['domain']
            typeAIId = request_data['typeAIId']
            projectId = request_data['projectId']
            surveyId = request_data['surveyId']
            companyId = request_data['companyId']
            job = q1.enqueue(detect_one_img_backgroud,request_data)
        data_to_mongo = ListDataDetect(data=list(data))
        data_to_mongo.save()
    except Exception as e:
        return jsonify(status=999, message=str(e))
    return jsonify(status=200,message="Success",id_queue_image_late=job.get_id())

#add model thủ công ( Sử dụng trong quá trình test)

@app.route('/add/model',methods=['GET'])
def add_model():
    path_model = request.args.get("path_model")
    id_product = request.args.get("id_product")
    name_product = request.args.get("name_product")
    model = Modeled(product_name=name_product,productId=id_product,model_path= path_model)
    model.save()
    return jsonify(code=200,msg="them model thanh cong")


#phần train




#get zip file

def get_zip(url_file,productId,productName,urlCallback):
    list_folder = None
    data_name = "dataset/data"+str(get_key())+"/"
    try:
        content = requests.get(url_file, stream=True).content
        filename = "raw_dataset/" + get_key() + ".zip"
        with open(filename, "wb") as file:
            file.write(content)
        # giai nen file zip
        with zipfile.ZipFile(filename, 'r') as zip_ref:
            zip_ref.extractall(data_name)
        print("Oke")
    except:
        callback2 = TrainCallBack(productId=productId,status=4, note="Dataset Gửi lên Sai").save()
        # callback khi data sai
        json_return = data_to_json(callback2)
        requests.post(urlCallback, json=json_return)
        return json_return
    try:
        list_folder = os.listdir(data_name + productName)
    except:
        callback2 = TrainCallBack(productId=productId, status=4, note="Dataset Gửi lên Sai").save()
        # callback khi data sai
        json_return = data_to_json(callback2)
        requests.post(urlCallback, json=json_return)
        os.remove(filename)
        return json_return
    os.remove(filename)
    path_product = data_name + productName
    return path_product,list_folder

def get_gpu_memory():
    import torch
    t = torch.cuda.get_device_properties(0).total_memory
    r = torch.cuda.memory_reserved(0)
    a = torch.cuda.memory_allocated(0)
    f = t - a  # free inside reserved
    return f

def pretrain_backgroud(productId,productName,filePath,urlCallback,userId):
    train = Train(productId=productId, productName=productName, filePath=filePath, urlCallback=urlCallback,userId=userId)
    train.save()
    # xu ly data
    try:
        path_product, list_folder = get_zip(filePath, productName=productName, productId=productId,
                                            urlCallback=urlCallback)  # list folder phải gồm images va labels
    except:
        list_folder = None
    if list_folder == None:
        callback2 = TrainCallBack(productId=productId, status=4,
                                  note="Khong the unzip file data",userId=userId).save()
        # callback khi data sai
        json_return = data_to_json(callback2)
        requests.post(urlCallback, json=json_return)
        return json_return
    if not 'images' or not 'labels' in list_folder:
        callback2 = TrainCallBack(productId=productId, status=4,
                                  note="Dataset Gửi lên Sai vì 2 thư mục con sai tên",userId=userId).save()
        # callback khi data sai
        json_return = data_to_json(callback2)
        requests.post(urlCallback, json=json_return)
        return json_return
    list_img = os.listdir(path_product + "/images/")
    num_train = int(len(list_img) * 80 / 100)
    time_predicts = TimePredict.objects(query_id=1).order_by('-last_save').first()
    if not time_predicts:
        time_predict = int(num_train * TIME_1_EPOCHS * EPOCHS / 60)
        total = time_predict
        time_start = int(time.time())
        save_time_predict = TimePredict(total=total,time_predict=time_predict,time_start = time_start)
        save_time_predict.save()
    else:
        if time_predicts.total == 0:
            time_predict = int(num_train * TIME_1_EPOCHS * EPOCHS / 60)
            total = time_predicts.total + time_predicts
            time_start = int(time.time())
            save_time_predict = TimePredict(total=total, time_predict=time_predict,time_start=time_start )
            save_time_predict.save()
        else:
            time_predict = int(num_train * TIME_1_EPOCHS * EPOCHS / 60)
            total = time_predicts.total + time_predict - int((time.time() - time_predicts.time_start) / 60)
            save_time_predict = TimePredict(total=total, time_predict=time_predict, time_start=time_predicts.time_start)
            save_time_predict.save()
    job = q.enqueue(train_background,args=(productName,path_product,productId,urlCallback,userId,time_predict,),job_timeout=86400, result_ttl=8640000)
    callback = TrainCallBack(productId=productId,userId=userId)
    callback.save()
    json_return = data_to_json(callback)
    json_return["timeFinish"] = total
    # json_return = {"productId":productId,"status":2,"note":"Đã Nhận Yêu Cầu Train"}
    # # # callback lan 1 : bao da nhan dc yeu cau
    print(json_return)
    requests.post(urlCallback, json=json_return)

def train_background(productName,path_product,productId,urlCallback,userId,time_predict):
    name = productName
    list_img = os.listdir(path_product + "/images/")
    dir_labelss = path_product + "/labels/"
    list_label = os.listdir(dir_labelss)
    fix_lb = FixLabels()
    try:
        for labelss in list_label:
            if ".txt" in labelss:
                print(fix_lb.fix_labels(dir_labelss+labelss))
    except:
        callback2 = TrainCallBack(productId=productId, status=4,
                                  note="loi data gui len", userId=userId).save()
        # callback khi data sai
        json_return = data_to_json(callback2)
        requests.post(urlCallback, json=json_return)
        time_start = int(time.time())
        time_predicts = TimePredict.objects(query_id=1).order_by('-last_save').first()
        total = time_predicts.total - time_predicts.time_predict
        time_predictss = TimePredict(total=total, time_predicts=time_predicts.time_predict, time_start=time_start)
        time_predictss.save()
        return json_return
    try:
        list_name_img = []
        list_name_label = []
        os.makedirs(path_product+ "/images/train")
        os.makedirs(path_product + "/images/val")
        os.makedirs(path_product + "/labels/train")
        os.makedirs(path_product + "/labels/val")
        for label in list_label:
            if ".txt" in label:
                list_name_label.append(label.split('.')[0])
        for img in list_img:
            if ".jpg" or ".png" or ".PNG" in img:
                if img.split('.')[0] in list_name_label:
                    list_name_img.append(img)
        num_train = int(len(list_name_img)*80/100)
        for i in range(0,num_train):
            os.rename(path_product + "/images/" + list_name_img[i], path_product + "/images/train/" + list_name_img[i])
            os.rename(path_product + "/labels/" + list_name_img[i].split('.')[0]+".txt",path_product + "/labels/train/" + list_name_img[i].split('.')[0]+".txt")
        for i in range(num_train, len(list_name_img)):
            os.rename(path_product + "/images/" + list_name_img[i],path_product + "/images/val/" + list_name_img[i])
            os.rename(path_product + "/labels/" + list_name_img[i].split('.')[0]+".txt",path_product + "/labels/val/" + list_name_img[i].split('.')[0]+".txt")
    except:
        callback2 = TrainCallBack(productId=productId, status=4,
                                  note="loi k xac dinh",userId=userId).save()
        # callback khi data sai
        json_return = data_to_json(callback2)
        requests.post(urlCallback, json=json_return)
        time_start = int(time.time())
        time_predicts = TimePredict.objects(query_id=1).order_by('-last_save').first()
        total = time_predicts.total - time_predicts.time_predict
        time_predictss = TimePredict(total=total, time_predicts=time_predicts.time_predict, time_start=time_start)
        time_predictss.save()
        return json_return

    # viet file config
    file_name_config = "config/" + productName + ".yaml"
    text_list = ["train: {0}/images/train\n".format(path_product),
                 "val: {0}/images/val\n".format(path_product),
                 "nc: 1\n", "names: ['{0}']\n".format(productName)]
    with open(file_name_config, "w") as f:
        f.writelines(text_list)
    new_train = TrainMODELS()

    # Kiểm tra xem model đã có hay chưa

    modeled = Modeled.objects(productId=productId).order_by('-last_day_train').first()
    if not modeled:
        weights = "yolov5x.pt"
    else:
        weights = modeled.model_path
    print(weights)
    callback3 = TrainCallBack(productId=productId, status=3,
                              note="Đang Train",userId=userId).save()
    # callback khi data sai
    json_return = data_to_json(callback3)
    requests.post(urlCallback, json=json_return)
    try:
        list_dir_model = new_train.process_train(file_name_config=file_name_config, batch=BATCH, epochs=EPOCHS,
                                                 weights=weights, project="run/train/{0}".format(productName),
                                                 name=productName, device=DEVICE, name_object=productName)
        modeleds = Modeled(productId=productId, product_name=productName, model_path=str(list_dir_model[1]))
        modeleds.save()
        callback4 = TrainCallBack(productId=productId, status=5,
                                  note="Train Complete",userId=userId).save()
        json_return = data_to_json(callback4)
        requests.post(urlCallback, json=json_return)
    except:
        callback5 = TrainCallBack(productId=productId, status=4,
                                  note="Train Errol",userId=userId).save()
        json_return = data_to_json(callback5)
        requests.post(urlCallback, json=json_return)
    time_predicts = TimePredict.objects(query_id=1).order_by('-last_save').first()
    total = time_predicts.total - time_predicts.time_predict
    time_start = int(time.time())
    time_predictss = TimePredict(total=total,time_predicts = time_predicts.time_predict,time_start=time_start)
    time_predictss.save()
    return "Complete"

@app.route('/train',methods=['POST'])
def train():
    #det data post json
    request_data = request.get_json()
    try:
        productId = request_data['productId']
        productName = request_data['productName']
        filePath = request_data['filePath']
        urlCallback = request_data['urlCallback']
        userId = request_data['userId']
    except:
        return jsonify(code=999, message="Error Requets")
    job = q2.enqueue(pretrain_backgroud, args=(productId, productName,filePath,urlCallback,userId,))

    # train = Train(productId=productId, productName=productName, filePath=filePath, urlCallback=urlCallback)
    # train.save()
    # callback = TrainCallBack(productId=productId)
    # callback.save()
    # json_return = data_to_json(callback)

    # callback lan 1 : bao da nhan dc yeu cau
    # requests.post(urlCallback, json=json_return)

    return jsonify(code=200,message="Success",id_queue =job.get_id())


#api check model da train hay chua
@app.route("/modeled",methods=["POST"])
def check_modeled():
    request_data = request.get_json()
    try:
        productId = request_data['productId']
    except:
        return jsonify(code=999, message="Error Requets")
    modeled = Modeled.objects(productId=productId).order_by('-last_day_train').first()
    if not modeled:
        return jsonify(status=0,note="Chua co san pham tren he thong",modeled=None)
    json_return = data_to_json(modeled)
    return jsonify(status=1,note="San Pham Da Duoc Train Roi",modeled=json_return)

if __name__ == "__main__":
    app.run(debug=True,port=6000)