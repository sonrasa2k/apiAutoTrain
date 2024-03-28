from flask import Flask,request,redirect,url_for,make_response
from flask import jsonify

from datetime import datetime

from flask_sqlalchemy import SQLAlchemy
import requests

import zipfile
import os
from train_custom import TrainMODELS

from num_object import NumObject

import cv2

app = Flask(__name__)
SERECT_KEY = "AI_OCTOPUS2021"
app.config["JWT_SECRET_KEY"] = SERECT_KEY  # Change this!
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///database.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)


#hàm random


def get_key():

    now = datetime.now()

    now = str(now).split(" ")
    key1 = "".join(now[0].split("-"))
    key2 = "".join(("".join(now[1].split(":")).split(".")))
    key = key2 + key1
    return key


# Khỏi tạo database

# class Train(db.Model):
#     id = db.Column(db.Text, primary_key=True)
#     name_object = db.Column(db.Text, nullable=False)
#     day_train = db.Column(db.DATETIME, nullable=False, default=datetime.now())
#     url_data = db.Column(db.Text, nullable=False)
#     model_id = db.Column(db.Text,nullable=False,primary_key=True)
#     modeled_id = db.relationship('Modeled', backref='train', lazy=True)

class Modeled(db.Model):
    id_model = db.Column(db.INTEGER, primary_key=True)
    name_product = db.Column(db.Text, nullable=False)
    id_product = db.Column(db.INTEGER, nullable=False,unique=False)
    path_model = db.Column(db.Text,nullable=False)
    last_day_train = db.Column(db.DATETIME,nullable=False,default=datetime.now())
    product_id = db.relationship('Detect', backref='modeled', lazy=True)

class RequestDetect(db.Model):
    __tablename__ = 'requestdetect'
    id = db.Column(db.Text, primary_key=True)
    listProduct = db.Column(db.Text, nullable=False)
    filePath = db.Column(db.Text,nullable=False)
    typeAIId = db.Column(db.INTEGER,nullable=False,default=3) #1/2/3 = 1+2
    products_id = db.relationship('Detect', backref='requestdetect', lazy=True)
class Detect(db.Model):
    id = db.Column(db.INTEGER, primary_key=True)
    product_id = db.Column(db.INTEGER, db.ForeignKey('modeled.id_product'),nullable=False)
    request_id = db.Column(db.Text,db.ForeignKey('requestdetect.id'),nullable=False)
    name_product = db.Column(db.Text,nullable=False)
    num_product = db.Column(db.INTEGER,nullable=False,default=0)


@app.route('/add/model',methods=['GET'])
def add_model():
    path_model = request.args.get("path_model")
    id_product = request.args.get("id_product")
    name_product = request.args.get("name_product")
    db.session.add(Modeled(name_product=name_product,id_product=id_product,path_model=path_model))
    db.session.commit()
    return jsonify(code=200,msg="them model thnah cong")

@app.route('/data/request')
def get_requests():
    data = []
    rows = RequestDetect.query.all()
    for row in rows:
        json = {"id":row.id,"list_product":row.listProduct,"filePath":row.filePath,'typeAIId':row.typeAIId}
        data.append(json)
    return jsonify(code=201,data=data)

@app.route('/train',methods=['POST'])
def train():
    return "None"
@app.route('/detect',methods=['POST'])
def detect():

    request_data = request.get_json()
    try:
        filePath = request_data['filePath']
        urlCallback = request_data['urlCallback']
        listProduct = request_data['listProduct']
        surveyDataId = request_data['surveyDataId']
        surveyDataGroupId = request_data['surveyDataGroupId']
        fileId = request_data['fileId']
        domain = request_data['domain']
        typeAIId= request_data['typeAIId']
    except:
        return jsonify(code=402,msg="Thiếu Tham Số")

    #cấu hình json trả về
    json_return = {
        "surveyDataId": surveyDataId,
        "surveyDataGroupId": surveyDataGroupId,
        "type": 1,  # 1 hop le , 0 khong hop le
        "note": "OKE",
        "data": [
            {
                "typeAIId": 1,
                "value": 2,  # 0 la khong tai được, 1 là ảnh chụp màn hình, 2 là oke
                "note": "OKE"
            },
            {
                "typeAIId": 2,
                "status": 1,  # 1 là đếm đc , 2 là không đếm được
                "note": "OKE",
                "value": []
            }
        ]
    }
    id = "dt" + get_key()
    db.session.add(RequestDetect(id=id,listProduct=str(listProduct),filePath=filePath))
    db.session.commit()

    img = cv2.imread(filePath, 0)
    if len(img)<1:
        json_return['type'] = 0
        json_return['note'] = "File image sai"
        json_return['data'][0]['value'] = 0
        json_return['data'][0]['note'] = "Không tải được hình do path sai"
        json_return['data'][1]['status'] = 2
        json_return['data'][1]['note'] = "Không đếm được do không tải được hình"
        return json_return
    #check type AIID
    for ai in typeAIId:
        if ai == 1:
            pass
        if ai == 2:
            #viet ham detect o day
            value = []
            num_object = NumObject()
            for product in listProduct:
                modeled = Modeled.query.filter_by(id_product = product['productId']).one_or_none()
                if not modeled:
                    json_return['type'] = 0
                    json_return['note'] = "Sản Phẩm Chưa Có Trên Hệ Thống"
                    json_return['data'][0]['value'] = 2
                    json_return['data'][0]['note'] = "OKE"
                    json_return['data'][1]['status'] = 2
                    json_return['data'][1]['note'] = "Sản Phẩm Chưa Có Trên Hệ Thống"
                    return json_return
                save_path ='runs/detect/'
                num_product = num_object.num_object(path_img=filePath,path_modeled=modeled.path_model,name_object=modeled.name_product,save_path=save_path)
                json_detect = {
                    "note":"OKE",
                    "product_id":product['productId'],
                    "quantity":num_product['num'],
                    "boxes":num_product['data'],
                    "product_name":num_product['name_object']
                }
                value.append(json_detect)
            json_return['data'][1]['value'] = value
    return json_return
if __name__ == '__main__':
    db.create_all()
    app.run(debug=True)
