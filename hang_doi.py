from flask import  jsonify
import json
def detect_background(list_request):
    list_result = []
    for request_data in list_request:
        try:
            filePath = request_data['filePath']
            urlCallback = request_data['urlCallback']
            listProduct = request_data['listProduct']
            surveyDataId = request_data['surveyDataId']
            surveyDataGroupId = request_data['surveyDataGroupId']
            fileId = request_data['fileId']
            domain = request_data['domain']
            typeAIId = request_data['typeAIId']
        except:
            return jsonify(code=402, msg="Thiếu Tham Số")
        request_detect = RequestDetect(filePath=filePath, urlCallback=urlCallback, listProduct=listProduct,
                                       surveyDataId=surveyDataId, surveyDataGroupId=surveyDataGroupId,
                                       fileId=fileId, domain=domain, typeAIId=typeAIId)
        request_detect.save()
        link_image = domain + '/' + filePath
        imgs = Get_Image()
        img = imgs.get_image_from_link(link_image)
        name = filePath.split('/')
        filename_img = 'public/' + name[len(name) - 1]
        cv2.imwrite(filename_img, img)
        try:
            if img == None:
                detects = Detect(surveyDataId=surveyDataId, surveyDataGroupId=surveyDataGroupId, types=0, fileId=fileId,
                                 note="Path Image Invalid")
                detects.save()
                jsons = data_to_json(detects)
                requests.post(urlCallback, json=jsons)
                return jsons
        except:
            if len(img) < 10:
                detects = Detect(surveyDataId=surveyDataId, surveyDataGroupId=surveyDataGroupId, fileId=fileId, types=0,
                                 note="Path Image Invalid")
                jsons = data_to_json(detects)
                requests.post(urlCallback, json=jsons)
                return jsons
        data = []
        detects = None
        for ai in typeAIId:
            if ai == 1:
                detect_screen = DetectScreen()
                data.append(detect_screen.check_screen(img))
                pass
            if ai == 2:
                value = []
                num_object = NumObject()
                for product in listProduct:
                    model = Modeled.objects(productId=product['productId']).first()
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
                                 data=data)
                detects.save()
                result = json.loads(detects.to_json())
                requests.post(urlCallback, json=result)
                os.remove(filename_img)
                list_result.append(result)
    list_result = json.dumps(list_result)
    return list_result