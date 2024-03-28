from fastai.vision.all import *


class DetectScreen:
    def __init__(self):
        self.pathImg = None
        self.path = 'screen2.pkl'
        self.load_inf = load_learner(self.path)
    def check_screen(self,img_path):
        try:
            kq = self.load_inf.predict(img_path)
        except:
            kq = []
        if len(kq) == 0:
            jsons = {
                "typeAIId": 1,
                "value": 0,  # 0 la khong tai được, 1 là ảnh chụp màn hình, 2 là oke
                "note": "HINH ANH KHONG TAI DUOC HOAC HE THONG LOI"
            }
            return jsons
        if kq[0] == "fake":
            jsons = {
                "typeAIId": 1,
                "value": 1,  # 0 la khong tai được, 1 là ảnh chụp màn hình, 2 là oke
                "note": "ANH CHUP MAN HINH"
            }
            return jsons
        jsons = {
                "typeAIId": 1,
                "value": 2,  # 0 la khong tai được, 1 là ảnh chụp màn hình, 2 là oke
                "note": "OKE"
            }
        return jsons
if __name__ == '__main__':
    a = DetectScreen()
    b = a.check_screen('dataset/Coke_Org_PET_390/images/Sprite 1 (100).jpg')
    print(b)
