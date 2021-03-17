import numpy as np
import yaml
import cv2

from feature import Info, Point

EPS = 0.00001

def _transform(src, angle, scale):
    cols, rows = src.shape[:2]
    # center = Point(cols*0.5, rows*0.5)
    rot_mat = cv2.getRotationMatrix2D((cols*0.5, rows*0.5), angle, scale)
    dst = cv2.warpAffine(src, rot_mat, (cols, rows))
    return dst
    
    
class ShapeInfoProducer(object):
    
    def __init__(self, src, mask=None, angle_range=[0], scale_range=[1], angle_step=15, scale_step=0.5):
        self.src = src
        if mask is None:
            self.mask = np.zeros_like(src, dtype=np.float32)*255
        else:
            self.mask = mask
        self.angle_range = angle_range
        self.scale_range = scale_range
        self.angle_step = angle_step
        self.scale_step = scale_step
        
        self.infos = list()
            
    def src_of(self, info):
        return _transform(self.src, info.angle, info.scale)
    
    def mask_of(self, info):
        return (_transform(self.mask, info.angle, info.scale)>0)
    
    def produce_infos(self):
        global EPS
        self.infos.clear()
        
        assert len(self.angle_range) <= 2
        assert len(self.scale_range) <= 2
        assert self.angle_step > EPS*10
        assert self.scale_step > EPS*10

        if len(self.angle_range) == 0:
            self.angle_range = [0, 0]
        elif len(self.angle_range) == 1:
            self.angle_range.append(self.angle_range[0])
        else:
            assert self.angle_range[0] <= self.angle_range[1]
            
        if len(self.scale_range) == 0:
            self.scale_range = [1, 0]
        elif len(self.scale_range) == 1:
            self.scale_range.append(self.scale_range[0])
        else:
            assert self.scale_range[0] <= self.scale_range[1]
        
        # for angle in range(, , self.angle_step):
        scale = self.scale_range[0]
        while scale < self.scale_range[1]+EPS:
            angle = self.angle_range[0]
            while angle < self.angle_range[1]+EPS:
                self.infos.append(Info(angle, scale))
                angle += self.angle_step
            scale += self.scale_step                
    
    def save_infos(self, infos, save_pth="infos.yaml"):
        infos_list = [{"angle": info.angle, "scale": info.scale} for info in infos]
        infos_dict = {"infos": infos_list}
        with open(save_pth, 'w', encoding="utf-8") as f:
            yaml.dump(infos_dict, f)
            
    def load_infos(self, info_pth="infos.yaml"):
        with open(info_pth, 'r', encoding='utf-8') as f:
            infos_list = yaml.load(f, Loader=yaml.FullLoader)
        infos = [Info(info['angle'], info['scale']) for info in infos_list['infos']]
        return infos


    