import numpy as np
import cv2

NEIGHBOR_THRESHOLD = 5

class Point(object):
    # x, y
    def __init__(self, x, y):
        self.x, self.y = x, y
        
class Feature(object):
    # x, y, label, theta
    def __init__(self, x=0, y=0, label=0, theta=0.0):
        self.x, self.y, self.label, self.theta = x, y, label, theta
        
    def read(self, feature:list()):
        self.x, self.y, self.label = feature
    
    def write(self, save_pth):
        pass
    
class Template(object):
    
    def __init__(self, 
                 width=None, height=None, 
                 tl_x=None, tl_y=None, 
                 pyramid_level=None, features=list()):
        self.width, self.height = width, height
        self.tl_x, self.tl_y = tl_x, tl_y
        self.pyramid_level = pyramid_level
        self.features = features
        
    def read(self, templ:dict()):
        self.width = templ['width']
        self.height = templ['height']
        self.tl_x = templ['tl_x']
        self.tl_y = templ['tl_y']
        self.pyramid_level = templ['pyramid_level']
        features = []
        for f in templ['features']:
            feat = Feature()
            feat.read(f)
            features.append(feat)
        self.features = features
    
    def write(self, save_pth):
        pass  


class Candidate(object):
    # x, y, label, score
    def __init__(self, x, y, label, score):
        self.x, self.y, self.label, self.score = x, y, label, score
        self.f = Feature(x=x, y=y, label=label)

def hysteresisGradient(magnitude, angle, threshold):
    quantized_angle = np.zeros_like(angle, np.uint8)
    # // Quantize 360 degree range of orientations into 16 buckets
    # // Note that [0, 11.25), [348.75, 360) both get mapped in the end to label 0,
    # // for stability of horizontal and vertical features.
    quantized_unfiltered = cv2.convertScaleAbs(angle, alpha=16.0/360.0)

    # // Zero out top and bottom rows
    # @todo is this necessary, or even correct?
    quantized_unfiltered[0, :] = 0
    quantized_unfiltered[-1, :] = 0
    quantized_unfiltered[:, 0] = 0
    quantized_unfiltered[:, -1] = 0
    
    # Mask 16 buckets into 8 quantized orientations
    quantized_unfiltered[1:-1, 1:-1] %= 7
    # cv2.imshow("quantized_unfiltered", quantized_unfiltered)
    # cv2.waitKey(0)
    # print(magnitude.shape, quantized_unfiltered.shape)
    # Filter the raw quantized image. Only accept pixels where the magnitude is above some
    # threshold, and there is local agreement on the quantization.
    for j in range(1, angle.shape[1]-1):
        for i in range(1, angle.shape[0]-1):

            if magnitude[i, j] > threshold:
                histogram = np.zeros(8, np.int32)
                patch3x3 = quantized_unfiltered[i-1:i+2, j-1:j+2].flatten()
                for index in patch3x3:
                    histogram[index] += 1
                
                max_votes = max(histogram)
                index = np.argmax(histogram)
                # Only accept the quantization if majority of pixels in the patch agree
                if max_votes >= NEIGHBOR_THRESHOLD:
                    quantized_angle[i, j] = pow(2, index)
    return quantized_angle


# def select_scattered_features(candidates, num_feat, distance):
    
    
    
class ColorGradientPyramid(object):
    
    def __init__(self, src, mask, weak_thresh, num_feat, strong_thresh):
        self.src = src
        self.mask = mask
        self.pyramid_level = 0
        self.weak_thresh = weak_thresh
        self.num_feat = num_feat
        self.strong_thresh = strong_thresh
        
        self.angle = None
        self.magnitude = None
        self.angle_ori = None
        
        self._update()
        
    def _update(self):
        # Compute horizontal and vertical image derivatives on 
        # all color channels separately
        KERNEL_SIZE = (7, 7)
        smoothed = cv2.GaussianBlur(
            self.src, KERNEL_SIZE, 0, sigmaY=0, borderType=cv2.BORDER_REPLICATE)
        if len(self.src.shape) == 2 or \
            (len(self.src.shape) == 3 and self.src.shape[2] == 1):
            sobel_dx = cv2.Sobel(
                smoothed, cv2.CV_32F, 1, 0, None, 3, 1.0, 0.0, borderType=cv2.BORDER_REPLICATE)
            sobel_dy = cv2.Sobel(
                smoothed, cv2.CV_32F, 0, 1, None, 3, 1.0, 0.0, borderType=cv2.BORDER_REPLICATE)
            magnitude = sobel_dx*sobel_dx + sobel_dy*sobel_dy
            
        else:
            # magnitude = np.zeros_like(self.src, np.float32)
            # src_shape = self.src.shape
            # sobel_dx = np.zeros((src_shape[1], src_shape[0]), np.float32)
            # sobel_dy = np.zeros((src_shape[1], src_shape[0]), np.float32)
            
            sobel_3dx = cv2.Sobel(
                smoothed, cv2.CV_16S, 1, 0, None, 3, 1.0, 0.0, borderType=cv2.BORDER_REPLICATE)
            sobel_3dy = cv2.Sobel(
                smoothed, cv2.CV_16S, 0, 1, None, 3, 1.0, 0.0, borderType=cv2.BORDER_REPLICATE)
            
            # Use the gradient orientation of the channel whose magnitude is largest
            sobel_3d = sobel_3dx*sobel_3dx+sobel_3dy*sobel_3dy
            magnitude = np.max(sobel_3d, axis=2)
            h, w = magnitude.shape[:2]
            inds = np.argmax(sobel_3d, axis=2)
            xx, yy = np.meshgrid(np.arange(w), np.arange(h))
            sobel_dx = sobel_3dx[yy, xx, inds]
            # sobel_dx_vis = cv2.convertScaleAbs(sobel_dx)
            sobel_dy = sobel_3dy[yy, xx, inds]

        sobel_ag = cv2.phase(sobel_dx.astype(np.float32), sobel_dy.astype(np.float32), angleInDegrees=True)
        self.angle = hysteresisGradient(magnitude, sobel_ag, self.weak_thresh**2)
        self.angle_ori = sobel_ag
        self.magnitude = magnitude.astype(np.float32)
        # cv2.imshow("angle", self.angle)
        # cv2.waitKey(0)
        # cv2.imshow("sobel ag", sobel_ag)
        # cv2.waitKey(0)
            
    def _select_scattered_features(self, candidates, num_feat, distance):
        distance_sq = distance**2
    
        features = []
        i, first_select = 0, True
        
        while True:
            c = candidates[i]
            keep = True
            for j in range(len(features)):
                f = features[j]
                keep = ((c.f.x-f.x)**2+(c.f.y-f.y)**2) >= distance_sq
                if not keep:
                    break
            if keep:
                features.append(c.f)
            i += 1
            if i == len(candidates):
                num_ok = len(features) >= num_feat
                if first_select:
                    if num_ok:
                        features.clear()
                        i = 0
                        distance += 1.0
                        distance_sq = distance**2
                        continue
                    else:
                        first_select = False

                i = 0
                distance -= 1.0
                distance_sq = distance ** 2
                if num_ok or distance < 3:
                    break
        return features
    
    def quantize(self):
        # dst = np.zeros_like(self.angle, np.uint8)
        return cv2.copyTo(self.angle, self.mask)
    
    def extract_template(self) -> Template:
        if self.mask is None:
            return None
        
        candidates = []
        mask = self.mask.astype(np.uint8)*255
        local_mask = cv2.erode(
            mask, kernel=None, anchor=(-1, -1), iterations=1, borderType=cv2.BORDER_REPLICATE)
        threshold_sq = self.strong_thresh**2
        nms_kernel_size = 5 
        nms_ksize = nms_kernel_size // 2
        magnitude_valid = np.ones_like(self.magnitude, np.uint8)
        # print("max mask: ", np.max(mask))
        # cv2.imshow("mask", mask)
        # cv2.waitKey(0)
        # magnitude_valid = np.maximum(self.magnitude, threshold_sq)
        h, w = self.magnitude.shape[:2]
        for i in range(nms_ksize, h-nms_ksize, 1):
            for j in range(nms_ksize, w-nms_ksize, 1):
                if local_mask[i, j] and magnitude_valid[i, j]:
                    score = self.magnitude[i-nms_ksize:i+nms_ksize, j-nms_ksize:j+nms_ksize].max()
                    if score == self.magnitude[i, j]:
                        magnitude_valid[i-nms_ksize:i+nms_ksize, j-nms_ksize:j+nms_ksize] = 0
                        magnitude_valid[i, j] = 1   
                        # print(score, threshold_sq, self.angle[i, j])
                        if score > threshold_sq and self.angle[i, j]>0:
                            label = -1 if self.angle[i, j] > 128 else int(np.log2(self.angle[i, j]))
                            candidate = Candidate(i, j, label, score)
                            candidate.f.theta = self.angle_ori[i, j]
                            # print(i, j, label, score, candidate.f.theta)
                            candidates.append(candidate)
        # We require a certain number of features
        num_cand = len(candidates)
        if num_cand < self.num_feat:
            if num_cand <= 4:
                print("too few features, abort")
                return None
            print("have no enough features, exaustive mode")
            
        candidates = sorted(candidates, key=lambda x: x.score)
        
        distance = num_cand /self.num_feat + 1
        templ = Template(-1, -1, None)
        templ.features = self._select_scattered_features(candidates, self.num_feat, distance)
        templ.pyramid_level = self.pyramid_level
        return templ

    def pyramid_down(self):
        # Some parameters need to be adjusted
        self.num_feat /= 2 # @todo Why not 4?
        self.pyramid_level += 1
        
        # Downsample the current inputs
        h, w = self.src.shape[:2]
        dsize = (w//2, h//2)
        # print(self.src.shape, dsize)
        self.src = cv2.pyrDown(self.src, dstsize=dsize)
        
        if self.mask is not None:
            self.mask = cv2.resize(
                self.mask.astype(np.uint8), dsize=dsize, fx=0.0, fy=0.0, interpolation=cv2.INTER_NEAREST)

        self._update()

class ColorGradient:
    # weak_threshold, num_features, strong_threshold
    
    def __init__(self, weak_thresh=30.0, num_feat=63, strong_thresh=60.0):
        self.weak_threshold = weak_thresh
        self.num_features = num_feat
        self.strong_threshold = strong_thresh
        
    def read(self, f_pth):
        pass
    
    def write(self, save_pth):
        pass  
    
    def process(self, src, mask) -> ColorGradientPyramid:
        return ColorGradientPyramid(src, mask, self.weak_threshold, self.num_features, self.strong_threshold) 

class Match(object):
    
    def __init__(self, x, y, similarity, class_id, template_id):
        self.x, self.y = x, y
        self.similarity = similarity
        self.class_id = class_id
        self.template_id = template_id
        
    def sort_matches(self, match):
        pass
    
    def is_equal(self):
        pass

class Info:
    # angle, scale
    def __init__(self, angle, scale):
        self.angle = angle
        self.scale = scale
