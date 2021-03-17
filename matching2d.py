import numpy as np
import cv2
import yaml
import sys
import os

from feature import *
# from shape_info_produce import ShapeInfoProducer

MAX_INT = sys.maxsize
EPS = 0.00001
LUT3 = 3
SIMILARITY_LUT = [[0, 4, LUT3, 4, 0, 4, LUT3, 4, 0, 4, LUT3, 4, 0, 4, LUT3, 4, 0, 0, 0, 0, 0, 0, 0, 0, LUT3, LUT3, LUT3, LUT3, LUT3, LUT3, LUT3, LUT3],
                  [0, LUT3, 4, 4, LUT3, LUT3, 4, 4, 0, LUT3, 4, 4, LUT3, LUT3,
                      4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, LUT3, LUT3, 4, 4, 4, 4, LUT3, LUT3, LUT3, LUT3, 4, 4,
                      4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, LUT3, LUT3, LUT3, LUT3, 4, 4, 4, 4, 4, 4, 4, 4, 0, LUT3,
                      0, LUT3, 0, LUT3, 0, LUT3, 0, LUT3, 0, LUT3, 0, LUT3, 0, LUT3],
                  [0, 0, 0, 0, 0, 0, 0, 0, LUT3, LUT3, LUT3, LUT3, LUT3, LUT3, LUT3,
                      LUT3, 0, 4, LUT3, 4, 0, 4, LUT3, 4, 0, 4, LUT3, 4, 0, 4, LUT3, 4],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, LUT3, 4,
                      4, LUT3, LUT3, 4, 4, 0, LUT3, 4, 4, LUT3, LUT3, 4, 4],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, LUT3,
                      LUT3, 4, 4, 4, 4, LUT3, LUT3, LUT3, LUT3, 4, 4, 4, 4],
                  [0, LUT3, 0, LUT3, 0, LUT3, 0, LUT3, 0, LUT3, 0, LUT3, 0, LUT3, 0, LUT3, 0, 0, 0, 0, LUT3, LUT3, LUT3, LUT3, 4, 4, 4, 4, 4, 4, 4, 4]]


def rotate_point2d(p, center, angle):
    p_arr = np.array(p, np.float32)
    cen_arr = np.array(center, np.float32)
    rot_mat = np.array([[np.cos(angle), -1*np.sin(angle)],
                        [np.sin(angle), np.cos(angle)]], np.float32)
    p_rot = np.dot(rot_mat, p_arr-cen_arr)+cen_arr
    return p_rot


def crop_templates(templates: list()):
    min_x, min_y, max_x, max_y = MAX_INT, MAX_INT, -1*MAX_INT, -1*MAX_INT
    # First pass: find min/max feature x,y over all pyramid levels and modalities
    for templ in templates:
        for f in templ.features:
            x = f.x << templ.pyramid_level
            y = f.y << templ.pyramid_level
            min_x = min(min_x, x)
            min_y = min(min_y, y)
            max_x = max(max_x, x)
            max_y = max(max_y, y)

    if min_x % 2 == 1:
        min_x -= 1
    if min_y % 2 == 1:
        min_y -= 1

    # Second pass: set width/height and shift all feature positions
    for templ in templates:
        templ.width = (max_x-min_x) >> templ.pyramid_level
        templ.height = (max_y-min_y) >> templ.pyramid_level
        templ.tl_x = min_x >> templ.pyramid_level
        templ.tl_y = min_y >> templ.pyramid_level
        for f in templ.features:
            f.x -= templ.tl_x
            f.y -= templ.tl_y
    return [min_x, min_y, max_x-min_x, max_y-min_y]


def spread(src, T):
    dst = np.zeros_like(src, np.uint8)
    h, w = src.shape[:2]
    for i in range(T):
        for j in range(T):
            dst[:h-j, :w-i] = np.bitwise_or(dst[:h-j, :w-i], src[j:, i:])
    return dst


def linearize(response_map, T):
    h, w = response_map.shape[:3]
    assert w % T == 0 and h % T == 0
    # linearized has T^2 rows, where each row is a linear memory
    mem_width = w // T
    mem_height = h // T

    linearized = np.zeros((T**2, mem_height*mem_width), np.uint8)
    # Outer two for loops iterate over top-left T^2 starting pixels
    for i in range(T):
        for j in range(T):
            row = i*T+j
            xx, yy = np.meshgrid(np.arange(i, h, T), np.arange(j, w, T))
            linearized[row, :] = response_map[xx.T, yy.T].flatten()
    return linearized


def compute_response_maps(src):
    h, w = src.shape[:2]
    assert (h * w) % 16 == 0

    # Least significant 4 bits of spread image pixel
    lsb4 = np.bitwise_and(src, 15)
    # Most significant 4 bits, right-shifted to be in [0, 16)
    msb4 = np.right_shift(np.bitwise_and(src, 240), 4)

    response_maps = []
    for i in range(8):
        lut_low = np.array(SIMILARITY_LUT[i][:16])
        lut_hight = np.array(SIMILARITY_LUT[i][16:])
        map_data = np.maximum(lut_low[lsb4], lut_hight[msb4])
        response_maps.append(map_data)
    return response_maps


def access_linear_memory(linear_memories, f, T, W):
    # Retrieve the TxT grid of linear memories associated with the feature label
    memory_grid = linear_memories[f.label]
    h, w = memory_grid.shape
    assert h == T**2
    assert f.x >= 0 and f.y >= 0

    grid_x = f.y % T
    grid_y = f.x % T
    grid_index = int(grid_y * T + grid_x)
    assert grid_index >= 0 and grid_index < h
    # memory = memory_grid[grid_index]

    lm_x, lm_y = f.y // T, f.x // T
    lm_index = int(lm_y * W + lm_x)
    assert lm_index >= 0 and lm_index < w
    return memory_grid[grid_index:, lm_index:].flatten()


def similarity(linear_memories, templ, size, T):
    # 63 features or less is a special case because the max similarity per-feature is 4.
    # 255/4 = 63, so up to that many we can add up similarities in 8 bits without worrying
    # about overflow. Therefore here we use _mm_add_epi8 as the workhorse, whereas a more
    # general function would use _mm_add_epi16.
    assert len(templ.features) < 8192
    # @todo Handle more than 255/MAX_RESPONSE features!!

    # Decimate input image size by factor of T
    H = size[0] // T
    W = size[1] // T
    # Feature dimensions, decimated by factor T and rounded up
    wf = int((templ.height - 1) / T + 1)
    hf = int((templ.width - 1) / T + 1)
    # Span is the range over which we can shift the template around the input image
    span_x, span_y = W - wf, H - hf

    # Compute number of contiguous (in memory) pixels to check when sliding feature over
    # image. This allows template to wrap around left/right border incorrectly, so any
    # wrapped template matches must be filtered out!
    template_positions = span_y * W + span_x + 1
    # @todo In old code, dst is buffer of size m_U. Could make it something like
    # (span_x)x(span_y) instead?
    dst = np.zeros((H, W), np.uint16).flatten()

    # Compute the similarity measure for this template by accumulating the contribution of
    # each feature
    for f in templ.features:
        if f.x < 0 or f.x >= size[1] or f.y < 0 or f.y >= size[0]:
            continue
        lm_ptr = access_linear_memory(linear_memories, f, T, W)
        dst[:template_positions] += lm_ptr[:template_positions]
    return dst.reshape((H, W))


def similarity_local(linear_memories, templ, size, T, center):
    # Similar to whole-image similarity() above. This version takes a position 'center'
    # and computes the energy in the 16x16 patch centered on it.
    # Compute the similarity map in a 16x16 patch around center
    W = int(size[1] // T)
    dst = np.zeros((16, 16), np.uint16)
    # Offset each feature point by the requested center. Further adjust to (-8,-8) from the
    # center to get the top-left corner of the 16x16 patch.
    # NOTE: We make the offsets multiples of T to agree with results of the original code.
    offset_x = (center.x / T - 8) * T
    offset_y = (center.y / T - 8) * T
    for f in templ.features:
        feat = Feature(f.x+offset_y, f.y+offset_x, f.label, f.theta)
        # Discard feature if out of bounds, possibly due to applying the offset
        if feat.x < 0 or feat.y < 0 or feat.x > size[1] or feat.y > size[0]:
            continue
        lm_ptr = access_linear_memory(linear_memories, feat, T, W)
        if lm_ptr is None:
            continue
        for i in range(16):
            ind = W*i
            dst[i] += lm_ptr[ind:ind+16]

    return dst


class ShapeDetector(object):

    def __init__(self, T=[4, 8], num_feat=0, weak_thresh=30.0, strong_thresh=60.0):
        self.num_feat = num_feat

        self.modality = ColorGradient(weak_thresh, num_feat, strong_thresh)
        self.pyramid_levels = len(T)
        self.T_at_level = T

        self.temp_pyramid_list = list()
        self.temp_class_map = dict()

    def add_template(self, src, class_id, mask, num_feat=0) -> int:

        if class_id in self.temp_class_map.keys():
            template_pyramids = self.temp_class_map[class_id]
        else:
            template_pyramids = []
        template_id = len(template_pyramids)
        # Extract a template at each pyramid level
        color_gradient_pyramid = self.modality.process(src, mask)

        tp = []
        if num_feat > 0:
            color_gradient_pyramid.num_features = num_feat

        for l in range(self.pyramid_levels):
            # @todo Could do mask subsampling here instead of in pyrDown()
            if l > 0:
                color_gradient_pyramid.pyramid_down()

            templ = color_gradient_pyramid.extract_template()
            if templ is not None:
                tp.append(templ)
            else:
                return -1
        # tp = None
        crop_templates(tp)
        template_pyramids.append(tp)
        self.temp_class_map[class_id] = template_pyramids

        return template_id

    def add_template_rotate(self, class_id, zero_id, theta, center):
        template_pyramids = self.temp_class_map[class_id]
        template_id = len(template_pyramids)

        to_rot_tp = template_pyramids[zero_id]
        theta_rad = theta*np.pi/180

        tp = []
        for l in range(self.pyramid_levels):
            if l > 0:
                center /= 2

            features = []
            for f in to_rot_tp[l].features:
                x = f.x + to_rot_tp[l].tl_x
                y = f.y + to_rot_tp[l].tl_y
                p_rot = rotate_point2d((x, y), center, theta_rad)

                f_new = Feature(int(p_rot[0] + 0.5),
                                int(p_rot[1]+0.5), theta=f.theta-theta)
                f_new.theta %= 360
                if f_new.theta < 0:
                    f_new.theta += 360

                f_new.label = int(f_new.theta * 16 / 360 + 0.5)
                f_new.label %= 7
                features.append(f_new)

            templ = Template(features=features)
            templ.pyramid_level = l
            tp.append(templ)
        crop_templates(tp)
        template_pyramids.append(tp)
        return template_id

    def get_templates(self, class_id, template_id):
        assert class_id in self.temp_class_map.keys()
        assert template_id < len(self.temp_class_map[class_id])
        return self.temp_class_map[class_id][template_id]

    def get_modalities(self):
        return self.modality

    def _match_class(self, lm_pyramid, sizes, threshold, class_id, template_pyramids):
        matches = []
        for temp_id, tp in enumerate(template_pyramids):
            # First match over the whole image at the lowest pyramid level
            # @todo Factor this out into separate function
            lowest_lm = lm_pyramid[-1]
            # Compute similarity maps for each ColorGradient at lowest pyramid level
            # candidates = []
            # lowest_start = len(tp) - 1
            lowest_T = self.T_at_level[-1]
            num_features = 0
            templ = tp[-1]
            num_features += len(templ.features)
            if len(templ.features) < 8192:
                similarities = similarity(
                    lowest_lm[0], templ, sizes[-1], lowest_T)
                similarities = similarities.astype(np.uint16)
            else:
                raise ValueError("feature size too large than 8192")

            # Find initial matches
            scores = (similarities*100) / (4*num_features)
            inds = np.argwhere(scores > threshold)
            offset = lowest_T / 2 + (lowest_T % 2 - 1)
            inds_ = inds*lowest_T + offset
            candidates = [Match(ind_[0], ind_[1], scores[ind[0], ind[1]], class_id, temp_id)
                          for ind_, ind in zip(inds_, inds)]

            # Locally refine each match by marching up the pyramid
            for l in range(self.pyramid_levels-2, -1, -1):
                lms = lm_pyramid[l]
                T = self.T_at_level[l]
                size = sizes[l]
                border = 8 * T
                offset = T / 2 + (T % 2 - 1)
                max_x = size[0] - tp[l].width - border
                max_y = size[1] - tp[l].height - border

                for match2 in candidates:
                    x, y = match2.x * 2 + 1, match2.y * 2 + 1
                    x, y = max(x, border), max(y, border)
                    x, y = min(x, max_x), min(y, max_y)
                    numFeatures = 0
                    templ = tp[l]
                    numFeatures += len(templ.features)
                    similarities2 = similarity_local(
                        lms[0], templ, size, T, Point(x, y))

                    scores = (similarities2 * 100.0) / (4 * numFeatures)
                    # Find best local adjustment
                    best_inds = np.argmax(scores, axis=1)
                    r = np.argmax(best_inds)
                    c = best_inds[r]
                    match2.similarity = np.max(scores)
                    # r = best_ind % scores.shape[1]
                    # c = best_ind // scores.shape[1]
                    match2.x = (x / T - 8 + c) * T + offset
                    match2.y = (y / T - 8 + r) * T + offset

                new_candidates = list(
                    filter(lambda x: x.similarity >= threshold, candidates))
            matches += new_candidates
        return matches

    def match(self, src, threshold, class_ids, mask=None):
        matches = list()
        # Initialize each ColorGradient with our sources
        quantizers = list()
        assert mask is None or len(mask.shape) == len(src.shape)

        quantizers.append(self.modality.process(src, mask))

        lm_pyramid = []
        sizes = []
        for l in range(self.pyramid_levels):
            T = self.T_at_level[l]
            if l > 0:
                for quant in quantizers:
                    quant.pyramid_down()

            lm_level = []
            for quant in quantizers:
                quantized = quant.quantize()
                spread_quantized = spread(quantized, T)
                response_maps = compute_response_maps(spread_quantized)

                memories = []
                for i in range(8):
                    linear_memory = linearize(response_maps[i], T)
                    memories.append(linear_memory)
                lm_level.append(memories)

            lm_pyramid.append(lm_level)
            sizes.append(quantized.shape)

        print("construct response map")

        if class_ids is None or len(class_ids) <= 0:
            for class_id, template_pyramids in self.temp_class_map.items():
                matches += self._match_class(lm_pyramid, sizes,
                                             threshold, class_id, template_pyramids)
        else:
            for class_id in class_ids:
                matches += self._match_class(lm_pyramid, sizes,
                                             threshold, class_id, self.temp_class_map[class_id])
        print("length of matches: ", len(matches))
        # Sort matches by similarity, and prune any duplicates introduced by pyramid refinement
        matches_sorted = sorted(matches, key=lambda x: (
            x.similarity, x.template_id), reverse=True)
        matches_filter = list()
        for ms in matches_sorted:
            if len(matches_filter) <= 0:
                matches_filter.append(ms)
                continue
            if ms.x == matches_filter[-1].x and ms.y == matches_filter[-1].y \
                    and ms.similarity == matches_filter[-1].similarity and ms.class_id == matches_filter[-1].class_id:
                continue
            matches_filter.append(ms)
        return matches_filter

    def read_classes(self, ids, fpath):
        with open(fpath, 'r') as f:
            temps_data = yaml.load(f, Loader=yaml.FullLoader)

        for cls_id in ids:
            cls_id = temps_data['class_id']
            self.pyramid_levels = temps_data['pyramid_levels']
            template_pyramids = []
            for template_pyramid in temps_data['template_pyramids']:
                templs = []
                for templ in template_pyramid['templates']:
                    tp = Template()
                    tp.read(templ)
                    templs.append(tp)
                template_pyramids.append(templs)
            self.temp_class_map[cls_id] = template_pyramids

    def write_classes(self, save_root):

        for cls_id, templs in self.temp_class_map.items():
            cls_dict = dict()
            cls_dict['class_id'] = cls_id
            cls_dict['pyramid_levels'] = self.pyramid_levels
            temp_pyramids = []
            for idx, tps in enumerate(templs):
                templates = []
                for tp in tps:
                    save_features = [[f.x, f.y, f.label] for f in tp.features]
                    tp_dict = {
                        "width": tp.width, "height": tp.height,
                        "tl_x": tp.tl_x, "tl_y": tp.tl_y,
                        "pyramid_level": tp.pyramid_level,
                        "features": save_features}
                    templates.append(tp_dict)
                tps_dict = {"template_id": idx, "templates": templates}
                temp_pyramids.append(tps_dict)
            cls_dict["template_pyramids"] = temp_pyramids

            save_pth = os.path.join(save_root, "{}_templ.yaml".format(cls_id))
            with open(save_pth, 'w') as f:
                yaml.dump(cls_dict, f, default_flow_style=True)
