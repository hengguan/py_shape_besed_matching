import cv2
import numpy as np
import os

from matching2d import ShapeDetector
from shape_info_produce import ShapeInfoProducer


def angle_test(mode="test", use_rot=True, img_pth="./data"):
    shape_det = ShapeDetector(T=[4, 8], num_feat=128)
    img = cv2.imread(img_pth)
    assert img is not None, "check your image path"

    if mode != "test":
        mask = np.ones((img.shape[0], img.shape[1]), np.uint8)*255
        print(mask.shape, img.shape)

        # padding to avoid rotating out
        PADDING = 99
        padded_img = np.pad(
            img, ((PADDING, PADDING+1), (PADDING, PADDING+1), (0, 0)), 'constant', constant_values=0)
        padded_mask = np.pad(
            mask, ((PADDING, PADDING+1), (PADDING, PADDING+1)), 'constant', constant_values=0)

        print("padding shape: ", padded_img.shape, padded_mask.shape)
        shape_info = ShapeInfoProducer(
            padded_img, padded_mask, angle_range=[0, 360], scale_range=[1], angle_step=1)
        shape_info.produce_infos()

        infos_have_templ = list()
        class_id = "test"
        is_first = True
        for info in shape_info.infos:
            to_show = shape_info.src_of(info)
            if is_first:
                templ_id = shape_det.add_template(
                    to_show, class_id, shape_info.mask_of(info))
                first_id = templ_id
                first_angle = info.angle
                if use_rot:
                    is_first = False
            else:
                h, w = shape_info.src.shape[:2]
                templ_id = shape_det.add_template_rotate(
                    class_id, first_id, info.angle-first_angle, np.array([w/2.0, h/2.0]))

            templ = shape_det.get_templates("test", templ_id)
            for f in templ[0].features:
                cv2.circle(
                    to_show, (f.y + templ[0].tl_y, f.x+templ[0].tl_x), 3, (0, 0, 255), -1)

            cv2.imshow("train", to_show)
            cv2.waitKey(1)

            if templ_id != -1:
                infos_have_templ.append(info)

        shape_det.write_classes("./data/")
        shape_info.save_infos(infos_have_templ, "./data/test_info.yaml")
        print("training finished!!!")
    else:
        ids = ["test"]
        shape_det.read_classes(ids, fpath='./data/test_templ.yaml')
        # templ = shape_det.get_templates("test", 341)
        # for f in templ[0].features:
        #     print(f.y, f.x)
        print("read training data finished!!!")
        # angle & scale are saved here, fetched by match id
        shape_info = ShapeInfoProducer(src=None)
        infos = shape_info.load_infos(info_pth="./data/test_info.yaml")

        PADDING = 250
        padded_img = np.pad(
            img, ((PADDING, PADDING+1), (PADDING, PADDING+1), (0, 0)), 'constant', constant_values=0)
        STRIDE = 16
        n, m = padded_img.shape[0] // STRIDE, padded_img.shape[1] // STRIDE
        match_img = padded_img[:n*16, :m*16]

        matches = shape_det.match(match_img, 80, ids)
        if len(matches) == 0:
            print("not match anything")
            exit(0)
        if len(img.shape) == 2 or img.shape[2] == 1:
            match_img = cv2.cvtColor(match_img, cv2.COLOR_GRAY2BGR)

        top5 = 1
        if top5 > len(matches):
            top5 = len(matches)

        for i in range(top5):
            match = matches[i]
            templ = shape_det.get_templates("test", match.template_id)
            print("match template id: ", match.template_id)
            # 270 is width of template image
            # // 100 is padding when training
            # // tl_x/y: template croping topleft corner when training
            r_scaled = 270 / 2.0 * infos[match.template_id].scale
            # scaling won't affect this, because it has been determined by warpAffine
            # // cv::warpAffine(src, dst, rot_mat, src.size()); last param
            train_img_half_width = 270 / 2.0 + 100
            train_img_half_height = 270 / 2.0 + 100

            x = match.x - templ[0].tl_x + train_img_half_width
            y = match.y - templ[0].tl_y + train_img_half_height

            for f in templ[0].features:
                cv2.circle(match_img, (int(f.y+match.x),
                                       int(f.x+match.y)), 3, (0, 255, 0), -1)

            cv2.putText(match_img, "{}".format(match.similarity),
                        (int(match.x+r_scaled-10), int(match.y-3)),
                        cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 255))

        cv2.imshow("match img", match_img)
        cv2.imwrite("./data/result.jpg", match_img)
        cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    angle_test("test", True, "./data/case2/test1.bmp")
