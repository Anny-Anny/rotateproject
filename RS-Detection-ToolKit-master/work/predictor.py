import json
import random
import cv2
from detectron2.utils.logger import setup_logger
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
import os
import math
from data.rotated_visualization import myVisualization

# TEST_PATH = "/home/dinghye/下载/科目四初赛第一阶段/test1/"
# CLASS_LIST = {'__background__', 1, 2, 3, 4, 5}
# TEST_PATH = "/home/xjw/下载/rotateproject/dataset/MyDataset/images/val"
# TEST_PATH = "/home/xjw/下载/rotateproject/dataset/smallDataset/images/val"
TEST_PATH = './input_path'
CLASS_LIST = ['__background__', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10']


CLASS_CHAR_LIST = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K']

def vis_single_img():
    setup_logger()

    cfg = get_cfg()
    cfg.merge_from_file("model/my_config.yaml")
    # cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # 模型阈值
    cfg.MODEL.WEIGHTS = "output2/model_final.pth"
    # cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5

    predictor = DefaultPredictor(cfg)
    MetadataCatalog.get("train").set(things_class=CLASS_LIST)

    # 356__1__2772___1848.png,627__1__1848___2772.png
    # for d in random.sample(os.listdir(TEST_PATH), 20):
    for d in ['76.png']:
        # im = cv2.imread(TEST_PATH + str(d) + ".png")

        # im = cv2.imread(os.path.join(TEST_PATH, d))
        im = cv2.imread(os.path.join('/home/xjw/下载/rotateproject/dataset/testset/',d))

        outputs = predictor(im)

        print(outputs)
        # print(TEST_PATH + str(d) + ".png")
        print(d)
        # ship_mentadata = MetadataCatalog.get("ship_train")
        # v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=0.8)
        # v = myVisualization(im[:, :, ::-1],  MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=0.8)
        v = myVisualization(im[:, :, ::-1], MetadataCatalog.get("train"), scale=0.2)
        v = v.draw_instance_predictions(outputs['instances'].to("cpu"))

        cv2.imshow("test", v.get_image()[:, :, ::-1])
        cv2.waitKey(0)


def tojs(predictor, d):
    im = cv2.imread(os.path.join(TEST_PATH, d))

    outputs = predictor(im)

    # print(outputs)
    # print(TEST_PATH + str(d) + ".png")
    # print(d)
    # output4 transform
    js = {}
    js["image_name"] = d
    labels = []
    if outputs['instances'] is not None:
        instances = outputs['instances'].get_fields()
        # instances = outputs['instances']['_fields']
        instan_num = len(instances['pred_boxes'])
        for i in range(instan_num):
            label = {}
            categoty_id = int(instances['pred_classes'].tolist()[i])
            label['category_id'] = CLASS_CHAR_LIST[categoty_id]
            # print('category_id', categoty_id, 'to', label['category_id'])
            label['points'] = rotateTo4Point(instances['pred_boxes'].tensor.tolist()[i])
            label['confidence'] = instances['scores'].tolist()[i]
            labels.append(label)
    js['labels'] = labels
    return js


def gen():
    js_res = []

    setup_logger()

    cfg = get_cfg()
    cfg.merge_from_file("./model/my_config.yaml")
    # cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # 模型阈值
    cfg.MODEL.WEIGHTS = "./model/model_final.pth"
    # cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5

    predictor = DefaultPredictor(cfg)
    MetadataCatalog.get("train").set(things_class=CLASS_LIST)

    # 356__1__2772___1848.png,627__1__1848___2772.png
    # for d in random.sample(os.listdir(TEST_PATH), 20):
    # for d in random.sample(os.listdir(TEST_PATH), 20):
    for d in os.listdir(TEST_PATH):
        js = tojs(predictor, d)
        if not len(js.get('labels')) == 0:
            js_res.append(js)

    return js_res


def write_to_json(filename, dic):
    with open(filename, 'w') as f:
        json.dump(dic, f, indent=4)


def rotateTo4Point(params):
    mbox_cx = params[0]
    mbox_cy = params[1]
    mbox_w = params[2]
    mbox_h = params[3]
    mbox_ang = params[4]
    # print(mbox_cx, mbox_cy, mbox_w, mbox_h, mbox_ang)

    bow_x = mbox_cx + mbox_w / 2 * math.cos(mbox_ang)
    bow_y = mbox_cy + mbox_w / 2 * math.sin(mbox_ang)

    tail_x = mbox_cx - mbox_w / 2 * math.cos(mbox_ang)
    tail_y = mbox_cy - mbox_w / 2 * math.sin(mbox_ang)

    bowA_x = bow_x + mbox_h / 2 * math.sin(mbox_ang)
    bowA_y = bow_y - mbox_h / 2 * math.cos(mbox_ang)

    bowB_x = bow_x - mbox_h / 2 * math.sin(mbox_ang)
    bowB_y = bow_y + mbox_h / 2 * math.cos(mbox_ang)

    tailA_x = tail_x + mbox_h / 2 * math.sin(mbox_ang)
    tailA_y = tail_y - mbox_h / 2 * math.cos(mbox_ang)

    tailB_x = tail_x - mbox_h / 2 * math.sin(mbox_ang)
    tailB_y = tail_y + mbox_h / 2 * math.cos(mbox_ang)

    return [[bowA_x, bowA_y], [bowB_x, bowB_y], [tailA_x, tailA_y], [tailB_x, tailB_y]]


if __name__ == '__main__':
    # res = gen(['356__1__2772___1848.png', '627__1__1848___2772.png'])

    res = gen()
    print('js result', res, len(res))
    write_to_json('./output_path/aircraft_results.json', res)

    #print(rotateTo4Point([10, 10, 10, 10, 1.57]))
    # vis_single_img()
