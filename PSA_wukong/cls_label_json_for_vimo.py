# codig=utf-8
import os
import json

# {
#   "version": "4.2.5",
#   "flags": {},
#   "shapes": [
#     {
#       "label": "棉",
#       "points": [
#         [
#           1,
#           0
#         ],
#         [
#           199,
#           199
#         ]
#       ],
#       "group_id": null,
#       "shape_type": "rectangle",
#       "flags": {}
#     }
#   ],
#   "imagePath": "46-scale_9_im_3_col.png",
#   "imageData": null,
#   "imageHeight": 0,
#   "imageWidth": 0
# }

def genreate_cls_json(txt_dir, txts, label_dict, json_for_vimo_dir):
    for ind, txt in enumerate(txts):
        file = open(os.path.join(txt_dir, txt))
        lines = [a[:-1] for a in file.readlines() if len(a) > 0]
        for line in lines:
            lab_dict = {"version": "4.2.5", "flags": {}, "shapes": [{ "label": None, "points": [[1,0],
                [199, 199]], "group_id": None, "shape_type": "rectangle", "flags": {}}], "imageData": None, "imageHeight": 0, "imageWidth": 0}
            im_name = os.path.basename(line)
            lab_dict["imagePath"] = im_name
            print(label_dict[str(ind)])
            lab_dict["shapes"][0]["label"] = label_dict[str(ind)]
            with open(os.path.join(json_for_vimo_dir, "{}.json".format(im_name[:-4])) , "w", encoding='utf-8') as fp:
                # fp.write(json.dumps(lab_dict,indent=4))
                json.dump(lab_dict, fp,ensure_ascii=False,indent = 4)



if __name__ == "__main__":
    # 类别名称
    label_dict = {'0': 'ok', '1': r"变形"}

    # 算法这边持有的分类txt
    txt_dir = '/Users/chenjia/Downloads/Learning/SmartMore/2022/DL/hebi/hebi_data/2022.02.14/jsons'
    txts = ['0214_no_bianxing.txt', '0214_yes_bianxing.txt']

    # 可给到vimo使用的json类型分类标注, 落盘此处
    json_for_vimo_dir = '/Users/chenjia/Desktop/0214bianxing_jsons/'
    if not os.path.exists(json_for_vimo_dir):
        os.makedirs(json_for_vimo_dir)

    genreate_cls_json(txt_dir, txts, label_dict, json_for_vimo_dir)



