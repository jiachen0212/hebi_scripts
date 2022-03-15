import os
import json
from PIL import Image

def generate_img_index(root_dir, extra_function=None):
    out_filename = r"all_img_index.txt"
    with open(os.path.join(root_dir,out_filename),"w",encoding="utf-8") as writer:
        for root, dirs, files in os.walk(root_dir):
            for filename in files:
                pure_filename, ext = os.path.splitext(filename)
                if ext in ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp'):
                    if extra_function:
                        extra_function(os.path.join(root,filename),writer)
                    else:
                        writer.write("{}\n".format(os.path.join(root, filename)))


def extra_empty_labelme_annotation(image_path,index_writer):
    """
    {
      "version": "4.6.0",
      "flags": {},
      "shapes": null,
      "imagePath": "",
      "imageData": null,
      "imageHeight": 8000,
      "imageWidth": 8192
    }
    """
    out_file_prefix, img_ext = os.path.splitext(image_path)
    _, image_filename = os.path.split(image_path)

    out_labelme_filepath = "{}.json".format(out_file_prefix)

    img = Image.open(image_path)
    w, h = img.size

    json_obj = dict()
    json_obj["version"] = "4.6.0"
    json_obj["flags"] = {}
    json_obj["shapes"] = []
    json_obj["imagePath"] = image_filename
    json_obj["imageData"] = None
    json_obj["imageHeight"] = h
    json_obj["imageWidth"] = w

    if not os.path.isfile(out_labelme_filepath):
        with open(out_labelme_filepath,"w",encoding="utf-8") as writer:
            json.dump(json_obj,writer)
    else:
        print("{} exist!".format(out_labelme_filepath))

    index_writer.write("{},{}\n".format(image_path,out_labelme_filepath))



if __name__ == "__main__":
    root_dir = r"D:\Work\projects\中航锂电\data\20220225\筛选验证数据"
    generate_img_index(root_dir, extra_empty_labelme_annotation)
