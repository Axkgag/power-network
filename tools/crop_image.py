import xml.etree.ElementTree as ET
import os
from PIL import Image
import random

image_folder = '../../datasets/train_collect/train/bin/before/del_image'
xml_folder = '../../datasets/train_collect/train/bin/before/del_xml'
output_path = '../../datasets/train_collect/train/bin/before/crop_del_img'
output_xml = '../../datasets/train_collect/train/bin/before/crop_del_xml'

def get_enhance(h2):
    for img_id, filename in enumerate(os.listdir(image_folder)):
        print(img_id, ": ", filename)
        image_path = os.path.join(image_folder, filename)
        xml_path = os.path.join(xml_folder, filename[:-4]+'.xml')
        if os.path.exists(xml_path):
            info_dict = parse_xml(xml_path)
            if info_dict["boxes"]:
                for box in info_dict["boxes"]:
                    if(abs(box[2]-box[0]) > 2 * h2) or (abs(box[3]-box[1]) > 2 * h2):
                        continue
                    if(box[0] == box[2]) or (box[1] == box[3]):
                        continue
                    center_x = (box[0]+box[2])/2+random.randint(0, 100)
                    center_y = (box[1]+box[3])/2+random.randint(0, 100)
                    box_id = box[4]
                    x1 = int(center_x - h2)
                    y1 = int(center_y - h2)
                    x2 = int(center_x + h2)
                    y2 = int(center_y + h2)
                    img_box = [x1, y1, x2, y2]
                    if (center_x+h2>info_dict["width"]) or (center_y+h2>info_dict["height"]) or (center_x-h2<0) or (center_y-h2<0):
                        x1,y1,x2,y2=cut_size(img_box,info_dict["width"],info_dict["height"])
                    filename1, ext=os.path.splitext(filename)
                    out_filename=filename1+"_"+str(box_id)+ext
                    out_path=os.path.join(output_path,out_filename)
                    crop_and_copy(image_path,x1,y1,x2,y2,out_path)
                    new_xml=filename1+"_"+str(box_id)+".xml"
                    new_xml_path=os.path.join(output_xml,new_xml)
                    get_new_xml(xml_path,x1,y1,x2,y2,h2,new_xml_path,out_filename)


def parse_xml(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    size = root.find('size')
    width = int(size.find('width').text)
    height = int(size.find('height').text)
    boxes = []
    for id, obj in enumerate(root.findall('object')):
        box = obj.find('bndbox')
        x1 = float(box.find('xmin').text)
        y1 = float(box.find('ymin').text)
        x2 = float(box.find('xmax').text)
        y2 = float(box.find('ymax').text)
        boxes.append((x1, y1, x2, y2, id))
    return {'width': width, 'height': height, 'boxes': boxes}


def cut_size(bbox, img_w, img_h):
    x0, y0, x1, y1 = bbox
    xmin = min(x0, x1)
    xmax = max(x0, x1)
    ymin = min(y0, y1)
    ymax = max(y0, y1)
    if xmin < 0:
        xmax-=xmin
        xmin=0
    if ymin<0:
        ymax-=ymin
        ymin=0
    if xmax>img_w:
        xmin-=(xmax-img_w)
        xmax=img_w
    if ymax>img_h:
        ymin-=(ymax-ymin)
        ymax=img_h
    return int(xmin), int(ymin), int(xmax), int(ymax)


def crop_and_copy(img_path,x1,y1,x2,y2,output_path):
    img =Image.open(img_path)
    region=img.crop((x1,y1,x2,y2))
    new_img=Image.new("RGB",(x2-x1,y2-y1),color="white")
    new_img.paste(region,(0,0))
    new_img.save(output_path)


def get_new_xml(xml_path, x1,y1,x2,y2,h2,out_path, out_filename):
    tree=ET.parse(xml_path)
    root_old=tree.getroot()
    root=ET.Element("annotation")
    width1=2*h2
    height1=2*h2

    folder=ET.SubElement(root,"folder")
    folder.text=root_old.find("folder").text
    filename = ET.SubElement(root, "filename")
    filename.text = out_filename
    fileurl = ET.SubElement(root, "fileurl")
    fileurl.text = out_path
    kind = ET.SubElement(root, "kind")
    kind.text = root_old.find("kind").text
    source = ET.SubElement(root, "source")
    database = ET.SubElement(source, "database")
    database.text = root_old.find("source").find("database").text
    id = ET.SubElement(source, "id")
    id.text = root_old.find("source").find("id").text
    annotation = ET.SubElement(source, "annotation")
    annotation.text = root_old.find("source").find("annotation").text
    size = ET.SubElement(root, "size")
    width = ET.SubElement(size, "width")
    height = ET.SubElement(size, "height")
    depth = ET.SubElement(size, "depth")
    width.text = str(width1)
    height.text=str(height1)
    depth.text=str(3)
    storage = ET.SubElement(source, "storage")
    storage.text = root_old.find("storage").text
    objectsum = ET.SubElement(source, "objectsum")
    objectsum.text = root_old.find("objectsum").text
    voltage = ET.SubElement(root, "voltage")
    voltage.text = root_old.find("voltage").text
    fileangle = ET.SubElement(root, "fileangle")
    fileangle.text = root_old.find("fileangle").text

    objs = root_old.findall("object")
    for obj in objs:
        bbox= obj.find("bndbox")
        if(float(bbox.find("xmin").text)>x1) and (float(bbox.find("ymin").text)>y1) and (float(bbox.find("xmax").text)<x2) and (float(bbox.find("ymax").text)<y2):
            new_xmin=float(bbox.find("xmin").text)-x1
            new_ymin=float(bbox.find("ymin").text)-y1
            new_xmax=float(bbox.find("xmax").text)-x1
            new_ymax=float(bbox.find("ymax").text)-y1
            object1 = ET.SubElement(root, "object")
            code = ET.SubElement(object1, "code")
            code.text = obj.find("code").text
            Serial = ET.SubElement(object1, "Serial")
            Serial.text = obj.find("Serial").text
            name=ET.SubElement(object1,"name")
            name.text=root_old.find("object").find("name").text
            pose = ET.SubElement(object1, "pose")
            pose.text = obj.find("pose").text
            truncated = ET.SubElement(object1, "truncated")
            truncated.text = obj.find("truncated").text
            difficult = ET.SubElement(object1, "difficult")
            difficult.text = obj.find("difficult").text
            Component = ET.SubElement(object1, "Component")
            Component.text = obj.find("Component").text
            Part = ET.SubElement(object1, "Part")
            Part.text = obj.find("Part").text
            DeDescription = ET.SubElement(object1, "DeDescription")
            DeDescription.text = obj.find("DeDescription").text
            DefectLevel = ET.SubElement(object1, "DefectLevel")
            DefectLevel.text = obj.find("DefectLevel").text
            Type = ET.SubElement(object1, "Type")
            Type.text = obj.find("Type").text
            Direction = ET.SubElement(object1, "Direction")
            Direction.text = obj.find("Direction").text
            bndbox=ET.SubElement(object1,"bndbox")
            xmin = ET.SubElement(bndbox, "xmin")
            ymin = ET.SubElement(bndbox, "ymin")
            xmax = ET.SubElement(bndbox, "xmax")
            ymax = ET.SubElement(bndbox, "ymax")

            xmin.text=str(new_xmin)
            ymin.text = str(new_ymin)
            xmax.text = str(new_xmax)
            ymax.text = str(new_ymax)

    new_tree=ET.ElementTree(root)
    new_tree.write(out_path,encoding="utf-8",xml_declaration=True)


if __name__ == '__main__':
    get_enhance(320)