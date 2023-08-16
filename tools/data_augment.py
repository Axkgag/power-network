import random
import os
import cv2
import numpy as np
import shutil
import xml.etree.ElementTree as ET
code_list = ["010101071", "010201041", "010202011", "010401041"]


def aug1(img):

    brightness_factor=np.random.uniform(0.5,2.5)
    contrast_factor=np.random.uniform(0.5,2.5)
    saturation_factor=np.random.uniform(0.5,2.5)
    hue_factor=np.random.uniform(-0.1,0.1)


    image=cv2.cvtColor(img,cv2.COLOR_RGB2HSV)
    image=image.astype(np.float32)
    image[:,:,2]=image[:,:,2]*brightness_factor
    image[:, :, 1] = image[:, :, 1] * contrast_factor
    image[:, :, 0] = (image[:, :, 0]+hue_factor*180)%180
    image[:, :, 1] = np.clip(image[:, :, 1], 0, 255)
    image[:, :, 2] = np.clip(image[:, :, 2], 0, 255)
    image=image.astype(np.uint8)
    image=cv2.cvtColor(image,cv2.COLOR_HSV2RGB)
    image=image.astype(np.float32)
    image=image*saturation_factor
    image=np.clip(image,0,255)
    image=image.astype(np.uint8)
    return image


def aug2(img):

    height,width,channels=img.shape


    for i in range(3):

        size=150
        x1=random.randint(0,width-size)
        y1=random.randint(0,height-size)
        x2=x1+size
        y2=y1+size


        region=img[y1:y2, x1:x2]
        region_small=cv2.resize(region,(16,16), interpolation=cv2.INTER_NEAREST)


        region_large=cv2.resize(region_small,(x2-x1,y2-y1), interpolation=cv2.INTER_NEAREST)
        mosaic=cv2.resize(region_large,(x2-x1,y2-y1),interpolation=cv2.INTER_NEAREST)


        img[y1:y2, x1:x2]=mosaic

    return img


if __name__ == '__main__':
    image_path='../../datasets/train_collect/train/image'
    xml_path='../../datasets/train_collect/train/xml'
    p1=1
    p2=1
    filenames=os.listdir(image_path)
    filenames.sort()
    xmls=os.listdir(xml_path)
    xmls.sort()
    for id, filename in enumerate(filenames):
        print(id, ": ", filename)
        source_xml=os.path.join(xml_path,xmls[id])
        tree=ET.parse(source_xml)
        root=tree.getroot()
        objs=tree.findall("object")
        for obj in objs:
            code=obj.find("code").text
            if code in code_list:
                print(code)
                if random.random()<=p1:
                    img=cv2.imdecode(np.fromfile(os.path.join(image_path,filename),dtype=np.uint8),-1)
                    filename1, ext=os.path.splitext(filename)
                    img2=aug1(img)
                    filename2=filename1+"_a1"+ext
                    cv2.imencode(".jpg",img2)[1].tofile(os.path.join(image_path,filename2))
                    xml2,ext2=os.path.splitext(xmls[id])
                    target_xml=xml2+"_a1"+ext2
                    shutil.copyfile(os.path.join(xml_path,xmls[id]),os.path.join(xml_path,target_xml))
                    new_tree=ET.parse(os.path.join(xml_path,target_xml))
                    new_root=new_tree.getroot()
                    new_root.find("filename").text=filename2
                    new_root.find("fileurl").text=os.path.join(image_path,filename2)
                    new_tree.write(os.path.join(xml_path,target_xml),encoding="utf-8",xml_declaration=True)
                if random.random()<=p2:
                    img=cv2.imdecode(np.fromfile(os.path.join(image_path,filename),dtype=np.uint8),-1)
                    filename1, ext=os.path.splitext(filename)
                    img2=aug2(img)
                    filename2=filename1+"_a2"+ext
                    cv2.imencode(".jpg",img2)[1].tofile(os.path.join(image_path,filename2))
                    xml2,ext2=os.path.splitext(xmls[id])
                    target_xml=xml2+"_a2"+ext2
                    shutil.copyfile(os.path.join(xml_path,xmls[id]),os.path.join(xml_path,target_xml))
                    new_tree=ET.parse(os.path.join(xml_path,target_xml))
                    new_root=new_tree.getroot()
                    new_root.find("filename").text=filename2
                    new_root.find("fileurl").text=os.path.join(image_path,filename2)
                    new_tree.write(os.path.join(xml_path,target_xml),encoding="utf-8",xml_declaration=True)
                break

