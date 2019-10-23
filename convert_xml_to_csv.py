import pandas as pd
import os
import glob
import xml.etree.ElementTree as et
import argparse


ap = argparse.ArgumentParser(description= "Covnert XML files to CSV")
ap.add_argument("--test_input",help="path to test XML files")
ap.add_argument("--train_input",help="path to train XML files")
args =ap.parse_args()
feature = []
for xml in glob.glob(os.path.join(os.getcwd(),args.test_input) + '/*.xml'):
    tree = et.parse(xml)
    root_node = tree.getroot()

    for object_node in root_node.findall("object"):
        file_name = root_node.find('filename').text
        width =  int(root_node.find('size')[0].text)
        height = int(root_node.find('size')[1].text)
        label_class =  object_node[0].text
        xmin = int(object_node[4][0].text)
        ymin = int(object_node[4][1].text)
        xmax = int(object_node[4][2].text)
        ymax =  int(object_node[4][3].text)
        feature.append([file_name, width, height, label_class, xmin, ymin, xmax, ymax])
    xml_list = pd.DataFrame(feature, columns=["filename", "width", "height", "class", "xmin", "ymin", "xmax", "ymax"])
    xml_list.to_csv("data/test_labels.csv", index = None)

feature = []
for xml in glob.glob(os.path.join(os.getcwd(),args.train_input) + "/*.xml"):
    tree = et.parse(xml)
    root_node = tree.getroot()

    for object_node in root_node.findall("object"):
        file_name = root_node.find('filename').text
        width =  int(root_node.find('size')[0].text)
        height = int(root_node.find('size')[1].text)
        label_class =  object_node[0].text
        xmin = int(object_node[4][0].text)
        ymin = int(object_node[4][1].text)
        xmax = int(object_node[4][2].text)
        ymax =  int(object_node[4][3].text)
        feature.append([file_name, width, height, label_class, xmin, ymin, xmax, ymax])
    xml_list = pd.DataFrame(feature, columns=["filename", "width", "height", "class", "xmin", "ymin", "xmax", "ymax"])
    xml_list.to_csv("data/train_labels.csv", index = None)


print("Finish convertering XML files to CSV....")
