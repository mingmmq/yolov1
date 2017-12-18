from bs4 import BeautifulSoup

def load_annotation(anno_filename):
    with open(anno_filename) as f:
        xml = f.readlines()
    xml = ''.join([line.strip('\t') for line in xml])
    return BeautifulSoup(xml, "lxml")

def draw_detection(path, image_name, data, grid_rows = 7):
    from PIL import ImageFont, Image, ImageDraw
    anno_path =  "/".join([path, "Annotations", image_name.replace(".jpg","")+".xml"])
    anno = load_annotation(anno_path)
    size = anno.find('size')
    image_path = "/".join([path, "JPEGImages", image_name])
    from PIL import Image, ImageDraw
    im = Image.open(image_path)
    draw = ImageDraw.Draw(im)

    width = float(size.findChild('width').contents[0])
    height = float(size.findChild('height').contents[0])


    h_gap = width/grid_rows
    v_gap = height/grid_rows
    for i in range(1,grid_rows):
        draw.line((i*h_gap,0, i*h_gap,height), fill=128)
        draw.line((0,i*v_gap, width, i*v_gap), fill=128)

    font = ImageFont.truetype("simsun.ttf", 15)
    for i in range(grid_rows):
        for j in range(grid_rows):
            draw.text((i*h_gap + 2,j * v_gap +2), str(j*grid_rows + i), font=font)


    objs = anno.findAll('object')
    for obj in objs:

        bbox = obj.findChildren('bndbox')[0]
        xmin = int(bbox.findChildren('xmin')[0].contents[0])
        ymin = int(bbox.findChildren('ymin')[0].contents[0])
        xmax = int(bbox.findChildren('xmax')[0].contents[0])
        ymax = int(bbox.findChildren('ymax')[0].contents[0])
        xcen = (xmin + xmax) / 2.0
        ycen = (ymin + ymax) / 2.0
        draw.ellipse((xcen-4, ycen-4, xcen +4,ycen +4), fill = 'blue', outline ='blue')
        draw.rectangle([xmin, ymin, xmax, ymax], outline="blue")

    draw.rectangle([data[2], data[3], data[4], data[5]], outline="orange")

    im.show()



def draw_image(path, image_name, grid_rows):
    from PIL import ImageFont, Image, ImageDraw
    anno_path =  "/".join([path, "Annotations", image_name.replace(".jpg","")+".xml"])
    anno = load_annotation(anno_path)
    size = anno.find('size')
    width = float(size.findChild('width').contents[0])
    height = float(size.findChild('height').contents[0])

    image_path = "/".join([path, "JPEGImages", image_name])
    from PIL import Image, ImageDraw
    im = Image.open(image_path)
    draw = ImageDraw.Draw(im)
    h_gap = width/grid_rows
    v_gap = height/grid_rows
    for i in range(1,grid_rows):
        draw.line((i*h_gap,0, i*h_gap,height), fill=128)
        draw.line((0,i*v_gap, width, i*v_gap), fill=128)

    font = ImageFont.truetype("simsun.ttf", 15)
    for i in range(grid_rows):
        for j in range(grid_rows):
            draw.text((i*h_gap + 2,j * v_gap +2), str(j*grid_rows + i), font=font)


    objs = anno.findAll('object')
    for obj in objs:

        bbox = obj.findChildren('bndbox')[0]
        xmin = int(bbox.findChildren('xmin')[0].contents[0])
        ymin = int(bbox.findChildren('ymin')[0].contents[0])
        xmax = int(bbox.findChildren('xmax')[0].contents[0])
        ymax = int(bbox.findChildren('ymax')[0].contents[0])
        xcen = (xmin + xmax) / 2.0
        ycen = (ymin + ymax) / 2.0
        draw.ellipse((xcen-4, ycen-4, xcen +4,ycen +4), fill = 'blue', outline ='blue')

    im.show()

    return im

def draw_image_by_label(path, image_name, label_list, grid_rows):
    from PIL import ImageFont, Image, ImageDraw
    anno_path =  "/".join([path, "Annotations", image_name.replace(".jpg","")+".xml"])
    anno = load_annotation(anno_path)
    size = anno.find('size')
    width = float(size.findChild('width').contents[0])
    height = float(size.findChild('height').contents[0])

    image_path = "/".join([path, "JPEGImages", image_name])
    from PIL import Image, ImageDraw
    im = Image.open(image_path)
    draw = ImageDraw.Draw(im)

    #draw lines of the grids
    h_gap = width/grid_rows
    v_gap = height/grid_rows
    for i in range(1,grid_rows):
        draw.line((i*h_gap,0, i*h_gap,height), fill=128)
        draw.line((0,i*v_gap, width, i*v_gap), fill=128)

    #write number of each grid
    font = ImageFont.truetype("simsun.ttf", 15)
    for i in range(grid_rows):
        for j in range(grid_rows):
            draw.text((i*h_gap + 2,j * v_gap +2), str(j*grid_rows + i), font=font)

    for label in label_list:
        xcen = (label % (grid_rows*grid_rows) % grid_rows)*h_gap + h_gap/2.0
        ycen = (label % (grid_rows*grid_rows) // grid_rows)*v_gap + v_gap/2.0

        draw.ellipse((xcen-4, ycen-4, xcen +4,ycen +4), fill = 'blue', outline ='blue')

    im.show()

    return im

def fromCsv(csv_file):
    import pandas as pd
    df = pd.read_csv(csv_file)
    img_names = df['img']
    labels = []
    for tag in df['tags'].str.split().tolist():
        labels.append([int(i) for i in tag])

    return img_names, labels

if __name__ == '__main__':
    path =  "../data/pascal/VOCdevkit/VOC2007"
    csv_file = "test_grid_voc2007.csv"
    img_names, labels = fromCsv(csv_file)
    # [draw_image(path, image, 7) for image in img_names]
    # [draw_image_by_label(path, image, label_list, 7) for image, label_list in zip(img_names, labels)]

    for image, label in zip(img_names, labels):
        draw_image(path, image, 7)
        draw_image_by_label(path, image, label, 7)

