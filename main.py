import PIL.Image
from MTCNN import MTCNN
import cv2
from face_regconition_model import iresnet, base_transform
import torch
import torchvision.transforms.functional as F
from torch.nn.functional import cosine_similarity

# image = cv2.imread('images/image2.jpg')
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def compare_faces(image1_path, image2_path, detection_model, recognition_model):
    image1 = cv2.imread(image1_path)
    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)

    image2 = cv2.imread(image2_path)
    image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)

    bboxes1, probs1 = detection_model.detect(image1)
    x1, y1, x2, y2 = int(bboxes1[0, 0]), int(bboxes1[0, 1]), int(bboxes1[0, 2]), int(bboxes1[0, 3])
    crop1 = image1[y1:y2, x1:x2]
    crop1_show = crop1
    crop1 = PIL.Image.fromarray(crop1)

    bboxes2, probs2 = detection_model.detect(image2)
    x1, y1, x2, y2 = int(bboxes2[0, 0]), int(bboxes2[0, 1]), int(bboxes2[0, 2]), int(bboxes2[0, 3])
    crop2 = image2[y1:y2, x1:x2]
    crop2_show = crop2
    crop2 = PIL.Image.fromarray(crop2)
    TF = base_transform(img_size=112, mode='test')
    crop1 = TF(crop1)
    crop2 = TF(crop2)

    hf_crop1 = F.hflip(crop1)
    hf_crop2 = F.hflip(crop2)

    ft1 = recognition_model(crop1[None].to('cuda'))
    ft1 = ft1[0]
    hf_ft1 = recognition_model(hf_crop1[None].to('cuda'))
    hf_ft1 = hf_ft1[0]
    feature1 = torch.concat([ft1, hf_ft1], dim=0)

    ft2 = recognition_model(crop2[None].to('cuda'))
    ft2 = ft2[0]
    hf_ft2 = recognition_model(hf_crop2[None].to('cuda'))
    hf_ft2 = hf_ft2[0]
    feature2 = torch.concat([ft2, hf_ft2], dim=0)

    score = cosine_similarity(feature1, feature2, dim=0)
    return score, crop1_show, crop2_show




if __name__ == '__main__':

    face_detection_model = MTCNN()

    recognition_model = iresnet(100)
    recognition_model.to('cuda')
    recognition_model.eval()
    recognition_model.load_state_dict(torch.load('checkpoint/resnet100.pth'))
    score, crop1_show, crop2_show = compare_faces(image1_path='images/image1.jpg',
                                                  image2_path='images/imag4.jpeg',
                                                  detection_model= face_detection_model,
                                                  recognition_model=recognition_model)
    print(score)
    cv2.imshow('image1', crop1_show)
    cv2.imshow('image2', crop2_show)
    cv2.waitKey(-1)
# bboxes, probs = face_detection_model.detect(image)
# for i in range(bboxes.shape[0]):
#     if probs[i] >= 0.5:
#         x1, y1, x2, y2 = int(bboxes[i, 0]), int(bboxes[i, 1]), int(bboxes[i, 2]), int(bboxes[i, 3])
#         image = cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
#         image = cv2.putText(image, '{:.2}'.format(probs[i]), (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),1, cv2.LINE_AA)

# image1 = cv2.imread('images/image1.jpg')
# image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
#
#
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# cv2.imshow('image', image)
# cv2.waitKey(-1)