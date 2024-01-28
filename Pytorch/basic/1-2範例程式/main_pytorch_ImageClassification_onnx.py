import cv2
import time
import numpy as np
from PIL import Image
import onnxruntime as ort
import torchvision.transforms as trns

onnxmodel_path='mobilenetv2.onnx'
class_def = 'imagenet_classes.txt'

def softmax(x):
    x = x.reshape(-1)
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def postprocess(result):
    return softmax(np.array(result)).tolist()

def main():
    # Run the model on the backend
    session = ort.InferenceSession(onnxmodel_path, None)
    # get the name of the first input of the model
    input_name = session.get_inputs()[0].name

    # Load ImageNet classes
    with open(class_def) as f:
        classes = [line.strip() for line in f.readlines()]

    # Define image transforms
    transforms = trns.Compose([trns.Resize((224, 224)),
                               trns.ToTensor(), 
                               trns.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    cap = cv2.VideoCapture(0)

    while True:
        ret, img = cap.read()
        if not ret:
            break
        # Read image and run prepro
        image = Image.fromarray(img)#.convert("RGB")
        image_tensor = transforms(image)
        image_tensor = image_tensor.unsqueeze(0)
        image_np = image_tensor.numpy()

        # model run
        outputs = session.run([], {input_name: image_np})[0] 
        print("Output size:{}".format(outputs.shape))

        # Result postprocessing
        idx = np.argmax(outputs)
        sort_idx = np.flip(np.squeeze(np.argsort(outputs)))
        idx = np.argmax(outputs)

        # outputs = np.sort(outputs[0,:])
        probs = postprocess(outputs)

        top_k=3
        cv2.putText(img, "Inference results:", (0, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
        print("Inference results:")
        for i, index in enumerate(sort_idx[:top_k]):
            py = 35 + 50*i
            text = "Label {}: {} ({:5f}) \n".format(index, classes[index],probs[index])
            cv2.putText(img, text, (0, py), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1, cv2.LINE_AA)
            print(text)

        cv2.imshow('demo', img)
        cv2.waitKey(1)

    cap.release()

if __name__ == '__main__':
    main()