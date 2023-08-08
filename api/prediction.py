import pickle
import cv2, os

out_path = "static/results"

with open("model.pickle", 'rb') as f:
    model = pickle.load(f)


with open("classes.txt", 'r') as f:
    classes = f.read().split()


def predict(img_path):
    image = cv2.imread(img_path)
    results = model(image)
    for res in results:
        label = res[0]
        x1, y1, x2, y2 = res[1:]
        cv2.rectangle(image, (x1, y1), (x2, y2), "red", 2)
        cv2.putText(image, label, (x1-10, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 2)

    cv2.imwrite(img_path.split('/')[-1], os.path.join(out_path,img_path.split('/')[-1]))