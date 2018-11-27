import numpy as np
import cv2 as cv
import os
import matplotlib.pyplot as plt
IMAGE_WIDTH = 352
IMAGE_HEIGHT = 1216

def plot():
    plt.figure(1)
    plt.ylabel("IoU")
    plt.plot(np.loadtxt("./IOU.txt"))

    plt.figure(2)
    plt.ylabel("loss")
    plt.plot(np.loadtxt("./loss.txt"))

    plt.figure(3)
    plt.ylabel("valid_set_loss")
    plt.plot(np.loadtxt("./valid_loss.txt"))

    plt.figure(4)
    plt.ylabel("valid_set_avg_IoU")
    plt.plot(np.loadtxt("./valid_IOU.txt"))

    plt.show()



def IoU(logits, reality):
    TP = 0
    FP = 0
    FN = 0
    pred = logits
    pred[pred >= 0.5] = 1
    pred[pred < 0.5] = 0
    pred = np.reshape(pred.astype(int), [-1])
    reality = np.reshape(reality.astype(int), [-1])
    for i in range(len(pred)):
        if pred[i] == 1 and reality[i] == 1:
            TP += 1
        elif pred[i] != 1 and reality[i] == 1:
            FN += 1
        elif pred[i] == 1 and reality[i] != 1:
            FP += 1
    # print(str(pred[i]) + ":" + str(reality[i]))
    print(str(TP) + ":" + str(FP) + ":" + str(FN))

    return TP * 1.0 / (TP + FN + FP)


def generate_test_IoU():
    iou = []
    l = os.listdir(os.path.join(os.path.dirname(__file__), "./data/image/test/"))
    l.sort()
    for i in range(45):
        logits = np.loadtxt("./pred_label/pred_{0}.txt".format(str(i)))
        result = np.loadtxt("./pred_label/result_label_{0}.txt".format(str(i)))
        iou.append(l[i])
        iou.append(IoU(logits, result))
    iou = np.reshape(np.asarray(iou),(45, 2))
    print(iou)

    np.savetxt('./test_iou.txt', iou)



# np.savetxt('./test_iou'.format(i), iou, fmt='%.3f')

def generate_final_result():
    l = os.listdir(os.path.join(os.path.dirname(__file__), "./data/image/test/"))
    l.sort()
    print(l)
    for i in range(45):
        logits = np.loadtxt("./pred_label/pred_{0}.txt".format(str(i)))
        result = np.loadtxt("./pred_label/result_label_{0}.txt".format(str(i)))
        pred = logits
        pred[pred >= 0.5] = 1
        pred[pred < 0.5] = 0
        pred = np.reshape(pred.astype(int), [-1])
        result = np.reshape(result.astype(int), [-1])
        cur_image = []
        for j in range(pred.shape[0]):
            if result[j] == -1:
                cur_image.append([0, 0, 0])
            elif pred[j] == 1:
                cur_image.append([255, 0, 255])
            elif pred[j] == 0:
                cur_image.append([0, 0, 255])
        cur_image = np.asarray(cur_image)
        cur_image = np.reshape(cur_image, (IMAGE_WIDTH, IMAGE_HEIGHT, 3))
        cv.imwrite('./result/pred_{0}.jpg'.format(l[i]), cur_image)


def main():
    generate_test_IoU()
    generate_final_result()
    plot()



if __name__ == '__main__':
    main()
