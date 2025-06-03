import cv2
import numpy as np
import test_digit

img_size = 28  # 28*28是mnist的图片训练集尺寸
kernel_connect = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], np.uint8)  # 膨胀化用的参数
ans = []  # 保存图片数组



def show(name):
    cv2.imshow("123", name)
    cv2.waitKey(0)


def split_digits(s, prefix_name):
    s = np.rot90(s)  # 使图片逆时针旋转90°
    # show(s)
    s_copy = cv2.dilate(s, kernel_connect, iterations=1)
    s_copy2 = s_copy.copy()
    contours, hierarchy = cv2.findContours(s_copy2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  # 该函数可以检测出图片中物品的轮廓
    # contours：list结构，列表中每个元素代表一个边沿信息。每个元素是(x, 1, 2)的三维向量，x表示该条边沿里共有多少个像素点，第三维的那个“2”表示每个点的横、纵坐标；
    # hierarchy：返回类型是(x, 4)的二维ndarray。x和contours里的x是一样的意思。如果输入选择cv2.RETR_TREE，则以树形结构组织输出，hierarchy的四列分别对应下一个轮廓编号、上一个轮廓编号、父轮廓编号、子轮廓编号，该值为负数表示没有对应项。

    # for it in contours:
    #     print(it)
    # print("##########################")

    idx = 0
    for contour in contours:
        idx = idx + 1
        [x, y, w, h] = cv2.boundingRect(contour)  # 当得到对象轮廓后，可用boundingRect()得到包覆此轮廓的最小正矩形，
        # show(cv2.boundingRect(contour))
        digit = s_copy[y:y + h, x:x + w]
        # show(digit)
        pad_len = (h - w) // 2
        # print(pad_len)
        if pad_len > 0:
            digit = cv2.copyMakeBorder(digit, 0, 0, pad_len, pad_len, cv2.BORDER_CONSTANT,value=0)
        elif pad_len < 0:
            digit = cv2.copyMakeBorder(digit, -pad_len, -pad_len, 0, 0, cv2.BORDER_CONSTANT, value=0)

        pad = digit.shape[0] // 4  # 避免数字与边框直接相连，留出4个像素左右。
        digit = cv2.copyMakeBorder(digit, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
        digit = cv2.resize(digit, (img_size, img_size), interpolation=cv2.INTER_AREA)  # 把图片缩放至28*28
        digit = np.rot90(digit, 3)  # 逆时针旋转270°将原本图片旋转为原来的水平方向
        # show(digit)
        cv2.imwrite(prefix_name + str(idx) + '.jpg', digit)
        ans.append(digit)
    test_digit.dj(ans)


if __name__ == '__main__':
    img = cv2.imread('/Users/lit./Desktop/iibproject/WechatIMG3116.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
    #show(thresh_img)
    split_digits(thresh_img, "split_img/split_img")

