import cv2
import numpy as np
import math

vidcap = cv2.VideoCapture('movie.mov')
fps = vidcap.get(cv2.CAP_PROP_FPS)
width = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
input_image = []
while True:
    success, image = vidcap.read()
    if not success:
        break
    input_image.append(image)


init_image = input_image[0].copy()
init_image = np.asarray(init_image)
init_image = cv2.cvtColor(init_image, cv2.COLOR_BGR2GRAY)
init_image = 255 - init_image
init_image = init_image/255

org_dim = init_image.shape

ref_image = cv2.imread('reference.jpg', 0)
ref_image = np.asarray(ref_image)
ref_image = 255 - ref_image
ref_image = ref_image/255
ref_dim = ref_image.shape
max = -100000
x = -1
y = -1
for i in range(0, org_dim[0] - ref_dim[0] + 1):
    for j in range(0, org_dim[1] - ref_dim[1] + 1):
        sub_img = init_image[i:i+ref_dim[0], j:j+ref_dim[1]]
        sub_img = np.asarray(sub_img)
        val = np.correlate(sub_img.flatten(), ref_image.flatten())
        if val > max:
            max = val
            x = i
            y = j

p_x = x
p_y = y


def exhaustive_search(p):
    print("Exhaustive Search")
    c_x = p_x
    c_y = p_y
    output_image = []
    c = 0

    for k in range(len(input_image)):
        max = -100000
        x = -1
        y = -1
        init_image = input_image[k].copy()
        init_image = np.asarray(init_image)
        init_image = cv2.cvtColor(init_image, cv2.COLOR_BGR2GRAY)
        init_image = 255 - init_image
        init_image = init_image / 255
        for i in range(int(c_x - p), int(c_x + p)):
            for j in range(int(c_y - p), int(c_y + p)):
                c += 1
                sub_img = init_image[i:i + ref_dim[0], j:j + ref_dim[1]]
                sub_img = np.asarray(sub_img)
                val = np.correlate(sub_img.flatten(), ref_image.flatten())
                if val > max:
                    max = val
                    x = i
                    y = j
        c_x = x
        c_y = y
        final_img = cv2.rectangle(input_image[k], (y, x), (y + ref_dim[1], x + ref_dim[0]), (0, 0, 255), 2)
        output_image.append(final_img)
    # for i in output_image:
    #     cv2.imshow("Boxed image", i)
    #     while (True):
    #         if cv2.waitKey(1) & 0xFF == ord('q'):
    #             break

    c = c / len(input_image)
    print(c)

    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    opVid = cv2.VideoWriter("exhaustive_search_result"+str(p)+".mp4", fourcc, fps, (width, height))
    for i in output_image:
        opVid.write(i)
    opVid.release()


def logarithmic_search(p):
    print("Log Search")
    c_x = p_x + ref_dim[0]/2
    c_y = p_y + ref_dim[1]/2
    output_image = []
    itr = int(math.ceil(math.log(p, 2)))
    c = 0
    for k in range(len(input_image)):

        init_image = input_image[k].copy()
        init_image = np.asarray(init_image)
        init_image = cv2.cvtColor(init_image, cv2.COLOR_BGR2GRAY)
        init_image = 255 - init_image
        init_image = init_image / 255
        for a in range(itr, 0, -1):
            d = pow(2, a - 1)
            max = -100000
            x = -1
            y = -1
            for i in range(int(c_x - d), int(c_x + d) + 1, d):
                for j in range(int(c_y - d), int(c_y + d) + 1, d):
                    c += 1
                    tempX = int(i - ref_dim[0]/2)
                    tempY = int(j - ref_dim[1]/2)
                    # print(i, j)
                    # print(tempX, tempY)
                    sub_img = init_image[tempX:tempX + ref_dim[0], tempY:tempY + ref_dim[1]]
                    sub_img = np.asarray(sub_img)
                    # print(sub_img.shape)
                    # print(ref_image.shape)
                    # cv2.imshow("Sub", sub_img)
                    # cv2.imshow("Ref", ref_image)
                    # cv2.imshow("Original", input_image[k])
                    # cv2.imshow("Curr", init_image)
                    # while (True):
                    #     if cv2.waitKey(1) & 0xFF == ord('q'):
                    #         break
                    val = np.correlate(sub_img.flatten(), ref_image.flatten())
                    if val > max:
                        max = val
                        x = i
                        y = j
            c_x = x
            c_y = y
        final_img = cv2.rectangle(input_image[k], (int(y - ref_dim[1]/2), int(x - ref_dim[0]/2)), (int(y - ref_dim[1]/2) + ref_dim[1], int(x - ref_dim[0]/2) + ref_dim[0]), (0, 0, 255), 2)
        output_image.append(final_img)

    # for i in output_image:
    #     cv2.imshow("Boxed image", i)
    #     while (True):
    #         if cv2.waitKey(1) & 0xFF == ord('q'):
    #             break
    c = c / len(input_image)
    print(c)
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    opVid = cv2.VideoWriter("Log_search_result" + str(p) +".mp4", fourcc, fps, (width, height))
    for i in output_image:
        opVid.write(i)
    opVid.release()


def hierarchical_search(p):
    c = 0
    print("Hierarchical Search")
    c_x = p_x + ref_dim[0] / 2
    c_y = p_y + ref_dim[1] / 2
    output_image = []
    for k in range(len(input_image)):
        init_image = input_image[k].copy()
        init_image = np.asarray(init_image)
        init_image = cv2.cvtColor(init_image, cv2.COLOR_BGR2GRAY)
        init_image = 255 - init_image
        init_image = init_image / 255

        h_org = []
        h_ref = []

        h_org.append(init_image)
        h_ref.append(ref_image)
        l_image = np.asarray(init_image).copy()
        l_ref = np.asarray(ref_image).copy()
        for i in range(2):
            blur_image = cv2.blur(l_image, (5, 5))
            small_image = cv2.resize(blur_image, (0, 0), fx=0.5, fy=0.5)
            blur_ref = cv2.blur(l_ref, (5, 5))
            small_ref = cv2.resize(blur_ref, (0, 0), fx=0.5, fy=0.5)
            h_org.append(small_image)
            h_ref.append(small_ref)
            l_image = np.asarray(small_image).copy()
            l_ref = np.asarray(small_ref).copy()

        l2_image = np.asarray(h_org[2])
        l2_ref = np.asarray(h_ref[2])
        max = -100000
        x = -1
        y = -1
        for i in range(int(c_x/4 - p/4), int(c_x/4 + p / 4 + 1)):
            for j in range(int(c_y / 4 - p / 4), int(c_y / 4 + p / 4 + 1)):
                c += 1
                tempX = int(i - l2_ref.shape[0] / 2)
                tempY = int(j - l2_ref.shape[1] / 2)
                # print(i, j)
                # print(tempX, tempY)
                sub_img = l2_image[tempX:tempX + l2_ref.shape[0], tempY:tempY + l2_ref.shape[1]]
                sub_img = np.asarray(sub_img)
                # print(sub_img.shape)
                # print(ref_image.shape)
                # cv2.imshow("Sub", sub_img)
                # cv2.imshow("Ref", ref_image)
                # cv2.imshow("Original", input_image[k])
                # cv2.imshow("Curr", init_image)
                # while (True):
                #     if cv2.waitKey(1) & 0xFF == ord('q'):
                #         break
                val = np.correlate(sub_img.flatten(), l2_ref.flatten())
                if val > max:
                    max = val
                    x = i
                    y = j
        c_x_n = c_x/2 + 2*(x-c_x/4)
        c_y_n = c_y/2 + 2*(y-c_y/4)
        l1_image = np.asarray(h_org[1])
        l1_ref = np.asarray(h_ref[1])
        max = -100000
        x = -1
        y = -1
        for i in range(int(c_x_n - 1), int(c_x_n + 2)):
            for j in range(int(c_y_n - 1), int(c_y_n + 2)):
                c += 1
                tempX = int(i - l1_ref.shape[0] / 2)
                tempY = int(j - l1_ref.shape[1] / 2)
                sub_img = l1_image[tempX:tempX + l1_ref.shape[0], tempY:tempY + l1_ref.shape[1]]
                sub_img = np.asarray(sub_img)
                # cv2.imshow("Sub", sub_img)
                # cv2.imshow("Ref", ref_image)
                # cv2.imshow("Original", input_image[k])
                # cv2.imshow("Curr", init_image)
                # while (True):
                #     if cv2.waitKey(1) & 0xFF == ord('q'):
                #         break
                val = np.correlate(sub_img.flatten(), l1_ref.flatten())
                if val > max:
                    max = val
                    x = i
                    y = j
        c_x_n = c_x + 2*(x-c_x/2)
        c_y_n = c_y + 2*(y-c_y/2)
        l_image = np.asarray(h_org[0])
        l_ref = np.asarray(h_ref[0])
        max = -100000
        x = -1
        y = -1
        for i in range(int(c_x_n - 1), int(c_x_n + 2)):
            for j in range(int(c_y_n - 1), int(c_y_n + 2)):
                c += 1
                tempX = int(i - l_ref.shape[0] / 2)
                tempY = int(j - l_ref.shape[1] / 2)
                # print(i, j)
                # print(tempX, tempY)
                sub_img = l_image[tempX:tempX + l_ref.shape[0], tempY:tempY + l_ref.shape[1]]
                sub_img = np.asarray(sub_img)
                # print(sub_img.shape)
                # print(ref_image.shape)
                # cv2.imshow("Sub", sub_img)
                # cv2.imshow("Ref", ref_image)
                # cv2.imshow("Original", input_image[k])
                # cv2.imshow("Curr", init_image)
                # while (True):
                #     if cv2.waitKey(1) & 0xFF == ord('q'):
                #         break
                val = np.correlate(sub_img.flatten(), l_ref.flatten())
                if val > max:
                    max = val
                    x = i
                    y = j
        c_x = x
        c_y = y
        final_img = cv2.rectangle(input_image[k], (int(y - ref_dim[1] / 2), int(x - ref_dim[0] / 2)),
                                  (int(y - ref_dim[1] / 2) + ref_dim[1], int(x - ref_dim[0] / 2) + ref_dim[0]),
                                  (0, 0, 255), 2)

        output_image.append(final_img)
    # for i in output_image:
    #     cv2.imshow("Boxed image", i)
    #     while (True):
    #         if cv2.waitKey(1) & 0xFF == ord('q'):
    #             break
    c = c/len(input_image)
    print(c)

    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    opVid = cv2.VideoWriter("Hierarchical_search_result" + str(p) +".mp4", fourcc, fps, (width, height))
    for i in output_image:
        opVid.write(i)
    opVid.release()


for p in range(8, 9):
    print(p)
    exhaustive_search(p)
    logarithmic_search(p)
    hierarchical_search(p)
