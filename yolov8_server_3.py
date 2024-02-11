from ultralytics import YOLO
import torch

from Moution_detect import Moution_detect

from tcp_sender import TCP_server
from glib import *
import copy
import multiprocessing as mp
from play_sound_track import Sound_track




def convert_to_global_cord_and_filter_bbox(cord_detect_nn, crop_cord, cord_track, p_x=30, p_y=30):
    lst_global, lst_filter = [], []
    for i, k in enumerate(cord_detect_nn):
        x1_global, y1_global, x2_global, y2_global = int(k[2]+crop_cord[0]), int(k[3]+crop_cord[1]), int(k[4]+crop_cord[0]), int(k[5]+crop_cord[1])
        lst_global.append([int(k[0]), k[1], x1_global, y1_global, x2_global, y2_global])
    for i, k in enumerate(cord_track):
        x_cnt_track, y_cnt_track = int(k[0] + ((k[2] - k[0]) / 2)), int(k[1] + ((k[3] - k[1]) / 2))
        for j, p in enumerate(lst_global):
            x_cnt_nn, y_cnt_nn = int(p[2] + ((p[4] - p[2]) / 2)), int(p[3] + ((p[5] - p[3]) / 2))

            if (abs(x_cnt_track - x_cnt_nn) <= p_x) and (abs(y_cnt_track - y_cnt_nn) <= p_y):
                lst_filter.append(p)

    return lst_filter



def track_objects_from_stream(q_in, q_to_neural, q_to_client_moution):
    size_nn_w, size_nn_h = 288, 288
    rescale_w, rescale_h = 600, 600
    p_w, p_h = 0.51, 0.51
    while True:

        neuro_image, contour_boxes, frame_count, img_mout = q_in.get()
        print("countor_bboxes", contour_boxes, neuro_image.shape)

        if (q_to_neural.qsize() == 0):
            for i, k in enumerate(contour_boxes):
                Xcnt, Ycnt, W, H = int(k[0] + ((k[2] - k[0])/2)), int(k[1] + ((k[3] - k[1])/2)), k[2] - k[0], k[3] - k[1]
                print("per_w_h", W/size_nn_w, H/size_nn_h, W/size_nn_w < p_w, H/size_nn_h < p_h, k)
                if (W/size_nn_w < p_w) and (H/size_nn_h < p_h):
                    obl_detection = cover_pt_by_area((Xcnt, Ycnt), area_w_h=[size_nn_w, size_nn_h], limit_box=[0, 0, 1920, 1080])  # Получаем координаты нарезаемой области
                    print("OBJ_DET_288x288", obl_detection)
                    print("shapes_288x288", neuro_image[obl_detection[1]:obl_detection[3], obl_detection[0]:obl_detection[2]].shape)
                else:
                    obl_detection = cover_pt_by_area((Xcnt, Ycnt), area_w_h=[rescale_w, rescale_h], limit_box=[0, 0, 1920, 1080])  # Получаем координаты нарезаемой области
                    print("OBJ_DET_600x600", obl_detection)
                    print("shapes_600x600", neuro_image[obl_detection[1]:obl_detection[3], obl_detection[0]:obl_detection[2]].shape)
                q_to_neural.put((neuro_image[obl_detection[1]:obl_detection[3], obl_detection[0]:obl_detection[2]], [obl_detection[0], obl_detection[1], obl_detection[2], obl_detection[3]], [frame_count, len([[obl_detection[0], obl_detection[1],obl_detection[2], obl_detection[3]]]), 1], contour_boxes, neuro_image))

        if q_to_client_moution.empty():
            q_to_client_moution.put(img_mout)

def start_tracking(get_tracker, q_to_neuro:mp.Queue, q_to_client_moution:mp.Queue):

    pr_tracking = mp.Process(target=track_objects_from_stream,args=(get_tracker, q_to_neuro, q_to_client_moution))
    pr_tracking.start()

def collect_n_cast_neuro(q_dets:mp.Queue, q_to_client_yolo:mp.Queue, q_warning:mp.Queue):
    model = YOLO("./runs/detect/train10/weights/best.pt")
    while True:
        arg = q_dets.get()
        images, cords, idx, track_cord, nn_image = arg[0], arg[1], arg[2], arg[3], arg[4]
        results = model(images, conf=0.6)
        #print("Result_DETECTION", results[0].boxes.xyxy.cpu().tolist(), results[0].boxes.cls.cpu().tolist(), results[0].boxes.conf.cpu().tolist())
        cls_list = []

        lst_det = [[cls, conf, k[0], k[1], k[2], k[3]] for cls, conf, k in zip(results[0].boxes.cls.cpu().tolist(), results[0].boxes.conf.cpu().tolist(), results[0].boxes.xyxy.cpu().tolist())]


        for result in convert_to_global_cord_and_filter_bbox(lst_det, cords, track_cord):

            cls_list.append(result[0])
            if result[0] == 0:
                cv.rectangle(nn_image, (result[2], result[3]), (result[4], result[5]), (0, 0, 255), 1)
                cv.putText(nn_image, f"drone[{round(result[1], 2)}]", (result[2] - 15, result[3] - 15), 3, 1.3,
                           (0, 0, 255), 2)
            if result[0] == 1:
                cv.rectangle(nn_image, (result[2], result[3]), (result[4], result[5]), (0, 255, 0), 1)
                cv.putText(nn_image, f"bird[{round(result[1], 2)}]", (result[2] - 15, result[3] - 15), 3, 1.3,
                           (0, 255, 0), 2)
            if result[0] == 2:
                cv.rectangle(nn_image, (result[2], result[3]), (result[4], result[5]), (0, 255, 255), 1)
                cv.putText(nn_image, f"plane[{round(result[1], 2)}]", (result[2] - 15, result[3] - 15), 3, 1.3,
                           (0, 255, 255), 2)


            if q_to_client_yolo.empty():
                q_to_client_yolo.put(nn_image)
        if (0 in cls_list or 2 in cls_list) and q_warning.empty():
            q_warning.put(True)


        for i, k in enumerate(track_cord):
            cv.circle(nn_image, (int(k[0] + ((k[2] - k[0])/2)), int(k[1] + ((k[3] - k[1])/2))), 170, (0, 0, 0), 8)

        cv.imshow("frame", nn_image)
        cv.waitKey(1)



if __name__ == '__main__':


    
    q_to_neuro = mp.Queue(50)
    # Выходная очередь нейросети, из которой забираются все обнаружения
    q_from_neuro = mp.Queue(8)

    q_to_tracker = mp.Queue(1)
    get_tracker = mp.Queue(1)
    q_from_tracker = None

    q_true_original_image, q_moution_detect_image, q_detect_yolo = mp.Queue(1), mp.Queue(1), mp.Queue(1)
    q_warning = mp.Queue(1)
    #Инициализация и запуск обработчиков нейросети:

    neuro_data_collector = mp.Process(target=collect_n_cast_neuro, args=(q_to_neuro, q_detect_yolo, q_warning))
    neuro_data_collector.start()



    cam = cv.VideoCapture("name_videofile_or_vebcam_or_ip_camera") 


    Moution_detect(q_to_tracker, get_tracker)
    start_tracking(get_tracker, q_to_neuro, q_moution_detect_image)
    Sound_track(q_warning, path_to_sound_file="ww.wav").run()
    tcp_serv_orig, tcp_serv_moution, tcp_setv_yolo = TCP_server(q_true_original_image, ip_adress="local", port=6005, recv_bytes=4096).run(), TCP_server(q_moution_detect_image, ip_adress="local", port=5005, recv_bytes=4096).run(), TCP_server(q_detect_yolo, ip_adress="local", port=9000, recv_bytes=4096).run()
    frame_count = 0

    while cam.isOpened():
        frame_valid, image = cam.read()
        print(image.shape)
        if frame_valid:
            frame_count += 1
            if q_to_tracker.empty():
                q_to_tracker.put((copy.deepcopy(image), frame_count))

            if q_true_original_image.empty():
                q_true_original_image.put(image)




            cv.imshow('original', cv.resize(image, (600, 600)))
            cv.waitKey(1)
