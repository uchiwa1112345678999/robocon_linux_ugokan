# -*- coding: utf-8 -*-

# そのうちフィルターもかけたい

import pyrealsense2 as rs
import numpy as np
import cv2
import serial
import re

def nothing(x):
    pass

exclude_depth=5.00

inWidth = 640
inHeight = 480

WHRatio = inWidth / float(inHeight)

cv2.namedWindow('show',cv2.WINDOW_NORMAL)

fps = 30
HFOV = 64
threshold=[0,180,0,256,0,256]

cv2.createTrackbar('H_min','show',id(threshold[0]),180,nothing)
cv2.createTrackbar('H_max','show',id(threshold[1]),180,nothing)
cv2.createTrackbar('S_min','show',id(threshold[2]),256,nothing)
cv2.createTrackbar('S_max','show',id(threshold[3]),256,nothing)
cv2.createTrackbar('V_min','show',id(threshold[4]),256,nothing)
cv2.createTrackbar('V_max','show',id(threshold[5]),256,nothing)

H_MAX = cv2.getTrackbarPos('H_max','show')
H_MIN = cv2.getTrackbarPos('H_min','show')

S_MAX = cv2.getTrackbarPos('S_max','show')
S_MIN = cv2.getTrackbarPos('S_min','show')

V_MAX = cv2.getTrackbarPos('V_max','show')
V_MIN = cv2.getTrackbarPos('V_min','show')




# ストリーミング初期化
config = rs.config()
config.enable_stream(rs.stream.color, inWidth, inHeight, rs.format.bgr8, fps)
config.enable_stream(rs.stream.depth, inWidth, inHeight, rs.format.z16, fps)


# ストリーミング開始
pipeline = rs.pipeline()
profile = pipeline.start(config)

# Skip 5 first frames to give the Auto-Exposure time to adjust
for x in range(5):
    pipeline.wait_for_frames()


#   距離[m] = depth * depth_scale 
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
clipping_distance_in_meters = exclude_depth # meter
clipping_distance = clipping_distance_in_meters / depth_scale


# Alignオブジェクト生成(位置合わせだった気がする)
align_to = rs.stream.color
align = rs.align(align_to)

try:
    while True:

        # Wait for a coherent pair of frames: depth and color
        frameset = pipeline.wait_for_frames()
        frameset = align.process(frameset)
        color_frame = frameset.get_color_frame()
        depth_frame = frameset.get_depth_frame()

        if not depth_frame or not color_frame:
            continue
        # imageをnumpy arrayに
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        # Depth画像前処理(1m以内を画像化)  参考本:https://mirai-tec.hatenablog.com/entry/2018/07/29/150902
        color = 0
        depth_image_3d = np.dstack((depth_image,depth_image,depth_image))
        color_image = np.where((depth_image_3d > clipping_distance) | (depth_image_3d <= 0), color, color_image)
        
        cv2.imshow("rmshow", color_image)
        # depth imageをカラーマップに変換
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(
            depth_image, alpha=0.08), cv2.COLORMAP_JET)

        #
        #
        #       マスクとかの処理
        #
        #
        one_channel_images = cv2.split(color_image)
        threshold_one_channel_images = []
        # H(Hue) : 色相
        ret, encoded = cv2.threshold(
            one_channel_images[0], H_MIN-1, 255, cv2.THRESH_BINARY)  # threshold以下なら0，それ以外なら255
        threshold_one_channel_images.append(encoded)
        ret, encoded = cv2.threshold(
            one_channel_images[0], H_MAX, 255, cv2.THRESH_BINARY_INV)  # threshold以上なら0，それ以外なら255
        threshold_one_channel_images.append(encoded)
        # S(Saturation) : 彩度
        ret, encoded = cv2.threshold(
            one_channel_images[1], S_MIN-1, 255, cv2.THRESH_BINARY)  # hreshold以下なら0，それ以外なら255
        threshold_one_channel_images.append(encoded)
        ret, encoded = cv2.threshold(
            one_channel_images[1], S_MAX, 255, cv2.THRESH_BINARY_INV)  # threshold以上なら0，それ以外なら255
        threshold_one_channel_images.append(encoded)
        # V(Value of brightness) : 明度
        ret, encoded = cv2.threshold(
            one_channel_images[2], V_MIN-1, 255, cv2.THRESH_BINARY)  # threshold以下なら0，それ以外なら255
        threshold_one_channel_images.append(encoded)
        ret, encoded = cv2.threshold(
            one_channel_images[2], V_MAX, 255, cv2.THRESH_BINARY_INV)  # threshold以上なら0，それ以外なら255
        threshold_one_channel_images.append(encoded)
        # 各チャンネルで次のbit演算をする
        # A,B|C
        # ======
        # 0,0|1
        # 0,1|0
        # 1,0|0
        # 1,1|1
        mask_images = []
        mask_images.append(cv2.bitwise_xor(
            threshold_one_channel_images[0], 255-threshold_one_channel_images[1]))
        mask_images.append(cv2.bitwise_xor(
            threshold_one_channel_images[2], 255-threshold_one_channel_images[3]))
        mask_images.append(cv2.bitwise_xor(
            threshold_one_channel_images[4], 255-threshold_one_channel_images[5]))

        # 各チャンネルのマスクで論理和を取る
        mask_image = cv2.bitwise_and(mask_images[0], mask_images[1])
        mask_image = cv2.bitwise_and(mask_image, mask_images[2])
        # 3チャンネル画像で，有効な画素には(255,255,255)で無効な画素には(0,0,0)が入っているような画像を作る
        mask_image = cv2.merge((mask_image, mask_image, mask_image))
        # 出力画像を生成
        dst_image = cv2.bitwise_and(color_image, mask_image)
        
        '''
        # 深度データ
        push_depth_mat = cv2.bitwise_and(depth_image, mask_image)

        #
        #
        #   深度リスト
        #
        #
        cunter = 0
        redist = 0
        width = push_depth_mat.shape[0]
        height = push_depth_mat.shape[1]
        channels = push_depth_mat.shape[2]
        for j in range(height):
            step = j*width
            for i in range(width):
                elm = i*push_depth_mat.bits
                for k in range(channels):
                    if push_depth_mat[step+elm+k] != 0:
                        cunter+=1
                        redist += push_depth_mat[step+elm+k]
        '''
        #
        #
        # 輪郭検出
        #
        #
        gray_dst=cv2.cvtColor(dst_image,cv2.COLOR_BGR2GRAY)
        _,img_na=cv2.threshold(gray_dst,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        count = -1
        contours_depth,_ = cv2.findContours(
            img_na, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            
        # print(len(contours_depth))
        if len(contours_depth) == 0:
            continue
        
        else:
            
            max_size_depth = contours_depth[0]# ここ、ものがないとエラー履く
            max_id_depth = -1
            for i in contours_depth:
                count += 1
                if len(max_size_depth) < len(i):
                    max_size_depth = i
                    max_id_depth += 1


        #
        #
        #  重心計算
        #
        #     
            on_depth = False
            mo_depth = None
            if max_id_depth != -1:
                mo_depth = cv2.moments(contours_depth[0])
                on_depth = True
                if mo_depth["m00"]==0:
                    mo_depth["m00"]=0.01
                    x, y = ((mo_depth["m10"]/mo_depth["m00"]),
                (mo_depth["m01"]/mo_depth["m00"]))
                #print(f'x,y:{x}, {y}')
                #print(contours_depth)
                x, y = (int(mo_depth["m10"]/mo_depth["m00"]),
                int(mo_depth["m01"]/mo_depth["m00"]))
                print(f'x,y:{x},{y}')
            else:
                print("None region")
            
            fl=int(x)
            fl.to_bytes(4,byteorder="little")
            ser =serial.Serial(
                port = '/dev/ttyACM0',
                baudrate = 9600,
            )
            ser.write(fl)
            #ser.close()
            

            cv2.imshow("WINDOW", color_image)
            cv2.imshow("rmshow", dst_image)

        
            
        
            if cv2.waitKey(1) & 0xff == 27:
                    break

finally:
    # ストリーミング停止
    pipeline.stop()
    cv2.destroyAllWindows()
