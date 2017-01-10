# -*- coding: utf-8 -*-

import cv2
import numpy as np
import numpy.linalg as LA

# 画像読み込み
img1_path = "C:/Users/m-harasawa/Desktop/002.jpg"
img2_path = "C:/Users/m-harasawa/Desktop/003.jpg"
img1 = cv2.imread(img1_path)
img2 = cv2.imread(img2_path)
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# 特徴抽出
detector = cv2.AKAZE_create()
kp1, des1 = detector.detectAndCompute(gray1, None)
kp2, des2 = detector.detectAndCompute(gray2, None)

# 特徴点のマッチング
bf = cv2.BFMatcher(cv2.NORM_HAMMING)
matches = bf.match(des1, des2)

# 特徴点の選択
dist = [m.distance for m in matches]
thres_dist = (sum(dist) / len(dist)) * 0.9
sel_matches = [m for m in matches if m.distance < thres_dist]
point1 = [[kp1[m.queryIdx].pt[0], kp1[m.queryIdx].pt[1]] for m in sel_matches]
point2 = [[kp2[m.trainIdx].pt[0], kp2[m.trainIdx].pt[1]] for m in sel_matches]
point1 = np.array(point1)
point2 = np.array(point2)

# ホモグラフィの計算
H, Hstatus = cv2.findHomography(point1,point2,cv2.RANSAC)

# ---stitiching---

def conv_uv(u, v, H): # ホモグラフィ変換
    ud, vd, nm = np.dot(H, np.array([u, v, 1]))
    ud, vd = ud/nm, vd/nm
    return ud, vd

def calc_w(u, v, w, h):
    return min([v, u, h-v, w-u])

def calc_w_h(w, h, H):
    # calculate img2 edges
    lt_u, lt_v = conv_uv(0, 0, H)
    rt_u, rt_v = conv_uv(w, 0, H)
    lb_u, lb_v = conv_uv(0, h, H)
    rb_u, rb_v = conv_uv(w, h, H)
    # select min u&v, max u&v
    min_v = min([lt_v, rt_v, 0])
    max_v = max([lb_v, rb_v, h])
    min_u = min([lt_u, rt_u, 0])
    max_u = max([lb_u, rb_u, w])
    # calc stitch image w&h
    is_w = int(round(max_u - min_u))
    is_h = int(round(max_v - min_v))
    return is_w, is_h, -int(min_v)

i1_h, i1_w = img1.shape[0], img1.shape[1]
i2_h, i2_w = img2.shape[0], img2.shape[1]

H = np.array(H)
iH = LA.inv(H) # 逆行列

# ホモグラフィ変換後のimg2の範囲を取得
is_w, is_h, offset = calc_w_h(i2_w, i2_h, iH)

# stitching imageのメモリ確保
simg = np.zeros((is_h, is_w, 3), np.uint8)

# img1を,v方向にオフセットして代入
for v1 in range(i1_h):
    for u1 in range(i1_w):
        simg[v1+offset, u1] = img1[v1, u1]

#v1のループ範囲をオフセット(v1にマイナス範囲を含めるため)
for v1 in range(-offset, is_h-offset):
    for u1 in range(is_w):
        i2_p = 0
        u2, v2 = conv_uv(u1, v1, H) #u1, v1をimg2画像の座標系で, u2, v2に変換
        # u2 が 0~i2_w 内に、v2が 0~i2_h 内にあるかチェック
        if (u2 > 0 and u2 < i2_w) and (v2 > 0 and v2 < i2_h):
            u2_, v2_ = float(int(u2)), float(int(v2))
            wu, wv = u2 - u2_, v2 - v2_
            p1 = img2[int(v2_), int(u2_)]
            if int(u2_) > i2_w-2 or int(v2_) > i2_h-2:
                #img2の範囲からはみ出した場合,補間なし
                i2_p = p1
            else:
                # バイリニア補間
                p2 = img2[int(v2_), int(u2_)+1]
                p3 = img2[int(v2_)+1, int(u2_)]
                p4 = img2[int(v2_)+1, int(u2_)+1]
                i2_p = (1-wu)*(1-wv)*p1+wu*(1-wv)*p2+(1-wu)*wv*p3+wu*wv*p4
            # ブレンディング
            if simg[v1+offset, u1].all(): # if overlap
                w1, w2 = float(calc_w(v1+offset, u1, i1_w, i1_h)), float(calc_w(v2, u2, i2_w, i2_h))
                if (w1 == 0 and w2 == 0) or w1*w2 < 0: # if edge
                    pass
                else:
                    wt1, wt2 = w1/(w1+w2), w2/(w1+w2)
                    p = wt1*simg[v1+offset, u1] + wt2*i2_p # blending
                    simg[v1+offset, u1] = p
            else:
                simg[v1+offset, u1] = i2_p
cv2.imwrite("C:/Users/m-harasawa/Desktop/stitch_image.jpg", simg)
cv2.imshow("image", simg)
cv2.waitKey(0)
cv2.destroyAllWindows()
