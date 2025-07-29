import cv2
import math
import numpy as np
class Main:
    @staticmethod
    def out_put():#実行用関数
        cv2.namedWindow('src')
        get_image = cv2.VideoCapture('highway.mov') # Open movie   
        #Car_line.edge_chacker(get_image)
        Car_speed.speed_checker(get_image)

class Car_speed():
    @staticmethod
    def speed_checker(cap: cv2.VideoCapture):
        frame_rate = cap.get(cv2.CAP_PROP_FPS)
        frame_index = 0
        detected_frames = []  # d==2になったフレームだけ記録
        last_d = 1  # 最初は白線を踏んでいないと仮定

        while True:
            ret, img_src = cap.read()
            if not ret:
                break

            height, width = img_src.shape[:2]
            half_height = height // 2
            ROI = img_src[half_height:, :]
            gray_scale = cv2.cvtColor(ROI, cv2.COLOR_BGR2GRAY)

            separate = 20
            trans_width = width // separate
            roi_box = []

            for i in range(separate):
                ROI_notice = gray_scale[:, i * trans_width:(i + 1) * trans_width]
                V = np.mean(ROI_notice)
                roi_box.append(V)

            Rmax = max(roi_box)
            Vavg = np.mean(roi_box)
            Rmax_index = roi_box.index(Rmax)

            opposite_candidate = []
            for i in range(separate):
                if roi_box[i] >= Vavg:
                    opposite_candidate.append((i, roi_box[i]))

            opposite_candidate.sort(key=lambda x: x[1], reverse=True)

            min_offset = int(0.55 * separate)
            max_offset = int(0.71 * separate)
            R2 = Rmax_index
            d = 1  # 初期は白線1本と仮定

            for i, v in opposite_candidate:
                distance = abs(i - Rmax_index)
                if min_offset <= distance <= max_offset:
                    R2 = i
                    d = 2  # 対辺が見つかったら白線2本検出（1セット）
                    break

            # 状態が 1→2 に変わった瞬間だけ記録
            if d == 2 and last_d != 2:
                detected_frames.append(frame_index)

            last_d = d  # 次のフレーム用に更新

            # 3セット検出されたら速度を計算
            if len(detected_frames) >= 3:
                frame_diff = detected_frames[2] - detected_frames[0]  # 最初と3番目のフレームの差
                time_sec = frame_diff / frame_rate
                distance_m = 20 * 3  # 1セット20m × 3セット
                speed = distance_m / time_sec  # m/s
                speed_kmh = speed * 3.6

                # 画面に表示
                cv2.putText(img_src, f"Speed: {speed_kmh:.2f} km/h", (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow('src', img_src)
            frame_index += 1  # 次のフレームへ

            if cv2.waitKey(30) == 27:  # ESCキーで終了
                break

        cap.release()
        cv2.destroyAllWindows()
Main.out_put()#実行