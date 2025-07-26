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

class Car_line:#車線の検出をする関数  
    figure_box={#変数一覧
        'auto' : 0,
        'width_pix' :5,
        'height_pix' : 5,
        'side_line' : 0,
        'ROI_number':80,
        'low_edge' :50,
        'high_edge' :150,
        'one_px' :1,
        'two_px' : 2,
        'small_line':0.5,
        'red_color' : (0,0,255),
        'line_width':8
    }

    @staticmethod
    def edge_chacker(cap:cv2.VideoCapture):
        while True:
            (ret, img_src) = cap.read()	# retは画像を取得成功フラグ
            if ret:		# フレームが得られていれば表示する
                fg=Car_line.figure_box
                line,half_height = Line_make.maker(img_src)     
                if line is not None:#線を検出した時
                    Car_line.deal_add_color(line,half_height,img_src)
            cv2.imshow('src', img_src)
            if cv2.waitKey(30) == 27:#ESCキーで止まる
                break        
        cap.release()
        cv2.destroyAllWindows()
    @staticmethod
    def deal_add_color(line,half_height,img_src):
        fg=Car_line.figure_box
        found_empty_line = True
        for line in line:
            x1, y1, x2, y2 = line[0]#検出した車線の始点と終点の座標       
            if x2 - x1 == fg['side_line']:  # 縦線
                slope = float('inf')
            else:
                slope = (y2 - y1) / (x2 - x1)#斜めってる車線対策
            if abs(slope) < fg['small_line']:
                continue  # 横線に近いからいらない
            cv2.line(img_src, (x1, y1 + half_height), (x2, y2 + half_height), fg['red_color'], fg['two_px'])#縦線に近かったら赤色で描写
        return found_empty_line
    '''
    車線検出の苦労したところ:メソッドとプロパティを調べるのに時間がかかった。
    工夫したところ:マジックナンバーをなるべく避けて可読性を上げた。もしかしたら再利用できるかもしれないから継承・オーバーライドするためにクラスで残しといた
    '''
class Car_speed(Car_line):#車線境界線（白い破線）を通過する時間間隔とその長さ1を利用して車速を求める
    @staticmethod
    def speed_checker(cap:cv2.VideoCapture):
        #  1.映像から1枚フレーム(画像)を切り出す
        frame_rate = cap.get(cv2.CAP_PROP_FPS)      
        #2. 画像最下行を等分しROI :Region of Interest (注目領域) とし、各 ROI 内の画素の平均値 V を求める。
        (ret, img_src) = cap.read()	# retは画像を取得成功フラグ
        height,width = img_src.shape[:2]#高さを取得
        half_height= height//2#画像最下行を等分
        ROI = img_src[half_height:, :]#下半分を検出するようにする
        gray_scale = cv2.cvtColor(ROI,cv2.COLOR_BGR2GRAY)

        separate = 20#20等分にする
        trans_width = width//separate
        roi_box=[]

        for i in range(separate):
            ROI_notice = gray_scale[:, i * trans_width:(i + 1) * trans_width]
            V = np.mean(ROI_notice)
            roi_box.append(V)
           
        #3. 各ROIの中で、(明るさの) 最大値Vmax を持つものをRmax とし、平均値Vavg 以上のROI を対辺候補とする。
        Rmax = max(roi_box)
        Vavg = np.mean(roi_box)
        Rmax_index = roi_box.index(Rmax)
        
        opposite_candidate = []
        for i in range(separate):
             if roi_box[i] >= Vavg:
                opposite_candidate.append((i,roi_box[i]))
        #4. 明るさ順にソートした対辺候補のリストから順にROIを取り出し、(a) その位置が Rmax±(0.55∼0.71)×画像幅 の範囲内にあるか？の条件を満足していれば、それをR2 と決定し次フレームへ。(d=2)
        opposite_candidate.sort(reverse=True)
        R2 = Rmax
        min_range = 0.55*separate
        max_range = 0.71*separate
        opposite_candidate.sort(key=lambda x: x[1], reverse=True)#明るさを基準にしたソート。明るさって基準を指定しないとだめだった
        for x in opposite_candidate:
            if abs(x)>=0.55 and abs(x)<=0.71:
                R2 = x
                break
            else:
                R2 = Rmax


        fg=Car_speed.figure_box
        pre_line = None
        current_line = 0
        while True:
            (ret, img_src) = cap.read()	# retは画像を取得成功フラグ
            if ret:
                line,half_height = Line_make.maker(img_src)
                fg = Car_line.figure_box
                if line is not None:
                    found_empty_line =Car_line.deal_add_color(line,half_height,img_src)
                if found_empty_line:
                    if pre_line is not None:
                        line_defferent = current_line - pre_line
                        time_sec = line_defferent / frame_rate
                        speed = fg['line_width'] / time_sec
                        speed_kmh = speed * 3.6
                    #print(f"Speed: {speed_kmh:.2f} km/h")
                        cv2.putText(img_src, f"Speed: {speed_kmh:.2f} km/h", (10, 50),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        pre_line = None  # リセット
                    else:
                        pre_line = current_line

            current_line += 1
            cv2.imshow('src', img_src)
            if cv2.waitKey(30) == 27:
                break

        cap.release()
        cv2.destroyAllWindows()

class Line_make:#線を引くまでの部分を再利用できるようにした
    @staticmethod
    def maker(img_src):
        height,width = img_src.shape[:2]#高さを取得
        half_height = height//2#高さの半分
        fg=Car_line.figure_box

        under_img = img_src[half_height:, :]#下半分を検出するようにする

        img_gray = cv2.cvtColor(under_img, cv2.COLOR_BGR2GRAY)#グレースケールに変換
        center_notice = cv2.GaussianBlur(img_gray, (fg['width_pix'], fg['height_pix']), fg['auto'])#この前実験の授業で習った中央以外ぼかすやつ。これで遠くにある変な線を取りにくくしたい
        edge = cv2.Canny(center_notice,fg['low_edge'],fg['high_edge'])#色のコントラストの度合いを拾う

        line = cv2.HoughLinesP(edge, fg['one_px'], np.pi/180, threshold=50, minLineLength=50, maxLineGap=10)#直線を引く
        return line,half_height
Main.out_put()#実行