import numpy as np
import cv2 as cv
import math

import time

def box_cvt_cent2corners_pts(box):
    pt_lt = (int(box[0]-box[2]/2),int(box[1]-box[3]/2))
    pt_rb = (int(box[0] + box[2] / 2), int(box[1] + box[3] / 2))
    return pt_lt,pt_rb

def box_cvt_cent2corners_pts_float(box):
    pt_lt = ((box[0]-box[2]/2),(box[1]-box[3]/2))
    pt_rb = ((box[0] + box[2] / 2), (box[1] + box[3] / 2))
    return pt_lt,pt_rb

def box_cvt_2corners(box):
    pt_lt = (int(box[0]),int(box[1]))
    pt_rb = (int(box[0] + box[2]), int(box[1] + box[3]))
    return (pt_lt[0],pt_lt[1],pt_rb[0],pt_rb[1])

def box_cvt_2corners_pts(box):
    pt_lt = (int(box[0]),int(box[1]))
    pt_rb = (int(box[0] + box[2]), int(box[1] + box[3]))
    return pt_lt,pt_rb

def sign(x):
    if x<0:
        s = -1
    else:
        s = 1
    return s

def make_border_mask(shape,pad = 25,style = 'linear'):
    n = pad
    k = 1 / n
    k_array = [0.0] * n
    for i in range(n):
        k_array[i] = i * k
    h_r, w_r = shape
    border_mask_h = np.ones(shape)
    border_mask_v = np.ones(shape)
    for i, k in enumerate(k_array):
        border_mask_v[:, i:i + 1] = k
        border_mask_h[i:i + 1, :] = k
        border_mask_h[h_r - n + i:h_r - n+1 + i, :] = 1-k
        border_mask_v[:, w_r - n + i:w_r - n+1 + i] = 1 - k
    border_mask = cv.multiply(border_mask_v, border_mask_h)
    # test_im = np.ones(shape,np.uint8)
    # test_im*=255
    # cv.multiply(test_im, border_mask, dtype=cv.CV_8U, dst=test_im)
    return border_mask

class Vect:
    '''
    Класс для удобной работы с геометрическими векторами
    '''
    def __init__(self, pt1 = [0,0],pt2 = [0,0]):
        self.vect = np.array([0,0])
        self.get_vect_by_pts(pt1,pt2)

    def get_vect_by_pts(self,pt1,pt2):
        self.vect[0] = pt2[0]-pt1[0]
        self.vect[1] = pt2[1]-pt1[1]
        return self.vect

    def get_length(self):
        return (self.vect[0]**2+self.vect[1]**2)**0.5

def get_scalar_mult(v1:Vect,v2:Vect):
    return np.dot(v1.vect,v2.vect)

def get_v_angle(v1:Vect,v2:Vect):
    cos_a = get_scalar_mult(v1,v2) / (v1.get_length()*v2.get_length())
    if cos_a>1:
        cos_a = 1.0
    elif cos_a < -1:
        cos_a = -1.0
    return (math.acos(cos_a))*180/math.pi

def pt2pt_2d_range(pt1,pt2):
    return ((pt2[0]-pt1[0])**2 + (pt2[1]-pt1[1])**2)**0.5


def draw_box(image,box,color = (100,100,100)):
    pt1 = (box[0],box[1])
    pt2 = (box[2],box[3])
    cv.rectangle(image,pt1,pt2,color,3)

def check_in_box(box,pt_xy):
    in_box = True
    in_box&=(pt_xy[0]>=box[0])&(pt_xy[0]<=box[2])
    in_box &= (pt_xy[1] >= box[1]) & (pt_xy[1] <= box[3])
    return in_box



class Image_meta:
    def __init__(self, az = 0,el = 0,px_size:[] = [4504,4504],angle_size:[]=[42.5,42.5]):
        self.timestamp = time.time()
        self.im_size = px_size      #Пиксельные размеры кадра
        self.angle_s = angle_size   #Угловые размеры кадра
        self.az = az
        self.el = el
        self.id = 0
        self.channel_count = 1
        self.deg_ppx_x = self.angle_s[0] / self.im_size[0]
        self.deg_ppx_y = self.angle_s[1] / self.im_size[1]
        self.px_center = (self.im_size[0] / 2, self.im_size[1] / 2)

    def set_new_im_size(self,im_size):
        if (self.im_size[0]==im_size[0])&(self.im_size[1]==im_size[1]):
            pass
        else:
            self.im_size[0] = im_size[0]
            self.im_size[1] = im_size[1]
            self.reinit_k()

    def set_new_angle_size(self, angle_s):
        if (self.angle_s[0]==angle_s[0])&(self.angle_s[1]==angle_s[1]):
            pass
        else:
            self.angle_s[0] = angle_s[0]
            self.angle_s[1] = angle_s[1]
            self.reinit_k()

    def set_sizes(self,im_size,angle_s):
        need2reinit = False
        if (self.im_size[0]==im_size[0])&(self.im_size[1]==im_size[1]):
            pass
        else:
            self.im_size[0] = im_size[0]
            self.im_size[1] = im_size[1]
            need2reinit = True
        if (self.angle_s[0]==angle_s[0])&(self.angle_s[1]==angle_s[1]):
            pass
        else:
            self.angle_s[0] = angle_s[0]
            self.angle_s[1] = angle_s[1]
            need2reinit = True
        if need2reinit:
            self.reinit_k()


    def reinit_k(self):
        self.deg_ppx_x = self.angle_s[0] / self.im_size[0]
        self.deg_ppx_y = self.angle_s[1] / self.im_size[1]
        self.px_center = (self.im_size[0] / 2, self.im_size[1] / 2)

    def print(self):
        print(f'Meta: frame_id = {self.id},az = {self.az}, el = {self.el},size = {self.im_size}, deg_size = {self.angle_s}, stamp = {self.timestamp}')

    def to_string(self):
        return f'Meta: frame_id = {self.id},az = {self.az}, el = {self.el},size = {self.im_size}, deg_size = {self.angle_s}, stamp = {self.timestamp}'

    def calc_view_field(self,distance):
        '''
        Считает размер зоны обзора в метрах на заданной дистанции
        :param distance:
        :return:
        '''
        half_h_angle = math.pi*self.angle_s[0]/360
        half_v_angle = math.pi * self.angle_s[1] / 360
        h_field = 2*distance*math.tan(half_h_angle)
        v_field = 2*distance*math.tan(half_v_angle)
        return h_field, v_field

    def calc_px_shift_by_m_shift(self,distance,shift):
        '''
        Считает смещение в пикселях, соответствующее смещению в метрах на заданной дистанции
        :param distance: расстояние до цели в метрах
        :param shift: смещение(или размер) в метрах
        :return:
        '''
        view_field = self.calc_view_field(distance)
        x_scale = self.im_size[0]/view_field[0]
        px_shift = shift*x_scale
        return px_shift


    def get_abs_p_pos(self,x,y):
        '''
        Получить угловые координаты цели из координат на кадре
        :param x:
        :param y:
        :return: Азимут, Угол места
        '''
        d_az = (x-self.px_center[0])*self.deg_ppx_x
        d_el = (self.px_center[0]-y)*self.deg_ppx_y
        return (self.az+d_az)%360, (self.el+d_el)

    def put_abs_p_pos(self,az,el):
        d_az = az - self.az
        # print(d_az)
        if d_az>=180:
            d_az-=360
        elif d_az<-180:
            d_az+=360


        d_el = el-self.el

        d_x = d_az/self.deg_ppx_x
        d_y = -d_el/self.deg_ppx_y
        # print(d_x,'/',d_y)
        return int(self.px_center[0]+d_x),int(self.px_center[1]+d_y)

def compare_meta(meta1:Image_meta,meta2:Image_meta):
    direction_eq = True
    sizes_eq = True
    direction_eq&=(meta1.az == meta2.az)
    direction_eq&=(meta1.el == meta2.el)
    sizes_eq&=(meta1.angle_s == meta2.angle_s)
    sizes_eq&=(meta1.im_size == meta2.im_size)
    return direction_eq,sizes_eq

class Meta2meta_converter:
    '''
    Класс для преобразования координат и размеров от одного источника к другому
    '''

    def __init__(self, meta1: Image_meta, meta2: Image_meta, alias1='src', alias2='dst'):
        self.aliases = [alias1, alias2]
        self.metas = [meta1,meta2]
        for m in self.metas:
            m.reinit_k()
        self.px_p_deg_x = [self.metas[0].im_size[0]/self.metas[0].angle_s[0],self.metas[1].im_size[0]/self.metas[1].angle_s[0]]
        self.px_p_deg_y = [self.metas[0].im_size[1] / self.metas[0].angle_s[1],
                           self.metas[1].im_size[1] / self.metas[1].angle_s[1]]
        #Коэффициенты перевода размеров из 1-го во 2-й и из 2-го в 1-й по X-координате:
        self.translate_ratios_x = [self.px_p_deg_x[1]/self.px_p_deg_x[0], self.px_p_deg_x[0]/self.px_p_deg_x[1]]
        # Коэффициенты перевода размеров из 1-го во 2-й и из 2-го в 1-й по Y-координате:
        self.translate_ratios_y = [self.px_p_deg_y[1] / self.px_p_deg_y[0], self.px_p_deg_y[0] / self.px_p_deg_y[1]]

    def set_aliases(self,alias1, alias2):
        self.aliases = [alias1, alias2]

    def translate_x_size(self,size,from_alias='src',to_alias = 'dst'):
        if (from_alias in self.aliases)&(to_alias in self.aliases):
            src_idx = self.aliases.index(from_alias)
            return size*self.translate_ratios_x[src_idx]
        else:
            print('!!!!!!!Неверно заданные имена систем координат')
    def translate_y_size(self,size,from_alias='src',to_alias = 'dst'):
        if (from_alias in self.aliases)&(to_alias in self.aliases):
            src_idx = self.aliases.index(from_alias)
            return size*self.translate_ratios_y[src_idx]
        else:
            print('!!!!!!!Неверно заданные имена систем координат')
    def translate_2d_size(self,w,h,from_alias='src',to_alias = 'dst'):
        w_new = self.translate_x_size(w,from_alias,to_alias)
        h_new = self.translate_y_size(h, from_alias, to_alias)
        return w_new,h_new

    def translate_pt(self,pt_x,pt_y,from_alias,to_alias):
        if (from_alias in self.aliases) & (to_alias in self.aliases):
            src_idx = self.aliases.index(from_alias)
            dst_idx = self.aliases.index(to_alias)
            az,el = self.metas[src_idx].get_abs_p_pos(pt_x,pt_y)
            new_x,new_y = self.metas[dst_idx].put_abs_p_pos(az,el)
            return new_x,new_y
        else:
            print('!!!!!!!Неверно заданные имена систем координат')

    def update_meta_direction(self,new_az,new_el,alias):
        if alias in self.aliases:
            idx = self.aliases.index(alias)
            self.metas[idx].az = new_az
            self.metas[idx].el = new_el
        else:
            print('!!!!!!!Неверно заданное имя системы координат')

def cover_pt_by_area(pt_xy,area_w_h = [1200,1200],limit_box = [0,0,4504,4504]):
    area_x1 = max(limit_box[0],int(pt_xy[0]-area_w_h[0]/2))
    area_x2 = area_x1+area_w_h[0]
    if area_x2>=limit_box[2]:
        area_x2 = limit_box[2]-1
        area_x1 = max(area_x2-area_w_h[0],limit_box[0])
    area_y1 = max(limit_box[1], int(pt_xy[1] - area_w_h[1] / 2))
    area_y2 = area_y1 + area_w_h[1]
    if area_y2 >= limit_box[3]:
        area_y2 = limit_box[3] - 1
        area_y1 = max(area_y2 - area_w_h[1], limit_box[1])
    return [area_x1,area_y1,area_x2,area_y2]

class Panoram_creator():
    '''
    Класс для преобразования координат и работы с панорамой
    '''
    def __init__(self, im_size:[] = [2000,1000],sect:[] = [0,180,0,45],angle_err = 1):
        '''
        :param im_size: Размер изображения для панорамы
        :param sect:    Параметры области построения панорамы: Az1 - левый край,Az2,El1-нижний край,El2
        :param sect_size: размер сканируемого сектора (ширина, высота), в градусах
        :param observ_size: размер области обзора (ширина, высота), в градусах
        :param frame_size: размер кадра, полученного от камеры
        '''
        # self.base_frame_size = frame_size
        # self.panoram_frame_size = (calc_panoram_size(sect_size[0],observ_size[0],frame_size[0]),
        #                            calc_panoram_size(sect_size[0],observ_size[0],frame_size[0]),)
        self.angle_err = angle_err

        self.az1 = sect[0]
        self.az2 = sect[1]
        self.el1 = sect[2]
        self.el2 = sect[3]

        #Рабочие углы области построения панорамы с учетом защитной рамки
        self.az1_w = (self.az1-self.angle_err)%360
        self.az2_w = (self.az2+self.angle_err)%360
        self.el1_w = self.el1-self.angle_err
        self.el2_w = self.el2+self.angle_err

        #Размер рабочей области в градусах
        self.d_az_w = calc_sector(self.az1_w,self.az2_w)[1]
        self.d_el_w = self.el2_w-self.el1_w

        self.im_size = im_size
        self.p_image = np.zeros((self.im_size[1],self.im_size[0]),dtype=np.uint8)

        k,shift = calc_fit_deg_to_px(self.im_size[0],self.im_size[1],self.d_az_w,self.d_el_w)

        self.k = k
        self.zero_px_shift = shift

        self.p_image[shift[1]:shift[1]+int(self.d_el_w*self.k),shift[0]:shift[0]+int(self.d_az_w*self.k)] = 255

    def put_new_on_panoram(self,image:np.array, meta:Image_meta):
        '''
        Разместить новое изображение на сформированной панораме
        :param image: Изображение cv
        :param meta:
        :return:
        '''
        print(f'zero: {self.zero_px_shift}')
        center_pos = [self.zero_px_shift[0]+self.k*calc_sector(self.az1_w,meta.az)[1],
                      self.zero_px_shift[1]+self.k*(self.el2_w-meta.el)]

        print(f'center_pos = {center_pos}')
        im_px_w = self.k*meta.angle_s[0]
        im_px_h = self.k*meta.angle_s[1]
        print(f'resize to {im_px_w},{im_px_h}')
        left_top = [int(center_pos[0]-im_px_w/2),int(center_pos[1]-im_px_h/2)]
        print(f'left_top_pos = {left_top}')

        # self.p_image[left_top[1]:left_top[1] + int(im_px_h), left_top[0]:left_top[0] + int(im_px_w)] =150
        self.p_image[left_top[1]:left_top[1]+int(im_px_h),left_top[0]:left_top[0]+int(im_px_w)] = cv.resize(image,dsize = (int(im_px_w),int(im_px_h)))
    def put_new_on_panoram_smooth(self, image: np.array, meta: Image_meta):
        '''
        Разместить новое изображение на сформированной панораме
        :param image: Изображение cv
        :param meta:
        :return:
        '''
        print(f'zero: {self.zero_px_shift}')
        center_pos = [self.zero_px_shift[0] + self.k * calc_sector(self.az1_w, meta.az)[1],
                      self.zero_px_shift[1] + self.k * (self.el2_w - meta.el)]

        print(f'center_pos = {center_pos}')
        im_px_w = self.k * meta.angle_s[0]
        im_px_h = self.k * meta.angle_s[1]
        print(f'resize to {im_px_w},{im_px_h}')
        left_top = [int(center_pos[0] - im_px_w / 2), int(center_pos[1] - im_px_h / 2)]
        print(f'left_top_pos = {left_top}')



        # self.p_image[left_top[1]:left_top[1] + int(im_px_h), left_top[0]:left_top[0] + int(im_px_w)] =150
        n = 30
        k = 1/n
        h,w = image.shape
        k_array = [0.0]*n
        for i in range(n):
            k_array[i] = i*k
        # print(k_array)

        resized = cv.resize(image, dsize=(int(im_px_w), int(im_px_h)))
        border_mask = make_border_mask(resized.shape,30)
        border_mask_inv = np.ones(resized.shape)-border_mask

        res = np.zeros(resized.shape,dtype = np.uint8)
        cv.multiply(resized,border_mask,dtype = cv.CV_8U,dst = res)
        buf = cv.multiply(self.p_image[left_top[1]:left_top[1] + int(im_px_h), left_top[0]:left_top[0] + int(im_px_w)],border_mask_inv,dtype=cv.CV_8U)
        res = cv.add(res,buf)
        # np.int
        # print(res)
        # cv.imshow('ww',res)
        # cv.waitKey()
        self.p_image[left_top[1]:left_top[1] + int(im_px_h), left_top[0]:left_top[0] + int(im_px_w)] = res
    def redraw(self,window:str):
        cv.imshow(window,self.p_image)
        cv.waitKey()

    def draw_borders(self,meta:Image_meta):
        # print(f'zero: {self.zero_px_shift}')
        center_pos = [self.zero_px_shift[0]+self.k*calc_sector(self.az1_w,meta.az)[1],
                      self.zero_px_shift[1]+self.k*(self.el2_w-meta.el)]

        # print(f'center_pos = {center_pos}')
        im_px_w = self.k*meta.angle_s[0]
        im_px_h = self.k*meta.angle_s[1]
        # print(f'resize to {im_px_w},{im_px_h}')
        left_top = [int(center_pos[0]-im_px_w/2),int(center_pos[1]-im_px_h/2)]
        # print(f'left_top_pos = {left_top}')
        right_but = (int(left_top[0]+im_px_w),int(left_top[1]+im_px_h))
        cv.rectangle(self.p_image, (left_top[0],left_top[1]),right_but,
                     0, 2)

def calc_fit_deg_to_px(w,h,d_az,d_el):
    '''

    :param w:   ширина изображения
    :param h:   высота изображения
    :param d_az: ширина области в градусах
    :param d_el: высота области в градусах
    :return:
    '''
    if (w/h)>(d_az/d_el):
        k =h/d_el
        shift = [int((w-k*d_az)/2),0]
    else:
        k = w/d_az
        shift = (0,int((h-k*d_el)/2))
    return k,shift



def calc_panoram_size(sect_da,observ_a,frame_size):
    return int(frame_size*sect_da/observ_a)


def calc_scan_areas(scan_area_bbox,window_w_h = (800,800),overlay=(0.1,0.1),force_int = True):
    '''
    Вычисление подобластей заданного размера (окон) для сканирования в заданной области
    с перекрытием не менее указанного в параметре overlay
    :param bbox: область сканирования
    :param window_w_h: (ширина, высота) окна сканирования
    :param overlay: коэффициенты перекрытия (overlay_x,overlay_y) по соответствующим осям
    :return: массив bbox_w окон
    '''
    # scan_area_w = scan_area_bbox[2] - scan_area_bbox[0]
    # scan_area_h = scan_area_bbox[3] - scan_area_bbox[1]
    # print(f'area w,h = {scan_area_w},{scan_area_h}')
    # over_px_x = overlay[0]*window_w_h[0]
    # print(f' over x = {over_px_x}')
    #
    # n_x = (scan_area_w - window_w_h[0]) / (window_w_h[0] - over_px_x)
    # print(f' n x = {n_x}')
    # n_x = math.ceil(n_x)
    # over_px_x = window_w_h[0] - (scan_area_w-window_w_h[0])/n_x
    # print(f' n_x = {n_x}, over_px_x = {over_px_x}')
    #
    #
    # over_px_y = overlay[1]*window_w_h[1]
    x_areas = calc_linear_scan_areas((scan_area_bbox[0],scan_area_bbox[2]),window_w_h[0],overlay[0],force_int)
    y_areas = calc_linear_scan_areas((scan_area_bbox[1],scan_area_bbox[3]),window_w_h[1],overlay[1],force_int)

    areas = []
    for y_a in y_areas:
        for x_a in x_areas:
            a_bbox = [x_a[0], y_a[0],x_a[1],y_a[1]]
            areas.append(a_bbox)
    # print('boxes')
    # print(areas)
    return areas


def calc_linear_scan_areas(scan_area: tuple, window_w=800, overlay=0.1, force_int = True):
    '''
    Вычисление подобластей заданного размера (окон) для сканирования в заданной области
    с перекрытием не менее указанного в параметре overlay вдоль оси
    :param scan_area: область сканирования x1<->x2
    :param window: размер окна сканирования
    :param overlay: коэффициенты перекрытия
    :return:
    '''
    # print("I'm alive")
    scan_area_w = scan_area[1]-scan_area[0]
    over = overlay * window_w
    # print(f' over= {over}')

    n_x = (scan_area_w - window_w) / (window_w - over)
    # print(f' n = {n_x}')
    n_x = math.ceil(n_x)
    over = window_w - (scan_area_w - window_w) / n_x
    # print(f' n_x = {n_x}, over_px_x = {over}')
    window_arr = []
    for i in range(n_x+1):
        start_x = scan_area[0]+i*(window_w-over)
        stop_x = start_x+window_w
        if force_int:
            window_arr.append((int(start_x),int(stop_x)))
        else:
            window_arr.append((start_x,stop_x))
    return window_arr

def calc_scan_points(sect_a1,sect_da, observ_a, overlay = 1/3, orient = 'h'):
    '''
    Вычисление углов поворота для сканирования заданной области с перекрытием не менее указанного в параметре overlay
    :param sect_a1: начальный угол
    :param sect_da: ширина сектора сканирования
    :param observ_a: угол обзора
    :param overlay: коэффициент перекрытия
    :return: массив абсолютных значений углов для установки камеры
    '''
    over = observ_a*overlay
    n = (sect_da-observ_a)/(observ_a-over)

    print(f'n = {n}, ceil(n) = {math.ceil(n)}')
    n = math.ceil(n)
    over = observ_a - (sect_da-observ_a)/n

    print(f'over = {over}')
    angles = [0.0]*(n+1)
    for i,a in enumerate(angles):
        print(i)
        if orient == 'h':
            angles[i] = (sect_a1+ observ_a/2+i*(observ_a-over))%360
        else:
            angles[i] = (sect_a1 + observ_a / 2 + i * (observ_a - over))
    print(angles)

    # shift_a = (1-overlay)*observ_a
    # no_overlay_a = (1-overlay)*observ_a
    # n_steps_norm = math.ceil(sect_da/shift_a)
    # shift_norm = sect_da/n_steps_norm
    # over_norm = sect_da-shift_norm
    # print(f'Сдвиг идеальный: {shift_a}, нормированный: {shift_norm}')
    # angles = [observ_a/2]*(n_steps_norm)

    # for i in range(n_steps_norm):
    #     angles[i]+=(i*(sect_da-over_norm))
    # # print(angles)
    # return angles
    return angles

def calc_scan_points_a1_a2(sect_a1,sect_a2,observ_a,overlay = 1/3,orient = 'h'):
    '''
    :param sect_a1: начальный угол
    :param sect_a2: конечный угол сектора сканирования
    :param observ_a: угол обзора
    :param overlay: коэффициент перекрытия
    :return: массив абсолютных значений углов для установки камеры
    '''
    if orient == 'h':
        a_c, d_a = calc_sector(sect_a1,sect_a2)
    else:
        d_a = sect_a2-sect_a1
    return calc_scan_points(sect_a1,d_a,observ_a,overlay,orient)


def calc_sector(a1,a2):
    '''Вычисляет центр сектора и его ширину. Сектор задан от a1 к a2 по часовой стрелке
    :return: a_center, d_a - середина сектора, ширина сектора
    '''
    da = (a2-a1)%360
    return (a1+da/2)%360,da

def check_in_sector(a,a0,d_a):
    '''проверяет, находится ли угол [a] внутри сектора,
    заданного центром [a0] и шириной [d_a]'''
    a_ref = (a-a0)%360
    if a_ref>=(-d_a/2) and a_ref<=(d_a/2):
        return True
    else:
        return False

def check_in_range(x,range:()):
    if (x>=range[0]) and (x<=range[1]):
        return True
    else:
        return False

def linear_cross(obj_1,obj_2):
    cross = 0
    if check_in_range(obj_1[0],(obj_2[0],obj_2[1])):
        cross = min(obj_1[1],obj_2[1])-obj_1[0]
    elif check_in_range(obj_2[0],(obj_1[0],obj_1[1])):
        cross = min(obj_1[1],obj_2[1])-obj_2[0]
    return cross

def bbox_cross_area(bbox1,bbox2):
    x_cross = linear_cross((bbox1[0],bbox1[2]),(bbox2[0],bbox2[2]))
    y_cross = linear_cross((bbox1[1],bbox1[3]),(bbox2[1],bbox2[3]))
    cross_area = x_cross*y_cross
    return cross_area

