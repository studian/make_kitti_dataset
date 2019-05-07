#!/usr/bin/env python
import sys, os, glob, cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tqdm import tqdm

def get_filepath_list_pair(_target_dir):
    subpath = []
    target_dir = os.path.normpath(_target_dir) # remove trailing separator.

    for fname in os.listdir(target_dir):
        full_dir = os.path.join(target_dir, fname)
        
        if(os.path.isdir(full_dir)):
            #print (full_dir)
            subpath.append(fname)
            
    return subpath

def make_sub_folder(root_folder, sub_folder):
    if not os.path.exists(root_folder+'/'+sub_folder):
        os.mkdir(root_folder+'/'+sub_folder)

    folder_path = root_folder+'/'+sub_folder
    return folder_path

def load_from_bin(bin_path):
    obj = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
    return obj

def depth_color(val, min_d=0, max_d=120):
    """ 
    print Color(HSV's H value) corresponding to distance(m) 
    close distance = red , far distance = blue
    """
    np.clip(val, 0, max_d, out=val) # max distance is 120m but usually not usual
    return (((val - min_d) / (max_d - min_d)) * 80).astype(np.uint8) 

def in_h_range_points(points, m, n, fov):
    """ extract horizontal in-range points """
    return np.logical_and(np.arctan2(n,m) > (-fov[1] * np.pi / 180), np.arctan2(n,m) < (-fov[0] * np.pi / 180))

def in_h_range_points2(points, m, n, fov):
    """ extract horizontal in-range points """
    return np.logical_and(np.arctan2(n,m) > (-fov[1] * np.pi / 180), np.arctan2(n,m) < (-fov[0] * np.pi / 180))

def in_v_range_points(points, m, n, fov):
    """ extract vertical in-range points """
    return np.logical_and(np.arctan2(n,m) < (fov[1] * np.pi / 180), np.arctan2(n,m) > (fov[0] * np.pi / 180))

def fov_setting(points, x, y, z, dist, h_fov, v_fov):
    """ filter points based on h,v FOV  """
    
    if h_fov[1] == 180 and h_fov[0] == -180 and v_fov[1] == 2.0 and v_fov[0] == -24.9:
        return points
    
    if h_fov[1] == 180 and h_fov[0] == -180:
        return points[in_v_range_points(points, dist, z, v_fov)]
    elif v_fov[1] == 2.0 and v_fov[0] == -24.9:        
        return points[in_h_range_points(points, x, y, h_fov)]
    else:
        h_points = in_h_range_points(points, x, y, h_fov)
        v_points = in_v_range_points(points, dist, z, v_fov)
        return points[np.logical_and(h_points, v_points)]

def fov_setting2(points, x, y, z, dist):
    """ filter points based on h,v FOV  """
    h_fov=(-45,45)
    return points[in_h_range_points2(points, x, y, h_fov)]

def in_range_points(points, size):
    """ extract in-range points """
    return np.logical_and(points > 0, points < size) 

def velo_points_filter(points, v_fov, h_fov):
    """ extract points corresponding to FOV setting """
    
    # Projecting to 2D
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    dist = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    #print(dist)
    #print(points[:, 3])

    if h_fov[0] < -90:
        h_fov = (-90,) + h_fov[1:]
    if h_fov[1] > 90:
        h_fov = h_fov[:1] + (90,)
    
    x_lim = fov_setting(x, x, y, z, dist, h_fov, v_fov)[:,None]
    y_lim = fov_setting(y, x, y, z, dist, h_fov, v_fov)[:,None]
    z_lim = fov_setting(z, x, y, z, dist, h_fov, v_fov)[:,None]

    # Stack arrays in sequence horizontally
    xyz_ = np.hstack((x_lim, y_lim, z_lim))
    xyz_ = xyz_.T

    # stack (1,n) arrays filled with the number 1
    one_mat = np.full((1, xyz_.shape[1]), 1)
    xyz_ = np.concatenate((xyz_, one_mat),axis = 0)

    # need dist info for points color
    dist_lim = fov_setting(dist, x, y, z, dist, h_fov, v_fov)
    color = depth_color(dist_lim, 0, 80)
    
    return xyz_, color

def velo_points_filter2(points):
    """ extract points corresponding to FOV setting """
    
    # Projecting to 2D
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    #dist = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    dist = points[:, 3]*100

    """
    if h_fov[0] < -90:
        h_fov = (-90,) + h_fov[1:]
    if h_fov[1] > 90:
        h_fov = h_fov[:1] + (90,)
        """
    
    x_lim = fov_setting2(x, x, y, z, dist)[:,None]
    y_lim = fov_setting2(y, x, y, z, dist)[:,None]
    z_lim = fov_setting2(z, x, y, z, dist)[:,None]

    # Stack arrays in sequence horizontally
    xyz_ = np.hstack((x_lim, y_lim, z_lim))
    xyz_ = xyz_.T

    # stack (1,n) arrays filled with the number 1
    one_mat = np.full((1, xyz_.shape[1]), 1)
    xyz_ = np.concatenate((xyz_, one_mat),axis = 0)

    # need dist info for points color
    dist_lim = fov_setting2(dist, x, y, z, dist)
    color = depth_color(dist_lim, 0, 99)
    
    return xyz_, color

def velo_points_filter3(points):
    """ extract points corresponding to FOV setting """
    
    # Projecting to 2D
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    rangeL = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    ReflectL = points[:, 3]*100
    
    dd1 = rangeL/80
    dd2 = ReflectL/99
    
    dist = dd2/dd1
    dist = dist*100

    """
    if h_fov[0] < -90:
        h_fov = (-90,) + h_fov[1:]
    if h_fov[1] > 90:
        h_fov = h_fov[:1] + (90,)
        """
    
    x_lim = fov_setting2(x, x, y, z, dist)[:,None]
    y_lim = fov_setting2(y, x, y, z, dist)[:,None]
    z_lim = fov_setting2(z, x, y, z, dist)[:,None]

    # Stack arrays in sequence horizontally
    xyz_ = np.hstack((x_lim, y_lim, z_lim))
    xyz_ = xyz_.T

    # stack (1,n) arrays filled with the number 1
    one_mat = np.full((1, xyz_.shape[1]), 1)
    xyz_ = np.concatenate((xyz_, one_mat),axis = 0)

    # need dist info for points color
    dist_lim = fov_setting2(dist, x, y, z, dist)
    color = depth_color(dist_lim, 0, 99)
    
    return xyz_, color
def calib_velo2cam(filepath):
    """ 
    get Rotation(R : 3x3), Translation(T : 3x1) matrix info 
    using R,T matrix, we can convert velodyne coordinates to camera coordinates
    """
    with open(filepath, "r") as f:
        file = f.readlines()    
        
        for line in file:
            (key, val) = line.split(':',1)
            if key == 'R':
                R = np.fromstring(val, sep=' ')
                R = R.reshape(3, 3)
            if key == 'T':
                T = np.fromstring(val, sep=' ')
                T = T.reshape(3, 1)
    return R, T

def calib_cam2cam(filepath, mode):
    """
    If your image is 'rectified image' :
        get only Projection(P : 3x4) matrix is enough
    but if your image is 'distorted image'(not rectified image) :
        you need undistortion step using distortion coefficients(5 : D)
        
    in this code, I'll get P matrix since I'm using rectified image
    """
    with open(filepath, "r") as f:
        file = f.readlines()
        
        for line in file:
            (key, val) = line.split(':',1)
            if key == ('P_rect_' + mode):
                P_ = np.fromstring(val, sep=' ')
                P_ = P_.reshape(3, 4)
                # erase 4th column ([0,0,0])
                P_ = P_[:3, :3]
    return P_

def velo3d_2_camera2d_points(points, v_fov, h_fov, vc_path, cc_path, mode='02'):
    """ print velodyne 3D points corresponding to camera 2D image """
    
    # R_vc = Rotation matrix ( velodyne -> camera )
    # T_vc = Translation matrix ( velodyne -> camera )
    R_vc, T_vc = calib_velo2cam(vc_path)
    
    # P_ = Projection matrix ( camera coordinates 3d points -> image plane 2d points )
    P_ = calib_cam2cam(cc_path, mode)

    """
    xyz_v - 3D velodyne points corresponding to h, v FOV in the velodyne coordinates
    c_    - color value(HSV's Hue) corresponding to distance(m)
    
             [x_1 , x_2 , .. ]
    xyz_v =  [y_1 , y_2 , .. ]   
             [z_1 , z_2 , .. ]
             [ 1  ,  1  , .. ]
    """  
    xyz_v, c_ = velo_points_filter(points, v_fov, h_fov)
    
    """
    RT_ - rotation matrix & translation matrix
        ( velodyne coordinates -> camera coordinates )
    
            [r_11 , r_12 , r_13 , t_x ]
    RT_  =  [r_21 , r_22 , r_23 , t_y ]   
            [r_31 , r_32 , r_33 , t_z ]
    """
    RT_ = np.concatenate((R_vc, T_vc),axis = 1)
    
    # convert velodyne coordinates(X_v, Y_v, Z_v) to camera coordinates(X_c, Y_c, Z_c) 
    for i in range(xyz_v.shape[1]):
        xyz_v[:3,i] = np.matmul(RT_, xyz_v[:,i])
        
    """
    xyz_c - 3D velodyne points corresponding to h, v FOV in the camera coordinates
             [x_1 , x_2 , .. ]
    xyz_c =  [y_1 , y_2 , .. ]   
             [z_1 , z_2 , .. ]
    """ 
    xyz_c = np.delete(xyz_v, 3, axis=0)

    # convert camera coordinates(X_c, Y_c, Z_c) image(pixel) coordinates(x,y) 
    for i in range(xyz_c.shape[1]):
        xyz_c[:,i] = np.matmul(P_, xyz_c[:,i])    

    """
    xy_i - 3D velodyne points corresponding to h, v FOV in the image(pixel) coordinates before scale adjustment
    ans  - 3D velodyne points corresponding to h, v FOV in the image(pixel) coordinates
             [s_1*x_1 , s_2*x_2 , .. ]
    xy_i =   [s_1*y_1 , s_2*y_2 , .. ]        ans =   [x_1 , x_2 , .. ]  
             [  s_1   ,   s_2   , .. ]                [y_1 , y_2 , .. ]
    """
    xy_i = xyz_c[::]/xyz_c[::][2]
    ans = np.delete(xy_i, 2, axis=0)
    
    """
    width = 1242
    height = 375
    w_range = in_range_points(ans[0], width)
    h_range = in_range_points(ans[1], height)

    ans_x = ans[0][np.logical_and(w_range,h_range)][:,None].T
    ans_y = ans[1][np.logical_and(w_range,h_range)][:,None].T
    c_ = c_[np.logical_and(w_range,h_range)]

    ans = np.vstack((ans_x, ans_y))
    """
    
    return ans, c_

def velo3d_2_camera2d_points2(points, vc_path, cc_path, mode='02'):
   """ print velodyne 3D points corresponding to camera 2D image """
   
   # R_vc = Rotation matrix ( velodyne -> camera )
   # T_vc = Translation matrix ( velodyne -> camera )
   R_vc, T_vc = calib_velo2cam(vc_path)
   
   # P_ = Projection matrix ( camera coordinates 3d points -> image plane 2d points )
   P_ = calib_cam2cam(cc_path, mode)

   """
   xyz_v - 3D velodyne points corresponding to h, v FOV in the velodyne coordinates
   c_    - color value(HSV's Hue) corresponding to distance(m)
   
            [x_1 , x_2 , .. ]
   xyz_v =  [y_1 , y_2 , .. ]   
            [z_1 , z_2 , .. ]
            [ 1  ,  1  , .. ]
   """  
   xyz_v, c_ = velo_points_filter2(points)
   
   """
   RT_ - rotation matrix & translation matrix
       ( velodyne coordinates -> camera coordinates )
   
           [r_11 , r_12 , r_13 , t_x ]
   RT_  =  [r_21 , r_22 , r_23 , t_y ]   
           [r_31 , r_32 , r_33 , t_z ]
   """
   RT_ = np.concatenate((R_vc, T_vc),axis = 1)
   
   # convert velodyne coordinates(X_v, Y_v, Z_v) to camera coordinates(X_c, Y_c, Z_c) 
   for i in range(xyz_v.shape[1]):
       xyz_v[:3,i] = np.matmul(RT_, xyz_v[:,i])
       
   """
   xyz_c - 3D velodyne points corresponding to h, v FOV in the camera coordinates
            [x_1 , x_2 , .. ]
   xyz_c =  [y_1 , y_2 , .. ]   
            [z_1 , z_2 , .. ]
   """ 
   xyz_c = np.delete(xyz_v, 3, axis=0)

   # convert camera coordinates(X_c, Y_c, Z_c) image(pixel) coordinates(x,y) 
   for i in range(xyz_c.shape[1]):
       xyz_c[:,i] = np.matmul(P_, xyz_c[:,i])    

   """
   xy_i - 3D velodyne points corresponding to h, v FOV in the image(pixel) coordinates before scale adjustment
   ans  - 3D velodyne points corresponding to h, v FOV in the image(pixel) coordinates
            [s_1*x_1 , s_2*x_2 , .. ]
   xy_i =   [s_1*y_1 , s_2*y_2 , .. ]        ans =   [x_1 , x_2 , .. ]  
            [  s_1   ,   s_2   , .. ]                [y_1 , y_2 , .. ]
   """
   xy_i = xyz_c[::]/xyz_c[::][2]
   ans = np.delete(xy_i, 2, axis=0)
   #print(np.shape(ans[0]))
   
   return ans, c_

def velo3d_2_camera2d_points3(points, vc_path, cc_path, mode='02'):
   """ print velodyne 3D points corresponding to camera 2D image """
   
   # R_vc = Rotation matrix ( velodyne -> camera )
   # T_vc = Translation matrix ( velodyne -> camera )
   R_vc, T_vc = calib_velo2cam(vc_path)
   
   # P_ = Projection matrix ( camera coordinates 3d points -> image plane 2d points )
   P_ = calib_cam2cam(cc_path, mode)

   """
   xyz_v - 3D velodyne points corresponding to h, v FOV in the velodyne coordinates
   c_    - color value(HSV's Hue) corresponding to distance(m)
   
            [x_1 , x_2 , .. ]
   xyz_v =  [y_1 , y_2 , .. ]   
            [z_1 , z_2 , .. ]
            [ 1  ,  1  , .. ]
   """  
   xyz_v, c_ = velo_points_filter3(points)
   
   """
   RT_ - rotation matrix & translation matrix
       ( velodyne coordinates -> camera coordinates )
   
           [r_11 , r_12 , r_13 , t_x ]
   RT_  =  [r_21 , r_22 , r_23 , t_y ]   
           [r_31 , r_32 , r_33 , t_z ]
   """
   RT_ = np.concatenate((R_vc, T_vc),axis = 1)
   
   # convert velodyne coordinates(X_v, Y_v, Z_v) to camera coordinates(X_c, Y_c, Z_c) 
   for i in range(xyz_v.shape[1]):
       xyz_v[:3,i] = np.matmul(RT_, xyz_v[:,i])
       
   """
   xyz_c - 3D velodyne points corresponding to h, v FOV in the camera coordinates
            [x_1 , x_2 , .. ]
   xyz_c =  [y_1 , y_2 , .. ]   
            [z_1 , z_2 , .. ]
   """ 
   xyz_c = np.delete(xyz_v, 3, axis=0)

   # convert camera coordinates(X_c, Y_c, Z_c) image(pixel) coordinates(x,y) 
   for i in range(xyz_c.shape[1]):
       xyz_c[:,i] = np.matmul(P_, xyz_c[:,i])    

   """
   xy_i - 3D velodyne points corresponding to h, v FOV in the image(pixel) coordinates before scale adjustment
   ans  - 3D velodyne points corresponding to h, v FOV in the image(pixel) coordinates
            [s_1*x_1 , s_2*x_2 , .. ]
   xy_i =   [s_1*y_1 , s_2*y_2 , .. ]        ans =   [x_1 , x_2 , .. ]  
            [  s_1   ,   s_2   , .. ]                [y_1 , y_2 , .. ]
   """
   xy_i = xyz_c[::]/xyz_c[::][2]
   ans = np.delete(xy_i, 2, axis=0)
   #print(np.shape(ans[0]))
   
   return ans, c_

def print_projection_cv2(points, color, image):
    """ project converted velodyne points into camera image """
    
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    for i in range(points.shape[1]):
        cv2.circle(hsv_image, (np.int32(points[0][i]),np.int32(points[1][i])),2, (int(color[i]),255,255),-1)

    return cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)

def print_projection_plt(points, color, image):
    """ project converted velodyne points into camera image """
    
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
       
    for i in range(points.shape[1]):
        cv2.circle(hsv_image, (np.int32(points[0][i]),np.int32(points[1][i])),2, (int(color[i]),255,255),-1)

    return cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB)

def print_reflection_interpol_projection_plt(interLidar, colorImage):
    """ project converted velodyne points into camera image """
    
    lab_image = cv2.cvtColor(colorImage, cv2.COLOR_BGR2LAB)
    #print(hsv_image[:,:,1])
    lab_image[155:,:,0] = interLidar[:,:]/99*255

    return cv2.cvtColor(lab_image, cv2.COLOR_LAB2BGR)

def merge_lidar_interpol_and_color(interLidar, colorImage, roi_height):
    """ project converted velodyne points into camera image """
    
    lab_image = cv2.cvtColor(colorImage, cv2.COLOR_BGR2LAB)
    #print(hsv_image[:,:,1])
    lab_image[roi_height:,:,0] = interLidar[:,:]/99*100
    color_image2 = cv2.cvtColor(lab_image, cv2.COLOR_LAB2BGR)
    fusion_roi_color = color_image2[roi_height:,:,:]
    org_roi_color = colorImage[roi_height:,:,:]

    return fusion_roi_color, org_roi_color

def print_range_interpol_projection_plt(interLidar, colorImage):
    """ project converted velodyne points into camera image """
    
    lab_image = cv2.cvtColor(colorImage, cv2.COLOR_BGR2LAB)
    lab_image[155:,:,0] = interLidar[:,:]/80*255
    
    return cv2.cvtColor(lab_image, cv2.COLOR_LAB2BGR)


def preprocessing_split_images(points, color, image, inter_mode, roi_height):
    mode = ['nearest', 'linear', 'cubic']
    # nearest : NearestNDInterpolator 
    # linear : LinearNDInterpolator
    # cubic : CloughTocher2DInterpolator  
    
    gray_image = np.zeros_like(image)
    valid_mask = np.zeros((gray_image.shape[0], gray_image.shape[1]), dtype=bool)
    
    for i in range(points.shape[1]):
        if(gray_image.shape[0] > np.int32(points[1][i]) and np.int32(points[1][i]) > roi_height and gray_image.shape[1] > np.int32(points[0][i])):
            gray_image[np.int32(points[1][i]), np.int32(points[0][i]),0] = np.float32(color[i])
            valid_mask[np.int32(points[1][i]), np.int32(points[0][i])] = True
            
    lidar_gray = gray_image[:,:,0]
    coords = np.array(np.nonzero(valid_mask)).T    
    values = lidar_gray[valid_mask]
    
    x = [i for i in range(roi_height, lidar_gray.shape[0])]
    y = [i for i in range(0, lidar_gray.shape[1])]
    grid_x,grid_y = np.meshgrid(x,y)
    
    from scipy.interpolate import LinearNDInterpolator, griddata
    it = griddata(coords, values, (grid_x, grid_y), method=inter_mode, fill_value=0)
    
    interpol_lidar_gray = it.T
        
    return lidar_gray[roi_height:], interpol_lidar_gray

def return_roi_preprocessing_interpolation_reflection_image(points, color, image, inter_mode, roi_height):
    #mode = ['nearest', 'linear', 'cubic']
    # nearest : NearestNDInterpolator 
    # linear : LinearNDInterpolator
    # cubic : CloughTocher2DInterpolator  
    
    gray_image = np.zeros_like(image)
    valid_mask = np.zeros((gray_image.shape[0], gray_image.shape[1]), dtype=bool)
    
    for i in range(points.shape[1]):
        if(gray_image.shape[0] > np.int32(points[1][i]) and np.int32(points[1][i]) > roi_height and gray_image.shape[1] > np.int32(points[0][i])):
            gray_image[np.int32(points[1][i]), np.int32(points[0][i]),0] = np.float32(color[i])
            valid_mask[np.int32(points[1][i]), np.int32(points[0][i])] = True
            
    lidar_gray = gray_image[:,:,0]
    coords = np.array(np.nonzero(valid_mask)).T    
    values = lidar_gray[valid_mask]
    
    x = [i for i in range(roi_height, lidar_gray.shape[0])]
    y = [i for i in range(0, lidar_gray.shape[1])]
    grid_x,grid_y = np.meshgrid(x,y)
    
    from scipy.interpolate import LinearNDInterpolator, griddata
    it = griddata(coords, values, (grid_x, grid_y), method=inter_mode, fill_value=0)
    
    interpol_lidar_gray = it.T
        
    return interpol_lidar_gray

def return_roi_preprocessing_reflection_image(points, color, image, roi_height):
    gray_image = np.zeros_like(image)
    
    for i in range(points.shape[1]):
        if(gray_image.shape[0] > np.int32(points[1][i]) and np.int32(points[1][i]) > roi_height and gray_image.shape[1] > np.int32(points[0][i])):
            gray_image[np.int32(points[1][i]), np.int32(points[0][i]),0] = np.float32(color[i])
            
    lidar_gray = gray_image[:,:,0]
        
    return lidar_gray[roi_height:]

def return_roi_preprocessing_range_image(points, color, image, roi_height):
    gray_image = np.zeros_like(image)
    
    for i in range(points.shape[1]):
        if(gray_image.shape[0] > np.int32(points[1][i]) and np.int32(points[1][i]) > roi_height and gray_image.shape[1] > np.int32(points[0][i])):
            gray_image[np.int32(points[1][i]), np.int32(points[0][i]),0] = np.float32(color[i])
            
    lidar_gray = gray_image[:,:,0]
        
    return lidar_gray[roi_height:]

def return_roi_color_image(colorImage, roi_height):
    org_roi_color = colorImage[roi_height:,:,:]

    return org_roi_color

def image_path2velo_path(image_path, lr_mode):
    velo_path = image_path.replace(".png", ".bin")
    velo_path = velo_path.replace('/image_' + lr_mode + '/', "/velodyne_points/")
    return velo_path


def preprocessing_total(image_path, v2c_filepath, c2c_filepath, lr_mode):
    velo_path = image_path2velo_path(image_path, lr_mode)

    image = cv2.imread(image_path)
    velo_points = load_from_bin(velo_path)

    min_height_coodi = 151

    ansR, rangeL = velo3d_2_camera2d_points(velo_points, v_fov=(-24.9, 2.0), h_fov=(-45,45), vc_path=v2c_filepath, cc_path=c2c_filepath, mode=lr_mode)

    ans, reflectL = velo3d_2_camera2d_points2(velo_points, vc_path=v2c_filepath, cc_path=c2c_filepath, mode=lr_mode)
    #reflect_mapping_image = print_projection_plt(points=ans, color=reflectL, image=image)

    roi_reflect_sparse = return_roi_preprocessing_reflection_image(points=ans, color=reflectL, image=image, roi_height=min_height_coodi)
    roi_range_sparse = return_roi_preprocessing_range_image(points=ansR, color=rangeL, image=image, roi_height=min_height_coodi)
    roi_org_color = return_roi_color_image(image, roi_height=min_height_coodi)

    return roi_org_color, roi_reflect_sparse, roi_range_sparse


def get_image_name(image_path):
    image_namess = image_path.split('/')
    image_names = image_namess[len(image_namess)-1]
    image_name = image_names.split('.')
    save_image_name = image_name[0]
    
    return save_image_name

def make_save_folder(image_root_folder, lr_mode):
    if not os.path.exists('./datasets'):
        os.mkdir('./datasets')
    folder_path = './datasets'

    if not os.path.exists(folder_path+'/'+image_root_folder):
            os.mkdir(folder_path+'/'+image_root_folder)
    folder_path += '/'+image_root_folder

    if not os.path.exists(folder_path+'/image_' + lr_mode):
            os.mkdir(folder_path+'/image_' + lr_mode)
    folder_path += '/image_' + lr_mode

    return folder_path


def make_save_sub_folder(folder_path, inter_mode):
    save_sub_folder_path = folder_path

    if not os.path.exists(save_sub_folder_path+'/color'):
        os.mkdir(save_sub_folder_path+'/color')

    save_sub_color_path = save_sub_folder_path+'/color'
    print(save_sub_color_path)

    if not os.path.exists(save_sub_folder_path+'/sparse_reflect'):
        os.mkdir(save_sub_folder_path+'/sparse_reflect')

    save_sub_sparse_reflect_path = save_sub_folder_path+'/sparse_reflect'
    print(save_sub_sparse_reflect_path)

    if not os.path.exists(save_sub_folder_path+'/sparse_range'):
        os.mkdir(save_sub_folder_path+'/sparse_range')

    save_sub_sparse_range_path = save_sub_folder_path+'/sparse_range'
    print(save_sub_sparse_range_path)
    
    return save_sub_color_path, save_sub_sparse_reflect_path, save_sub_sparse_range_path

def return_final_roi_image(input_image):
    size_img = np.array(input_image)
    #print(size_img.shape)
    #print(len(size_img.shape))
    image_height = size_img.shape[0]
    half_w = size_img.shape[1] // 2

    if(len(size_img.shape) == 3): 
        output_image = size_img[(image_height-216):, (half_w-612):(half_w+612),:]
        return output_image

    output_image = size_img[(image_height-216):, (half_w-612):(half_w+612)]
    return output_image
