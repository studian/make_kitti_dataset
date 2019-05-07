import numpy as np
from collections import Counter
from path import Path
from skimage.io import imread, imsave
from tqdm import tqdm
import glob
import os, cv2
from data_utils import make_save_folder, make_save_sub_folder, get_image_name, make_sub_folder, get_filepath_list_pair, image_path2velo_path
import argparse

def load_velodyne_XYZ_points(file_name):
    # adapted from https://github.com/hunse/kitti
    points = np.fromfile(file_name, dtype=np.float32).reshape(-1, 4)
    points[:,3] = 1
    
    return points

def load_velodyne_XYZ_Intensity_points(file_name):
    # adapted from https://github.com/hunse/kitti
    points = np.fromfile(file_name, dtype=np.float32).reshape(-1, 4)
    points2 = np.array(points, dtype=np.float32)
    
    points[:,3] = 1
    
    return points, points2

def read_calib_file(path):
    # taken from https://github.com/hunse/kitti
    float_chars = set("0123456789.e+- ")
    data = {}
    with open(path, 'r') as f:
        for line in f.readlines():
            key, value = line.split(':', 1)
            value = value.strip()
            data[key] = value
            if float_chars.issuperset(value):
                # try to cast to float array
                try:
                    data[key] = np.array(list(map(float, value.split(' '))))
                except ValueError:
                    # casting error: data[key] already eq. value, so pass
                    pass

    return data

def sub2ind(matrixSize, rowSub, colSub):
    m, n = matrixSize
    return rowSub * (n-1) + colSub - 1


def generate_depth_map(calib_dir, velo_file_name, im_shape, cam=2):
    # load calibration files
    cam2cam = read_calib_file(calib_dir+'/calib_cam_to_cam.txt')
    velo2cam = read_calib_file(calib_dir+'/calib_velo_to_cam.txt')
    velo2cam = np.hstack((velo2cam['R'].reshape(3,3), velo2cam['T'][..., np.newaxis]))
    velo2cam = np.vstack((velo2cam, np.array([0, 0, 0, 1.0])))

    # compute projection matrix velodyne->image plane
    R_cam2rect = np.eye(4)
    R_cam2rect[:3,:3] = cam2cam['R_rect_00'].reshape(3,3)
    P_rect = cam2cam['P_rect_0'+str(cam)].reshape(3,4)
    P_velo2im = np.dot(np.dot(P_rect, R_cam2rect), velo2cam)

    # load velodyne points and remove all behind image plane (approximation)
    # each row of the velodyne data is forward, left, up, reflectance
    velo = load_velodyne_XYZ_points(velo_file_name)    
    velo = velo[velo[:, 0] >= 0, :]

    # project the points to the camera
    velo_pts_im = np.dot(P_velo2im, velo.T).T
    velo_pts_im[:, :2] = velo_pts_im[:,:2] / velo_pts_im[:,-1:]
    
    # check if in bounds
    # use minus 1 to get the exact same value as KITTI matlab code
    velo_pts_im[:, 0] = np.round(velo_pts_im[:,0]) - 1
    velo_pts_im[:, 1] = np.round(velo_pts_im[:,1]) - 1
    val_inds = (velo_pts_im[:, 0] >= 0) & (velo_pts_im[:, 1] >= 0)
    val_inds = val_inds & (velo_pts_im[:,0] < im_shape[1]) & (velo_pts_im[:,1] < im_shape[0])
    velo_pts_im = velo_pts_im[val_inds, :]

    # project to image
    depth = np.zeros((im_shape))
    depth[velo_pts_im[:, 1].astype(np.int), velo_pts_im[:, 0].astype(np.int)] = velo_pts_im[:, 2]

    # find the duplicate points and choose the closest depth
    inds = sub2ind(depth.shape, velo_pts_im[:, 1], velo_pts_im[:, 0])
    dupe_inds = [item for item, count in Counter(inds).items() if count > 1]
    for dd in dupe_inds:
        pts = np.where(inds == dd)[0]
        x_loc = int(velo_pts_im[pts[0], 0])
        y_loc = int(velo_pts_im[pts[0], 1])
        depth[y_loc, x_loc] = velo_pts_im[pts, 2].min()
    depth[depth < 0] = 0
    return depth

def generate_intensity_map(calib_dir, velo_file_name, im_shape, cam=2):
    # load calibration files
    cam2cam = read_calib_file(calib_dir+'/calib_cam_to_cam.txt')
    velo2cam = read_calib_file(calib_dir+'/calib_velo_to_cam.txt')
    velo2cam = np.hstack((velo2cam['R'].reshape(3,3), velo2cam['T'][..., np.newaxis]))
    velo2cam = np.vstack((velo2cam, np.array([0, 0, 0, 1.0])))

    # compute projection matrix velodyne->image plane
    R_cam2rect = np.eye(4)
    R_cam2rect[:3,:3] = cam2cam['R_rect_00'].reshape(3,3)
    P_rect = cam2cam['P_rect_0'+str(cam)].reshape(3,4)
    P_velo2im = np.dot(np.dot(P_rect, R_cam2rect), velo2cam)

    # load velodyne points and remove all behind image plane (approximation)
    # each row of the velodyne data is forward, left, up, reflectance
    velo, velo2 = load_velodyne_XYZ_Intensity_points(velo_file_name)
    
    velo = velo[velo[:, 0] >= 0, :]
    velo2 = velo2[velo2[:, 0] >= 0, :]

    # project the points to the camera
    velo_pts_im = np.dot(P_velo2im, velo.T).T
    velo_pts_im[:, :2] = velo_pts_im[:,:2] / velo_pts_im[:,-1:]
    temp =  velo_pts_im
    temp[:,2] = velo2[:,3].T
    
    # check if in bounds
    # use minus 1 to get the exact same value as KITTI matlab code
    velo_pts_im[:, 0] = np.round(velo_pts_im[:,0]) - 1
    velo_pts_im[:, 1] = np.round(velo_pts_im[:,1]) - 1
    val_inds = (velo_pts_im[:, 0] >= 0) & (velo_pts_im[:, 1] >= 0)
    val_inds = val_inds & (velo_pts_im[:,0] < im_shape[1]) & (velo_pts_im[:,1] < im_shape[0])
    velo_pts_im = velo_pts_im[val_inds, :]

    # project to image
    intensity = np.zeros((im_shape))
    intensity[velo_pts_im[:, 1].astype(np.int), velo_pts_im[:, 0].astype(np.int)] = velo_pts_im[:, 2]

    # find the duplicate points and choose the closest depth
    inds = sub2ind(intensity.shape, velo_pts_im[:, 1], velo_pts_im[:, 0])
    dupe_inds = [item for item, count in Counter(inds).items() if count > 1]
    for dd in dupe_inds:
        pts = np.where(inds == dd)[0]
        x_loc = int(velo_pts_im[pts[0], 0])
        y_loc = int(velo_pts_im[pts[0], 1])
        intensity[y_loc, x_loc] = velo_pts_im[pts, 2].min()
    intensity[intensity < 0] = 0
    return intensity

def return_preprocessing_interpolation_reflection_image(load_reflect, inter_mode):
    gray_image = np.zeros_like(load_reflect)
    valid_mask = np.zeros((gray_image.shape[0], gray_image.shape[1]), dtype=bool)
    #gray_image[np.int32(points[1][i]), np.int32(points[0][i]),0] = np.float32(color[i])
    valid_mask[load_reflect[:,:]>0] = True
    load_reflect_copy = load_reflect.copy()

    lidar_gray = gray_image[:,:]
    coords = np.array(np.nonzero(valid_mask)).T    
    values = load_reflect_copy[valid_mask]
    print(coords)
    print(values)

    h = [i for i in range(0, lidar_gray.shape[0])]
    w = [i for i in range(0, lidar_gray.shape[1])]
    grid_x, grid_y = np.meshgrid(h,w)

    from scipy.interpolate import LinearNDInterpolator, griddata
    #mode = ['nearest', 'linear', 'cubic']
    # nearest : NearestNDInterpolator 
    # linear : LinearNDInterpolator
    # cubic : CloughTocher2DInterpolator
    it = griddata(coords, values, (grid_x, grid_y), method=inter_mode, fill_value=0)

    interpol_lidar_gray = it.T
    return interpol_lidar_gray


def main(first_sub_num, save_main_path):
    #cam_modes = [2, 3]
    cam_modes = [2]
    #roi_height = 151
    roi_height = 224
    roi_width2 = 592

    main_path = './shadow_dataset'
    save_main_path = make_sub_folder('.', save_main_path)

    subpath1 = get_filepath_list_pair(main_path)
    cnt1 = first_sub_num

    full_sub_path1 = main_path+'/'+subpath1[cnt1]
    save_sub_path1 = make_sub_folder(save_main_path, subpath1[cnt1])    
    print(full_sub_path1)
    print(save_sub_path1)

    calib_dir = full_sub_path1 #current_path + 'dataset/'
    
    sub_path2 = get_filepath_list_pair(full_sub_path1)
    for cnt2 in range(len(sub_path2)):
        full_sub_path2 = main_path+'/'+subpath1[cnt1]+'/'+sub_path2[cnt2]
        save_sub_path2 = make_sub_folder(save_sub_path1, sub_path2[cnt2])
        print(full_sub_path2)
        print(save_sub_path2)

        current_path = './'

        velodyne_list = glob.glob(full_sub_path2 + '/velodyne_points/data/*.bin')
        velodyne_list.sort()

        for velodyne_path in tqdm(velodyne_list):
            file_name_split1 = velodyne_path.split('/')
            file_name = file_name_split1[len(file_name_split1)-1].split('.')[0]
    
            for cam_mode in cam_modes:
                image_folder_name = 'image_0' + str(cam_mode)
                folder_path = make_sub_folder(save_sub_path2, image_folder_name)
                save_color_sub_path = make_sub_folder(folder_path, 'color')
                save_reflect_sparse_sub_path = make_sub_folder(folder_path, 'reflect_sparse')
                save_range_sparse_sub_path = make_sub_folder(folder_path, 'range_sparse')

                save_reflect_linear_sub_path = make_sub_folder(folder_path, 'interpol_reflect_linear')
                save_reflect_nearest_sub_path = make_sub_folder(folder_path, 'interpol_reflect_nearest')

                open_color_image_path = velodyne_path.replace('/velodyne_points/', '/'+image_folder_name+'/')
                open_color_image_path = open_color_image_path.replace('.bin', '.png')

                if(os.path.exists(velodyne_path) and os.path.exists(open_color_image_path)):
                    open_color_image = cv2.imread(open_color_image_path)
                    height, width, channel = open_color_image.shape

                    save_image_name = get_image_name(open_color_image_path) 
                    save_color_image_path = save_color_sub_path+'/'+save_image_name+'.png'
                    save_reflect_sparse_image_path = save_reflect_sparse_sub_path+'/'+save_image_name+'.png'      
                    save_range_sparse_image_path = save_range_sparse_sub_path+'/'+save_image_name+'.png' 

                    save_reflect_linear_image_path = save_reflect_linear_sub_path+'/'+save_image_name+'.png' 
                    save_reflect_nearest_image_path = save_reflect_nearest_sub_path+'/'+save_image_name+'.png' 

                    depth_image = generate_depth_map(calib_dir, velodyne_path, open_color_image.shape[:2], cam=cam_mode)
                    intensity_image = generate_intensity_map(calib_dir, velodyne_path, open_color_image.shape[:2], cam=cam_mode)
      
                    width_2 = int(width/2)

                    final_color_image = open_color_image[-roi_height:, width_2-roi_width2:width_2+roi_width2,:]
                    final_depth_image = depth_image[-roi_height:, width_2-roi_width2:width_2+roi_width2]
                    final_intensity_image = intensity_image[-roi_height:, width_2-roi_width2:width_2+roi_width2]

                    linear_reflect = return_preprocessing_interpolation_reflection_image(load_reflect=final_intensity_image[:,:]*255, inter_mode='linear')
                    linear_reflect_resized = cv2.resize(linear_reflect, (roi_width2, 112), interpolation = cv2.INTER_AREA)

                    nearest_reflect = return_preprocessing_interpolation_reflection_image(load_reflect=final_intensity_image[:,:]*255, inter_mode='nearest')
                    nearest_reflect_resized = cv2.resize(nearest_reflect, (roi_width2, 112), interpolation = cv2.INTER_AREA)
                    #print(final_intensity_image.shape)

                    reflection_resized = np.zeros((112, 592), dtype='f')
                    range_resized = np.zeros((112, 592), dtype='f')
                    color_resized = np.zeros((112, 592, 3), dtype='f')
                    #color_resized = cv2.resize(final_color_image, (roi_width2, 112), interpolation = cv2.INTER_AREA)

                    for h in range(0, 224, 2):
                        for w in range(0, 1184, 2):
                            reflection_resized[int(h/2), int(w/2)] = final_intensity_image[h, w]*255
                            range_resized[int(h/2), int(w/2)] = final_depth_image[h, w]/80.*255
                            color_resized[int(h/2), int(w/2), :] = final_color_image[h, w, :]


                    #reflection_resized = cv2.resize(final_intensity_image, (roi_width2, 112), interpolation = cv2.INTER_AREA)
                    #range_resized = cv2.resize(final_depth_image, (roi_width2, 112), interpolation = cv2.INTER_AREA)
        
                    cv2.imwrite(save_color_image_path, color_resized)
                    cv2.imwrite(save_range_sparse_image_path, range_resized)
                    cv2.imwrite(save_reflect_sparse_image_path, reflection_resized)

                    cv2.imwrite(save_reflect_linear_image_path, linear_reflect_resized)
                    cv2.imwrite(save_reflect_nearest_image_path, nearest_reflect_resized)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--first_sub_num")
    parser.add_argument("--save_main_path")


    args = parser.parse_args()
    first_sub_num = int(args.first_sub_num)
    save_main_path = args.save_main_path

    main(first_sub_num, save_main_path)
    print('Finished......Saving complete......')


