import os
import glob
import json
import imageio
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
#from cv2.reg import MapperGradSimilar, MapperGradAffine
#import cv2

import itk

from tqdm import tqdm
from scipy.stats import skew, kurtosis
from scipy.optimize import curve_fit, fsolve

#%%
labels = {0:  "Background",
          1:  "Skeletal Muscle",
          2:  "Visceral Fat",
          3:  "Subcutaneous Fat"}

more_labels = {0:  "Background",
                        1:  "Visceral Fat",
                        2:  "Subcutaneous Fat",
                        3:  "Dorsal Muscle",
                        4:  "Shoulder Muscle",
                        5:  "Intercostal Muscle",
                        6:  "Pectoral Muscle"}

#imagedb sav folder
imagedb_folder = ""
imagedb_images = pd.read_spss(imagedb_folder+"imagedb.images.sav")

cutoff_threshold = -950
histogram_bins = 50
histogram_min_hu = -250
histogram_max_hu = 250

def gaussian(x, a, x0, sigma):
    return a*np.exp(-((x-x0)**2)/(2*sigma**2))

def configured_gaussian(a, x0, sigma):
    return lambda x : a*np.exp(-((x-x0)**2)/(2*sigma**2))

def findIntersection(func1, func2, x0):
    return fsolve(lambda x: func1(x) - func2(x), x0)

def get_linked_IDs(link_file_folder):
    with open(os.path.join(link_file_folder, "linked_IDs.json"), 'r') as json_file:
        linked_IDs = json.load(json_file)
    return linked_IDs

def recode_seg(orig_seg):
    recoded_seg = np.zeros(orig_seg.shape,dtype=np.uint8)
    #recode the tissue labels to match S-EFOV. Originally: 0:Background,1:Skeletal Muscle,2:Visceral Fat,3:Subcutaneous Fat
    for orig_label,new_label in enumerate([0,2,4,1]): 
        recoded_seg[np.where(orig_seg == orig_label)] = new_label
    return recoded_seg

def rescale_intensities(img, min_value=0, max_value=255):
    #print(img.min(), img.max())
    shift = min_value + 1000
    img = img + shift
    scale = max_value / 1500
    img = img * scale
    img[img<0] = 0
    img[img>255] = 255
    return img.astype(np.uint8)
    
def register_images(moving_image, fixed_image, seg):
    #Will attempt to register moving_image on top of fixed_image
    #Will then apply the same transformation as applied to img0 to seg
    parameter_object = itk.ParameterObject.New()
    #parameter_map = parameter_object.GetDefaultParameterMap('bspline', 3, 20.0)
    parameter_map = parameter_object.GetDefaultParameterMap('rigid', 3)
    parameter_object.AddParameterMap(parameter_map)
    parameter_object.SetParameter("DefaultPixelValue", "-1000")
    parameter_object.SetParameter("Transform", "SimilarityTransform")
    
    result_image, result_transform_parameters = itk.elastix_registration_method(fixed_image, moving_image,
                                                                                parameter_object=parameter_object,
                                                                                log_to_console=False)
    #apply the same transformation to the segmentation
    result_transform_parameters.SetParameter("DefaultPixelValue", "0")
    result_transform_parameters.SetParameter("FinalBSplineInterpolationOrder", "0")
    registered_seg = itk.transformix_filter(seg,
                                            result_transform_parameters)
    
    return result_image, registered_seg

    
def process_segmentations(segmentation_folder, linked_IDs, ct_folder, debug_path=None):  
    
    areas = pd.DataFrame(data = 0, index = range(len(linked_IDs)), 
                         columns = ["NMS"])
    num_patients = len(linked_IDs)
    
    histograms = np.zeros((num_patients, 2, len(more_labels), histogram_bins))
    bin_edges = np.zeros((num_patients, 2, len(more_labels), histogram_bins+1))
    bin_centers = np.zeros((num_patients, 2, len(more_labels), histogram_bins))
    
    segmentation_info = []
    for i in [0,1]:
        with open(os.path.join(segmentation_folder, str(i), "segmentation_info.json"), 'r') as json_file:
            segmentation_info.append(json.load(json_file))
    
    for i,NMS in enumerate(linked_IDs):
        print(f"Calculating areas: {i+1}/{num_patients}")
        areas.at[i, "NMS"] = NMS
        seriesUIDs = linked_IDs[NMS]
        segs = []
        ct_imgs = []
        areas_of_voxel = []
        #Read both scan moments and store them
        for j,seriesUID in enumerate(seriesUIDs):
            #determine voxel size
            voxel_size_x = imagedb_images["voxel_size_x"][imagedb_images["series_uid"] == seriesUID].values[0]            
            areas_of_voxel.append(voxel_size_x**2)
            
            #load segmentation
            segmentation_path = os.path.join(segmentation_folder,str(j),seriesUID+".tif")
            #seg = imageio.imread(segmentation_path)
            seg = itk.imread(segmentation_path, itk.F)
            segs.append(seg)
            
            
            #Load the CT images and pick the correct slice for later registration
            seg_slice_idx = segmentation_info[j][seriesUID]["index"]
            
            ct_path = os.path.join(ct_folder, str(j), seriesUID+".tif")
            #ct_slice = imageio.volread(ct_path)[seg_slice_idx].squeeze()
            ct_img = itk.imread(ct_path, itk.F)#
            #ct_img.SetSpacing([voxel_size_x, voxel_size_x, voxel_size_x])
            ct_img = ct_img[seg_slice_idx-2:seg_slice_idx+3]
            ct_imgs.append(ct_img)
                                               
            
        
        #Find the image with the smallest voxel size, and register that image over top of the other one
        if(areas_of_voxel[0] < areas_of_voxel[1]):
            fixed_img = ct_imgs[1]
            moving_img = ct_imgs[0]
            seg = segs[0]
            #seg.SetSpacing([math.sqrt(areas_of_voxel[0]), math.sqrt(areas_of_voxel[0])])
            segidx = 0
            areas_of_voxel[0] = areas_of_voxel[1]
        else:
            fixed_img = ct_imgs[0]
            moving_img = ct_imgs[1]
            seg = segs[1]
            #seg.SetSpacing([math.sqrt(areas_of_voxel[1]), math.sqrt(areas_of_voxel[1])])
            segidx = 1
            areas_of_voxel[1] = areas_of_voxel[0]    
        
        #register each of the five slices seperately and overwrite the unregistered slice
        for j in range(5):     
            reg_ct_slice, reg_seg_slice = register_images(moving_img[j], fixed_img[j], seg[j])
            segs[segidx][j] = reg_seg_slice
            ct_imgs[segidx][j] = reg_ct_slice
        
        
        if debug_path!=None:
            plt.imsave(os.path.join(debug_path, f"{NMS}_{seriesUIDs[0]}_0_ct.png"), ct_imgs[0][2], cmap="gray")
            plt.imsave(os.path.join(debug_path, f"{NMS}_{seriesUIDs[1]}_1_ct.png"), ct_imgs[1][2], cmap="gray")
            plt.imsave(os.path.join(debug_path, f"{NMS}_{seriesUIDs[0]}_0_seg_pre.png"), recode_seg(segs[0][2]), cmap="gray")
            plt.imsave(os.path.join(debug_path, f"{NMS}_{seriesUIDs[1]}_1_seg_pre.png"), recode_seg(segs[1][2]), cmap="gray")
        
        #Set all background voxels outside of the FOV of the smaller segmentation to zero in the larger segmentation, and then follow it up with the smaller segmentation.
        if segidx == 0:
            segs[1][ct_imgs[0] <= cutoff_threshold] = 0 #larger segmentation
            segs[0][ct_imgs[1] <= cutoff_threshold] = 0 #smaller segmentation
        else:
            segs[0][ct_imgs[1] <= cutoff_threshold] = 0 #larger segmentation
            segs[1][ct_imgs[0] <= cutoff_threshold] = 0 #smaller segmentation
            
        if debug_path!=None:
            plt.imsave(os.path.join(debug_path, f"{NMS}_{seriesUIDs[0]}_0_seg_post.png"), recode_seg(segs[0][2]), cmap="gray")
            plt.imsave(os.path.join(debug_path, f"{NMS}_{seriesUIDs[1]}_1_seg_post.png"), recode_seg(segs[1][2]), cmap="gray")
                    
        if(np.array(segs)[0].max() > 3):
            seg_labels = more_labels
        else:
            seg_labels = labels
        
        #process the segmentations and scans to obtain statistics.
        for j,seriesUID in enumerate(seriesUIDs):
            #convert ITK images to numpy arrays
            seg = np.array(segs[j])
            seg = np.rint(seg).astype(np.int8)
            
            ct_slices = np.array(ct_imgs[j])
            
            #save debug images
            # if i%50==0 and debug_path!=None:
            #     plt.imsave(os.path.join(debug_path, f"{NMS} {j}.png"), seg[2], cmap="plasma")
            
            #for each tissue type calculate stats
            for k in range(len(seg_labels)):
                area = (seg == k).sum()*areas_of_voxel[j]
                
                #store area in dataframe
                areas.at[i, seg_labels[k]+" "+str(j)] = area
                
                #determine radiodensity metrics
                tissue = ct_slices[seg==k]
                areas.at[i, f"mean HU {seg_labels[k]} {j}"] = tissue.mean()
                areas.at[i, f"SD HU {seg_labels[k]} {j}"] = np.std(tissue)
                areas.at[i, f"min HU {seg_labels[k]} {j}"] = tissue.min()
                areas.at[i, f"10th percentile HU {seg_labels[k]} {j}"] = np.percentile(tissue, 10)
                areas.at[i, f"25th percentile HU {seg_labels[k]} {j}"] = np.percentile(tissue, 25)
                areas.at[i, f"50th percentile HU {seg_labels[k]} {j}"] = np.percentile(tissue, 50)
                areas.at[i, f"75th percentile HU {seg_labels[k]} {j}"] = np.percentile(tissue, 75)
                areas.at[i, f"90th percentile HU {seg_labels[k]} {j}"] = np.percentile(tissue, 90)
                areas.at[i, f"max HU {seg_labels[k]} {j}"] = tissue.max()
                areas.at[i, f"skewness HU {seg_labels[k]} {j}"] = skew(tissue, axis=None)
                areas.at[i, f"kurtosis HU {seg_labels[k]} {j}"] = kurtosis(tissue, axis=None)
                
                #calculate histogram
                histogram, bins = np.histogram(np.clip(tissue, histogram_min_hu, histogram_max_hu), bins=histogram_bins, density=False)
                histograms[i,j,k] = histogram
                bin_edges[i,j,k] = bins
                bin_centers[i,j,k] = np.mean(np.vstack([bins[:-1], bins[1:]]), axis=0)
            
            #post-process histograms
            if(len(seg_labels)>len(labels)):
                fat_idx = 2 #more labels
                muscle_idx = [3,4,5,6]
            else:
                fat_idx = 3
                muscle_idx = [1]
                
            #Fit a gaussian to the fat distribution
            x_data = bin_centers[i,j,fat_idx]
            y_data = histograms[i,j,fat_idx]
            ampl = np.max(y_data)
            mean = np.mean(x_data)
            sigma = np.std(x_data)
            parameters, covariance = curve_fit(gaussian, x_data, y_data, p0=[ampl,mean,sigma])
            #configure gaussian for intersection calculation
            fat_gaussian = configured_gaussian(parameters[0], parameters[1], parameters[2])

            for k in muscle_idx:
                #Fit a gaussian to the muscle distribution
                x_data = bin_centers[i,j,k]
                y_data = histograms[i,j,k]
                ampl = np.max(y_data)
                mean = np.mean(x_data)
                sigma = np.std(x_data)
                parameters, covariance = curve_fit(gaussian, x_data, y_data, p0=[ampl,mean,sigma])
                #configure gaussian for intersection calculation
                muscle_gaussian = configured_gaussian(parameters[0], parameters[1], parameters[2])
                
                #Calculate threshold (below which will be considered fat)
                threshold = parameters[1]-2*parameters[2]
                intersection = findIntersection(fat_gaussian, muscle_gaussian, 0.0)
                if intersection > threshold:
                    threshold = intersection
                
                #get fat tissue area from CT slices
                tissue = ct_slices[seg==k]
                area = (tissue<threshold).sum()*areas_of_voxel[j]
                areas.at[i, f"Fat area in {seg_labels[k]} {j}"] = area
                
    return areas, seg_labels, histograms, bin_edges, bin_centers

def calculate_area_difference(areas, linked_IDs, seg_labels):
    num_patients = len(linked_IDs)
    for i,NMS in enumerate(linked_IDs):
        print(f"Processing differences: {i+1}/{num_patients}")
        
        for j in range(len(seg_labels)):
            label = seg_labels[j]
            
            #calculate absolute difference
            areas_0 = areas.loc[:, label+" 0"].to_numpy()
            areas_1 = areas.loc[:, label+" 1"].to_numpy()
            area_abs_diff = np.absolute(areas_0 - areas_1)
            
            areas.loc[:, label+" abs diff"] = area_abs_diff
            
            #calculate percentage difference
            area_perc_diff = (area_abs_diff/areas_0)*100
            
            areas.loc[:, label+" perc diff"] = area_perc_diff
    return areas
#%%
ct_folder = ""
segmentation_folder = ""
output_folder = segmentation_folder

linked_IDs = get_linked_IDs(segmentation_folder)

#%%
areas, seg_labels, histograms, bin_edges, bin_centers = process_segmentations(segmentation_folder, linked_IDs, ct_folder, debug_path="D:\\registration_debug")


#%%
areas = calculate_area_difference(areas, linked_IDs, seg_labels)

#%% Save to excel sheet
areas.to_excel(os.path.join(output_folder,"registered_fov_compensated_tissue_areas_updated.xlsx"))

#%%Save histogram arrays
np.save(os.path.join(output_folder,"registered_fov_compensated_bin_counts.npy"), histograms)
np.save(os.path.join(output_folder,"registered_fov_compensated_bin_edges.npy"), bin_edges)
np.save(os.path.join(output_folder,"registered_fov_compensated_bin_centers.npy"), bin_centers)


#%% Plot a histogram
histograms = np.load(os.path.join(output_folder,"registered_fov_compensated_bin_counts.npy"))
bin_edges = np.load(os.path.join(output_folder,"registered_fov_compensated_bin_edges.npy"))
bin_centers = np.load(os.path.join(output_folder,"registered_fov_compensated_bin_centers.npy"))
title = ""

patient_num = 280
gaussians  = {}
plt.figure()

if len(seg_labels) == 4:
    fat_idx = 1
    tissue_idxs = [1,3]
else:
    fat_idx = 2
    tissue_idxs = [fat_idx,4,6]

for tissue_num in tissue_idxs:
    bar_width = bin_edges[patient_num,0,tissue_num,1]-bin_edges[patient_num,0,tissue_num,0]
    x_data = bin_centers[patient_num,0,tissue_num]
    # x_data = bin_edges[patient_num,0,tissue_num,1:]
    y_data = histograms[patient_num,0,tissue_num]
    
    #Fit a gaussian
    ampl = np.max(y_data)
    mean = np.mean(x_data)
    sigma = np.std(x_data)
    parameters, covariance = curve_fit(gaussian, x_data, y_data, p0=[ampl,mean,sigma])
    plt.bar(x_data, y_data, width=bar_width, label = seg_labels[tissue_num])
    
    gaussians[tissue_num] = configured_gaussian(parameters[0], parameters[1], parameters[2])
    y_fit = gaussian(x_data, parameters[0], parameters[1], parameters[2])
    plt.plot(x_data, y_fit, "r-")
    
    if(tissue_num == fat_idx):    
        threshold = parameters[1]+2*parameters[2]
    else:
        threshold = parameters[1]-2*parameters[2]
        intersection = findIntersection(gaussians[fat_idx], gaussians[tissue_num], 0.0)
        if intersection > threshold:
            threshold = intersection
    title = title+f"Estimated optimal {seg_labels[tissue_num]} threshold: {threshold}\n"
    plt.axvline(threshold, color="b")

plt.legend()
plt.title(title)
plt.show()


















