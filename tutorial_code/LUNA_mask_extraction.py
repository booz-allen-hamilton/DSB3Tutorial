import SimpleITK as sitk
import numpy as np
import csv
from glob import glob
import pandas as pd

#Some helper functions

def make_mask(center,diam,z,width,height,spacing,origin):
    '''
Center : centers of circles px -- list of coordinates x,y,z
diam : diameters of circles px -- diameter
widthXheight : pixel dim of image
spacing = mm/px conversion rate np array x,y,z
origin = x,y,z mm np.array
z = z position of slice in world coordinates mm
    '''
    mask = np.zeros([height,width]) # 0's everywhere except nodule swapping x,y to match img
    #convert to nodule space from world coordinates

    # Defining the voxel range in which the nodule falls
    v_center = (center-origin)/spacing
    v_diam = int(diam/spacing[0]+5)
    v_xmin = np.max([0,int(v_center[0]-v_diam)-5])
    v_xmax = np.min([width-1,int(v_center[0]+v_diam)+5])
    v_ymin = np.max([0,int(v_center[1]-v_diam)-5]) 
    v_ymax = np.min([height-1,int(v_center[1]+v_diam)+5])

    v_xrange = range(v_xmin,v_xmax+1)
    v_yrange = range(v_ymin,v_ymax+1)

    # Convert back to world coordinates for distance calculation
    x_data = [x*spacing[0]+origin[0] for x in range(width)]
    y_data = [x*spacing[1]+origin[1] for x in range(height)]

    # Fill in 1 within sphere around nodule
    for v_x in v_xrange:
        for v_y in v_yrange:
            p_x = spacing[0]*v_x + origin[0]
            p_y = spacing[1]*v_y + origin[1]
            if np.linalg.norm(center-np.array([p_x,p_y,z]))<=diam:
                mask[int((p_y-origin[1])/spacing[1]),int((p_x-origin[0])/spacing[0])] = 1.0
    return(mask)

def matrix2int16(matrix):
    ''' 
matrix must be a numpy array NXN
Returns uint16 version
    '''
    m_min= np.min(matrix)
    m_max= np.max(matrix)
    matrix = matrix-m_min
    return(np.array(np.rint( (matrix-m_min)/float(m_max-m_min) * 65535.0),dtype=np.uint16))

############
#
# Getting list of image files
luna_path = "/home/jonathan/LUNA2016/"
luna_subset_path = luna_path+"subset1/"
output_path = "/home/jonathan/tutorial/"
file_list=glob(luna_subset_path+"*.mhd")


#####################
#
# Helper function to get rows in data frame associated 
# with each file
def get_filename(case):
    global file_list
    for f in file_list:
        if case in f:
            return(f)
#
# The locations of the nodes
df_node = pd.read_csv(luna_path+"annotations.csv")
df_node["file"] = df_node["seriesuid"].apply(get_filename)
df_node = df_node.dropna()

#####
#
# Looping over the image files
#
fcount = 0
for img_file in file_list:
    print "Getting mask for image file %s" % img_file.replace(luna_subset_path,"")
    mini_df = df_node[df_node["file"]==img_file] #get all nodules associate with file
    if len(mini_df)>0:    # some files may not have a nodule--skipping those 
        biggest_node = np.argsort(mini_df["diameter_mm"].values)[-1]   # just using the biggest node
        node_x = mini_df["coordX"].values[biggest_node]
        node_y = mini_df["coordY"].values[biggest_node]
        node_z = mini_df["coordZ"].values[biggest_node]
        diam = mini_df["diameter_mm"].values[biggest_node]

        #
        # extracting image
        #
        itk_img = sitk.ReadImage(img_file)
        img_array = sitk.GetArrayFromImage(itk_img) #indexes are z,y,x
        num_z,height,width = img_array.shape        #heightXwidth constitute the transverse plane
        imgs = np.ndarray([3,height,width],dtype=np.uint16)
        masks = np.ndarray([3,height,width],dtype=np.uint8)
        center = np.array([node_x,node_y,node_z])  #nodule center
        origin = np.array(itk_img.GetOrigin()) #x,y,z  Origin in world coordinates (mm)
        spacing = np.array(itk_img.GetSpacing())# spacing of voxels in world coor. (mm)
        v_center =np.rint((center-origin)/spacing)  # nodule center in voxel space
        #
        # for each slice in the image, convert the image data to the uint16 range
        # and generate a binary mask for the nodule location
        #
        i = 0
        for i_z in range(int(v_center[2])-1,int(v_center[2])+2):
            mask = make_mask(center,diam,i_z*spacing[2]+origin[2],width,height,spacing,origin)
            masks[i] = mask
            imgs[i] = matrix2int16(img_array[i_z])
            i+=1
        #

        np.save(output_path+"images_%d.npy" % (fcount) ,imgs)
        np.save(output_path+"masks_%d.npy" % (fcount) ,masks)
        fcount+=1
