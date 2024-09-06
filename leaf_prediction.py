import os
import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.enums import Resampling
import geopandas as gpd

import mrcnn.model as modellib

from sklearn.utils import resample
from tqdm.auto import tqdm
from joblib import Parallel, delayed
import time
from shapely.geometry import shape as shapely_shape
from pyproj import Proj, transform
from rasterio.features import geometry_mask, shapes


def leaf_predictions(ortho_path,file_name,model,output_leaf_file,img_size):
    # Constants
    # Paths and file names
    
    id_file = file_name[:8]

    # Load the image
    f_res_x = 0.0025
    f_res_y = 0.0025
    f_res_m = 0.0025

    start_time = time.time()
    # Creating windows for the predictions

    def remove_outliers(x, a=0.01, b=0.99, replace=0, na_rm=True):
        qnt = np.quantile(x, [a, b])
        H = 1.5 * (qnt[1] - qnt[0])
        y = np.copy(x)
        y[x < (qnt[0] - H)] = replace
        y[x > (qnt[1] + H)] = replace
        return y
    def real_res(stack, UTMzone):
        nc = stack.width
        nr = stack.height
        bounds = stack.bounds

        xmn = bounds.left
        xmx = bounds.right
        ymn = bounds.bottom
        ymx = bounds.top

        # Example
        x = [xmn, xmx]
        y = [ymn, ymx]

        # Define the original projection (assuming WGS84)
        in_proj = Proj(init='epsg:4326')

        # Define the target UTM projection
        out_proj = Proj(proj='utm', zone=UTMzone, ellps='WGS84')

        # Transform coordinates
        UTMxmn, UTMymn = transform(in_proj, out_proj, xmn, ymn)
        UTMxmx, UTMymx = transform(in_proj, out_proj, xmx, ymx)

        # Calculate resolution
        res_x = (UTMxmx - UTMxmn) / nc
        res_y = (UTMymx - UTMymn) / nr

        return (res_x, res_y)

    def check_and_zero_out(ss, overlapping, f_res_m):
        # Compute the pixel overlap
        pixel_overlap = int(overlapping // f_res_m)

        # Iterate over each channel
        for channel in range(ss.shape[2]):
            # Define the region to check
            region = ss[pixel_overlap:ss.shape[0]-pixel_overlap, pixel_overlap:ss.shape[1]-pixel_overlap, channel]
            
            # Check if there is any non-zero value in the specified region
            overlap = np.any(region != 0)
            
            # If no non-zero value is found, set the entire channel to zero
            if not overlap:
                ss[:, :, channel] = 0

        return ss


    print('Creating windows for the predictions')
    with rasterio.open(os.path.join(ortho_path, file_name)) as src:
        nc, nr = src.width, src.height
        x_img, y_img = img_size, img_size
        overlapping = 0.15
        def post_process_window(D1,img_size_x,img_size_y):
            # Check if the image is black
            # Apply any other post-processing steps here
            # For example, convert the image to RGB and normalize it
            Im_RGB = np.stack((D1[2], D1[1], D1[0]), axis=-1)
            Im_RGB = Im_RGB / max_V
            Im_RGB[np.isnan(Im_RGB)] = 0
            Im_RGB[Im_RGB < 0] = 0
            Im_RGB[Im_RGB > 1] = 1
            Im_RGB = Im_RGB ** 0.7
            
            # Convert to 0-255 scale
            vec_arr = (Im_RGB * 255).astype(np.uint8)

            # Example sizes, set as needed
            Col_Mat = np.zeros((img_size_x, img_size_y, 3), dtype=np.uint8)
            Col_Mat[:, :, 0] = vec_arr[:img_size_x, :img_size_y, 0]
            Col_Mat[:, :, 1] = vec_arr[:img_size_x, :img_size_y, 1]
            Col_Mat[:, :, 2] = vec_arr[:img_size_x, :img_size_y, 2]

            # Check if the image is completely black
            black_img = False

            # Condition 01: All pixels are the same
            if np.all(Col_Mat == Col_Mat[0, 0, :]):
                black_img = True

            # Condition 02: Most pixels are the same color
            unique, counts = np.unique(Col_Mat.reshape(-1, Col_Mat.shape[2]), axis=0, return_counts=True)
            limit_size = 2400  # Example threshold, adjust as needed
            if limit_size > (img_size_x * img_size_y) - counts.max():
                black_img = True
            if(not black_img):
                return vec_arr
            else:
                return None
        def process_window(x, y,img_size_x,img_size_y):
            
            with rasterio.open(os.path.join(ortho_path, file_name)) as src:
                window = Window(y,x,img_size_y, img_size_x)
                transform_wind = src.window_transform(window)
                D1 = src.read(window=window, out_shape=(src.count, img_size_x, img_size_y), resampling=Resampling.bilinear)
                # Uncomment and implement post-processing if needed
                ix = post_process_window(D1,img_size_x,img_size_y)
                if ix is not None:
                    return ix, transform_wind
                else:
                    return None

        # Sampling to get the max value for RGB normalization
        num_samples = 200
        bands_data = src.read([1, 2, 3])

        # Reshape the bands data into a 2D array
        bands_2d = bands_data.reshape(bands_data.shape[0], -1).T

        # Perform resampling
        samples = resample(bands_2d, n_samples=num_samples, replace=False, random_state=0)
        values_t = np.asarray(samples).flatten()
        values_t = remove_outliers(values_t)
        max_V = np.max(values_t)
        #print(max_V)

        # Moving window parameters
        x_meter = img_size * f_res_x
        y_meter = img_size * f_res_y
        dinamic_distance_m_x = x_meter - overlapping
        dinamic_distance_m_y = y_meter - overlapping
    
        centers = []
        overlap_x = int(overlapping / f_res_x)
        overlap_y = int(overlapping / f_res_y)
        height,width = nr,nc
        for i in range(0, int(height), int(x_img) - overlap_x):
            for j in range(0, int(width), int(y_img) - overlap_y):
                if i + x_img > height and j + y_img > width :
                  centers.append((i, j,height-i,width - j))   
                elif i + x_img > height:
                   centers.append((i, j,height-i, y_img))
                elif j + y_img > width:
                   centers.append((i, j,x_img,width - j)) 
                else:
                    centers.append((i, j,x_img,y_img))

        boxes = Parallel(n_jobs=7)(delayed(process_window)(i, j,img_size_x,img_size_y) for i, j,img_size_x,img_size_y in tqdm(centers))


    # results is now a list of tuples, each containing the image data and its transform function

    boxes = [box for box in boxes if box is not None]


    #Loading the Model




    # Iterating over the windows for the predictions
    print('Iterating over the windows for the predictions')
    with rasterio.open(os.path.join(ortho_path, file_name)) as src:
        profile = src.profile

    multi_poly = []
    last_instance = 0

    for data in tqdm(boxes):

                a = np.array(data[0])
                results = model.detect([a], verbose=1)
            
                r = results[0]

                #results = model.detect([a], verbose=1)
                ss = r['masks']
                # Segment leaves
                n_leaf = ss.shape[2]
                
                if n_leaf > 1:
                    ss = check_and_zero_out(ss,overlapping, f_res_m)
                    label_gray = np.arange(last_instance, last_instance + n_leaf)
                    last_instance = max(label_gray)
                    out_gray = np.zeros((ss.shape[0], ss.shape[1]), dtype=np.int32)
                    
                    for i in range(n_leaf):
                        out_gray[ss[:, :, i]] = label_gray[i]
                    
                    mask_tt = out_gray
                    mask_temp = np.zeros((ss.shape[0], ss.shape[1]))

                    mask_temp[:mask_tt.shape[0], :mask_tt.shape[1]] = mask_tt
                    
                    with rasterio.Env():
                            mask_profile = {
                                'width': mask_temp.shape[1],
                                'height': mask_temp.shape[0],
                                'count': 1,  # Number of bands
                                'transform': data[1],
                                'crs': profile['crs'],  # Change to your desired CRS
                                'nodata': np.nan,
                                'dtype': 'float32'}
                            with rasterio.open("mask2.tif", 'w', **mask_profile) as dst:
                                dst.write(mask_temp, 1)

                        # Convert raster to polygons
                    with rasterio.open("mask2.tif") as src:
                            image = src.read(1).astype('uint8')
                            mask_polygons = list(shapes(image, transform=data[1]))

                    polygons = []
                    for geom, val in mask_polygons:
                            if val != 0:
                                polygons.append(shapely_shape(geom))

                    if polygons:
                            if not multi_poly:
                                multi_poly = polygons
                            else:
                                multi_poly.extend(polygons)



    # Saving the multipoly array 
    multi_poly_gdf = gpd.GeoDataFrame({'geometry': multi_poly}, crs=profile['crs'])
    multi_poly_gdf.to_file(output_leaf_file, driver='GPKG')


    end_time = time.time()
    print("\n ... processing time of leaf segmentation", round(end_time - start_time, 2), "hours \n ...")