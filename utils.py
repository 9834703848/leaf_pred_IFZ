import json
from rasterio.mask import mask
from rasterio.warp import reproject, Resampling
from skimage.measure import label
from skimage.color import label2rgb
from scipy.ndimage import binary_erosion


from shapely.geometry import shape as shapely_shape, mapping
import numpy as np
import rasterio
from rasterio.features import shapes
from tqdm import tqdm
from pyproj import Proj, transform
from rasterio.warp import reproject, Resampling
from skimage.morphology import erosion
import numpy as np
import pandas as pd

from rasterio.features import rasterize
from skimage.morphology import erosion, label
from skimage.color import label2rgb

from sklearn.impute import SimpleImputer

import random

from keras.backend import argmax as k_argmax

from shapely.geometry import Point





def real_res(stack_path, UTMzone):
        # Open the raster file
        with rasterio.open(stack_path) as src:
            nc = src.width
            nr = src.height
            bounds = src.bounds

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

        return (res_x,res_y)


def read_raster(file_path):
    with rasterio.open(file_path) as src:
        return src.read(), src.transform, src.crs

def write_raster(output_path, data, transform, crs):
    with rasterio.open(output_path, 'w', driver='GTiff', height=data.shape[1], width=data.shape[2], count=1, dtype=data.dtype, crs=crs, transform=transform) as dst:
        dst.write(data, 1)


def crop_raster(raster_path, geometry):
    with rasterio.open(raster_path) as src:
        # Ensure geometry is in the same CRS as the raster
        if geometry.crs != src.crs:
            geometry = geometry.to_crs(src.crs)
        
        # Convert geometry to GeoJSON-like dict
        geojson_list = [mapping(geom) for geom in geometry.geometry]
        
        # Apply mask
        out_image, out_transform = mask(src, geojson_list, crop=True)
        out_image[out_image == src.nodata] = np.nan
        
        return out_image, out_transform
    


def rasterize_numpy(mask_temp,out_transform,profile):
            with rasterio.Env():
                    mask_profile = {
                        'width': mask_temp.shape[1],
                        'height': mask_temp.shape[2],
                        'count': 6,  # Number of bands
                        'transform': out_transform,
                        'crs': profile['crs'],  # Change to your desired CRS
                        'nodata': np.nan,
                        'dtype': 'float32'}
                    with rasterio.open("mask2.tif", 'w', **mask_profile) as dst:
                        dst.write(mask_temp)

                # Convert raster to polygons
            with rasterio.open("mask2.tif") as src:
                    return src

import numpy as np
def handle_infinite_and_log_transform(data):
    if(np.any(np.isinf(np.abs(data)))):
        data[np.isinf(np.abs(data))] = np.nan
        data = np.log10(data)
        data[np.isinf(data)] = np.nan
    return data
def calculate_nsvdi(Bands, M1):
    Bands = np.array(Bands)
    # M1 = np.reshape(M1,(M1.shape[0],M1.shape[1]))
    r = np.where(Bands == 'RED')[0]
    g = np.where(Bands == 'GREEN')[0]
    b = np.where(Bands == 'BLUE')[0]
    
    nc = M1.shape[1]  # Number of columns
    nr = M1.shape[0]  # Number of rows
    
    rgb_mat = np.column_stack((
        M1[:,:, r].flatten(),
        M1[:,:, g].flatten(),
        M1[:,:, b].flatten()
    ))
    #print(rgb_mat.shape,M1.shape,r,g,b,Bands)
    max_rgb = np.apply_along_axis(np.max, axis=1, arr=rgb_mat)
    min_rgb = np.apply_along_axis(np.min, axis=1, arr=rgb_mat)
    #print(max_rgb.shape)
    v_mat = max_rgb
    mm = max_rgb - min_rgb
    s_mat = mm / v_mat
    
    NSVDI = (s_mat - v_mat) / (s_mat + v_mat)
    
    NSVDI = NSVDI.reshape(nr, nc)
    
    return NSVDI

def process_angle_raster(a2cam_path, shape_geometry, T1,T1_transform,src, pos=None, resampling_method=Resampling.nearest):
        # Crop and mask the raster
        with rasterio.open(a2cam_path) as src:
            out_image, out_transform = mask(src, shape_geometry, crop = True)
            a2cam_nodata = src.nodata
            if a2cam_nodata is not None:
                out_image[out_image == a2cam_nodata] = np.nan
        T1_shape = T1.shape
        if out_image.shape != T1.shape:
            # Prepare for resampling
            resampled_image = np.empty(T1.shape, dtype=np.float32)
            with rasterio.open(a2cam_path) as src_pix_size:
                for band in range(1, 2):  # Assuming single-band image
                    reproject(
                        source=rasterio.band(src_pix_size, band),
                        destination=resampled_image,
                        src_transform=out_transform,
                        src_crs=src_pix_size.crs,
                        dst_transform=T1_transform,
                        dst_crs=src_pix_size.crs,
                        resampling=Resampling.nearest
                    )
        
            resampled_image = resampled_image.ravel()
            resampled_image = resampled_image[pos]
            return resampled_image
                
        else:
            return out_image[0]

def process_angle_raster_2(a2cam_path, shape_geometry, T1, T1_transform,pos=None, resampling_method=Resampling.nearest,output_path='pix_size_resampled.tif'):
    # Read the input raster and apply the mask
    with rasterio.open(a2cam_path) as src:
        out_image, out_transform = mask(src, shape_geometry, crop = True)
        a2cam_nodata = src.nodata
        if a2cam_nodata is not None:
            out_image[out_image == a2cam_nodata] = np.nan
    T1_shape = T1.shape
    if out_image.shape != T1.shape:
        # Prepare for resampling
        resampled_image = np.empty(T1.shape, dtype=np.float32)
        with rasterio.open(a2cam_path) as src_pix_size:
            for band in range(1, 2):  # Assuming single-band image
                reproject(
                    source=rasterio.band(src_pix_size, band),
                    destination=resampled_image,
                    src_transform=out_transform,
                    src_crs=src_pix_size.crs,
                    dst_transform=T1_transform,
                    dst_crs=src_pix_size.crs,
                    resampling=Resampling.nearest
                )
      
             
        return resampled_image[0]
             
    else:
        return out_image[0]


def crop_and_align_raster(input_tif, mask_tif, T1, T1_transform, resampling_method=Resampling.nearest):
    """
    Crop one GeoTIFF using another GeoTIFF as the mask and resample the cropped raster to align with a target raster grid.

    Parameters:
    input_tif (str): Path to the input GeoTIFF file to be cropped.
    mask_tif (str): Path to the mask GeoTIFF file used for cropping.
    T1 (numpy.ndarray): Target raster array for alignment.
    T1_transform (Affine): Affine transformation for the target raster.
    resampling_method (Resampling): Resampling method to be used during reproject.

    Returns:
    numpy.ndarray: Cropped and aligned raster array.
    """

    with rasterio.open(mask_tif) as mask_src:
        # Read the mask data
        mask_data = mask_src.read(1)
        
        # Create a mask geometry where the mask data is not equal to the nodata value
        mask_geom = []
        mask_indices = np.where(mask_data != mask_src.nodata)
        
        if mask_indices[0].size > 0:
            # Convert the mask indices to a list of coordinates
            mask_coords = zip(mask_indices[1], mask_indices[0])
            
            # Use the affine transformation to convert pixel coordinates to CRS coordinates
            for x_idx, y_idx in mask_coords:
                x, y = mask_src.xy(y_idx, x_idx)
                mask_geom.append((x, y))

    from shapely.geometry import Polygon
    mask_polygon = Polygon(mask_geom)
    mask_geojson = [mask_polygon.__geo_interface__]

    with rasterio.open(input_tif) as src:
        # Crop the source raster using the mask geometry
        a2cam, a2cam_transform = mask(src, mask_geojson, crop=True)
        a2cam[a2cam == src.nodata] = np.nan

        if a2cam.size != T1.size:
            a2cam_resampled = np.empty((T1.shape[1], T1.shape[2]), dtype=a2cam.dtype)
            reproject(
                source=a2cam,
                destination=a2cam_resampled,
                src_transform=a2cam_transform,
                src_crs=src.crs,
                dst_transform=T1_transform,
                dst_crs=src.crs,
                resampling=resampling_method
            )
            a2cam = a2cam_resampled

    return a2cam,a2cam_transform

import numpy as np

def MSAVIhy(Bands, M1):
    n = np.where(Bands == 'NIR')[0]
    r = np.where(Bands == 'RED')[0]
    MSAVIhy = (2 * M1[n] + 1 - np.sqrt((2 * M1[n] + 1) ** 2 - 8 * (M1[n] - M1[r]))) / 2
    return MSAVIhy

def GVIMSS(Bands, M1):
    re = np.where(Bands == 'REDEDGE')[0]
    g = np.where(Bands == 'GREEN')[0]
    r = np.where(Bands == 'RED')[0]
    n = np.where(Bands == 'NIR')[0]
    GVIMSS = (-0.283 * M1[g] - 0.660 * M1[r] + 0.577 * M1[re] + 0.388 * M1[n])
    return GVIMSS

def D678500(Bands, M1):
    r = np.where(Bands == 'RED')[0]
    b = np.where(Bands == 'BLUE')[0]
    D678500 = M1[r] - M1[b]
    return D678500

def NSVDI(Bands, M1):
    r = np.where(Bands == 'RED')[0]
    g = np.where(Bands == 'GREEN')[0]
    b = np.where(Bands == 'BLUE')[0]
    nc, nr = M1.shape[2], M1.shape[1]
    rgb_mat = np.vstack((M1[r].flatten(), M1[g].flatten(), M1[b].flatten())).T
    max_rgb = np.max(rgb_mat, axis=1)
    min_rgb = np.min(rgb_mat, axis=1)
    v_mat = max_rgb
    mm = max_rgb - min_rgb
    s_mat = mm / v_mat
    NSVDI = (s_mat - v_mat) / (s_mat + v_mat)
    NSVDI = NSVDI.reshape((nr, nc))
    return NSVDI
import rasterio
from rasterio.windows import from_bounds
import geopandas as gpd

def read_raster_window(shape_temp, raster_path):
   
    # Ensure shape_temp is a GeoDataFrame
    if not isinstance(shape_temp, gpd.GeoDataFrame):
        raise ValueError("shape_temp must be a GeoDataFrame")
    
    # Get the bounds of the GeoDataFrame
    bounds = shape_temp.total_bounds  # (xmin, ymin, xmax, ymax)
    
    # Open the raster file
    with rasterio.open(raster_path) as src:
        # Create a window from the bounds
        window = from_bounds(bounds[0], bounds[1], bounds[2], bounds[3], transform=src.transform)
        
        # Read the data within the window
        S1 = src.read(window=window)
        
        # Get the transform for the window
        window_transform = src.window_transform(window)
    
    # Get the coordinates of the window
    xmin, ymin, xmax, ymax = bounds
    
    return S1, window_transform, bounds



import rasterio
from rasterio.features import rasterize

def rasterize_shapes(shapes, out_shape, transform):
    """
    Rasterize geometries with different labels.

    Parameters:
    shapes (GeoDataFrame): GeoDataFrame with geometries and their labels.
    out_shape (tuple): Shape of the output raster (height, width).
    transform (Affine): Affine transform for the output raster.

    Returns:
    numpy.ndarray: Rasterized array with labels.
    """
    # Create a list of (geometry, value) tuples
    shapes['label'] = range(0, len(shapes))
    shapes_with_labels = [(geom, label) for geom, label in zip(shapes.geometry, shapes.label)]
    
    # Rasterize the shapes
    raster = rasterize(
        shapes_with_labels, 
        out_shape=out_shape,
        transform=transform,
        fill= 0,  # Background value for areas not covered by shapes
        all_touched=False,  # Rasterize all pixels touched by geometries
        dtype='int32'
    )
    return raster
def label_the_leaf_regions(mask_pred,pos,new_plot_leaf):
    # Example mask shape (replace with your actual mask shape)
        mask_shape = mask_pred.shape # Example shape, replace with your actual mask shape
        rleaf_new_2 = np.full((mask_shape[0], mask_shape[1]), np.nan, dtype=np.float32).flatten()
        # Iterate over each pixel coordinate in the mask
        with rasterio.open('T1.tif') as src:
            # Read CRS and affine transform from the raster
            raster_crs = src.crs
            raster_transform = src.transform
            raster_nodata = src.nodata
            
            # Iterate over each pixel coordinate in the raster
            for i in range(len(pos)):
                    cor_y = pos[i] % src.width  # Column index (0-based)
                    cor_x = pos[i] // src.width 
                    # Convert pixel coordinates to the coordinate system of the geometries
                    # Use the affine transformation to convert from pixel coordinates to CRS coordinates
                    x, y = rasterio.transform.xy(raster_transform, cor_x, cor_y)
                    
                    # Create a Point object in the same coordinate system as the geometries
                    point = Point(x, y)
                    
                    # Check if the point is within any geometry in new_plot_leaf
                    for idx, geom in enumerate(new_plot_leaf['geometry']):
                        if geom.contains(point):
                            rleaf_new_2[pos[i]] = idx + 1
                            break
        return rleaf_new_2.reshape(mask_shape)

def erode(array, footprint):
    return erosion(array, footprint)



def clustering_mask(mask_pred):
        mask_sp = np.zeros_like(mask_pred)
        mask_sp[mask_pred == 2] = 1
        mask_pred = label(mask_sp)
        labels = np.unique(mask_pred)
        label_to_color = {label_r: "#{:06x}".format(random.randint(0, 0xFFFFFF)) for label_r in labels}
        label_to_color_rgb = {label_r: color for label_r, color in label_to_color.items()}
        prediction_matrix = mask_pred
        leaf_color_matrix = np.zeros((prediction_matrix.shape[0], prediction_matrix.shape[1])).astype(str)
        for label_r, color in label_to_color_rgb.items():
            leaf_color_matrix[mask_pred == label_r] = color
        leaf_color_matrix[mask_pred == 0] = '#000000'
        return leaf_color_matrix


def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    return [int(hex_color[i:i+2], 16) for i in (0, 2, 4)]


def random_point_in_polygon(polygon):
    minx, miny, maxx, maxy = polygon.bounds
    while True:
        p = Point(random.uniform(minx, maxx), random.uniform(miny, maxy))
        if polygon.contains(p):
            return p

# Define a function to generate non-overlapping random circle polygons within a given polygon
def generate_non_overlapping_circles_within_polygon(polygon, num_circles, radius, num_points=30):
    circles = []
    attempts = 0
    max_attempts = 1000  # Maximum attempts to find non-overlapping circles

    while len(circles) < num_circles and attempts < max_attempts:
        center = random_point_in_polygon(polygon)
        circle = center.buffer(radius, resolution=num_points)

        # Check for overlap with existing circles
        if polygon.contains(circle) and not any(circle.intersects(existing_circle) for existing_circle in circles):
            circles.append(circle)
        
        attempts += 1
    
    return circles



import numpy as np
from PIL import Image, ImageColor
import matplotlib.pyplot as plt

# Assuming I1 is a list of 2D numpy arrays for R's I1[[1]], I1[[2]], and I1[[3]]
def convert_to_rgb(I1):
    nr, nc = I1[0].shape
    # Find max value in each channel
    Im_RGB = np.stack((I1[2], I1[1], I1[0]), axis=-1)
    max_V = np.max(Im_RGB, axis=(0, 1), keepdims=True)
    # Im_RGB = Im_RGB / max_V
    Im_RGB[np.isnan(Im_RGB)] = 0
    Im_RGB[Im_RGB < 0] = 0
    Im_RGB[Im_RGB > 1] = 1
    Im_RGB = Im_RGB ** 0.7
    
    # Convert to 0-255 scale
    
    return Im_RGB

def save_image(Im_RGB, filename):
    # Check if the image is already in uint8 format
    if Im_RGB.dtype != np.uint8:
        # If the image is not in uint8 format, convert it
        Im_RGB = (Im_RGB * 255).astype(np.uint8)
    
    # Convert the NumPy array to a PIL image and save it
    img = Image.fromarray(Im_RGB)
    img.save(filename)
    
def save_image2(image_data,output_file_path):
    plt.imshow(image_data, cmap='gray')  # Display the image in grayscale

    # Remove side axes
    plt.axis('off')

    # Save the image to a file
    plt.savefig(output_file_path, bbox_inches='tight', pad_inches=0)

    # Close the plot
    plt.close()

def apply_colormap(Im_RGB, mask, color_map):
    Col_Mat = Im_RGB.copy()
    for color, value in color_map.items():
       if(color != 0 and color != '0'):
        Col_Mat[mask == color] = ImageColor.getrgb(value)
    rgb_save = Col_Mat
    return rgb_save
def colormap(Im_RGB, mask):
    Col_Mat = Im_RGB.copy()
    colours = np.unique(mask)
    for color in colours:
       if(color!='#000000' and color !='0'):
          Col_Mat[mask == color] = ImageColor.getrgb(str(color))

    return Col_Mat
def find_unique_labels(mask):

    return np.unique(mask)



import numpy as np

def resize_array(array1, array2):
    """
    Resize array1 to match the shape of array2 by padding with zeros or cropping.

    Parameters:
        array1 (np.ndarray): The array to be resized.
        array2 (np.ndarray): The array defining the target shape.

    Returns:
        np.ndarray: Resized array1 with the same shape as array2.
    """
    shape1 = array1.shape
    shape2 = array2.shape
    slices = []
    paddings = []

    # Calculate padding or cropping for each dimension
    for s1, s2 in zip(shape1, shape2):
        if s1 < s2:
            # Padding needed
            pad_total = s2 - s1
            pad_before = pad_total // 2
            pad_after = pad_total - pad_before
            slices.append(slice(None))
            paddings.append((pad_before, pad_after))
        elif s1 > s2:
            # Cropping needed
            crop_total = s1 - s2
            crop_before = crop_total // 2
            crop_after = crop_total - crop_before
            slices.append(slice(crop_before, s1 - crop_after))
            paddings.append((0, 0))
        else:
            # No padding or cropping needed
            slices.append(slice(None))
            paddings.append((0, 0))

    # Apply padding if necessary
    if any(p[0] > 0 or p[1] > 0 for p in paddings):
        array1_resized = np.pad(array1, paddings, mode='constant', constant_values=0)
    else:
        array1_resized = array1
    
    # Apply cropping if necessary
    array1_resized = array1_resized[tuple(slices)]
    
    return array1_resized



from PIL import Image, ImageDraw
from rasterio.windows import from_bounds
import math


def get_cropped_regio(ortho_file_path,shape_temp):
            raster_path = ortho_file_path


            # Load the vector data from the GPKG file
            gdf = shape_temp

            # Assume there's only one region in the GPKG file (or select one specifically)
            region_geometry = gdf.geometry.iloc[0]

            # Get the bounding box of the region
            original_bounds = region_geometry.bounds  # (xmin, ymin, xmax, ymax)

            # Increase the region bounds by some buffer to include surrounding area
            buffer = 0.000001  # Adjust this value as needed
            cropped_bounds = (
                original_bounds[0] - buffer,
                original_bounds[1] - buffer,
                original_bounds[2] + buffer,
                original_bounds[3] + buffer
            )

            # Open the raster and crop the region with a buffer
            with rasterio.open(raster_path) as src:
                # Get the window for the cropped bounds
                window = from_bounds(*cropped_bounds, transform=src.transform)
                
                # Read the cropped raster data
                cropped_image = src.read(window=window)
                
                # Get the transform for the cropped image
                cropped_transform = src.window_transform(window)

            # Convert the raster data to an image
            I1 = cropped_image # Replace with actual I1 data
            Im_RGB = convert_to_rgb(I1)
            Im_RG = (Im_RGB * 255).astype(np.uint8)
            image = Image.fromarray(Im_RG)

            # Extract polygon coordinates
            polygon_coords = np.array(region_geometry.exterior.coords)

            # Transform the polygon coordinates to pixel coordinates
            pixel_coords = []
            for coord in polygon_coords:
                x_geo, y_geo = coord
                x_pixel =  math.ceil((x_geo - cropped_bounds[0]) / cropped_transform[0])
                y_pixel =  math.ceil((y_geo - cropped_bounds[3]) / cropped_transform[4])
                pixel_coords.append((x_pixel, y_pixel))

            # Draw the polygon on the image


            draw = ImageDraw.Draw(image)
            draw.polygon(pixel_coords, outline="red", width=2)
            
            return image


import geopandas as gpd
from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import unary_union

def convert_multipolygon_to_polygon(geometry):
    if isinstance(geometry, MultiPolygon):
        # Merge polygons into a single geometry
        merged_polygon = unary_union(geometry)
        # Check if the result is a Polygon
        if isinstance(merged_polygon, Polygon):
            return merged_polygon
        else:
            # Return the merged geometry as is if it's still complex
            return merged_polygon
    return geometry 