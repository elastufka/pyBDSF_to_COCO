import numpy as np
import pandas as pd
from utils import *
#from multiprocessing.pool import Pool, ThreadPool
#from multiprocessing import cpu_count
from astropy.time import Time
from datetime import datetime
import json
import argparse

def get_args_parser():
    parser = argparse.ArgumentParser('pyBDSF_to_COCO', add_help=False)
    parser.add_argument('--catalog', type=str, help="""Full path to catalog file (.txt or .fits)""")
    parser.add_argument('--image', type=str, help="""Full path to image file (.fits)""")
    parser.add_argument('--crop_coords', default=None, type=str, help="""Full path to file where coordinates (SkyCoords) of image crops is located""") 
    parser.add_argument('--crop_dir', default='.', type=str, help="""Folder where image crops are located""") 
    parser.add_argument('--output_file', type=str, help="""Full path in which to write COCO-format json""")
    parser.add_argument('--solar', type=bool, default=False, help="""Full path in which to write COCO-format json""")
    return parser

def catalog_coords_to_pix(df, wcs):
    """Convert coordinates in catalog from world to pixel. Rather slow for 10K rows right now, can try terality to speed up."""
    df["bottom_left_coord"] = df.apply(lambda x: ra_dec_to_Skycoord(x["RA"]["DEG"],x["DEC"]["DEG"],x["IM_MAJ"]["DEG"]),axis=1)
    df["top_right_coord"] = df.apply(lambda x: ra_dec_to_Skycoord(x["RA"]["DEG"],x["DEC"]["DEG"],x["IM_MAJ"]["DEG"],fn=np.add), axis=1)
    pixarr = [pixvals_from_Skycoord(wcs, row.bottom_left_coord.values[0],row.top_right_coord.values[0]) for i,row in df.iterrows()] #due tomultiindex, row0 is Nan
    pixarr = np.array(pixarr)

    df["source_xmin"] = pixarr[:,0]
    df["source_ymin"] = pixarr[:,1]
    df["source_width"] = pixarr[:,4]
    df["source_height"] = pixarr[:,5]
    df["source_bbox"] = df.apply(lambda x: pix_to_bbox(x.source_xmin,x.source_ymin, x.source_width, x.source_height), axis=1)
    
    #segmentation
    df["segmentation"] = df.apply(lambda x: sky_ellipse_to_path(x["RA"]["DEG"],x["DEC"]["DEG"],x["IM_MAJ"]["DEG"],x["IM_MIN"]["DEG"],x["IM_PA"]["DEG"], wcs), axis=1)
    return df 

def create_image_info(image_id, file_name, image_size, license_id=1, coco_url="", flickr_url=""):
    """from https://github.com/waspinator/pycococreator/blob/master/pycococreatortools/pycococreatortools.py"""
    #try:
    #    date_obs = header["DATE_OBS"]
    #except KeyError:
    date_obs = Time(datetime.now()).isot
    image_info = {
            "id": image_id,
            "file_name": file_name,
            "width": image_size[0],
            "height": image_size[1],
            "date_captured": date_obs,
            "license": license_id,
            "coco_url": coco_url,
            "flickr_url": flickr_url
    }

    return image_info

def create_annotation_info(annotation_id, image_id, bounding_box=None, segmentation=None):
    if not isinstance(bounding_box, list):
        bbox = bounding_box.values[0]
        seg = [segmentation.values[0][0].tolist(),segmentation.values[0][1].tolist()]
    else: 
        bbox = bounding_box
        seg = [segmentation[:,1].tolist(), segmentation[:,0].tolist()] #(y,x) -> (x,y)
    area = int(np.product(bbox[-2:]))
    annotation_info = {
        "id": annotation_id,
        "image_id": image_id,
        "category_id": 1,
        "iscrowd": 0,
        "area": area,
        "bbox": bbox,
        "segmentation": seg
    } 
    return annotation_info

def create_crop_info(i, crop_dir, crop_shape, cdf):
    crop_name = f"{crop_dir}/crop_{i}.npy"
    output_file = f"{crop_dir}/coco_crop_{i}.json"
    crop_json = []
    image_info = create_image_info(i, crop_name, crop_shape)
    crop_json.append(image_info)

    for i, row in cdf.iterrows():
        crop_json.append(create_annotation_info(i,crop_name, row.source_bbox, row.segmentation))
    with open(output_file, 'w') as f: 
        json.dump(crop_json, f)
    return output_file

def main_solar(args, crop_shape = (256,256)):
    from torch_utils.utils import ma_mask_to_COCO
    arrs = np.load(args.image)

    def solar_prep(i,arrs):
        arr = arrs[i]
        thresh = (np.max(arr)-np.min(arr))/3 + np.min(arr)
        mask = np.ma.masked_greater_equal(arr,thresh).mask
        bboxes, segs = ma_mask_to_COCO(mask)
        #print(len(bboxes), len(segs))
        cdf = pd.DataFrame({"source_bbox":bboxes, "segmentation":segs})
        return i, args.crop_dir, crop_shape, cdf
    
    async_results = mp_execute_async(create_crop_info, 4, arrs, ifunc = solar_prep, fnargs = [arrs])
    output = []
    for async_result in async_results:
        output.append(async_result.get())

def run_main(args): #for now
    wcs, header, _ = mapdata_from_fits(args.image)
    df = read_catalog(args.catalog)
    #df = df.head(200) #for now
    df = catalog_coords_to_pix(df, wcs)
    #df = pd.read_csv("/home/glados/unix-Documents/AstroSignals/temp_df.csv", header=[0,1])
    df.reset_index(inplace=True, drop=True)

    if not args.crop_coords:
        img_id = 0
        image_info = create_image_info(img_id, args.image, (wcs.celestial.array_shape), header)
        #print(image_info)
        #do in parallel create_annotation_info
        count = len(df)
        vals = df["source_bbox"]
        segmentations = df["segmentation"]
        async_results = mp_execute_async(create_annotation_info, 4, range(1,count), ifunc = single_async_prep, fnargs = [img_id, vals, segmentations])
    
    else: 
        coordlist = np.load(args.crop_coords, allow_pickle=True) #numpy array
        if not isinstance(coordlist[0][0][0], SkyCoord):
            #deal with this later
            print("not a SkyCoord")
        else:
            async_results = mp_execute_async(create_crop_info, 4, coordlist, ifunc = crop_async_prep, fnargs = [coordlist, args.crop_dir, df, wcs])

    output = []
    for async_result in async_results:
        output.append(async_result.get())
    if not args.crop_coords:
        with open(args.output_file, 'w') as f: 
            json.dump([image_info, annotations], f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('pyBDSF_to_COCO', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.solar:
        main_solar(args)
    else:
        run_main(args)