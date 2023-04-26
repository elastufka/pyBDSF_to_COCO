import os
import numpy as np
import pandas as pd
from utils import *
#from multiprocessing.pool import Pool, ThreadPool
#from multiprocessing import cpu_count
from astropy.time import Time
from datetime import datetime as dt
import json
import argparse

def get_args_parser():
    parser = argparse.ArgumentParser('pyBDSF_to_COCO', add_help=False)
    parser.add_argument('--catalog', type=str, help="""Full path to catalog file (.txt or .fits)""")
    parser.add_argument('--image', type=str, help="""Full path to image file (.fits), or else a folder where all images are located.""")
    parser.add_argument('--crop_coords', default=None, type=str, help="""Full path to file where coordinates (SkyCoords) of image crops is located""") 
    #parser.add_argument('--crop_dir', default='.', type=str, help="""Folder where image crops are located""") 
    parser.add_argument('--output_file', type=str, help="""Full path in which to write COCO-format json""")
    parser.add_argument('--test', default=False, type=bool, help="""test""")


    #args for info and category dictionaries
    parser.add_argument('--dataset_year', default=None, type=int, help="""Year dataset created""") 
    parser.add_argument('--dataset_version', default="", type=str, help="""Version of dataset""") 
    parser.add_argument('--description', default="", type=str, help="""Description of the dataset""")
    parser.add_argument('--contributor', default=None, type=str, help="""Contributor(s) to the dataset""") 
    parser.add_argument('--dataset_url', type=str, help="""URL to dataset""")
    parser.add_argument('--dataset_date', default=None, type=str, help="""Date on which dataset was created""")

    parser.add_argument('--category_names', default=None, type=str, help="""path to csv file containing category names, or a comma-separated string with names of categories""")
    return parser

def catalog_coords_to_pix(df, wcs):
    """Convert coordinates in catalog from world to pixel. Rather slow for 10K rows right now, can try terality to speed up."""
    df["bottom_left_coord"] = df.apply(lambda x: ra_dec_to_Skycoord(x["RA"]["DEG"],x["DEC"]["DEG"],x["IM_MAJ"]["DEG"]),axis=1)
    df["top_right_coord"] = df.apply(lambda x: ra_dec_to_Skycoord(x["RA"]["DEG"],x["DEC"]["DEG"],x["IM_MAJ"]["DEG"],fn=np.add), axis=1)
    pixarr = [pixvals_from_Skycoord(wcs, row.bottom_left_coord.values[0],row.top_right_coord.values[0]) for i,row in df.iterrows()] #due tomultiindex, row0 is Nan
    pixarr = np.array(pixarr)

    df["source_xmin"] = pixarr[:,0]
    df["source_ymin"] = pixarr[:,1]
    #df["source_xmax"] = pixarr[:,2]
    #df["source_ymax"] = pixarr[:,3]
    df["source_width"] = pixarr[:,4]
    df["source_height"] = pixarr[:,5]
    df["source_bbox"] = df.apply(lambda x: pix_to_bbox(x.source_xmin,x.source_ymin, x.source_width, x.source_height), axis=1)
    #df["source_bbox"] = df.apply(lambda x: pix_to_bbox(x.source_xmin,x.source_ymin, x.source_xmax, x.source_ymax), axis=1)
    
    #segmentation
    df["segmentation"] = df.apply(lambda x: sky_ellipse_to_path(x["RA"]["DEG"],x["DEC"]["DEG"],x["IM_MAJ"]["DEG"],x["IM_MIN"]["DEG"],x["IM_PA"]["DEG"], wcs), axis=1)
    return df 

def create_info(year = None, version = None, description = "", contributor = None, url="", date_created=None):
    if not year:
        year = dt.now().year
    if not contributor:
        try:
            contributor = os.environ["USER"]
        except KeyError:
            contributor = ""
    if not date_created:
        date_created = Time(dt.now()).isot
    info = {"year": year,
            "version": version,
            "description": description,
            "contributer": contributor,
            "url": url,
            "date_created": date_created}
    return info

def create_categories(names):
    if isinstance(names, str):
        if names.endswith('.csv'):
            cats = pd.read_csv(names) #need to get the values...
        else: 
            cats = names.split(',')
    categories = []
    for i, n in enumerate(cats):
        categories.append({"id":i+1,"name":n,"supercategory":""})
    return categories

def create_image_info(image_id, file_name, image_size, license_id=1, coco_url="", flickr_url=""):
    """from https://github.com/waspinator/pycococreator/blob/master/pycococreatortools/pycococreatortools.py"""
    #try:
    #    date_obs = header["DATE_OBS"]
    #except KeyError:
    date_obs = Time(dt.now()).isot
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
    else: 
        bbox = bounding_box
    if not isinstance(segmentation, list):
        segmentation = [segmentation.values[0][0].tolist(),segmentation.values[0][1].tolist()]
    area = int(np.product(bbox[-2:]))
    annotation_info = {
        "id": annotation_id,
        "image_id": image_id,
        "category_id": 1,
        "iscrowd": 0,
        "area": area,
        "bbox": bbox,
        "segmentation": segmentation
    } 
    return annotation_info

def create_crop_info(i, image, crop_shape, cdf):
    image = image[:image.rfind("/")]
    crop_name = f"{image}/crop_{i}.npy"
    crop_json = []
    image_info = create_image_info(i, crop_name, crop_shape)
    crop_json.append(image_info)

    for j, row in cdf.iterrows():
        crop_json.append(create_annotation_info(j,i, row.source_bbox, row.segmentation))
    #with open(output_file, 'w') as f: 
    #    json.dump(crop_json, f)
    return crop_json

def run_main(args): #for now
    info = create_info(args.dataset_year, args.dataset_version, args.description, args.contributor, args.dataset_url, args.dataset_date)
    categories = create_categories(args.category_names)
    if args.image.endswith(".fits"):
        wcs, header, _ = mapdata_from_fits(args.image)
    else: 
        pass #for now do nothing
    df = read_catalog(args.catalog)

    if args.test:
        df = df.head(50) #for now
    df = catalog_coords_to_pix(df, wcs)
    #df = pd.read_csv("/home/glados/unix-Documents/AstroSignals/temp_df.csv", header=[0,1])

    if np.isnan(df.RA.DEG[0]):
        df.drop(0, inplace = True)
    df.reset_index(inplace=True, drop=True)

    if not args.crop_coords:
        img_id = 0
        image_info = [create_image_info(img_id, args.image, (wcs.celestial.array_shape))]
        #do in parallel create_annotation_info
        count = len(df)
        vals = df["source_bbox"]
        segmentations = df["segmentation"]
        async_results = mp_execute_async(create_annotation_info, 4, range(1,count), ifunc = single_async_prep, fnargs = [img_id, vals, segmentations])
    
    else: 
        image_info=[]
        coordlist = np.load(args.crop_coords, allow_pickle=True) #numpy array
        if not isinstance(coordlist[0][0][0], SkyCoord):
            #deal with this later
            print("not a SkyCoord")
        else:
            async_results = mp_execute_async(create_crop_info, 4, coordlist, ifunc = crop_async_prep, fnargs = [coordlist, args.image, df, wcs])

    annotations = []
    for async_result in async_results:
        res = async_result.get()
        if args.crop_coords:
            image_info.append(res[0])
            annotations.extend(res[1:])
        else: 
            annotations.append(res)

    annotations = unique_annotation_ids(annotations)
    mdict = {"info":info,"images":image_info,"annotations":annotations,"categories":categories}

    with open(args.output_file, 'w') as f: 
        json.dump(mdict, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('pyBDSF_to_COCO', parents=[get_args_parser()])
    args = parser.parse_args()
    run_main(args)