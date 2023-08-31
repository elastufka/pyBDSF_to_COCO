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
    parser.add_argument('--image', default=None, type=str, help="""Full path to image file (.fits), or else a folder where all images are located. If no image is passed, it is assumed that the catalog contains a column image_name of corresponding image names.""")
    parser.add_argument('--crop_coords', default=None, type=str, help="""Full path to file where coordinates (SkyCoords) of image crops is located""") 
    #parser.add_argument('--crop_dir', default='.', type=str, help="""Folder where image crops are located""") 
    parser.add_argument('--output_file', type=str, help="""Full path in which to write COCO-format json""")
    parser.add_argument('--crop_prefix', type=str, help="""Prefix to use for crop names in front of crop_n.npy or crop_n.fits""")
    parser.add_argument('--start_imid', default=0, type=int, help="""Index at which to start Image IDs. Useful if dataset will contain multiple images.""")
    parser.add_argument('--start_annid', default=0, type=int, help="""Index at which to start annotation IDs. Useful if dataset will contain multiple images.""")
    parser.add_argument('--test', default=False, type=bool, help="""test""")
    parser.add_argument('--no_PA', default=False, type=bool, help="""if no PA information, use PA=0""")

    #formatting args
    parser.add_argument('--bbox_fmt', default="xywh", type=str, help="""Bounding box format: xywh (x0,y0,width,height) or xyxy (x0,y0,x1,y1)""")
    parser.add_argument('--seg_fmt', default="[xyxy]", type=str, help="""Segmentation format: [[x],[y]] list of x, list of y or [xyxy] list of x0,y0,x1,y1,...xn,yn""")
    
    #how to deal with error estimates in catalog? not yet implemented
    parser.add_argument('--sigma_n', default=1.0, type=float, help="""Account for n sigma error on selected parameters""")
    parser.add_argument('--sigma_apply', default="ra, dec, maj, min", type=str, help="""Parameters for which to account for n sigma error""")

    #args for info and category dictionaries
    parser.add_argument('--keymap', default=None, type=str, help="""JSON file where map from PyBDSF standard keys are mapped to the keys used in the given dataset""")     
    parser.add_argument('--annotations_only', default=False, type=bool, help="""only write Images and Annotations dicts to json""")     
    parser.add_argument('--dataset_year', default=None, type=int, help="""Year dataset created""") 
    parser.add_argument('--dataset_version', default="", type=str, help="""Version of dataset""") 
    parser.add_argument('--description', default="", type=str, help="""Description of the dataset""")
    parser.add_argument('--contributor', default=None, type=str, help="""Contributor(s) to the dataset""") 
    parser.add_argument('--dataset_url', type=str, help="""URL to dataset""")
    parser.add_argument('--dataset_date', default=None, type=str, help="""Date on which dataset was created""")

    parser.add_argument('--category_names', default=None, type=str, help="""path to csv file containing category names, or a comma-separated string with names of categories""")
    return parser

def catalog_coords_to_pix(df, wcs, keylist, bbox_fmt = "xywh"):
    """Convert coordinates in catalog from world to pixel. Rather slow for 10K rows right now, can try terality to speed up."""
    rakey, deckey, majkey, minkey, pakey = keylist
    if pakey not in df.keys():
        df[pakey,'DEG'] = 0
    
    if not wcs:
        #get the wcs from each individual fits file... this will probably be slow
        assert("image_name" in df.columns)

    df["bottom_left_coord"] = df.apply(lambda x: ra_dec_to_Skycoord(x[rakey]["DEG"],x[deckey]["DEG"],x[majkey]["DEG"]),axis=1)
    df["top_right_coord"] = df.apply(lambda x: ra_dec_to_Skycoord(x[rakey]["DEG"],x[deckey]["DEG"],x[minkey]["DEG"],fn=np.add), axis=1)
    pixarr = [pixvals_from_Skycoord(wcs, row) for i,row in df.iterrows()] #due tomultiindex, row0 is Nan
    pixarr = np.array(pixarr)

    df["source_xmin"] = pixarr[:,0]
    df["source_ymin"] = pixarr[:,1]
    #if bbox_fmt == "xyxy":
    #    df["source_xmax"] = pixarr[:,2]
    #    df["source_ymax"] = pixarr[:,3]
    #else: 
    #    df["source_xmax"] = pixarr[:,4]
    #    df["source_ymax"] = pixarr[:,5]
    #df["source_width"] = pixarr[:,4]
    #df["source_height"] = pixarr[:,5]
    #df["source_bbox"] = df.apply(lambda x: pix_to_bbox(x.source_xmin,x.source_ymin, x.source_xmax, x.source_ymax), axis=1)
     
    #segmentation. adjust for sigma later
    df["segmentation"] = df.apply(lambda x: sky_ellipse_to_path(x[rakey]["DEG"],x[deckey]["DEG"],x[majkey]["DEG"],x[minkey]["DEG"],x[pakey]["DEG"], wcs), axis=1)
    df["source_bbox"] = bbox_from_ellipse(df.segmentation)
    return df 

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

def create_annotation_info(annotation_id, image_id, iscrowd, bounding_box=None, segmentation=None):
    if not isinstance(bounding_box, list):
        bbox = bounding_box.values[0]
    else: 
        bbox = bounding_box
    if isinstance(segmentation, pd.Series): #it's a series
        segmentation = segmentation.values.tolist() 
        if len(segmentation) == 1:
            segmentation = [segmentation[0][0],segmentation[0][1]]
        segmentation = segmentation_xyxy(segmentation)
    if isinstance(iscrowd, pd.Series): #it's a series
        iscrowd = iscrowd.values[annotation_id] if len(iscrowd) > 1 else iscrowd.values[0]
    elif not isinstance(iscrowd, int): 
        try:
            iscrowd = iscrowd[annotation_id]
        except IndexError:
            iscrowd = iscrowd[0]

    area = int(np.product(bbox[-2:]))
    annotation_info = {
        "id": annotation_id,
        "image_id": image_id,
        "category_id": 1,
        "iscrowd": iscrowd,
        "area": area,
        "bbox": bbox,
        "segmentation": segmentation
    } 
    return annotation_info

def create_crop_info(i, image, prefix, start_imid, start_annid, crop_shape, cdf):
    #image = image[:image.rfind("/")]
    crop_name = f"{prefix}crop_{i}.npy" #{image}/
    crop_json = []
    image_info = create_image_info(i+start_imid, crop_name, crop_shape)
    crop_json.append(image_info)

    for j, row in cdf.iterrows():
        crop_json.append(create_annotation_info(j+start_annid,i+start_imid, row.iscrowd, row.source_bbox, row.segmentation))
    #with open(output_file, 'w') as f: 
    #    json.dump(crop_json, f)
    return crop_json

def run_main(args): #for now
    if not args.annotations_only:
        info = create_info(args.dataset_year, args.dataset_version, args.description, args.contributor, args.dataset_url, args.dataset_date)
        categories = create_categories(args.category_names)
    if args.image.endswith(".fits"):
        wcs, _ = mapdata_from_fits(args.image)
    else: 
        wcs = None

    df = read_catalog(args.catalog)

    keydict = None
    if args.keymap:
        keydict = read_keymap(args.keymap)
    rakey, deckey, majkey, minkey, pakey = keys_used(keydict)
    keylist = [rakey, deckey, majkey, minkey, pakey]

    if args.test:
        df = df.sample(n=500).reset_index(drop=True) #for now
        
    df = catalog_coords_to_pix(df, wcs, keylist, bbox_fmt = args.bbox_fmt)
    #df = pd.read_csv("/home/glados/unix-Documents/AstroSignals/temp_df.csv", header=[0,1])

    if np.isnan(df[rakey].DEG[0]):
        df.drop(0, inplace = True)
    df.reset_index(inplace=True, drop=True)

    if args.image and not args.crop_coords:
        img_id = args.start_imid
        image_info = [create_image_info(img_id, args.image, (wcs.celestial.array_shape))]
        #do in parallel create_annotation_info
        count = len(df)
        vals = df["source_bbox"]
        segmentations = df["segmentation"]
        iscrowd = check_overlap(vals, segmentations) 
        #print(count, len(vals), len(segmentations), len(iscrowd), type(iscrowd))
        async_results = mp_execute_async(create_annotation_info, 4, range(1,count), ifunc = single_async_prep, fnargs = [img_id, vals, segmentations, iscrowd, args.seg_fmt])
        #annotations = [create_annotation_info(*single_async_prep(i, img_id, vals, segmentations, args.seg_fmt)) for i in range(1,count)]
    else: 
        image_info=[]
        if not args.image: 
            coordlist = df.image_name.values
        else:
            coordlist = np.load(args.crop_coords, allow_pickle=True) #numpy array
        if not isinstance(coordlist[0][0][0], SkyCoord):
            #deal with this later
            print("not a SkyCoord")
        else:
            #get crop shape of first crop
            crop_shape = get_crop_shape(coordlist[0], wcs)
            async_results = mp_execute_async(create_crop_info, 4, coordlist, ifunc = crop_async_prep, fnargs = [coordlist, args.image, args.crop_prefix, args.start_imid, args.start_annid, df, wcs, crop_shape, keylist])

    annotations = []
    for async_result in async_results: #why is this slow?
        res = async_result.get()
        if args.crop_coords:
            image_info.append(res[0])
            annotations.extend(res[1:])
        else: 
            annotations.append(res)

    annotations = unique_annotation_ids(annotations)
    if not args.annotations_only:
        mdict = {"info":info,"images":image_info,"annotations":annotations,"categories":categories}
    else: 
        mdict = {"images":image_info,"annotations":annotations}
    with open(args.output_file, 'w') as f: 
        json.dump(mdict, f)
    print(f"{len(annotations)} COCO annotations for {len(image_info)} images written to {args.output_file}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser('pyBDSF_to_COCO', parents=[get_args_parser()])
    args = parser.parse_args()
    run_main(args)