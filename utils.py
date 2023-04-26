import numpy as np
import pandas as pd 
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy import units as u
from photutils.aperture import SkyEllipticalAperture
from matplotlib import pyplot as plt
import matplotlib.patches as patches
from multiprocessing.pool import Pool
from multiprocessing import cpu_count
import json

def read_catalog(filename):
    """should this always return a multiindex dataframe?"""
    if filename.endswith(".txt"):
        reader = read_txt_catalog
    elif filename.endswith(".fits"):
        reader = read_FITS_catalog
    return reader(filename)

def read_txt_catalog(filename):
    return pd.read_fwf(filename, skiprows=8, header=[0,1])

def read_FITS_catalog(filename):
    pass

def mapdata_from_fits(filename):
    with fits.open(filename) as f: 
        header = f[0].header
        wcs = WCS(header)
        dat = f[0].data.squeeze()
    return wcs, header, dat

def img_bl_tr(img, wcs = None):
    """Determine bottom left and top right of image in astrophysical coordinates"""
    if not wcs: #assume FITS image
        wcs, dat = mapdata_from_fits(img)
        imshape = dat.shape
        return wcs.celestial.array_index_to_world(0,0), wcs.celestial.array_index_to_world(imshape[-2],imshape[-1]) 
    else: #img are the crop coordinates out of a bigger image
        bl = wcs.world()
        tr = wcs.world_to_pixel()

def ra_dec_to_Skycoord(ra, dec, source_axis, fn = np.subtract):
    """Determine coordinate bottom-left or top-right"""
    return SkyCoord(fn(ra,source_axis/2)*u.deg, fn(dec,source_axis/2)*u.deg)

def pix_to_bbox(xmin, ymin, width, height):
    """pixel coords to int to COCO bounding box format. input is dataframe row (Series)"""
    bbox = []
    if np.isnan(xmin.values[0]):
        return [np.nan, np.nan, np.nan, np.nan]
    for x in [xmin, ymin, width, height]:
        bbox.append(int(x.values[0]))
    return bbox

def sorted_coords(x1,x2,y1,y2):
    """sort bottom-left and top right coords"""
    x1, x2 = min([x1,x2]), max([x1,x2])
    y1, y2 = min([y1,y2]), max([y1,y2])
    return x1,x2,y1,y2

def pixvals_from_Skycoord(wcs, bottom_left_coord, top_right_coord):
    if np.isnan(bottom_left_coord.ra.value):
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
    x1, y1 = wcs.celestial.world_to_pixel(bottom_left_coord)
    x1 = int(np.round(x1,0))
    y1 = int(np.round(y1,0))
    x2, y2 = wcs.celestial.world_to_pixel(top_right_coord)
    x2 = int(np.round(x2,0))
    y2 = int(np.round(y2,0))
    x1, x2, y1,y2 = sorted_coords(x1,x2,y1,y2)
    width = x2-x1
    height = y2-y1
    return x1,y1,x2,y2,width, height

def sky_ellipse_to_path(ra, dec, maj, minor, pa, wcs, units=[u.deg, u.deg, u.deg, u.deg], pix_offset=None):
    """Catalog values of a 2D ellipse to pixel mask"""
    if np.isnan(ra):
        return None
    semimajor = maj/2
    semiminor = minor/2
    position_coord = SkyCoord(ra*units[0], dec*units[0])
    ap = SkyEllipticalAperture(position_coord, semimajor*units[1], semiminor*units[2], pa*units[3])
    appix = ap.to_pixel(wcs.celestial)
    _, ax = plt.subplots()
    patch = appix.plot(ax)
    path = path_from_patch(patch)
    plt.close()
    return path

def path_from_patch(patch):
    path = patch[0].get_path()
    original_vert = path.vertices.copy()
    vertices = patch[0].get_patch_transform().transform(original_vert)
    #I interpret poly as list of polygons, defined by the vertexes, like [[x1,y1,x2,y2…xN,yN],…[x1,y1,x2,y2…xN,yN] ], where the coordinates are of the same scale as the image.
    #The masks, encoded this way, are shown correctly by CocoAPI [1]
    #from https://github.com/facebookresearch/Detectron/issues/100
    return vertices.T.tolist() #hope this doesn't have to be int

def get_crop_shape(example_crop_coords, wcs):
    _,_,_,_,w,h = pixvals_from_Skycoord(wcs, example_crop_coords[0][0], example_crop_coords[1][0])
    return w,h

def adjust_bbox_pixvals(bbox, xmin, ymin):
    if len(bbox) == 1:
        bbox = bbox[0]
    return [bbox[0]-xmin, bbox[1]-ymin, bbox[2],bbox[3]]

def adjust_segmentation_pixvals(seg, x1,y1):
    return [np.array(seg[0])-x1, np.array(seg[1])-y1] 

def single_crop_catalog(df, cc, wcs):
    """Return catalog information for a single crop"""
    bl = cc[0][0]
    blra = bl.ra.value
    tr = cc[1][0]
    trra = tr.ra.value
    bldec =bl.dec.value
    trdec = tr.dec.value
    dd = df.where(df.RA.DEG > min((blra,trra))).where(df.RA.DEG < max((blra,trra))).dropna(how='all')
    res = dd.where(dd.DEC.DEG > min((bldec,trdec))).where(dd.DEC.DEG < max((bldec,trdec))).dropna(how='all')
    #shift all pixel coordinates relative to crop position in primary image
    x1,y1,_,_,_,_ = pixvals_from_Skycoord(wcs, bl, tr)
    res["source_xmin"] -= x1
    res["source_ymin"] -= y1
    #res["source_bbox"] = [adjust_bbox_pixvals(row.source_bbox, x1,y1) for _,row in res.iterrows()]
    #res.apply(lambda x: adjust_bbox_pixvals(x["source_bbox"], x1,y1), axis=1)
    res["segmentation"] = [adjust_segmentation_pixvals(row.segmentation.values[0],x1,y1) for _,row in res.iterrows()] #res.apply(lambda x: [x["segmentation"][0] - x1, x["segmentation"][1] - y1], axis=1)
    return res

def single_async_prep(i,img_id, vals, segmentations):
    val = vals[i]
    seg = segmentations[i]
    return i, img_id, val, seg

def crop_async_prep(i, coordlist, crop_dir, df, wcs):
    cc = coordlist[i]
    crop_shape = get_crop_shape(cc, wcs)
    cdf = single_crop_catalog(df, cc, wcs)
    cdf.reset_index(inplace=True,drop=True)
    return i, crop_dir, crop_shape, cdf

def mp_execute_async(func, nprocesses, iterover, ifunc=None, fnargs=None):
    async_results = []
    if not isinstance(nprocesses, int):
        nprocesses = cpu_count()
    with Pool(processes=nprocesses) as pool:
        for i, _ in enumerate(iterover):
            mpargs = [i]
            mpargs.extend(fnargs) #i will always be first arg passed to prep function
            if ifunc:
                mpargs = (ifunc(*mpargs))
            
            async_results.append(pool.apply_async(func, args=mpargs)) 
        pool.close()
        pool.join()
    return async_results

def unique_annotation_ids(annotations):
    aid = [a["id"] for a in annotations]
    if set(aid) != len(aid): #annotations are not unique
        for i,_ in enumerate(annotations):
            annotations[i]['id'] = i
    return annotations

def plot_image_catalog(image, annotations_json, bounding_boxes = True, segmentations = False, **kwargs):
    """Plot the image and the associated bounding boxes and/or segmentations on top for verification"""
    try:
        wcs ,_, dat = mapdata_from_fits(image)
    except OSError:
        dat = np.load(image)
    with open(annotations_json) as f:
        data = json.load(f)
    jdf = pd.DataFrame(data[1:]) #[0] is image annotations

    if bounding_boxes:
        bboxes = jdf.bbox
    if segmentations:
        segs = jdf.segmentation

    fig,ax = plt.subplots() #should this be plotted in wcs? can it?
    ax.imshow(dat, **kwargs)
    if bounding_boxes:
        for box in bboxes:
            #draw rectangle
            rect = patches.Rectangle((box[0],box[1]),box[2],box[3], edgecolor='w', facecolor='none')
            ax.add_patch(rect)
    if segmentations:
        for seg in segs:
            #draw ellipse
            ax.plot(seg[0],seg[1],'r')    

    return fig
