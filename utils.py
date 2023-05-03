import numpy as np
import pandas as pd 
from astropy.io import fits
from astropy.wcs import WCS
from astropy.table import Table
from astropy.coordinates import SkyCoord
from astropy import units as u
from photutils.aperture import SkyEllipticalAperture
from matplotlib import pyplot as plt
import matplotlib.patches as patches
from multiprocessing.pool import Pool
from multiprocessing import cpu_count
import json
from shapely.geometry import Polygon

def read_catalog(filename):
    """should this always return a multiindex dataframe?"""
    if filename.endswith(".txt"):
        reader = read_txt_catalog
    elif filename.endswith(".fits"):
        reader = read_FITS_catalog
    elif filename.endswith("csv") or filename.endswith("ascii"):
        reader = read_csv_catalog
    elif filename.endswith("cat"):
        reader = read_cat
    return reader(filename)

def read_txt_catalog(filename):
    return pd.read_fwf(filename, skiprows=8, header=[0,1])

def read_FITS_catalog(filename):
    """specifically for MGCLS catalogs right now"""
    tab = Table.read(filename, format='fits')
    cat = tab.to_pandas()

    #unit conversions, renaming columns, multiindex
    cat['smax_asec'] = [float(row.smax_asec)*u.arcsec.to(u.deg) for _, row in cat.iterrows()] 
    cat['smin_asec'] = [float(row.smin_asec)*u.arcsec.to(u.deg) for _, row in cat.iterrows()]
    cat.rename(columns={"RA_deg":"RA","Dec_deg":"DEC","smax_asec":"Maj","smin_asec":"Min","spa_deg":"PA"}, inplace=True)
    units=["DEG","DEG","Jy","","","DEG","DEG","DEG"]
    cat.columns = pd.MultiIndex.from_tuples([*zip(cat.columns, units)])

    return cat

def read_cat(filename):
    cat= pd.read_csv(filename, sep=" ")
    #cat["src"] = [row.srcid[:row.srcid.find("_")] for _, row in cat.iterrows()]
    #cat["srcid"] = [int(row.srcid[row.srcid.find("_")+1:]) for _, row in cat.iterrows()]
    #cat["image_name"] = [f"gaussians_{row.src}/"]
    cat['majoraxis'] = [float(row.majoraxis[:-6])*u.arcsec.to(u.deg) for _, row in cat.iterrows()]
    cat['minoraxis'] = [float(row.minoraxis[:-6])*u.arcsec.to(u.deg) for _, row in cat.iterrows()]
    cat['PA'] = [float(row.PA[:-3]) for _, row in cat.iterrows()]
    cat.rename(columns={"ra":"RA","dec":"DEC","majoraxis":"Maj","minoraxis":"Min"}, inplace=True)
    units=["DEG","DEG","Jy","","","DEG","DEG","DEG"]
    cat.columns = pd.MultiIndex.from_tuples([*zip(cat.columns, units)])
    return cat

def read_csv_catalog(filename, skiprows=8):
    # for ascii, sep= " "
    return pd.read_csv(filename, skiprows=skiprows, header=[0,1])

def read_keymap(filename):
    """default PyBDSF keys: https://pybdsf.readthedocs.io/en/latest/write_catalog.html"""
    with open(filename) as f: 
        keydict = json.load(f)
    return keydict

def keys_used(keydict):
    if not keydict:
        return "RA","DEC","Maj","Min","PA"
    else:
        rakey = keydict["RA"]
        deckey = keydict["DEC"]
        majkey = keydict["Maj"]
        minkey = keydict["Min"]
        pakey = keydict["PA"]
        return rakey, deckey, majkey, minkey, pakey

def mapdata_from_fits(filename):
    with fits.open(filename) as f: 
        header = f[0].header
        wcs = WCS(header)
        dat = f[0].data #.squeeze()
    return wcs, dat

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

def pixvals_from_Skycoord(wcs, row):
    blc = row.bottom_left_coord.values[0]
    trc = row.top_right_coord.values[0]
    if np.isnan(blc.ra.value):
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
    if wcs is None:
        wcs, _ = mapdata_from_fits(row.image_name)
        #get wcs from file name
    x1, y1 = wcs.celestial.world_to_pixel(blc)
    x1 = int(np.round(x1,0))
    y1 = int(np.round(y1,0))
    x2, y2 = wcs.celestial.world_to_pixel(trc)
    x2 = int(np.round(x2,0))
    y2 = int(np.round(y2,0))
    x1, x2, y1,y2 = sorted_coords(x1,x2,y1,y2)
    width = x2-x1
    height = y2-y1
    return x1,y1,x2,y2,width, height

# def pixvals_from_Skycoord(wcs, bottom_left_coord, top_right_coord):
#     if np.isnan(bottom_left_coord.ra.value):
#         return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
#     x1, y1 = wcs.celestial.world_to_pixel(bottom_left_coord)
#     x1 = int(np.round(x1,0))
#     y1 = int(np.round(y1,0))
#     x2, y2 = wcs.celestial.world_to_pixel(top_right_coord)
#     x2 = int(np.round(x2,0))
#     y2 = int(np.round(y2,0))
#     x1, x2, y1,y2 = sorted_coords(x1,x2,y1,y2)
#     width = x2-x1
#     height = y2-y1
#     return x1,y1,x2,y2,width, height

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
    return vertices.T.tolist() #hope this doesn't have to be int #replace .T

def get_crop_shape(example_crop_coords, wcs, image_name = None):
    row = pd.DataFrame({"bottom_left_coord":example_crop_coords[0][0], "top_right_coord":example_crop_coords[0][0], "image_name": image_name})
    _,_,_,_,w,h = pixvals_from_Skycoord(wcs, row[0])
    return w,h

def adjust_bbox_pixvals(bbox, xmin, ymin):
    if len(bbox) == 1:
        bbox = bbox[0]
    return [bbox[0]-xmin, bbox[1]-ymin, bbox[2],bbox[3]]

def adjust_segmentation_pixvals(seg, x1,y1):
    return [np.array(seg[0])-x1, np.array(seg[1])-y1]
    #return [[x-x1, y-y1] for x,y in seg] 

def segmentation_xyxy(segmentation):
    """re-format segmentation from [list of x, list of y] to single list of [x1,y1,x2,y2,...xn,yn]"""
    m2 = []
    for x,y in zip(*segmentation):
        m2.append(x)
        m2.append(y)
    return [m2]

def check_overlap(segs):
    """check if any segmentations in a single crop/image overlap"""
    iscrowd = np.zeros(len(segs)).astype(int)
    #use shapely Polygon
    polygons = [Polygon(np.array(s).T) for s in segs] #replace .T

    for i,p in enumerate(polygons):
        inter = [j for j,p2 in enumerate(polygons) if i !=j and p.intersects(p2)]
        if inter != []:
            for k in inter:
                #smallcat.append({"image_id":n,"bbox":jdf.bbox[k],"segmentation":jdf.segmentation[k]})
                iscrowd[k] = 1
    return iscrowd.tolist()

def single_crop_catalog(df, cc, wcs, rakey="RA", deckey="DEC"):
    """Return catalog information for a single crop"""
    if wcs is None:
        res = df.where(df.image_name == cc).dropna(how='all')#
    else:
        bl = cc[0][0]
        blra = bl.ra.value
        tr = cc[1][0]
        trra = tr.ra.value
        bldec =bl.dec.value
        trdec = tr.dec.value
        dd = df.where(df[rakey].DEG > min((blra,trra))).where(df[rakey].DEG < max((blra,trra))).dropna(how='all')
        res = dd.where(dd[deckey].DEG > min((bldec,trdec))).where(dd[deckey].DEG < max((bldec,trdec))).dropna(how='all')
        #shift all pixel coordinates relative to crop position in primary image
        x1,y1,_,_,_,_ = pixvals_from_Skycoord(wcs, bl, tr)
        res["source_xmin"] -= x1
        res["source_ymin"] -= y1
        res["source_bbox"] = [adjust_bbox_pixvals(row.source_bbox, x1,y1) for _,row in res.iterrows()]
        #res.apply(lambda x: adjust_bbox_pixvals(x["source_bbox"], x1,y1), axis=1)
        res["segmentation"] = [adjust_segmentation_pixvals(row.segmentation.values[0],x1,y1) for _,row in res.iterrows()] #res.apply(lambda x: [x["segmentation"][0] - x1, x["segmentation"][1] - y1], axis=1)
    
    res["iscrowd"] = check_overlap(res["segmentation"])
    return res

def single_async_prep(i,img_id, vals, segmentations, segmentation_fmt):
    val = vals[i]
    seg = segmentations[i]
    iscrowd = check_overlap(segmentations)
    if segmentation_fmt == "[xyxy]":
        seg = segmentation_xyxy(seg)
    return i, img_id, iscrowd, val, seg

def crop_async_prep(i, coordlist, crop_dir, prefix, start_imid, start_annid, df, wcs):
    cc = coordlist[i]
    image_name = None
    if wcs is None:
        image_name = df["image_name"][i]
    crop_shape = get_crop_shape(cc, wcs, image_name)
    cdf = single_crop_catalog(df, cc, wcs)
    cdf.reset_index(inplace=True,drop=True)
    return i, crop_dir, prefix, start_imid, start_annid, crop_shape, cdf

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

def train_val_split(full_json, train_dir, val_dir, json_only=False, **kwargs):
    """do sklearn train_test_split, move images into the correct folders, and split the json"""
    with open(full_json) as f: 
        sdict = json.load(f)
    from sklearn.model_selection import train_test_split
    import shutil

    idf = pd.DataFrame(sdict["images"])
    adf = pd.DataFrame(sdict["annotations"])
    X_train,X_val = train_test_split(idf.id.values, **kwargs)

    trainims, trainanns = [],[]
    valims, valanns = [],[]
    akeys = ['id','image_id','iscrowd','category_id']
    ikeys = ['id','width','height']
    #move images
    for x in X_train:
        imfile = idf.where(idf.id == x).dropna(how='all')
        for k in ikeys:
            imfile[k] = imfile[k].astype(int)
        if not json_only:
            shutil.move(imfile.file_name.values[0], train_dir)
        #adjust imfile name
        imfile['file_name'] = [f[f.rfind("/")+1:] for f in imfile['file_name']]
        trainims.extend(imfile.to_dict(orient='records'))
        anns = adf.where(adf.image_id == x).dropna(how='all')
        for k in akeys:
            anns[k] = anns[k].astype(int)
        trainanns.extend(anns.to_dict(orient='records'))
        

    for x in X_val:
        imfile = idf.where(idf.id == x).dropna(how='all')
        for k in ikeys:
            imfile[k] = imfile[k].astype(int)
        if not json_only:
            shutil.move(imfile.file_name.values[0], val_dir)
        imfile['file_name'] = [f[f.rfind("/")+1:] for f in imfile['file_name']]
        valims.extend(imfile.to_dict(orient='records'))
        anns = adf.where(adf.image_id == x).dropna(how='all')
        for k in akeys:
            anns[k] = anns[k].astype(int)
        valanns.extend(anns.to_dict(orient='records'))


    #split JSON
    traininfo = sdict['info'].copy()
    valinfo = sdict['info'].copy()
    traininfo['description'] += " train"
    valinfo['description'] += " val"
    train_json =  {"info":traininfo,"images":trainims,"annotations":trainanns,"categories":sdict['categories']}
    val_json =  {"info":valinfo,"images":valims,"annotations":valanns,"categories":sdict['categories']}
    
    #return train_json, val_json
    
    with open(f"{full_json[:full_json.rfind('.')]}_train.json",'w') as f:
         json.dump(train_json,f)
         
    with open(f"{full_json[:full_json.rfind('.')]}_val.json",'w') as f:
         json.dump(val_json,f)

def plot_image_catalog(image, annotations_json, bounding_boxes = True, segmentations = False, **kwargs):
    """Plot the image and the associated bounding boxes and/or segmentations on top for verification"""
    try:
        wcs, dat = mapdata_from_fits(image)
        dat = dat.squeeze()
    except OSError:
        dat = np.load(image)
        crop_number = int(image[image.rfind("_")+1:image.rfind(".")])
    if isinstance(annotations_json, list): #it's already the dict item
        jdf = pd.DataFrame(annotations_json)
        try:
            jdf = jdf.where(jdf.image_id == crop_number).dropna(how='all')
        except UnboundLocalError:
            pass
    else:
        with open(annotations_json) as f:
            data = json.load(f)
        jdf = pd.DataFrame(data["annotations"]) #[0] is image annotations

    if bounding_boxes:
        bboxes = jdf.bbox
        #print(bboxes)
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
            if len(seg) == 1:
                segx = [x for i,x in enumerate(seg[0])if i%2 == 0]
                segy = [x for i,x in enumerate(seg[0])if i%2 == 1]
            else: 
                segx = seg[0]
                segy = seg[1]
            ax.plot(segx,segy,'r')    

    return fig
