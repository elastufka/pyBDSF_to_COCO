# PyBDSF_to_COCO

Convert a source catalog written by [PyBDSF](https://pybdsf.readthedocs.io) to [COCO](https://cocodataset.org/#home) dataset format. WCS coordinates used in PyBDSF catalogs will be converted to pixel values.

Current limitations:
- Bounding boxes are currently based on source major axis, not on segmentation
- segmentation is determined from ellipse shape with no uncertainties accounted for
- Only text format catalogs are used at the moment
- catalog keys specific for catalog for [MIGHTEE](https://www.mighteesurvey.org/data-access) survey
- naming output JSON associated with image crops does not yet make use of the location on disk of such images and their naming convention
- is_crowd is always 0

## Usage

### single image and associated catalog

```bash
pyBDSF_to_COCO.py --image /path/to/fitsfile.fits --catalog /path/to/catalog.txt --output_file /path/to/catalog_coco.json 
```

### single image, associated catalog, SkyCoords of (bottom left, upper right) cutouts from that image

```bash
pyBDSF_to_COCO.py --image /path/to/fitsfile.fits --catalog /path/to/catalog.txt --output_file /path/to/catalog_coco.json  --crop_coords /path/to/crop_coords.npy --crop_dir /path/to/image/crops 
```

### Verify that output JSON and source catalog match

```python
from pyBDSF_to_COCO.utils import plot_image_catalog

plot_image_catalog("image.fits", "annotations_json", bounding_boxes = True, segmentations = False)
```

![Bounding boxes from catalog overplotted on image](example_bbox.png)


```python
from pyBDSF_to_COCO.utils import plot_image_catalog

plot_image_catalog("image.fits", "annotations_json", bounding_boxes = True, segmentations = True)
```

![eBounding boxes and segmentations from catalog overplotted on image](example_seg_and_box.png)

## Dependencies
- astropy
- photutils

## Example output

Two sources from a crop of the COSMOS image

```json
[{"id": 8, "file_name": "../data/MIGHTEE/early_science/coco_json/crop_8.npy", "width": 256, "height": 256, "date_captured": "2023-XX-XXTXX:XX:XX.XXX", "license": 1, "coco_url": "", "flickr_url": ""}, 
{"id": 0, "image_id": "../data/MIGHTEE/early_science/coco_json/crop_8.npy", "category_id": 1, "iscrowd": 0, "area": 72, "bbox": [39, 95, 9, 8], "segmentation": [[41.10545424643942, 40.2399265243223, 39.656333744040694, 39.48320544158412, 39.31007713912766, 39.56154847354128, 40.18223580229392, 40.80292313104644, 41.742150083003935, 42.79306282377547, 43.843975564546895, 44.92077172373786, 45.78629944585509, 46.651827167972215, 47.23541994825382, 47.408548250710396, 47.58167655316686, 47.330205218753235, 46.7095178900006, 46.08883056124807, 45.14960360929058, 44.09869086851904, 43.04777812774762, 41.97098196855666, 41.10545424643942, 41.10545424643942], [102.29997567520627, 101.63154316731902, 100.67108282805066, 99.63012489888774, 98.58916696972506, 97.5527010547878, 96.74899674139351, 95.945292427999, 95.43996861342566, 95.34431700244818, 95.24866539147092, 95.57049551444834, 96.23892802233581, 96.90736053022306, 97.86782086949142, 98.90877879865434, 99.94973672781703, 100.98620264275428, 101.78990695614857, 102.59361126954309, 103.09893508411642, 103.1945866950939, 103.29023830607116, 102.96840818309374, 102.29997567520627, 102.29997567520627]]},
{"id": 1, "image_id": "../data/MIGHTEE/early_science/coco_json/crop_8.npy", "category_id": 1, "iscrowd": 0, "area": 81, "bbox": [44, 217, 9, 9], "segmentation": [[45.45666250794545, 44.83225062824397, 44.52950075158151, 44.61509002422031, 44.70067929685911, 45.16761973379528, 45.91307312365609, 46.65852651351702, 47.62162988891498, 48.590270910334425, 49.55891193175387, 50.45400535027284, 51.07841722997421, 51.70282910967569, 52.00557898633815, 51.91998971369935, 51.83440044106055, 51.36746000412438, 50.62200661426357, 49.87655322440264, 48.91344984900468, 47.944808827585234, 46.97616780616579, 46.08107438764682, 45.45666250794545, 45.45666250794545], [222.9831076392138, 221.91549851303125, 220.71825036718906, 219.6550432091326, 218.59183605107614, 217.74947603591477, 217.31348317956167, 216.8774903232088, 216.88346151120868, 217.33008165871297, 217.77670180621726, 218.6275063564317, 219.69511548261426, 220.76272460879682, 221.959972754639, 223.02317991269547, 224.08638707075193, 224.9287470859133, 225.3647399422664, 225.80073279861926, 225.7947616106194, 225.3481414631151, 224.9015213156108, 224.05071676539637, 222.9831076392138, 222.9831076392138]]}]
```

## Loading into PyTorch

... coming soon