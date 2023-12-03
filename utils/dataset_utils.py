"""
    Generic data loading routines for the SEN12MS-CR dataset of corresponding Sentinel 1,
    Sentinel 2, cloudy Sentinel 2 data and land cover maps.

    The SEN12MS-CR class is meant to provide a set of helper routines for loading individual
    image patches as well as triplets of patches from the dataset. These routines can easily
    be wrapped or extended for use with many deep learning frameworks or as standalone helper
    methods. For an example use case please see the "main" routine at the end of this file.

    NOTE: Some folder/file existence and validity checks are implemented but it is
          by no means complete.

    Authors: Patrick Ebel (patrick.ebel@tum.de), Lloyd Hughes (lloyd.hughes@tum.de),
    based on the exemplary data loader code of https://mediatum.ub.tum.de/1474000, with minimal modifications applied.
"""

import os
import rasterio

import numpy as np

from enum import Enum
from glob import glob


class S1Bands(Enum):
    VV = 1
    VH = 2
    ALL = [VV, VH]
    NONE = []


class S2Bands(Enum):
    B01 = aerosol = 1
    B02 = blue = 2
    B03 = green = 3
    B04 = red = 4
    B05 = re1 = 5
    B06 = re2 = 6
    B07 = re3 = 7
    B08 = nir1 = 8
    B08A = nir2 = 9
    B09 = vapor = 10
    B10 = cirrus = 11
    B11 = swir1 = 12
    B12 = swir2 = 13
    ALL = [B01, B02, B03, B04, B05, B06, B07, B08, B08A, B09, B10, B11, B12]
    RGB = [B04, B03, B02]
    NONE = []

class S2CloudyBands(Enum):
    B01 = aerosol = 1
    B02 = blue = 2
    B03 = green = 3
    B04 = red = 4
    B05 = re1 = 5
    B06 = re2 = 6
    B07 = re3 = 7
    B08 = nir1 = 8
    B08A = nir2 = 9
    B09 = vapor = 10
    B10 = cirrus = 11
    B11 = swir1 = 12
    B12 = swir2 = 13
    ALL = [B01, B02, B03, B04, B05, B06, B07, B08, B08A, B09, B10, B11, B12]
    RGB = [B04, B03, B02]
    NONE = []

class LCBands(Enum):
    IGBP = igbp = 1
    LCCS1 = landcover = 2
    LCCS2 = landuse = 3
    LCCS3 = hydrology = 4
    ALL = [IGBP, LCCS1, LCCS2, LCCS3]
    NONE = []


class Seasons(Enum):
    SPRING = "ROIs1158_spring"
    SUMMER = "ROIs1868_summer"
    FALL = "ROIs1970_fall"
    WINTER = "ROIs2017_winter"
    ALL = [SPRING, SUMMER, FALL, WINTER]


class Sensor(Enum):
    s1 = "s1"
    s2 = "s2"
    lc = "lc"
    s2cloudy = "s2_cloudy"


IGBPClasses = {
    1: "Evergreen needleleaf forests",
    2: "Evergreen broadleaf forests",
    3: "Deciduous needleleaf forests",
    4: "Deciduous broadleaf forests",
    5: "Mixed forests",
    6: "Closed shrublands",
    7: "Open shrublands",
    8: "Woody savannas",
    9: "Savannas",
    10: "Grasslands",
    11: "Permanent wetlands",
    12: "Croplands",
    13: "Urban and built-up lands",
    14: "Cropland/natural vegetation mosaics",
    15: "Snow and ice",
    16: "Barren",
    17: "Water bodies",
    255: "Unclassified"
}

IGBPSimpleClasses = {
    1: "Forest",
    2: "Shrub Land",
    3: "Savannas",
    4: "Grasslands",
    5: "Wetlands",
    6: "Croplands",
    7: "Urban",
    8: "Snow & Ice",
    9: "Barren",
    10: "Water",
    255: "Unclassified"
}

IGBPSimpleClassList = ["Forest", "Shrub Land", "Savannas", "Grasslands", "Wetlands", "Croplands", "Urban", "Snow & Ice", "Barren", "Water"]

IGBPMapping = {
    1: 1,
    2: 1,
    3: 1,
    4: 1,
    5: 1,
    6: 2,
    7: 2,
    8: 3,
    9: 3,
    10: 4,
    11: 5,
    12: 6,
    13: 7,
    14: 6,
    15: 8,
    16: 9,
    17: 10,
    255: 255
}


# Note: The order in which you request the bands is the same order they will be returned in.


class SEN12MSDataset:
    def __init__(self, base_dir):
        self.base_dir = base_dir

        if not os.path.exists(self.base_dir):
            raise Exception(
                "The specified base_dir for SEN12MS dataset does not exist")

    """
        Returns a list of scene ids for a specific season.
    """

    def get_scene_ids(self, season):
        season = Seasons(season).value
        path = os.path.join(self.base_dir, season + "_s1")

        if not os.path.exists(path):
            raise NameError("Could not find season {} in base directory {}".format(
                season, self.base_dir))

        scene_list = [os.path.basename(s)
                      for s in glob(os.path.join(path, "*"))]
        scene_list = [int(s.split('_')[1]) for s in scene_list]
        return set(scene_list)

    """
        Returns a list of patch ids for a specific scene within a specific season
    """

    def get_patch_ids(self, season, scene_id):
        season = Seasons(season).value
        path = os.path.join(self.base_dir, season + "_s1", f"s1_{scene_id}")

        if not os.path.exists(path):
            raise NameError(
                "Could not find scene {} within season {}".format(scene_id, season))

        patch_ids = [os.path.splitext(os.path.basename(p))[0]
                     for p in glob(os.path.join(path, "*"))]
        patch_ids = [int(p.rsplit("_", 1)[1].split("p")[1]) for p in patch_ids]

        return patch_ids

    """
        Return a dict of scene ids and their corresponding patch ids.
        key => scene_ids, value => list of patch_ids
    """

    def get_season_ids(self, season):
        season = Seasons(season).value
        ids = {}
        scene_ids = self.get_scene_ids(season)

        for sid in scene_ids:
            ids[sid] = self.get_patch_ids(season, sid)

        return ids

    """
        Returns raster data and image bounds for the defined bands of a specific patch
        This method only loads a sinlge patch from a single sensor as defined by the bands specified
    """

    def get_patch(self, season, scene_id, patch_id, bands, crop=256):
        season = Seasons(season).value
        sensor = None

        if isinstance(bands, (list, tuple)):
            b = bands[0]
        else:
            b = bands

        if isinstance(b, S1Bands):
            sensor = Sensor.s1.value
            bandEnum = S1Bands
        elif isinstance(b, S2CloudyBands):
            sensor = Sensor.s2cloudy.value
            bandEnum = S2CloudyBands
        elif isinstance(b, S2Bands):
            sensor = Sensor.s2.value
            bandEnum = S2Bands
        elif isinstance(b, LCBands):
            sensor = Sensor.lc.value
            bandEnum = LCBands
        else:
            raise Exception("Invalid bands specified")

        if isinstance(bands, (list, tuple)):
            bands = [b.value for b in bands]
        else:
            bands = bands.value

        scene = "{}_{}".format(sensor, scene_id)
        filename = "{}_{}_p{}.tif".format(season, scene, patch_id)
        patch_path = os.path.join(self.base_dir, season + "_" + sensor, scene, filename)

        with rasterio.open(patch_path) as patch:
            data = patch.read(bands)
            bounds = patch.bounds

        if len(data.shape) == 2:
            data = np.expand_dims(data, axis=0)

        if data.shape[1] != crop:
          middle = data.shape[1] // 2
          crop_middle = crop // 2
          data = data[:, middle - crop_middle:middle + crop_middle, middle - crop_middle:middle + crop_middle]

        return data, bounds

    """
        Returns a quadruplet of patches. S1, S2, S2 cloudy and LC as well as the geo-bounds of the patch
    """

    def get_s1s2s2cloudylc_quadruplet(self, season, scene_id, patch_id, s1_bands=S1Bands.ALL, s2_bands=S2Bands.ALL, s2cloudy_bands=S2CloudyBands.ALL, lc_bands=LCBands.ALL, crop=256):
        s1, bounds = self.get_patch(season, scene_id, patch_id, s1_bands, crop=crop)
        s2, _ = self.get_patch(season, scene_id, patch_id, s2_bands, crop=crop)
        s2cloudy, _ = self.get_patch(season, scene_id, patch_id, s2cloudy_bands, crop=crop)
        lc, bounds_lc = self.get_patch(season, scene_id, patch_id, lc_bands, crop=crop)

        return s1, s2, s2cloudy, lc, bounds

    """
        Returns a quadruplet of numpy arrays with dimensions D, B, W, H where D is the number of patches specified
        using scene_ids and patch_ids and B is the number of bands for S1, S2, S2 cloudy or LC
    """

    def get_quadruplets(self, season, scene_ids=None, patch_ids=None, s1_bands=S1Bands.ALL, s2_bands=S2Bands.ALL, s2cloudy_bands=S2CloudyBands.ALL, lc_bands=LCBands.ALL, crop=256):
        season = Seasons(season)
        scene_list = []
        patch_list = []
        bounds = []
        s1_data = []
        s2_data = []
        s2cloudy_data = []
        lc_data = []

        # This is due to the fact that not all patch ids are available in all scenes
        # And not all scenes exist in all seasons
        if isinstance(scene_ids, list) and isinstance(patch_ids, list):
            raise Exception("Only scene_ids or patch_ids can be a list, not both.")

        if scene_ids is None:
            scene_list = self.get_scene_ids(season)
        else:
            try:
                scene_list.extend(scene_ids)
            except TypeError:
                scene_list.append(scene_ids)

        if patch_ids is not None:
            try:
                patch_list.extend(patch_ids)
            except TypeError:
                patch_list.append(patch_ids)

        for sid in scene_list:
            if patch_ids is None:
                patch_list = self.get_patch_ids(season, sid)

            for pid in patch_list:
                s1, s2, s2cloudy, lc, bound = self.get_s1s2s2cloudylc_quadruplet(
                    season, sid, pid, s1_bands, s2_bands, s2cloudy_bands, lc_bands, crop=crop)
                s1_data.append(s1)
                s2_data.append(s2)
                s2cloudy_data.append(s2cloudy)
                lc_data.append(lc)
                bounds.append(bound)

        return np.stack(s1_data, axis=0), np.stack(s2_data, axis=0), np.stack(s2cloudy_data, axis=0), np.stack(lc_data, axis=0), bounds
