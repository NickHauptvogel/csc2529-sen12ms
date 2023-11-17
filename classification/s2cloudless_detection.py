from s2cloudless import S2PixelCloudDetector
from dataset import SEN12MS, ToTensor, Normalize
import torchvision.transforms as transforms
import numpy as np

if __name__ == '__main__':

    # define mean/std of the training set (for data normalization)
    bands_mean = {'s1_mean': [-11.76858, -18.294598],
                  's2_mean': [1226.4215, 1137.3799, 1139.6792, 1350.9973, 1932.9058,
                              2211.1584, 2154.9846, 2409.1128, 2001.8622, 1356.0801]}

    bands_std = {'s1_std': [4.525339, 4.3586307],
                 's2_std': [741.6254, 740.883, 960.1045, 946.76056, 985.52747,
                            1082.4341, 1057.7628, 1136.1942, 1132.7898, 991.48016]}

    # load test dataset
    imgTransform = transforms.Compose([ToTensor(), Normalize(bands_mean, bands_std)])

    test_dataGen = SEN12MS("/content/drive/MyDrive/CSC2529/data", "/content/drive/MyDrive/CSC2529/csc2529-sen12ms/splits",
                           imgTransform=imgTransform,
                           label_type="single_label", subset="test",
                           use_s1=False, use_s2=True, use_RGB=False,
                           IGBP_s=True,
                           exper_suffix="_cloudy",
                           crop_size=224)

    detector = S2PixelCloudDetector()

    # Open write file
    with open("cloud_cover.txt", "w") as f:
        for sample in test_dataGen:
            s2 = sample['image'].numpy()
            s2 = np.transpose(s2, (1, 2, 0))
            # Add a dummy batch dimension
            s2 = np.expand_dims(s2, axis=0)
            mask = detector.get_cloud_masks(s2)

            # Get percentage of cloud cover
            cloud_cover = np.sum(mask) / (mask.shape[1] * mask.shape[2])

            # Write to file
            f.write(f"{sample['id']}\t{cloud_cover}\n")
