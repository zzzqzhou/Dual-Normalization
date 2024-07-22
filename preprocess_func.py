import nibabel as nib
import numpy as np
import os
import SimpleITK as sitk
from bezier_curve import bezier_curve
from tqdm import tqdm

modality_name_list = {'t1': '_t1.nii', 
                      't1ce': '_t1ce.nii', 
                      't2': '_t2.nii', 
                      'flair': '_flair.nii'}

def resize_image_itk(itkimage, newSize, resamplemethod=sitk.sitkNearestNeighbor):
    resampler = sitk.ResampleImageFilter()
    originSize = itkimage.GetSize()
    originSpacing = itkimage.GetSpacing()
    newSize = np.array(newSize, float)
    factor = originSize / newSize
    newSpacing = originSpacing * factor
    newSize = newSize.astype(np.int)
    resampler.SetReferenceImage(itkimage)
    resampler.SetSize(newSize.tolist())
    resampler.SetOutputSpacing(newSpacing.tolist())
    resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
    resampler.SetInterpolator(resamplemethod)
    itkimgResampled = resampler.Execute(itkimage)
    return itkimgResampled

def save_img(slice, label, dir):
    np.savez_compressed(dir, image=slice, label=label)

def norm(slices):
    max = np.max(slices)
    min = np.min(slices)
    slices = 2 * (slices - min) / (max - min) - 1
    return slices

def nonlinear_transformation(slices):

    points_1 = [[-1, -1], [-1, -1], [1, 1], [1, 1]]
    xvals_1, yvals_1 = bezier_curve(points_1, nTimes=100000)
    xvals_1 = np.sort(xvals_1)

    nonlinear_slices_1 = np.interp(slices, xvals_1, yvals_1)
    nonlinear_slices_1[nonlinear_slices_1 == 1] = -1

    return slices, nonlinear_slices_1


def save_test_npz(data_root, modality, target_root):
    list_dir = os.listdir(data_root)
    tbar = tqdm(list_dir, ncols=70)
    count = 0

    for name in tbar:
        nib_img = nib.load(os.path.join(data_root, name, name + modality_name_list[modality]))
        nib_mask = nib.load(os.path.join(data_root, name, name + '_seg.nii'))

        affine = nib_img.affine.copy()
        
        slices = nib_img.get_fdata()
        masks = nib_mask.get_fdata()
        masks[masks != 0] = 1

        slices = norm(slices)

        if not os.path.exists(os.path.join(target_root, modality)):
            os.makedirs(os.path.join(target_root, modality))
        
        for i in range(slices.shape[2]):
            save_img(slices[:, :, i], masks[:, :, i], os.path.join(target_root, modality, 'test_sample{}.npz'.format(count)))
            count += 1

def main(data_root, modality, target_root):
    list_dir = os.listdir(data_root)
    tbar = tqdm(list_dir, ncols=70)
    count = 0
    for name in tbar:
        nib_img = nib.load(os.path.join(data_root, name, name + modality_name_list[modality]))
        nib_mask = nib.load(os.path.join(data_root, name, name + '_seg.nii'))
        
        affine = nib_img.affine.copy()
        
        slices = nib_img.get_fdata()
        masks = nib_mask.get_fdata()
        masks[masks != 0] = 1

        slices = norm(slices)

        slices, nonlinear_slices_1 = nonlinear_transformation(slices)

        if not os.path.exists(os.path.join(target_root, modality + '_ss')):
            os.makedirs(os.path.join(target_root, modality + '_ss'))
        if not os.path.exists(os.path.join(target_root, modality + '_sd')):
            os.makedirs(os.path.join(target_root, modality + '_sd'))

        for i in range(slices.shape[2]):
            """
            Source-Similar
            """
            save_img(slices[:, :, i], masks[:, :, i], os.path.join(target_root, modality + '_ss', 'sample{}_0.npz'.format(count)))
            """
            Source-Dissimilar
            """
            save_img(nonlinear_slices_1[:, :, i], masks[:, :, i], os.path.join(target_root, modality + '_sd', 'sample{}_0.npz'.format(count)))
            count += 1


if __name__ == '__main__':
    data_root = 'Your Nii Training Data Folder'
    target_root = 'Your Npz Training Data Folder'
    modality = 't2'
    main(data_root, modality, target_root)

    data_root = 'Your Nii Test Data Folder'
    target_root = 'Your Npz Test Data Folder'
    modality_list = ['flair', 't1', 't1ce']
    for modality in modality_list:
        save_test_npz(data_root, modality, target_root)