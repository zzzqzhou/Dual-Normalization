import nibabel as nib
import numpy as np
import os
import SimpleITK as sitk
from bezier_curve import bezier_curve
from tqdm import tqdm

modality_name_list = {'t1': '_t1.nii.gz', 
                      't1ce': '_t1ce.nii.gz', 
                      't2': '_t2.nii.gz', 
                      'flair': '_flair.nii.gz'}

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

    points_2 = [[-1, -1], [-0.5, 0.5], [0.5, -0.5], [1, 1]]
    xvals_2, yvals_2 = bezier_curve(points_2, nTimes=100000)
    xvals_2 = np.sort(xvals_2)
    yvals_2 = np.sort(yvals_2)

    points_3 = [[-1, -1], [-0.5, 0.5], [0.5, -0.5], [1, 1]]
    xvals_3, yvals_3 = bezier_curve(points_3, nTimes=100000)
    xvals_3 = np.sort(xvals_3)

    points_4 = [[-1, -1], [-0.75, 0.75], [0.75, -0.75], [1, 1]]
    xvals_4, yvals_4 = bezier_curve(points_4, nTimes=100000)
    xvals_4 = np.sort(xvals_4)
    yvals_4 = np.sort(yvals_4)

    points_5 = [[-1, -1], [-0.75, 0.75], [0.75, -0.75], [1, 1]]
    xvals_5, yvals_5 = bezier_curve(points_5, nTimes=100000)
    xvals_5 = np.sort(xvals_5)

    """
    slices, nonlinear_slices_2, nonlinear_slices_4 are source-similar images
    nonlinear_slices_1, nonlinear_slices_3, nonlinear_slices_5 are source-dissimilar images
    """
    nonlinear_slices_1 = np.interp(slices, xvals_1, yvals_1)
    nonlinear_slices_1[nonlinear_slices_1 == 1] = -1
    
    nonlinear_slices_2 = np.interp(slices, xvals_2, yvals_2)

    nonlinear_slices_3 = np.interp(slices, xvals_3, yvals_3)
    nonlinear_slices_3[nonlinear_slices_3 == 1] = -1

    nonlinear_slices_4 = np.interp(slices, xvals_4, yvals_4)

    nonlinear_slices_5 = np.interp(slices, xvals_5, yvals_5)
    nonlinear_slices_5[nonlinear_slices_5 == 1] = -1

    return slices, nonlinear_slices_1, nonlinear_slices_2, \
           nonlinear_slices_3, nonlinear_slices_4, nonlinear_slices_5


def main(data_root, modality, target_root):
    list_dir = os.listdir(data_root)
    tbar = tqdm(list_dir, ncols=70)
    count = 0
    for name in tbar:
        nib_img = nib.load(os.path.join(data_root, name, name + modality_name_list[modality]))
        nib_mask = nib.load(os.path.join(data_root, name, name + '_seg.nii.gz'))
        
        affine = nib_img.affine.copy()
        
        slices = nib_img.get_fdata()
        masks = nib_mask.get_fdata()
        masks[masks != 0] = 1

        slices = norm(slices)
        slices, nonlinear_slices_1, nonlinear_slices_2, \
        nonlinear_slices_3, nonlinear_slices_4, nonlinear_slices_5 = nonlinear_transformation(slices)

        if not os.path.exists(os.path.join(target_root, modality + '_ss')):
            os.makedirs(os.path.join(target_root, modality + '_ss'))
        if not os.path.exists(os.path.join(target_root, modality + '_sd')):
            os.makedirs(os.path.join(target_root, modality + '_sd'))

        for i in range(slices.shape[2]):
            """
            Source-Similar
            """
            save_img(slices[:, :, i], masks[:, :, i], os.path.join(target_root, modality + '_ss', 'sample{}_0.npz'.format(count)))
            save_img(nonlinear_slices_2[:, :, i], masks[:, :, i], os.path.join(target_root, modality + '_ss', 'sample{}_1.npz'.format(count)))
            save_img(nonlinear_slices_4[:, :, i], masks[:, :, i], os.path.join(target_root, modality + '_ss', 'sample{}_2.npz'.format(count)))
            """
            Source-Dissimilar
            """
            save_img(nonlinear_slices_1[:, :, i], masks[:, :, i], os.path.join(target_root, modality + '_sd', 'sample{}_0.npz'.format(count)))
            save_img(nonlinear_slices_3[:, :, i], masks[:, :, i], os.path.join(target_root, modality + '_sd', 'sample{}_1.npz'.format(count)))
            save_img(nonlinear_slices_5[:, :, i], masks[:, :, i], os.path.join(target_root, modality + '_sd', 'sample{}_2.npz'.format(count)))
            count += 1

if __name__ == '__main__':
    data_root = 'Your Data Dir.'
    target_root = 'Your Target Data Dir.'
    modality = 'Your Brats Modality'
    main(data_root, modality, target_root)