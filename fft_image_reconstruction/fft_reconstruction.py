import SimpleITK as sitk
import numpy as np

image = sitk.ReadImage("echoWithWaveNoise.png", sitk.sitkFloat32)

# execute fourier transform
fft_filter = sitk.ForwardFFTImageFilter()
fft_image = fft_filter.Execute(image)

fft_array = np.array(sitk.GetArrayFromImage(fft_image))
fft_abs = np.abs(fft_array)
fft_image = sitk.GetImageFromArray(fft_abs)
fft_image.CopyInformation(image)

# rescale to view in fourier space
rescale_filter = sitk.RescaleIntensityImageFilter()
rescale_filter.SetOutputMinimum(0)
rescale_filter.SetOutputMaximum(16384)

rescaled_image = rescale_filter.Execute(fft_image)

fft_image_cast = sitk.Cast(rescaled_image, sitk.sitkUInt8)
sitk.WriteImage(fft_image_cast, "fft_magnitude.png")

# zero out influence
coords = [(20, 20), (21, 21)]
for y, x in coords:
    fft_array[y, x] = 0 + 0j
    fft_array[256 - y, 256 - x] = 0 + 0j

# reconstruct original image
fft_reconstructed = sitk.GetImageFromArray(fft_array)
ifft_filter = sitk.InverseFFTImageFilter()
restored_image = ifft_filter.Execute(fft_reconstructed)

restored_image_cast = sitk.Cast(sitk.RescaleIntensity(restored_image), sitk.sitkUInt8)
sitk.WriteImage(restored_image_cast, "echoReconstructed.png")