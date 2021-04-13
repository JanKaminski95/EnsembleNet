from skimage import io,filters,exposure
from skimage.util.shape import view_as_windows

img = plt.imread('./new_tifs/NeuroTrace_530-615_20191105_Axio20X_section2_Z4_NeuroTrace.tif')
img = img/img.max()
img = exposure.equalize_hist(img)
img = img*255
img = img.astype(np.uint8)

window_size = (512,512)
crop = view_as_windows(img,window_size,480)
for i in range(crop.shape[0]):
  for j in range(crop.shape[1]):
    patch = crop[i,j]
    io.imsave("./new/P28_S3/images/crop_P28_S3_row%icolumn%i.jpg"%(i+1,j+1),patch)

img = plt.imread('./new_tifs/20200807_Widefield_Zstack_20Xtiles_P1_section2-MAX_NeuroTrace.tif')
img = img/img.max()
img = exposure.equalize_hist(img)
img = img*255
img = img.astype(np.uint8)

window_size = (512,512)
crop = view_as_windows(img,window_size,480)
for i in range(crop.shape[0]):
  for j in range(crop.shape[1]):
    patch = crop[i,j]
    io.imsave("./new/P1_S2/images/crop_P1_S2_row%icolumn%i.jpg"%(i+1,j+1),patch)

img = plt.imread('./new_tifs/20200807_Widefield_Zstack_20Xtiles_P1_section3-MAX_NeuroTrace.tif')
img = img/img.max()
img = exposure.equalize_hist(img)
img = img*255
img = img.astype(np.uint8)
window_size = (512,512)
crop = view_as_windows(img,window_size,480)
for i in range(crop.shape[0]):
  for j in range(crop.shape[1]):
    patch = crop[i,j]
    io.imsave("./new/P1_S3/images/crop_P1_S3_row%icolumn%i.jpg"%(i+1,j+1),patch)

img = plt.imread('./new/2mask_P1_S2.tif')
img = img/img.max()
img = img*255
img = img.astype(np.uint8)
window_size = (512,512)
crop = view_as_windows(img,window_size,480)
for i in range(crop.shape[0]):
  for j in range(crop.shape[1]):
    patch = crop[i,j]
    io.imsave("./new/P1_S2/targets/crop_mask_P1_S2_row%icolumn%i.jpg"%(i+1,j+1),patch)

img = plt.imread('./new/mask_P1_S3.tif')
img = img/img.max()
img = img*255
img = img.astype(np.uint8)
window_size = (512,512)
crop = view_as_windows(img,window_size,480)
for i in range(crop.shape[0]):
  for j in range(crop.shape[1]):
    patch = crop[i,j]
    io.imsave("./new/P1_S3/targets/crop_mask_P1_S3_row%icolumn%i.jpg"%(i+1,j+1),patch)

img = plt.imread('./new/mask_P28_S2.tif')
img = img/img.max()
img = img*255
img = img.astype(np.uint8)
window_size = (512,512)
crop = view_as_windows(img,window_size,480)
for i in range(crop.shape[0]):
  for j in range(crop.shape[1]):
    patch = crop[i,j]
    io.imsave("./new/P28_S2/targets/crop_mask_P28_S2_row%icolumn%i.jpg"%(i+1,j+1),patch)

