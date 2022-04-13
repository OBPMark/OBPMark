import cv2
import numpy as np
import argparse
from astropy.io import fits
import os


def arguments():
    # the fist argument is to specify if is conversion png to bitfile or bitfile to png
    parser = argparse.ArgumentParser(description='Convert png to bitfile or bitfile to png')
    # force the user to specify the conversion
    parser.add_argument('-c', '--conversion', type=str, required=True, help='conversion png to bitfile or bitfile to png, p is for png to bitfile, b is for bitfile to png, f is for bitfile to fits, t is for fits to bitfile, pf is for png to fits')
    # the rest of the arguments are a list of the files to be converted
    parser.add_argument('-f', '--files', required=True , nargs='+', help='list of files to be converted')
    # add argument to select the bit depth of the bitfile
    parser.add_argument('-d', '--bit_depth', type=int, default=16, help='bit depth of the bitfile, default is 16')
    # add brightness argument for the 32 bit bitfile
    parser.add_argument('-b', '--brightness', type=int, default=0, help='brightness of the bitfile, default is 0, only used for 32 bit bitfile')
    args = parser.parse_args()
    return args


def png_to_bitfile(file):
    # extract the name of the file without the extension and the path
    name = file.split('/')[-1].split('.')[0]
    # add the output folder to the name
    name = 'output/' + name
    image_data_array = []

    # read the png file
    img = cv2.imread(file, cv2.IMREAD_UNCHANGED)
    # read each pixel
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            # read each pixel and store the average value from RGB in a 16bit variable
            pixel = img[i,j]
            image_data_array.append(int(np.average(pixel)))

    image_data = np.array(image_data_array, dtype=np.uint16)
    # write the bitfile
    image_data.tofile(name + '.bin')



def bitfile_to_png(file, bit_depth, brightness):
    # read the bitfile
    if bit_depth == 16:
        np_bit_depth = np.uint16 
    elif bit_depth == 8:
        np_bit_depth = np.uint8
    elif bit_depth == 32:
        np_bit_depth = np.uint32
    else:
        print('Error: invalid bit depth')
        return
    image_data = np.fromfile(file, dtype=np_bit_depth)
    # extract the name of the file without the extension and the path
    name = file.split('/')[-1].split('.')[0]
    # add the output folder to the name
    name = 'output/' + name
    # if bit_depth is 32 the image_data will be the square root of each elements of image_data
    if bit_depth == 32:
        # loop through each element of image_data
        for i in range(len(image_data)):
            # calculate the square root of each element
            value = int(np.sqrt(image_data[i]))
            value = value * brightness
            # if the result is greater than 65535, set it to 65535
            if value > 65535:
                value = 65535
            # if the result is less than 0, set it to 0
            if value < 0:
                value = 0
            image_data[i] = value
            
            
    # create the image array when the image is a square with the square root of the length of the bitfile, each value is the average of the RGB values
    img = np.zeros((int(np.sqrt(len(image_data))), int(np.sqrt(len(image_data))), 3), dtype=np_bit_depth)
    # read each pixel
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            # read each value of the bitfile and store it in the image array
            img[i,j] = image_data[i*img.shape[1] + j]
    # write the png file forcing that the png uses the same bit depth as the bitfile
    cv2.imwrite(name +"_rec" + '.png', img.astype(np.uint16) )

def bitfile_to_fits(file, bit_depth):
    # takes the binay file and converts it to a fits file
    # read the bitfile
    if bit_depth == 16:
        np_bit_depth = np.uint16 
    elif bit_depth == 8:
        np_bit_depth = np.uint8
    elif bit_depth == 32:
        np_bit_depth = np.uint32
    else:
        print('Error: invalid bit depth')
        return
    # store the bitfile in a numpy array with correct shape
    image_data = np.fromfile(file, dtype=np_bit_depth)
    # image_data shape is (n, n)
    # modify the shape to (n, n, 1)
    image_data = image_data.reshape(int(np.sqrt(len(image_data))), int(np.sqrt(len(image_data))), 1)
    # extract the name of the file without the extension and the path
    name = file.split('/')[-1].split('.')[0]
    # add the output folder to the name
    name = 'output/' + name
    # create the fits file
    hdu = fits.PrimaryHDU(image_data)
    # write the fits file
    hdul = fits.HDUList([hdu])
    hdul.writeto(name + '.fits', overwrite=True)


    

def main():

    args = arguments()
    # check if the conversion is png to bitfile or bitfile to png
    if args.conversion == 'p':
        # convert png to bitfile
        # create a output folder if it doesn't exist
        if not os.path.exists('output'):
            os.makedirs('output')
        # convert each file in the list
        for file in args.files:
           png_to_bitfile(file)

    elif args.conversion == 'b':
        # convert bitfile to png
        for file in args.files:
            bitfile_to_png(file, args.bit_depth, args.brightness)
    elif args.conversion == 'f':
        # convert bitfile to fits
        for file in args.files:
            bitfile_to_fits(file, args.bit_depth)

    else:
        print('Error: invalid conversion')
            
    
  
if __name__ == "__main__":
    main()