import cv2
import numpy as np
import argparse
import os


def arguments():
    # the fist argument is to specify if is conversion png to bitfile or bitfile to png
    parser = argparse.ArgumentParser(description='Convert png to bitfile or bitfile to png')
    # force the user to specify the conversion
    parser.add_argument('-c', '--conversion', type=str, required=True, help='conversion png to bitfile or bitfile to png, p is for png to bitfile, b is for bitfile to png')
    # the rest of the arguments are a list of the files to be converted
    parser.add_argument('-f', '--files', required=True , nargs='+', help='list of files to be converted')
    # add argument to select the bit depth of the bitfile
    parser.add_argument('-d', '--bit_depth', type=int, default=16, help='bit depth of the bitfile, default is 16')
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


def bitfile_to_png(file, bit_depth):
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
    # create the image array when the image is a square with the square root of the length of the bitfile, each value is the average of the RGB values
    img = np.zeros((int(np.sqrt(len(image_data))), int(np.sqrt(len(image_data))), 3), dtype=np_bit_depth)
    # read each pixel
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            # read each value of the bitfile and store it in the image array
            img[i,j] = image_data[i*img.shape[1] + j]
    # write the png file forcing that the png uses the same bit depth as the bitfile
    cv2.imwrite(name +"_rec" + '.png', img.astype(np.uint16) )

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
            bitfile_to_png(file, args.bit_depth)
    else:
        print('Error: invalid conversion')
            
    
  
if __name__ == "__main__":
    main()