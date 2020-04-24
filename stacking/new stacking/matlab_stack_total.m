directory = "/Users/gfx/aws_test"
dataset_file = strcat(directory, "/", "FOV100.hdf5")
save_file = strcat(directory, "/", "stacked_w_matlab.hdf5")

stack_hdf(dataset_file, save_file)