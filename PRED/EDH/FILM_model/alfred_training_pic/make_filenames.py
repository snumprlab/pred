#make filenames_file

import os
os.getcwd() #make sure at pytorch directory

f = open("../train_test_inputs/alfred_train_files_with_gt.txt" , "w")
f.write("episode # is " + str(number_of_this_episode) + "\n")
f.close()

