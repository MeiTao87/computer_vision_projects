## Pipeline: face detect and classify at the same time --> notification
* need to change the data generator
* generate the label according to how many sub folders.
* sample lable: classes(number of sub folders), [(x, y, w, h, pc) * 2]  (2 is the number of anchors) 

## ToDo:

* collect and annotate data
* add augmentation (img and bbox) in data_generator