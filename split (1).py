import splitfolders
dr = 'asl_alphabet'
splitfolders.ratio(dr,"split_asl_dataset_48x48" ,ratio=(0.8,0.2))

# Splited as "train" folder (for training data) and "val" folder (for testing data)