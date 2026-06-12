import kagglehub

# 
output_dir = "./datasets/"
# Download latest version
path = kagglehub.dataset_download("arjunashok33/miniimagenet")
# path = kagglehub.dataset_download("wenewone/cub2002011")
# path = kagglehub.dataset_download("apollo2506/eurosat-dataset", output_dir=output_dir + "eurosat")

print("Path to dataset files:", path)