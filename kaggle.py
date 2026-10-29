import kagglehub

# Download latest version
path = kagglehub.dataset_download("dmitryyemelyanov/chinese-traffic-signs")

print("Path to dataset files:", path)