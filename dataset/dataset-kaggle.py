import kagglehub

# Download latest version
path = kagglehub.dataset_download("chrislekas/european-football-dataset-europes-top-5-leagues")

print("Path to dataset files:", path)