_HOSTNAMES = ["eng"]

CommonScreensDatasetInfo = {
    "image_dir": "/data/graham/datasets/common-screens/s3/data/jpeg",
    "header_path": "/data/graham/datasets/common-screens/s3/metadata/common-screens-with-meta-2022-header.txt",
    "metadata_path": "/data/graham/datasets/common-screens/s3/metadata/common-screens-with-meta-2022-12.csv",
}

Mind2WebDatasetInfo = {
    "task_dir": "/data/graham/datasets/mind2web/data/raw_dump",
    "train": {
        "dataset_path": "osunlp/Mind2Web",
    },
    "test": {
        "dataset_path": "/data/graham/code/mind2web/data/Mind2Web/data/test_set",
        "data_files": "**/*.json",
    },
}
