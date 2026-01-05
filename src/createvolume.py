import modal

vol = modal.Volume.from_name("sam2_input_data", create_if_missing=True, version=1)

with vol.batch_upload() as batch:
    batch.put_directory("SAM2images", "SAM2images", recursive=True)
