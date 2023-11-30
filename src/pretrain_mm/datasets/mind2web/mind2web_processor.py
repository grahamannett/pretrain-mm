class Mind2WebTaskProcessor:
    @staticmethod
    def preprocessor(sample: dict):
        """
        this is a task preprocessor for the Mind2Web dataset such that it works for the processor meaning it is only image + text
        """
        return {
            "text": sample["text"] + sample["label"],
            "images": sample["image"],
        }

    @staticmethod
    def postprocessor(sample):
        """
        helper function
        """
        sample["input_ids"] = sample.input_ids.squeeze(0)
        sample["attention_mask"] = sample.attention_mask.squeeze(0)
        sample["image_patches"] = [img.squeeze(0) for img in sample.image_patches]
        sample["image_patches_indices"] = sample.image_patches_indices.squeeze(0)
        return sample
