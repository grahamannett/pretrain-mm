class RicoDatasetConfig:
    name: str


# MAIN FOCUS IS SCREENSPOT DATASET.  NOT ACTUALLY RICO DATASET
# https://huggingface.co/datasets/rootsautomation/ScreenSpot
ScreenSpotConfig = RicoDatasetConfig(name="rootsautomation/ScreenSpot")


# https://huggingface.co/datasets/rootsautomation/RICO-ScreenQA
ScreenQAConfig = RicoDatasetConfig(name="rootsautomation/RICO-ScreenQA")
# https://huggingface.co/datasets/rootsautomation/RICO-ScreenQA-Complex
ScreenQAComplexConfig = RicoDatasetConfig(name="rootsautomation/RICO-ScreenQA-Complex")
# https://huggingface.co/datasets/rootsautomation/RICO-ScreenQA-Short
ScreenQAShortConfig = RicoDatasetConfig(name="rootsautomation/RICO-ScreenQA-Short")

# https://huggingface.co/datasets/rootsautomation/RICO-Screen2Words
Screen2WordsConfig = RicoDatasetConfig(name="rootsautomation/RICO-Screen2Words")

# https://huggingface.co/datasets/rootsautomation/RICO-SCA
SCAConfig = RicoDatasetConfig(name="rootsautomation/RICO-SCA")
