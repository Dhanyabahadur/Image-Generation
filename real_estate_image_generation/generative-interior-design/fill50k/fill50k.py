import pandas as pd
from huggingface_hub import hf_hub_url
import datasets
import os

_VERSION = datasets.Version("0.0.4")

_DESCRIPTION = """\
Interior design dataset for ControlNet training: conditioning + prompt -> furnished room.
Local dataset generated from Airbnb room images with furniture removal and LLaVA-1.5 captions.

Training Structure:
- conditioning_image: Segmentation/depth maps of empty rooms (GUIDANCE)
- text: Captions describing the desired furnished rooms (PROMPT)  
- image: Original furnished room images (TARGET)

Training Goal: conditioning_image + text -> image
"""

_HOMEPAGE = "TODO"
_LICENSE = "TODO"
_CITATION = "TODO"

_FEATURES = datasets.Features(
    {
        "image": datasets.Image(),
        "conditioning_image": datasets.Image(),
        "text": datasets.Value("string"),
    },
)

_DEFAULT_CONFIG = datasets.BuilderConfig(name="default", version=_VERSION)

# METADATA_URL = hf_hub_url(
#     "fusing/fill50k",
#     filename="train.jsonl",
#     repo_type="dataset",
# )

# IMAGES_URL = hf_hub_url(
#     "fusing/fill50k",
#     filename="images.zip",
#     repo_type="dataset",
# )

# CONDITIONING_IMAGES_URL = hf_hub_url(
#     "fusing/fill50k",
#     filename="conditioning_images.zip",
#     repo_type="dataset",
# )

# _DEFAULT_CONFIG = datasets.BuilderConfig(name="default", version=_VERSION)


class Fill50k(datasets.GeneratorBasedBuilder):
    """
    Uses our local folders:
    - target_images/: Furnished room images (targets)
    - conditioning_images/: Segmentation/depth of empty rooms (conditioning)
    - train.jsonl, train_segmentation.jsonl, train_depth.jsonl: Training data
    """

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name="segmentation", 
            version=_VERSION, 
            description="Dataset with segmentation conditioning"
        ),
        datasets.BuilderConfig(
            name="depth", 
            version=_VERSION, 
            description="Dataset with depth conditioning"
        ),
        _DEFAULT_CONFIG
    ]
    
    # BUILDER_CONFIGS = [_DEFAULT_CONFIG]
    DEFAULT_CONFIG_NAME = "default"

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=_FEATURES,
            supervised_keys=None,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """
        Setup data loading based on config.
        Matches original author's approach but uses our local files.
        """
        # Determine which training file to use based on config
        config_name = self.config.name if self.config.name else "default"
        
        if config_name == "depth":
            metadata_path = 'train_depth.jsonl'
            conditioning_type = "depth"
            print(f"🔧 Loading DEPTH conditioning dataset")
        elif config_name == "segmentation": 
            metadata_path = 'train_segmentation.jsonl'
            conditioning_type = "segmentation"
            print(f"🎨 Loading SEGMENTATION conditioning dataset")
        else:  # default
            metadata_path = 'train.jsonl'
            conditioning_type = "segmentation"  # default uses segmentation
            print(f"📋 Loading DEFAULT dataset (segmentation)")
        
        # Verify file exists
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Training file not found: {metadata_path}")
            
        print(f"📁 Using training file: {metadata_path}")
        print(f"🎯 Training: {conditioning_type} + prompt → furnished room")

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # Match original author's parameter names
                gen_kwargs={
                    "metadata_path": metadata_path,
                    "images_dir": "",  # Empty - we use full paths in JSON
                    "conditioning_images_dir": "",  # Empty - we use full paths in JSON
                },
            ),
        ]
        
#         metadata_path = 'train.jsonl' #dl_manager.download(METADATA_URL)
#         images_dir = '' #dl_manager.download_and_extract(IMAGES_URL)
#         conditioning_images_dir = '' #dl_manager.download_and_extract(
#         #    CONDITIONING_IMAGES_URL
#         #)

#         return [
#             datasets.SplitGenerator(
#                 name=datasets.Split.TRAIN,
#                 # These kwargs will be passed to _generate_examples
#                 gen_kwargs={
#                     "metadata_path": metadata_path,
#                     "images_dir": images_dir,
#                     "conditioning_images_dir": conditioning_images_dir,
#                 },
#             ),
#         ]

#     def _generate_examples(self, metadata_path, images_dir, conditioning_images_dir):
#         metadata = pd.read_json(metadata_path, lines=True)

#         for _, row in metadata.iterrows():
#             text = row["text"]

#             image_path = row["image"]
#             image_path = os.path.join(images_dir, image_path)
#             image = open(image_path, "rb").read()

#             conditioning_image_path = row["conditioning_image"]
#             conditioning_image_path = os.path.join(
#                 conditioning_images_dir, row["conditioning_image"]
#             )
#             conditioning_image = open(conditioning_image_path, "rb").read()

#             yield row["image"], {
#                 "text": text,
#                 "image": {
#                     "path": image_path,
#                     "bytes": image,
#                 },
#                 "conditioning_image": {
#                     "path": conditioning_image_path,
#                     "bytes": conditioning_image,
#                 },
#             }


    def _generate_examples(self, metadata_path, images_dir, conditioning_images_dir):
        """
        
        Expected JSON format from our prepare_train_jsonl_corrected.py:
        {
          "text": "Description of furnished room",
          "image": "target_images/xxx.jpg",                    # Furnished room (target)
          "conditioning_image": "conditioning_images/xxx_segmentation.png",  # Conditioning
          "input_image": "images/xxx.jpg"                      # Empty room (not used in training)
        }
        """
        
        print(f"📋 Loading training data from: {metadata_path}")
        
        # Use pandas like original author
        import pandas as pd
        metadata = pd.read_json(metadata_path, lines=True)
        
        print(f"📊 Found {len(metadata)} training examples")
        
        loaded_count = 0
        skipped_count = 0
        
        for idx, row in metadata.iterrows():
            try:
                # Extract data (matching original author's approach)
                text = row["text"]
                image_path = row["image"]  # target_images/xxx.jpg
                conditioning_image_path = row["conditioning_image"]  # conditioning_images/xxx_seg/depth.png
                
                # Build full paths (original used os.path.join with empty dirs)
                image_path = os.path.join(images_dir, image_path) if images_dir else image_path
                conditioning_image_path = os.path.join(conditioning_images_dir, conditioning_image_path) if conditioning_images_dir else conditioning_image_path
                
                # Check if files exist
                if not os.path.exists(image_path):
                    print(f"⚠️  Missing target image: {image_path}")
                    skipped_count += 1
                    continue
                    
                if not os.path.exists(conditioning_image_path):
                    print(f"⚠️  Missing conditioning image: {conditioning_image_path}")
                    skipped_count += 1
                    continue
                
                # Load images as bytes (matching original author's approach)
                with open(image_path, "rb") as f:
                    image_bytes = f.read()
                    
                with open(conditioning_image_path, "rb") as f:
                    conditioning_image_bytes = f.read()
                
                # Yield in original author's format
                yield row["image"], {
                    "text": text,
                    "image": {
                        "path": image_path,
                        "bytes": image_bytes,
                    },
                    "conditioning_image": {
                        "path": conditioning_image_path,
                        "bytes": conditioning_image_bytes,
                    },
                }
                
                loaded_count += 1
                
                # Progress update
                if loaded_count % 10 == 0:
                    print(f"   📊 Loaded {loaded_count} examples...")
                
            except Exception as e:
                print(f"❌ Error processing example {idx}: {e}")
                skipped_count += 1
                continue
        
        print(f"\n✅ Dataset loading complete!")
        print(f"📊 Successfully loaded: {loaded_count} examples")
        print(f"⏭️  Skipped: {skipped_count} examples")
        print(f"🚀 Ready for ControlNet training!") 