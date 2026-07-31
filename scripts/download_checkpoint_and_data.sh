# Script for downloading model checkpoint and example inputs/outputs.

mkdir -p model/egoallo

# egoallo_checkpoint_april13.zip (552 MB)
gdown https://drive.google.com/file/d/14bDkWixFgo3U6dgyrCRmLoXSsXkrDA2w/view?usp=drive_link --fuzzy
unzip egoallo_checkpoint_april13.zip -d model/egoallo/
mv model/egoallo/egoallo_checkpoint_april13/* model/egoallo/
rmdir model/egoallo/egoallo_checkpoint_april13
rm egoallo_checkpoint_april13.zip

# egoallo_example_trajectories.zip (8.17 GB)
gdown https://drive.google.com/file/d/14zQ95NYxL4XIT7KIlFgAYTPCRITWxQqu/view?usp=drive_link --fuzzy
unzip egoallo_example_trajectories.zip
rm egoallo_example_trajectories.zip

# evaluation_data.zip (27 MB) -- OptiTrack ground truth + per-method predictions
mkdir -p evaluation/data
gdown https://drive.google.com/file/d/1I0FfBCEsV5LmAtGimnHunU5VQih4-LVQ/view?usp=sharing --fuzzy
unzip evaluation_data.zip -d evaluation/data/
rm evaluation_data.zip
