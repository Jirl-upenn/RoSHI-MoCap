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
