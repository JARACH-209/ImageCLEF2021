# ImageCLEF2021

## Setting up rclone
### Installing:
```
curl -O https://downloads.rclone.org/rclone-current-linux-amd64.zip
unzip rclone-current-linux-amd64.zip

cd rclone-*-linux-amd64

mkdir -p ~/.local/bin/
cp rclone ~/.local/bin/

mkdir -p ~/.local/share/man/man1
cp rclone.1 ~/.local/share/man/man1
```
### Configuring
```
rclone config
```
Set up a new remote with the CLEF Shared Drive (Team Drive in rclone lingo)

## Setting up the dataset
### For raw dataset
```
mkdir ~/Datasets
cd ~/Datasets
rclone copy RemoteName:Dataset/ImageCLEF/ ./ImageCLEF/ --exclude 'ImageSlices' -P
rm *.rar
cd ImageCLEF
mkdir Dataset
mv *.tar ./Dataset
tar -xvf ./Dataset/*.tar
```
### For ImageSlices (warning, will take a lot of time)
```
cd ~/Datasets/ImageCLEF/
rclone copy RemoteName:/Dataset/ImageCLEF/ImageSlices ./ImageSlices -P
```
