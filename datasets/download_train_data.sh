# tomotwin tomograms
mkdir -p ./tomotwin_data/tomograms
cd ./tomotwin_data/tomograms
wget https://zenodo.org/api/records/6637357/files-archive
unzip files-archive
tar -xvf *.tar.gz
#rm files-archive
#rm *.tar.gz
cd ../..

# shrec2021 data
mkdir -p ./shrec2021/full_dataset
cd ./shrec2021/full_dataset
wget https://dataverse.nl/api/access/datafile/309089
unzip 309089
rm 309089
