# tomotwin tomograms
mkdir -p data/tomotwin_data/tomograms
cd data/tomotwin_data/tomograms
wget https://zenodo.org/api/records/6637357/files-archive
unzip files-archive
tar -xvf *.tar.gz
#rm files-archive
#rm *.tar.gz
cd ..

# shrec2021 data
cd ..
mkdir shrec2021
cd shrec2021
mkdir full_dataset
cd full_dataset
wget https://dataverse.nl/api/access/datafile/309089
unzip 309089
rm 309089
