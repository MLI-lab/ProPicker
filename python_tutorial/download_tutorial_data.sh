# make base directory for empiar datasets
mkdir -p ./empiar
cd ./empiar
# download particle coordinages
wget -r -np -nH --cut-dirs=2 -R "index.html*" -e robots=off https://ftp.ebi.ac.uk/empiar/world_availability/10988/data/DEF/particle_lists/
# download tomograms
wget -r -np -nH --cut-dirs=2 -R "index.html*" -e robots=off https://ftp.ebi.ac.uk/empiar/world_availability/10988/data/DEF/tomograms/
# download ground truth annotations
wget -r -np -nH --cut-dirs=2 -R "index.html*" -e robots=off https://ftp.ebi.ac.uk/empiar/world_availability/10988/data/DEF/labels/