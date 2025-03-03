# download tomotwin demo which contains the ckpt we want
mkdir /tmp/tomotwin_demo
cd /tmp/tomotwin_demo
wget https://zenodo.org/records/7225386/files/tomotwin_demo.zip?download=1
unzip tomotwin_demo.zip?download=1
# move checkpoint to current directory
cd -
mv /tmp/tomotwin_demo/data/best_f1_after600.pth ./tomotwin.pth
# remove the demo
rm -rf /tmp/tomotwin_demo

