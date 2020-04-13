cd ../Learning
echo "Running LLE"
python LLE.py
echo "Running Isomap"
python Isomap.py
cd ../Projection
echo "Re-running PCA with RBF kernel"
python PCA.py
