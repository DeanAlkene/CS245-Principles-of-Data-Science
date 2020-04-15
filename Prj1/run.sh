cd ../Projection
echo "Running PCA"
python PCA.py
echo "Running LDA"
python LDA.py
cd ../Learning
echo "Running tSNE"
python tSNE.py
echo "Running LLE"
python LLE.py
