echo "Running LDA"
python LDA.py
cd ../Learning
echo "Running tSNE"
python tSNE.py
cd ../Projection
echo "Re-running PCA with RBF kernel"
python PCA.py
