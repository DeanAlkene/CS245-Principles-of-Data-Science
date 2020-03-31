import numpy as np
import pandas as pd
import sklearn

file_name = "Animals_with_Attributes2/Features/ResNet101/AwA2-features.txt"
data = pd.read_csv(file_name, sep=' ', nrows=100)
