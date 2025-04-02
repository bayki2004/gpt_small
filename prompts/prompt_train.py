import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split

# Load the data
author_title = pd.read_csv('data/author_title.csv')
