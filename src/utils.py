from numpy import load
import pandas as pd

data = load('trainx16x32_0.npz')
lst = data.files
for item in lst:
    print(data[item])
    df = pd.DataFrame(data[item])

df.to_csv('/content/drive/MyDrive/Colab Notebooks/_Nornikel/Homeworks_MIPT/MLE/storage/sample.csv', index=False, sep='\t')  