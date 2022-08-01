
import os
import pandas as pd

def data_label(path):

    df_label = pd.read_csv('label.csv', header = None)

    files_dir = os.listdir(path)

    path_list = []
     
    label_list = []
   
    for file_dir in files_dir:
        
        if os.path.splitext(file_dir)[1] == ".jpg":
            path_list.append(file_dir)
            index = int(os.path.splitext(file_dir)[0])
            label_list.append(df_label.iat[index, 0])
 

    path_s = pd.Series(path_list)
    label_s = pd.Series(label_list)
    df = pd.DataFrame()
    df['path'] = path_s
    df['label'] = label_s
    df.to_csv(path+'\\dataset.csv', index=False, header=False)
 
 
def main():

    train_path = 'C:\\Users\\dqu\\Desktop\\CV_Project\\train'
    val_path = 'C:\\Users\\dqu\\Desktop\\CV_Project\\validation'
    data_label(train_path)
    data_label(val_path)
 
if __name__ == "__main__":
     main()
