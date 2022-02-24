import pandas as pd
import numpy as np
import re

def sorted_df(list_filenames):
    for file in list_filenames:
        df = pd.read_csv(file)
        if df.isnull().values.any()==True:
            df = df.dropna(inplace=True)
            df = df.reset_index(drop=True)
        df = df.sort_values(by=['Id'])
        df = df.reset_index(drop=True)
        df.to_csv(file)
    
#    final_list=[]
#    df = pd.read_csv(list_filenames[0])
#    df = df.sort_values(by=['Id'])
#    sorted_list = df['Id'].to_list()
#    final_list.append(df)
#    for i in range(1, len(list_filenames)):
#        df = pd.read_csv(list_filenames[i])
#        df.Id = df.Id.astype("category")
#        df.Id.cat.set_categories(sorted_list, inplace=True)
#        df.sort_values(["Id"])
#        df["Id"] = df["Id"].astype(str)
#        final_list.append(df)
#    for j in range(len(final_list)):
#        final_list[j].to_csv(list_filenames[j])
    

def str_to_list(string):
    if string[2]==" ":
        string = list(string)
        string[2]=""
        string = ''.join(string)
    string = re.sub(' +', ' ', string)
    string = string.replace(' ', ',')
    string = eval(string)[0]
    return string

def get_name(filename):
    df = pd.read_csv(filename)
    df = df.iloc[:, :2]
    return df

def get_conf(filename):
    df = pd.read_csv(filename)
    conf = pd.DataFrame(df.iloc[:, -1], columns=['Confidence'])
    conf['Confidence'] = conf['Confidence'].apply(str_to_list)
    return conf

def conf_average(l1,l2):
    lit=np.array(l1)
    lit2=np.array(l2)
    result=lit*lit2
    return result

def get_final_class(list_conf):
    # big_df = pd.DataFrame(columns=['final_class'])
    if len(list_conf)>1:
        for i in range(len(list_conf)-1):
            for j in range(len(list_conf[0])):
                lit1 = list_conf[i].loc[j, 'Confidence']
                lit2 = list_conf[i+1].loc[j, 'Confidence']
                list_conf[i+1].loc[j, 'Confidence'] = conf_average(lit1, lit2).tolist()
    
    df_final = list_conf[-1]
    df_final['max'] = df_final['Confidence'].apply(get_max)
    df_final['max_index'] = df_final['Confidence'].apply(get_max_index)
    return df_final

def get_max(list_):
    max_value = max(list_)
    return max_value

def get_max_index(list_):
    max_value = max(list_)
    max_index = list_.index(max_value)
    return max_index

def get_class(dico, df):
    df['Category_pred']=''
    for i in range(len(df)):
        df.loc[i, 'Category_pred'] = dico[df.loc[i, "max_index"]]
    return df

def process(list_filenames):
    sorted_df(list_filenames)
    names = get_name(list_filenames[0])
    list_df = []
    for file in list_filenames:
        df = get_conf(file)
        list_df.append(df)
    df_final = get_final_class(list_df)
    df_final = get_class(dico, df_final)
    df = pd.concat([df_final, names], axis=1)
    df = df.loc[:, ['Id', 'Category_pred']]
    df = df.rename(columns={'Id': 'Id', 'Category_pred': 'Category'})
    df.to_csv('final_submission.csv', index=False)
    # return df

def get_files(models_names):
	list_filenames=[]
	for model in models_names:
    	list_filenames.append("/home/miashs3/SuperAlbert/results/{}/predictions.csv".format(model))
    return list_filenames

if __name__ == "__main__":
	models_names = ['resnet50', 'autres noms de modèle']
	print("Aggregation of : {}".format(models_names))
	list_filenames = get_files(models_names)
	print("Start processing...")
    process(list_filenames)
    print("Processing Done : final_submission.csv created !")