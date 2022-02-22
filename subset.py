import os
import random
import json
from os import listdir, walk
from os.path import isfile, join
import pandas as pd
import shutil


def open_json(file):
    with open(file) as json_data:
        data_dict = json.load(json_data)
    return data_dict


def subset(nb_classes, nb_images, folder='train', organ='flower', folder_src='/Users/yannis/Challenge_Deep/', folder_dist='/Users/yannis/Challenge_Deep/'):
    # On récupère dans une liste tous les sous-dossiers inclus dans "train"
    classes = []
    path = folder_src+folder
    for (repertoire, sousRepertoires, fichiers) in walk(path):
         classes.append(repertoire)
    
    if path in classes:
        classes.remove(path)
    
    # Vérification
    if nb_classes > len(classes):
        return "Nombre de classes trop élevé !"
    # print(classes)
    
    # Dans chaque dossier, on vérifie s'il y a assez d'images, sinon on supprime le dossier
    for dir_ in classes:
        files = [f for f in listdir(dir_) if isfile(join(dir_, f))]
        if len(files) < nb_images:
            classes.remove(dir_)
        
    if len(classes) < nb_classes:
        return "Nombre d'images par classe trop élevé"
    
    path_json = folder_src+"dataset.json"
    df = pd.read_json(path_json, orient='index')
    
    if organ:
        dict_ = {}
        for dir_ in classes:
            # print(dir_)
            flowers=[]
            files = [f for f in listdir(dir_) if isfile(join(dir_, f))]
            for file in files:
                id_f = file.split('.')[0]
                if df.loc[id_f, 'organ']==organ:
                    flowers.append(id_f)
            # print(flowers)
            if len(flowers) >= nb_images:
                dir_ = '/'.join(dir_.split('/')[-2:])
                dict_[dir_] = flowers
            
            
    else:
        dict_ = {}
        for dir_ in classes:
            list_files = []
            files = [f for f in listdir(dir_) if isfile(join(dir_, f))]
            for file in files:
                id_f = file.split('.')[0]
                list_files.append(id_f)
            if len(list_files) >= nb_images:
                dir_ = '/'.join(dir_.split('/')[-2:])
                dict_[dir_] = list_files
    
    if len(dict_)<nb_classes:
        return "Nombre d'images par classe trop élevé"
    
    classes = random.sample(list(dict_.keys()), nb_classes)
    
    new_dict={}
    
    for classe in classes:
        new_dict[classe] = dict_[classe]
    for key in new_dict.keys():
        new_dict[key] = random.sample(dict_[key], nb_images)
        
    # print(new_dict)    
    
    if not os.path.exists(folder_dist+'new_train'):
        os.makedirs(folder_dist+'new_train')
        print("Le dossier new_train a été créé !")
    else:
        print("Les fichiers vont maintenant être copiés dans le dossier new_train")
    
    new_df = pd.DataFrame(columns=df.columns.to_list())
    # print(new_dict)
    
    for key in new_dict.keys():
        fold = key.split('/')[1]
        if not os.path.exists(folder_dist+'new_train/{}'.format(fold)):
            os.makedirs(folder_dist+'new_train/{}'.format(fold))
        
        for file in new_dict[key]:
            # print(file)
            shutil.copy(folder_src+'{}/{}.jpg'.format(key, file), folder_dist+'new_train/{}/'.format(fold))
            new_df.loc[file] = df.loc[file, :]
    
    new_df.to_json(folder_dist+'new_dataset.json', orient='index')
    print('Le fichier new_dataset.json a été généré avec succès !')
    
    return "La copie a été réalisée avec succès"


if __name__ == "__main__":
    subset(nb_classes=1081, nb_images=1, folder='train', organ='', folder_src='/home/data/challenge_2022_miashs/', folder_dist='/home/miashs3/SuperAlbert/')