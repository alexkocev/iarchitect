import numpy as np
import pandas as pd

def get_data():
    """Method to get the tuile from csv files"""
    path = '../iarchitect/data/'

    # Import description csv as df_tuile
    csv_tuile = 'Tableau des caractéristiques - Caractéristiques'
    df_tuile = pd.read_csv(path + csv_tuile + '.csv')
    df_tuile.columns = df_tuile.head(1).values.tolist()[0]
    df_tuile = df_tuile.iloc[1: , :]

    # Import nemesis csv as df_nemesis
    csv_nemesis = 'Tableau des compatibilités - Matrice'
    df_nemesis = pd.read_csv(path + csv_nemesis + '.csv')
    df_nemesis.rename(columns={'Unnamed: 0':'Categorie'}, inplace=True)

    return df_tuile, df_nemesis

def clean_data(df_tuile, df_nemesis):
    """Clean df_tuile and df_nemesis and merge the two data frames"""
    ### DATA CLEANING in df_nemesis
    df_nemesis.drop_duplicates(inplace=True)

    df_nemesis = df_nemesis.set_index('Categorie')

    # Replace all Nan and positive values with 0
    df_nemesis = df_nemesis.fillna(0)
    df_nemesis = df_nemesis.mask(df_nemesis > 0, 0)

    # Update all negative values to 1
    df_nemesis = df_nemesis.mask(df_nemesis < 0, 1)

    # Reset the index and create Categorie_id
    df_nemesis = df_nemesis.reset_index()
    df_nemesis['Categorie_id'] = df_nemesis.index

    # Create dictionaries for Categories
    cat_dict = df_nemesis[['Categorie']].to_dict()['Categorie']
    cat_dict = {value : key for (key, value) in cat_dict.items()}

    # data selection
    df_nemesis = df_nemesis.drop(columns = ['Categorie', '& +20 espèces'])


    ### DATA CLEANING in df_tuile
    df_tuile.drop_duplicates(inplace=True)

    # Remove seasonality by excluding 'Nom' with numbers
    df_tuile = df_tuile[~df_tuile['Nom'].str.contains('\\d', regex=True)]

    # Reset the index and create Nom_id
    df_tuile = df_tuile.reset_index()
    df_tuile['Nom_id'] = df_tuile.index

    # Replace Categorie with Categorie_id
    df_tuile = df_tuile.replace({"Categorie": cat_dict})
    df_tuile = df_tuile.rename(columns={'Categorie':'Categorie_id'})

    # Create dictionaries for Name
    name_dict = df_tuile[['Nom']].to_dict()['Nom']
    name_dict = {value : key for (key, value) in name_dict.items()}

    # Data selection
    df_tuile = df_tuile[['Nom_id', 'Profit','Rendement (kg/pied)','Popularité', 'Categorie_id']]


    ### MERGE DATA FRAMES
    df_tuile = df_tuile.merge(df_nemesis, how='left', on='Categorie_id')
    df_tuile = df_tuile.drop(columns = 'Categorie_id')
    df_tuile = df_tuile.fillna(0)

    # Return a numpy array
    tuile = df_tuile.to_numpy()
    return tuile

if __name__ == '__main__':
    df_tuile, df_nemesis = get_data()
    tuile = clean_data(df_tuile, df_nemesis)
