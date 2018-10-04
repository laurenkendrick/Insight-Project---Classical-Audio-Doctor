from __future__ import print_function
import pickle
import pandas as pd
import numpy as np
import subprocess
import sklearn.datasets
from sklearn.decomposition import PCA
import argparse
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import matplotlib.pyplot as plt
from datetime import datetime
import os
import sklearn.ensemble
import lime
import lime.lime_tabular
from xgboost import XGBClassifier
from sklearn import cross_validation
import codecs
from sklearn import svm
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_selection import SelectKBest
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import youtube_dl
import string
import glob
#From input url, extract data and views
DEVELOPER_KEY='AIzaSyDyYbI1TRfwsjwSb2hxiX2URUnRoAeibd0'
YOUTUBE_API_SERVICE_NAME = 'youtube'
YOUTUBE_API_VERSION = 'v3'
youtube = build(YOUTUBE_API_SERVICE_NAME, YOUTUBE_API_VERSION,
    developerKey=DEVELOPER_KEY)
def youtube_video(query):
  # Call the search.list method to retrieve results matching the specified
  # query term.
  search_response = youtube.videos().list(
    id=query,
    part='snippet,statistics'
  ).execute()

  videos = []

  # Add each result to the dataframe of videos
  global videos_df
  videos_ = []
  tags=[]
  
  for search_result in search_response.get('items', []):
      response2 = youtube.channels().list(
                  part='statistics, snippet',
                  id=search_result['snippet']['channelId']).execute()
      videos_.append({'VideoId': query, 'Title': search_result['snippet']['title'], 
                      'Description': search_result['snippet']['description'],
                      'channelID':search_result['snippet']['channelId'],
                      'channelSubscribers':response2['items'][0]['statistics']['subscriberCount'],
                      'publishedAt':search_result['snippet']['publishedAt'],
                     'viewCount':search_result['statistics']['viewCount']})
      if 'tags' in search_result['snippet'].keys():
            tags.append(search_result['snippet']['tags'])
      else:
            tags.append([])

  videos_df=pd.DataFrame(videos_)
  videos_df['tags']=tags

def remove_prefix(text, prefix):
    if text.startswith(prefix):
        return text[len(prefix):]
    return text

def download_audio(videoid):
  ydl_opts = {
    'format': 'bestaudio/best',
    'postprocessors': [{
        'key': 'FFmpegExtractAudio',
        'preferredcodec': 'm4a',
    }],
  }
  #for p in videos_df['VideoId']:
  #    print(p)
      #!youtube-dl https://www.youtube.com/watch?v={p} -f "bestaudio[ext=m4a]" --id
      #completed=subprocess.run('!youtube-dl https://www.youtube.com/watch?v={p} -f "bestaudio[ext=m4a]" --id')
  with youtube_dl.YoutubeDL(ydl_opts) as ydl:
      ydl.download(videoid)



def get_dataset(input2):
  query=remove_prefix(input2,'https://www.youtube.com/watch?v=')
  query=query.split('&',1)[0] #NEW 10/2
  youtube_video(query)
  print(query)
  videos_df['Description']=videos_df['Description'].astype(str)
  videos_df['Title']=videos_df['Title'].astype(str)
  videos_df['viewCount']=videos_df['viewCount'].astype(float)
  videos_df['channelSubscribers']=videos_df['channelSubscribers'].astype(float)
  videos_df['publishedAt']=videos_df['publishedAt'].apply(lambda x: datetime.strptime(x,'%Y-%m-%dT%H:%M:%S.%fZ'))
  videos_df['Desc_word_count']=videos_df['Description'].str.split().str.len()
  a=datetime.now()
  videos_df['dayslive']=a-videos_df['publishedAt']
  videos_df['dayslive']=videos_df['dayslive'].apply(lambda x: (x.days * 86400 + x.seconds)/86400)
  title=videos_df['Title'][0]
  #ydl_opts = {
  #  'format': 'bestaudio/best',
  #  'postprocessors': [{
  #      'key': 'FFmpegExtractAudio',
  #      'preferredcodec': 'm4a',
  #  }],
  #}
  #for p in videos_df['VideoId']:
  #    print(p)
  #    #!youtube-dl https://www.youtube.com/watch?v={p} -f "bestaudio[ext=m4a]" --id
  #    #completed=subprocess.run('!youtube-dl https://www.youtube.com/watch?v={p} -f "bestaudio[ext=m4a]" --id')
  #    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
  #        ydl.download([input2])
  download_audio([input2])


  import moviepy.editor as mpy
  from pyAudioAnalysis import audioFeatureExtraction
  for file in glob.glob('*-{0}.m4a'.format(query)):
      clip=mpy.AudioFileClip(file)
  #clip = mpy.AudioFileClip('{0}-{1}.m4a'.format(title,query)) #Possible to modify duration of clip in this command
  sample_rate = clip.fps
  audio_data = clip.to_soundarray()
  print(len(audio_data))
  print(sample_rate)
  x1=[i[0] for i in audio_data]
  del audio_data
  del clip
  print("x1 complete")
  #x2=[i[1] for i in audio_data]
  F, f_names = audioFeatureExtraction.stFeatureExtraction(x1, sample_rate, 0.050*sample_rate, 0.025*sample_rate) #Not sure if this is the right frame size/overlap for classical music
  print("F exists")
  del x1
  sound_df=pd.DataFrame(data=F.transpose(),columns=f_names)
  sound_specs=sound_df.describe()
  del sound_df
  print("sound_specs exists")
  sound_specs = sound_specs.stack().to_frame().T
  sound_specs.columns = ['{}_{}'.format(*c) for c in sound_specs.columns]
  sound_specs = sound_specs.loc[:, ~sound_specs.columns.str.startswith('count_')]
  sound_specs['VideoId']=query #p  #NEW 10/2
  all_df=pd.merge(videos_df, sound_specs, on = 'VideoId')
  print("all_df exists")
  
  del sound_specs
  #Create text columns
  from nltk import word_tokenize, pos_tag, ne_chunk
  import nltk
  from collections import Counter
  #nltk.download('punkt')
  def count_ne(example):
        y=pos_tag(word_tokenize(example))
        counts = Counter(tag for word,tag in y)
        return counts['NNP']
  all_df['Description_ne_counts']= list(map(lambda x: count_ne(x), all_df['Description']))
  all_df['Title_ne_counts']= list(map(lambda x: count_ne(x), all_df['Title']))
  def count_ne_tags(example):
        ind=0
        for item in example:
            y=pos_tag(word_tokenize(item))
            counts = Counter(tag for word,tag in y)
            if counts['NNP']>0: 
                ind+=1
        return ind
  all_df['tags_ne_counts']= list(map(lambda x: count_ne_tags(x), all_df['tags']))
  translator = str.maketrans('', '', string.punctuation)
  all_df['tags'] = list(map(lambda x: [item.lower() for item in x], all_df['tags']))
  all_df['Title'] = list(map(lambda x: x.translate(translator).lower(), all_df['Title']))
  all_df['Description'] = list(map(lambda x: x.translate(translator).lower(), all_df['Description']))
  #remove special characters
  textrefs=pd.read_csv('Text lists.csv')
  composers=textrefs['Composer last name'].dropna()
  composers=list(map(lambda x: x.lower().rstrip(),composers))
  artists=textrefs['Famous violinists and pianists'].dropna()
  artists=list(map(lambda x: x.lower().rstrip(),artists))
  instruments=textrefs['Musical instrument'].dropna()
  instruments=list(map(lambda x: x.lower().rstrip(),instruments))
  genres=textrefs['Genre'].dropna()
  genres=list(map(lambda x: x.lower().rstrip(),genres))
  piece=textrefs['Piece type'].dropna()
  piece=list(map(lambda x: x.lower().rstrip(),piece))
  complete=textrefs['completeness'].dropna()
  complete=list(map(lambda x: x.lower().rstrip(),complete))
  def count_vars(example,elements):
        c=Counter(example.split())
        num=0
        for word in elements:
            num+=c[str(word)]
        return num
  def count_vars_tags(example,elements):
        c=Counter(example)
        num=0
        for word in elements:
            num+=c[str(word)]
        return num
    #count_vars(videodata['Description'][0],composers)
  all_df['tags_composer'] = list(map(lambda x: count_vars_tags(x,composers), all_df['tags']))
  all_df['Title_composer'] = list(map(lambda x: count_vars(x,composers), all_df['Title']))
  all_df['Description_composer'] = list(map(lambda x: count_vars(x,composers), all_df['Description']))
  all_df['tags_artists'] = list(map(lambda x: count_vars_tags(x,artists), all_df['tags']))
  all_df['Title_artists'] = list(map(lambda x: count_vars(x,artists), all_df['Title']))
  all_df['Description_artists'] = list(map(lambda x: count_vars(x,artists), all_df['Description']))
  all_df['tags_instruments'] = list(map(lambda x: count_vars_tags(x,instruments), all_df['tags']))
  all_df['Title_instruments'] = list(map(lambda x: count_vars(x,instruments), all_df['Title']))
  all_df['Description_instruments'] = list(map(lambda x: count_vars(x,instruments), all_df['Description']))
  all_df['tags_genres'] = list(map(lambda x: count_vars_tags(x,genres), all_df['tags']))
  all_df['Title_genres'] = list(map(lambda x: count_vars(x,genres), all_df['Title']))
  all_df['Description_genres'] = list(map(lambda x: count_vars(x,genres), all_df['Description']))
  all_df['tags_piece'] = list(map(lambda x: count_vars_tags(x,piece), all_df['tags']))
  all_df['Title_piece'] = list(map(lambda x: count_vars(x,piece), all_df['Title']))
  all_df['Description_piece'] = list(map(lambda x: count_vars(x,piece), all_df['Description']))
  all_df['tags_complete'] = list(map(lambda x: count_vars_tags(x,complete), all_df['tags']))
  all_df['Title_complete'] = list(map(lambda x: count_vars(x,complete), all_df['Title']))
  all_df['Description_complete'] = list(map(lambda x: count_vars(x,complete), all_df['Description']))
  all_df['tags_count']=list(map(lambda x: len(x), all_df['tags']))
  all_df['Title_length']=list(map(lambda x: len(x), all_df['Title']))
  from textblob import TextBlob
  all_df['Title_pos']=list(map(lambda x: TextBlob(x).sentiment[0], all_df['Title']))
  all_df['Description_pos']=list(map(lambda x: TextBlob(x).sentiment[0], all_df['Description']))
  all_df['tags_pos']=list(map(lambda x: TextBlob(" ".join(str(i) for i in x)).sentiment[0], all_df['tags']))

  #Project features to PCAs and convert to log for channelSubscribers, viewCount, dayslive
  all_data=all_df.drop(['publishedAt','tags','VideoId'], axis=1)
  #import saved PCA models
  pca = pickle.load(open('pca_model_mfcc.sav', 'rb'))
  X=all_data[['mean_mfcc_1','mean_mfcc_2','mean_mfcc_3','mean_mfcc_4','mean_mfcc_5','mean_mfcc_6','mean_mfcc_7','mean_mfcc_8','mean_mfcc_9','mean_mfcc_10','mean_mfcc_11','mean_mfcc_12','mean_mfcc_13','std_mfcc_1','std_mfcc_2','std_mfcc_3','std_mfcc_4','std_mfcc_5','std_mfcc_6','std_mfcc_7','std_mfcc_8','std_mfcc_9','std_mfcc_10','std_mfcc_11','std_mfcc_12','std_mfcc_13']]
  Xreg=pca.transform(X)
  print(Xreg)
  mfccdata=pd.DataFrame(Xreg,columns=['mfccPC1','mfccPC2','mfccPC3'])
  all_data=pd.concat([all_data,mfccdata],axis=1)
  pca2= pickle.load(open('pca_model_text.sav', 'rb'))
  Xtext=all_data[['tags_ne_counts','Title_ne_counts','Description_ne_counts','tags_pos','Title_pos','Description_pos','tags_composer','Title_composer','Description_composer','tags_instruments','Title_instruments','Description_instruments','tags_artists','Title_artists','Description_artists','tags_genres','Title_genres','Description_genres', 'tags_piece', 'Title_piece', 'Description_piece','tags_complete', 'Title_complete', 'Description_complete']]
  Yreg=pca2.transform(Xtext)
  textdata=pd.DataFrame(Yreg,columns=['textPC1','textPC2','textPC3','textPC4','textPC5','textPC6'])
  all_data=pd.concat([all_data,textdata],axis=1)
  #Create classifier column
  #Create a classifier variable equal to 1 if viewCount is greater than 1, 0 otherwise
  all_data['views_cat'] = np.where(all_data['viewCount']<100, 0, 1) 
  #Create another category for viral videos
  all_data.loc[all_data['viewCount'] >=20000, 'views_cat'] = 2 

  #Get rid of excess columns and reorder to match correct dataset
  all_data_reduced_final=all_data[['Desc_word_count', 'channelSubscribers', 'mean_energy_entropy',
       'mean_spectral_entropy', 'mean_spectral_spread', 'mean_zcr',
       'std_chroma_std', 'std_energy_entropy', 'dayslive', 'tags_count',
       'Title_length', 'textPC1', 'textPC2', 'textPC3', 'textPC4', 'textPC5',
       'textPC6', 'mfccPC1', 'mfccPC2', 'mfccPC3','views_cat']]
  Y_test=all_data_reduced_final['views_cat']
  X_test=all_data_reduced_final.drop(['views_cat','mean_zcr','mfccPC3','textPC2','std_energy_entropy'],axis=1)
  #Create a bunch dataset
  testingdataset = sklearn.datasets.base.Bunch(data=X_test.values, target=Y_test.values, target_names=Y_test.name, feature_names=X_test.columns)
  return testingdataset

def get_results(data_bunch):
  #Run input data through model
  filename = 'final_xgb_model.sav'
  loaded_model = pickle.load(open(filename, 'rb'))
  target_test=data_bunch.target
  test=data_bunch.data
  result2 = loaded_model.predict(test)
  return result2

def get_lime_results(data_bunch):
  #Run input data through model
  filename = 'final_xgb_model.sav'
  loaded_model = pickle.load(open(filename, 'rb'))
  traindata=pd.read_csv('databasetraindata.csv')
  traindata=traindata.drop(['Unnamed: 0'],axis=1)
  Y_train=traindata['views_cat']
  X_train=traindata.drop(['views_cat','mean_zcr','mfccPC3','textPC2','std_energy_entropy'],axis=1)
  trainingdataset = sklearn.datasets.base.Bunch(data=X_train.values, target=Y_train.values, target_names=Y_train.name, feature_names=X_train.columns)
  target_train=trainingdataset.target
  train=trainingdataset.data
  target_test=data_bunch.target
  test=data_bunch.data
  explainer = lime.lime_tabular.LimeTabularExplainer(train, feature_names=trainingdataset.feature_names, class_names=['No one watching','Some views','Viral views'], discretize_continuous=False)
  exp = explainer.explain_instance(test[0], loaded_model.predict_proba, num_features=10, top_labels=3)
  return exp


def get_imp_table(data_bunch,exp,pred):
    filename = 'final_xgb_model.sav'
    loaded_model = pickle.load(open(filename, 'rb'))
    # Create dataframe of feature ranking
    rankingtable=[]
    d_features={}
    d_features.update({'Desc_word_count':['Description length','Try writing more details in the description']})
    d_features.update({'tags_count':['Number of tags','Try adding relevant keywords to tags']})
    d_features.update({'Title_length':['Title length','Try adding more details to the title']})
    d_features.update({'mean_energy_entropy':['Mean energy entropy','Try normalizer or compression tool for volume issue or denoiser to address static or buzz']})
    d_features.update({'mean_spectral_spread':['Mean spectral spread','Try denoiser or check vibrato sound']})
    d_features.update({'mean_spectral_entropy':['Mean spectral entropy','Adjust mic during recording to pick up different registers across instrument(s)']})
    d_features.update({'std_chroma_std':['Standard deviation of chroma features','Check for uneven playback speeds, out-of-tune playing, or excess noise']})
    d_features.update({'mfccPC1':['First PCA component of pitch and power spectra','Check recording for hum or low frequency noise, then apply denoiser tool']})
    d_features.update({'mfccPC2':['Second PCA component of pitch and power spectra','Check recording for noise or buzz, then apply denoiser tool']})
    d_features.update({'textPC1':['Keywords in description','Try adding named entities to the description, especially composer and musical instruments']})
    d_features.update({'textPC3':['Keywords in title and tags','Check that keywords in tags reflect title and vice versa']})
    d_features.update({'textPC4':['Composer and tags','Check that composer is included across title, description, and tags, and that instruments, artists, named entities, and genre are included in tags']})
    d_features.update({'textPC5':['Description named entities and tag completeness','Add named entities associated with video to description and that relevant musical keywords are present in tags']})
    d_features.update({'textPC6':['Instruments','Check instrument included in title, description, and tags']})
    d_features.update({'channelSubscribers':['Channel subscribers','Recruit subscribers']})
    d_features.update({'dayslive':['Length of time posted','Wait for more views']})
    import emoji

#    d_features['Desc_word_count']='Try adding more description'
#    d_features['tags_count']='Try adding more tags relevant to your video'
#    d_features['Title_length']='Check title is detailed and relevant'
#    d_features['mean_zcr']='Try denoiser tool to remove static'
#    d_features['mean_energy_entropy']='Try normalizing or compression tool to increase volume, or denoiser to alter static/buzz sound'
#    d_features['mean_spectral_spread']='Try denoiser or check vibrato sound'
#    d_features['mean_spectral_entropy']='Try adjusting mic placement during recording to pick up different instruments or low and mid-frequencies of solo instrument'
#    d_features['std_energy_entropy']='Check for sudden noises or silence and manually patch with visual editing'
#    d_features['std_chroma_std']='Check for uneven playback speeds, out-of-tune playing, or excess noise'
#    d_features['mfccPC1']='If score is low, recording may have a hum, buzz, or other low frequency noise'
#    d_features['mfccPC2']='Try denoiser if recording is noisy or buzzy'
#    d_features['mfccPC3']='Power may be unbalanced across low and high frequencies.  Try higher quality compression'
#    d_features['textPC1']='Add more keywords and named entities to the description, especially the composer and musical instruments'
#    d_features['textPC2']='Add more named entities to the title and tags, and include musical instruments in the tags, title, and description'
#    d_features['textPC3']='Check that relevant named entities appearing in tags may be missing from title, or vice versa'
#    d_features['textPC4']='Mention composer in tags, title and description'
#    d_features['textPC5']='Composer, instruments, and artists may be missing from tags if this score is low'
#    d_features['textPC6']='Check that the musical instruments are mentioned in the description, title, and tags. Composer may be mentioned inconsistently acrosstext fields. Consider mentioning genre.'
#    d_features['channelSubscribers']='Recruit subscribers'
#    d_features['dayslive']='Wait for more views'
    
    indices=data_bunch.feature_names.tolist()
    print(indices)
    #current=np.where(pred>0,2,1)
    current=min(pred+1,2)
    current=np.asscalar(current)
    explanations=exp.local_exp
    explanations=explanations[current]
    #print(explanations)
    d_scores={}
    for i in range(len(explanations)):
      d_scores[explanations[i][0]]=explanations[i][1]
    for f in range(len(indices)):
        if f in d_scores:
            rankingtable.append({'Index':f,
                'Feature':d_features[indices[f]][0],
                'Current score':d_scores[f],
                'Suggestion':d_features[indices[f]][1],
                'Current performance':emoji.emojize(":smile:",use_aliases=True) if d_scores[f]>=0.025 else emoji.emojize(':rage:',use_aliases=True) if d_scores[f]<=-0.025 else emoji.emojize(':cold_sweat:',use_aliases=True) if d_scores[f]<=0 else emoji.emojize(':neutral_face:',use_aliases=True)})
        #sort ranking table so that worst scores are at the top
    rankingtable=pd.DataFrame(rankingtable)
    rankingtable=rankingtable.sort_values(by=['Current score'],ascending=True)
    rankingtable['Current score']=rankingtable['Current score'].round(2)
    rankingtable=rankingtable.drop(['Index'],axis=1)
    rankingtable=rankingtable.drop(['Current score'],axis=1)
    #rankingtable=rankingtable.to_dict('list')
    #return rankingtable.sort_values(by=['Your_score_on_the_next_highest_category'],ascending=True)
    return rankingtable
