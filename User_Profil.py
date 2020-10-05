# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 22:06:27 2020

@author: Administrateur
"""


import tkinter as tk
from tkinter import messagebox
import tweepy
from tweepy import OAuthHandler #authentificatin
from tweepy import API 


from wordcloud import WordCloud
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
#pour less clés de l'API Twitter
import twitter_credentials
import mots_vides
# =============================================================================
# numpy pour le calcul scientifique en Python pour la vectorisation
# =============================================================================
import numpy as np
# =============================================================================
# pandas fournissant des structures de données hautes performances et faciles
# à utiliser et des outils d'analyse de données
# =============================================================================
#Data Analysis
import pandas as pd
from tweepy import Cursor

#Data Preprocessing and Feature Engineering
from textblob import TextBlob
import re
#import Natural Languagr Tollkit
import nltk
#import la liste des stop words
from nltk.corpus import stopwords

STOPWORDS2 = nltk.corpus.stopwords.words('english')
newList = mots_vides.newList
for e in newList :
    STOPWORDS2.append(e)




#pour les mots vides
#import natural  language toolkit
from gensim.parsing.preprocessing import remove_stopwords
nltk.download('stopwords', quiet=True)
STOPWORDS = set(stopwords.words('english'))
#STOPWORDS = set(stopwords.words('french'))
#print (STOPWORDS)
#import enchant
#d = enchant.Dict("en_US")

#Pour les slogand
import requests
prefixStr = '<div class="translation-text">'
postfixStr = '</div'

#pour la traduction
from googletrans import Translator

# spacy for lemmatization
# Initialize spacy 'en' model, keeping only tagger component (for efficiency)
# python3 -m spacy download en
import spacy

nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
# Plotting tools
import matplotlib.pyplot as plt




from pprint import pprint

# Gensim
import gensim
import gensim.corpora as corpora
from gensim import models
from gensim.models.coherencemodel import CoherenceModel

# spacy for lemmatization
# Initialize spacy 'en' model, keeping only tagger component (for efficiency)
# python3 -m spacy download en
import spacy

nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
# Plotting tools


# Enable logging for gensim - optional

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)


#Le fichier XML
from lxml import etree
from datetime import datetime


#text classification
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn import metrics

data = fetch_20newsgroups()
cat = data.target_names
#print (data.target_names)
categories = cat
train = fetch_20newsgroups(subset='train', categories=categories)
test = fetch_20newsgroups(subset='test', categories=categories)
model = make_pipeline(TfidfVectorizer(), MultinomialNB())
model.fit(train.data, train.target)
labels = model.predict(test.data)
mat = confusion_matrix(test.target, labels)
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False,
            xticklabels=train.target_names, yticklabels=train.target_names)
plt.xlabel('true label')
plt.ylabel('predicted label');


print('Accuracy achieved is ' + str(np.mean(labels == test.target)))
print(metrics.classification_report(test.target, labels, target_names=test.target_names)),
metrics.confusion_matrix(test.target, labels)

# Enable logging for gensim - optional
'''
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)

import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)'''

# =============================================================================
# # # # ######################## TWITTER CLIENT ######################### # # #
# =============================================================================
class TwitterClient():
    #le constructeur
    def __init__(self, twitter_user=None):
        #authentifier pour communiquer avec l'API Twitter
        self.auth = TwitterAuthenticator().authenticate_twitter_app()
        #initialize Tweepy API
        self.twitter_client = API(self.auth, wait_on_rate_limit=True)
        #variable qui permet de definir le nome du client ayant le 1er timeline
        self.twitter_user = twitter_user
        
    def get_user_timeline_tweets(self, num_tweets, user_name):
        tweets = []
        for tweet in Cursor(self.twitter_client.user_timeline,screen_name=user_name).items(num_tweets):
            tweets.append(tweet)
        return tweets

   #getter de la variable ci-dessus de l'API
    def get_twitter_client_api(self):
        return self.twitter_client


# =============================================================================
# # # # # ################# TWITTER AUTHENTICATER ################# # # # #
# =============================================================================
# # # # TWITTER AUTHENTICATER # # # #
class TwitterAuthenticator():
    #S’authentifier à l’API avec les clés et les jetons d'accès
    def authenticate_twitter_app(self):
        auth = OAuthHandler(twitter_credentials.consumer_key, twitter_credentials.consumer_secret)
        auth.set_access_token(twitter_credentials.access_token, twitter_credentials.access_token_secret)
        return auth

# =============================================================================
#  # # # # #################### TWITTER Analyzer ##################### # # #
# =============================================================================
class TweetAnalyzer():
    """
    Functionalité pour l'analyse et la catégorisation de contenue des tweets 
    """
   
        


    def regex_or(*items):
        return '(?:'+'|'.join(items)+')'


    #fonction pour normaliser les slogans
    def sologan(self, tweet):
         r = requests.post('https://www.noslang.com/', {'action': 'translate', 'p': tweet, 'noswear': 'noswear', 'submit': 'Translate'})
         startIndex = r.text.find(prefixStr)+len(prefixStr)
         endIndex = startIndex + r.text[startIndex:].find(postfixStr)
         a= r.text[startIndex:endIndex]
         if (a.find("None of the words you entered are in our database.")) == (-1):
                   tweet = (a)
         return tweet
 
    #Fonction pour la traduction en anglais
    def traduction(self, tweet):
         translator = Translator()
         result = translator.translate(tweet)
         return result.text
     
    #fonction pour nettoyer les tweets 
    def clean_tweet(self, tweet):
        #Hashtag = "#[a-zA-Z0-9_]+"
        tweet=  re.sub('\S*@\S*\s?', '', tweet) #Supprimer les emails
        tweet = re.sub('https?:\/\/\S+', '', tweet) #Supprimer les URL
        #tweet = re.sub(url, '', tweet) #Supprimer les URL
        tweet = re.sub('@[A-Za-z0-9]+', '', tweet) #supprimer les @mentions
        tweet = re.sub('[@＠][a-zA-Z0-9_]+', '', tweet) #supprimer les @mentions
        tweet = re.sub("\'", "", tweet) # Remove distracting single quotes
        tweet = re.sub('#', '', tweet) #supprimer les #
        tweet = re.sub("^\d+\s|\s\d+\s|\s\d+$", " ", tweet) #delete numbers
        tweet = re.sub('/', '', tweet) #supprimer les /
        tweet = re.sub('RT[\s]+', '', tweet) #Supprimer RT
        tweet = re.sub('^retweet+', '', tweet) #Supprimer RT
        tweet = re.sub('[^\w\s@/:%_-]', '', tweet) #Supprimer emoji et ponctuations
        #return ' '.join(re.sub(r"(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", tweet).split())
        return tweet 
    
    
  
        
    #analyser le point de polarité des point de sentiments
    def analyze_sentiment(self, tweet):
        analysis = TextBlob(tweet)
        
        if analysis.sentiment.polarity > 0:
            return 1
        elif analysis.sentiment.polarity == 0:
            return 0
        else:
            return -1
        
        
        
    #calculer la subjectivité
    def analyze_sub(self, tweet):
     if TextBlob(tweet).sentiment.subjectivity >= 0.5:
         return 'Subjective'
     else:
         return 'objective'
     
        
        
    #convertir les grands données de format json to data frame
    def tweets_to_data_frame(self, tweets):
        #créer l'objet dataframe (df)
        #la boucle : pour chaque signle tweet dans la liste tweetsde fichier enregistré, nommée la comumns tweets
        df = pd.DataFrame(data=[tweet.text for tweet in tweets], columns=['tweets'])
        df['tweets'] = df['tweets'].apply(self.traduction) 
        #suppression de Stop-words comme “a”, “an”, “the”, “of”,
        df['tweets'] = df['tweets'].apply(lambda x: ' '.join([word for word in x.split() if word not in (STOPWORDS)]))
        num=0
        for i in df['tweets']: 
           r = requests.post('https://www.noslang.com/', {'action': 'translate', 'p': 
           i, 'noswear': 'noswear', 'submit': 'Translate'})
           startIndex = r.text.find(prefixStr)+len(prefixStr)
           endIndex = startIndex + r.text[startIndex:].find(postfixStr)
           a= r.text[startIndex:endIndex]
           if (a.find("None of the words you entered are in our database.")) == (-1):
               df.loc[num,'tweets'] = (a)
          # print(df.loc[num,'tweets'].text[startIndex:endIndex])
           print(df.loc[num,'tweets'])
           num = num+1
        
        #analyse de données
        df['date'] = np.array([tweet.created_at for tweet in tweets])
        df['tweet_id']=np.array([tweet.id for tweet in tweets])
        
        return df
    
    #Fonction pour supprimer complètement les ponctuations et les caractères inutiles des tweets
    def sent_to_words(self, sentences):
     for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True pour effacer punctuations
    
        
    #deux mots qui apparaissent fréquemment ensemble
    def construire_bigrams(self, texts):
     return [bigram_mod[doc] for doc in texts]

    #accepter que les noms et les adjectives
    def lemmatization(self, texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
       """https://spacy.io/api/annotation"""
       #la sorties est une liste contenat que les adj et les noms
       texts_out = []
       #Pour chaque mot de text
       for sent in texts:
          # ajouter les deux " " dans chaque mot
          doc = nlp(" ".join(sent)) 
          #accepter que les adj et noms dans la liste de sortie
          texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
       return texts_out  
    
    
# =============================================================================
# # # # # ################# personal information ################# # # # #
# =============================================================================
class UserInfo():
    
    def persoInfo(self,user_name,api):
        persoInfo = []
        info= api.get_user(user_name)
        user_id = info.id_str #Représentation sous forme de chaîne de l'identifiant unique de cet utilisateur. 
        name=info.name #Le nom de l'utilisateur, tel qu'il l'a défini.
        location = info.location #Emplacement défini par l'utilisateur pour le profil de ce compte
        persoInfo.append(user_id)
        persoInfo.append(name)
        persoInfo.append(location)
        
        return persoInfo
    
    def compteInfo(self,user_name,api):
        donneesInfo = []
        info= api.get_user(user_name)
        screenName=info.screen_name #Le nom d'écran, le descripteur ou l'alias avec lequel cet utilisateur s'identifie.
        description= info.description #La chaîne UTF-8 définie par l'utilisateur décrivant son compte.
        followers_count=info.followers_count #Le nombre d'abonnés de ce compte actuellement. 
        friends_count=info.friends_count #Le nombre d'utilisateurs que ce compte suit (AKA leurs «suiveurs»).
        cration_dat=info.created_at #Heure UTC à laquelle le compte utilisateur a été créé sur Twitter.
       
        donneesInfo.append(screenName)
        donneesInfo.append(description)
        donneesInfo.append(followers_count)
        donneesInfo.append(friends_count)
        donneesInfo.append(cration_dat)
        
        return donneesInfo 

# =============================================================================
# #  # # # # ######################## Mywindow ######################### # # #
# =============================================================================


class MyWindow(tk.Tk):
    
    def __init__(self):
        tk.Tk.__init__(self)
        frame=tk.Frame(self,bg='#4065A4')
        
        self.__name = tk.StringVar()
        #self.iconbitmap("C:/Users/pc/Desktop/cerist.ico")
        self.config(background='#4065A4')
        label = tk.Label( frame, text="veuillez saisir le nom d'utilisateur SVP :",bg='#4065A4',font=("Times New Roman",12),fg="white")
        label.pack()
        label = tk.Label(frame, text=" " ,bg='#4065A4')
        label.pack()
        name = tk.Entry(frame, textvariable=self.__name )
        name.focus_set()
        name.pack()
        label = tk.Label(frame, text=" " ,bg='#4065A4')
        label.pack()
        button = tk.Button( frame, text="Connect!", command=self.doSomething)
        button.pack(
            )
        
        self.geometry( "300x200" )
        self.title( "User Name" )
        frame.pack(expand=tk.YES)
        
    def doSomething(self):
        if (self.__name.get() ==""):
             messagebox.showinfo("Alert", "veuillez ne pas laisser la case vide")
        else :
            return( self.__name.get() )
        
# =============================================================================
#  # # # # #################### TWITTER TOPICS ##################### # # #
# =============================================================================        
class TweetTopics():   
    
    def predict_category(s, train=train, model=model):
        pred = model.predict([s])
        return train.target_names[pred[0]]


    #Pour avoir le meilleur nombre de topics
    #fonction qui retourne les modele (selon le nombre de topics) et la valeur de coherence pour chaque modèle
    def compute_coherence_values(self, Dict, corpus, texts, start, limit, step, max):
        #liste des valeurs de cohérences
        coherence_values = []
        #liste des modèles
        model_list = []
        #pour chaque nombre de topics appliquer LDA
        for num_topics in range(start, limit, step):
            model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=num_topics, 
                                           random_state=100,
                                           update_every=1,
                                           chunksize=100,
                                           passes=10,
                                           alpha='auto',
                                           per_word_topics=True) 
            #Avoir la list de modeles LDA pour avoir acces directement au meilleure modèle 
            model_list.append(model)
            #Calculer la cohérence
            coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=Dict, coherence='u_mass')
            co = coherencemodel.get_coherence()
            #ajouter la valeur de coherence de ce modèle a la liste des coherences
            coherence_values.append(coherencemodel.get_coherence())
            #arreter l'algorithmes ssi la coherence est maximale
            if (co < max):
                max = co
            else :
                #max >= 0.75:
                return model_list, coherence_values
            #print(coherence_values)
        return model_list, coherence_values
    
    #fonction 
    def format_topics_sentences(self, ldamodel, corpus, texts):
        # Init output
        sent_topics_df = pd.DataFrame(pd.np.empty((0, 3)))
     
        # Get main topic in each document
        for i, row in enumerate(ldamodel[corpus]):
            row = sorted(row[0], key=lambda x: (x[1]), reverse=True)
           
            # Get the Dominant topic, Perc Contribution and Keywords for each document
            for j, (topic_num, prop_topic) in enumerate(row):
               
                if j == 0:  # => dominant topic
                    wp = ldamodel.show_topic(topic_num)
                    topic_keywords = ", ".join([word for word, prop in wp])
                    sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic,4), topic_keywords]), ignore_index=True)
                else:
                    break
        sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']
       
        # Add original text to the end of the output
        contents = pd.Series(texts)
        sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
        return(sent_topics_df)    
    
    
# =============================================================================
# # # # # ################# XML FILE ################# # # # #
# =============================================================================
class XML_USER():
    
    def create_xml(self, Perso_Info, Compte,topicsIntersets):
        from datetime import datetime
        tag_Perso = ["User_id", "Name", "Location"]
        tag_Compte = ["ScreenName", "Description", "Followers_count", "Friends_count", "Creation_date"]
        now = datetime.now()
        User = etree.Element("User_Profile")
        #User.text  = "\n"
        
        dataTime = etree.SubElement(User,'Date')
        Day = etree.SubElement(dataTime,'Day')
        Day.text = now.strftime("%d")
        Month = etree.SubElement(dataTime,'Month')
        Month.text = now.strftime("%B")
        Year = etree.SubElement(dataTime,'Year')
        Year.text = now.strftime("%Y")
        Hour = etree.SubElement(dataTime,'Time')
        Hour.text = now.strftime("%H:%M:%S")
      
  
        usrPersoo = etree.SubElement(User,'Personal_data')
        #usrPerso = ET.SubElement(usrPerso,"Dimension Données_Personnelles")
        i=0
        for user in range(len( Perso_Info)):
                usr = etree.SubElement(usrPersoo,tag_Perso[i])
                usr.text = str(Perso_Info[user])
                i=i+1
        

        usrCompte = etree.SubElement(User,'Account_Data')
        i=0
        for user in range(len( Compte)):
                usr = etree.SubElement(usrCompte,tag_Compte[i])
                usr.text = str(Compte[user])
                i=i+1
          
        SujetInteret = etree.SubElement(User,'Topics_of_interest')
        for i in range(len( topicsIntersets)):
                Topic = etree.SubElement(SujetInteret,'Interest'+str(i+1))
                interet = etree.SubElement(Topic,'Name_Topic')
                interet.text = str(topicsIntersets[i][0])
                score = etree.SubElement(Topic,'Score')
                score.text = str(topicsIntersets[i][1])
                sous_topic = etree.SubElement(Topic,'Sub_Topic')
                
                nom_Stopics = etree.SubElement(sous_topic,'Key_words')
                nom_Stopics.text = str(topicsIntersets[i][2])
        
        
        tree = etree.ElementTree(User)           
        tree.write(str(Perso_Info[1])+".xml",encoding='utf-8', xml_declaration=True, pretty_print=True)

# =============================================================================
# #  # # # # ######################## MAIN ######################### # # #
# =============================================================================
 #cree les objets
if __name__ == '__main__':
    
    twitter_client = TwitterClient()
    tweet_analyzer = TweetAnalyzer()
    #créer object de la class TweetTopics
    tweet_topics = TweetTopics()
    userInfo=UserInfo()
    fileXml= XML_USER()
 
    #api variable contenant l'object client crée dans la class twitter_client
    api = twitter_client.get_twitter_client_api()
   
    '''window = MyWindow()
    window.mainloop()
    #computer science : compscifact
    #politique : nprpolitics
    #food : TwitterFood
    #foot : Mahrez22
    user_name =window.doSomething()'''
    #user_name = 'tebbouneamadjid'
    #user_name= 'CompSciFact'
    user_name = 'Cuisineetmets'
    #user_name = "pfe63694949"
    print(user_name)
    #user_name = "DJessica987"
    #tartecosmetics
    #commencer le streaming  avec la fonction (user_timeline) de Twitter client
    #tweets = api.user_timeline(screen_name=user_name, count=1000)
    #tweets = twitter_client.get_user_timeline_tweets(1000, user_name)
    #récuperer les informations personnelles de l'utilisteur'''
    information_user=userInfo.persoInfo(user_name,api)
    print (information_user)
    #récuperer les informationsde compte de l'utilisteur
    compte_user = userInfo.compteInfo(user_name,api)
    
    
    '''df = tweet_analyzer.tweets_to_data_frame(tweets)
    df['tweets'] = df['tweets'].apply(tweet_analyzer.clean_tweet)
   #analyse des sentiments
    df['subjectivite'] = df['tweets'].apply(tweet_analyzer.analyze_sub)
    df['sentiment'] = np.array([tweet_analyzer.analyze_sentiment(tweet) for tweet in df['tweets']])
    df = df[df['tweets'].notnull()]
    #df['subjectivité'] = np.array([tweet_analyzer.analyze_sub(tweet) for tweet in df['tweets']])
    #plot the word cloud
    allWords = ' '.join( [twts for twts in df['tweets']])
    wordCloud = WordCloud(width = 500, height = 300, random_state = 21, max_font_size = 119).generate(allWords)
    plt.imshow(wordCloud, interpolation= "bilinear")
    plt.axis('off')
    plt.show()
    #Afficher les valeurs de sentimens
    df['sentiment'].value_counts()
    #plot et visualisation
    plt.title("Sentiment Analysis")
    plt.xlabel("Sentiment")
    plt.ylabel("Count")
    df['sentiment'].value_counts().plot(kind='bar')
    plt.show()
    df['replay'] = np.array([' ' for tweet in df['tweets']])
    a= -1
    for tweet_id in df['tweet_id'] :
        a= a+1
        replies = tweepy.Cursor(api.search, q='to:{}'.format(user_name),
                                since_id=tweet_id, tweet_mode='extended').items()
        liste = [] 
        while True:
            try:
                reply = replies.next()
                #print ('type')
                #print (type(reply))
                if not hasattr(reply, 'in_reply_to_status_id_str'):
                    continue
                if reply.in_reply_to_status_id == tweet_id:
                    liste.append(reply.full_text)
                    print (a)
                    #df.loc[a, 'replay'] =df.loc[a, 'replay']+reply.full_text
                    print(format(reply.full_text))
                    #logging.info("reply of tweet:{}".format(reply.full_text))
            except tweepy.TweepError as e:
                logging.error("Tweepy error occured:{}".format(e))
                break
            except StopIteration:
                break
            except Exception as e:
                logger.error("Failed while fetching replies {}".format(e))
                break
        li = liste
        if li :
            listresultat = []
            print('\nlist de cmnt ', li)
            for elements in li:
                print ('\ncomnt',elements) 
                elements = tweet_analyzer.traduction(elements)            
                elements = remove_stopwords(elements)
                elements = tweet_analyzer.sologan(elements)
                elements = tweet_analyzer.clean_tweet(elements)
                listresultat.append(elements)
               
            df.at[a, 'replay'] = listresultat
            print(df.at[a, 'replay'])
            
            
         

            
            
            
   

    #Supprimer les tweets subjectives ayant une polarité = -1
    i = -1
    for tweet in df['tweets']:
            i = i +1
            if (df['subjectivite'][i] == 'Subjective' and df['sentiment'][i] == -1):
              print ('a supp ', tweet)
              df = df[df['tweets'] != tweet]
    #convertir les tweets et les commentaires en miniscule
    df['tweets'] = df['tweets'].str.lower()
    #Appliquer la nouvelle liste des mots vides
    df['tweets'] = df['tweets'].apply(lambda x: ' '.join([word for word in x.split() if word not in (STOPWORDS2)]))
    #supprimer les rows ayant des tweets ou les comentaires vides
    df = df[df['tweets'] != '']
    #Exporter les tweets dans un fichier CSV
    df.to_csv('TweetDataSet.csv', index=True, encoding='utf-8')
    print(df.head(1000))'''




#Partie 2
    df = pd.read_csv("foodcommentaire.csv") 
    #ajouter le colonne Kays_comments pour les mot clés représentant le commentaire
    df['Keys_comments'] = np.array([' ' for tweet in df['tweets']])
    #ajouter la colonne Comments_CV pour la valeur e cohérence de chaque tweetéComnt apres une application du modèle LDA
    df['Comments_CV'] = np.array(['' for tweet in df['tweets']])
    #Le nom de topic de résultat de modèle LDA appliqué sur le corpus tweetséComents
   
    #Appliquer le modèle LDA pour les commentaires   
    #la ligne de commentaire
    print('\nApplication de LDA sur le corpus Commentaires & Tweets...')
    b = 0
    #Pour chaque commentaire
    for rep in df['replay']: 
            print('\n', b+1)
            #si le commentaire n'est pas vide  pour le tweet
            if (df.at[b,'replay'] != ' '):
                #regrouper les commentaires de tweet et le tweet dans la liste 
                liste = [df.at[b,'replay']]
                liste.append(df.at[b,'tweets'])
                #puis ajouter le corpus dans df2
                df2 = pd.DataFrame(data = [l for l in liste], columns=['A'])
            #Si le commentaire est vide
            else :                #sinon si ya pas de commentaires pour ce tweet appliquer LDA uniquement pour le tweet
                liste = (df.at[b,'tweets'])
                #ajouter que le tweet dans df2
                df2 = pd.DataFrame(data = [liste], columns=['A'])
            #Supprimer les ponctuations et les caractères inutiles pour une 2ème fois
            #avoir liste de mots
            data_words = list(tweet_analyzer.sent_to_words(df2['A']))
            # Construire les modèles bigram & trigram avec gensim
            bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100) #seuil(threshold) plus élevé moins de phrases.
            trigram = gensim.models.Phrases(bigram[data_words], threshold=100)  
            # La façon la plus rapide pour obtenir une phrase matraquée en trigramme/bigramme
            bigram_mod = gensim.models.phrases.Phraser(bigram)
           
            #print(trigram_mod[bigram_mod[data_words[0]]])
            #Construire le bigram
            data_words_bigrams = tweet_analyzer.construire_bigrams(data_words)
            # Appliquer la  lemmatisation conserver que  noun, adj
            data_lemmatized = tweet_analyzer.lemmatization(data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])
            #print(data_lemmatized)
            #Les deux principales entrées du modèle de sujet LDA sont le dictionnaire ( id2word) et le corpus. 
            # Créer le  Dictionnaire
            id2word = corpora.Dictionary(data_lemmatized)
            # Créer le  Corpus
            #récupere le text lemmatiqé
            texts = data_lemmatized
            # calculer la fréqunce de chaque terme dans le corpus
            corpus = [id2word.doc2bow(text) for text in texts]
            #TF-IDF
            tfidf = models.TfidfModel(corpus)
            tfidf_corpus = tfidf[corpus]
            #si le corpus n'st pas vide...
            #print('corpus', len(corpus))
            if corpus != [[]] and corpus!= [[], []]:
                #print('\ncorpus : ',corpus)
                #pour un tweet et ses commentaires on parle d'un sujet unique
                numTopic = 1
                
                # Modélisation de sujet via LDA
                #chunksizeest le nombre de documents à utiliser dans chaque bloc de formation.
                #update_everydétermine la fréquence à laquelle les paramètres du modèle doivent être mis à jou
                #passescorrespond au nombre total de passes de formation.
                lda_model = gensim.models.ldamodel.LdaModel(corpus=tfidf_corpus,
                                                       id2word=id2word,
                                                       num_topics=numTopic, 
                                                       random_state=100,
                                                       update_every=1,
                                                       chunksize=100,
                                                       passes=50,
                                                       alpha=0.001,
                                                       per_word_topics=True)
                #Alpha (densité de sujet de document), Nous avons utilisé 0.001 car chaque tweet est assez court et ne comportera probablement qu'un seul sujet.
                
                #sauvgarder le sac à mots de chaque tweet dans la ligne qui convient
                Topic = lda_model.print_topics()
                #Inferer le nom de topics
                
                       
                for i in range (len(Topic)):
                    #☺print('\n\n 3yiiiiiiiiiiiiiiit\n', Topic)
                    mot = Topic[i][1]
                    index=0
                    topic =' '
                    listeTopic = []
                    while index < len(mot):
                        if (mot[index] == '"') :
                            index = index+1
                            while (mot[index] != '"'):
                                topic = topic + mot[index]
                                index = index+1
                            listeTopic.append(topic)
                        topic=''
                        index = index+1
                        #afficher les keywirds de topic
                    pprint(lda_model.print_topics())
                    df.at[b, 'Keys_comments'] = listeTopic
                doc_lda = lda_model[corpus]
                #pour juger de la qualité d'un modèle de sujet
                # Calculer Perplexity
                #print('\nPerplexity: ', lda_model.log_perplexity(corpus))  # a measure of how good the model is. lower the better.  
                # Calculer le score de Coherence
               
                '''coherence_model_lda = CoherenceModel(model=lda_model, texts=data_lemmatized, dictionary=id2word, coherence='c_v')
                
                coherence_lda = coherence_model_lda.get_coherence()
                print('Coherence Score: ', coherence_lda)
                df.at[b, 'Comments_CV'] = coherence_lda'''
               
            b = b+1
            # Visualize the topics
            #pyLDAvis.enable_notebook()
            #vis = pyLDAvis.gensim.prepare(lda_model, corpus, id2word, sort_topics = True)
            #vis_data = pyLDAvis.sklearn.prepare(lda_model, corpus, id2word)
            #pyLDAvis.show(vis)
   
    
   
    #Pour les tweets comme corpus
    print('\nApplication de LDA sur le corpus de Tweets...')  
    #Supprimer les ponctuations et les caractères inutiles pour une 2ème fois pour les tweets
    #avoir liste de mots
    data_words = list(tweet_analyzer.sent_to_words(df['tweets']))
    #Construire les modèles bigram & trigram avec gensim
    bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100) # higher threshold fewer phrases.
    trigram = gensim.models.Phrases(bigram[data_words], threshold=100)  
    # La façon la plus rapide pour obtenir une phrase matraquée en trigramme/bigramme
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    trigram_mod = gensim.models.phrases.Phraser(trigram)
    #Construire Bigrams
    data_words_bigrams = tweet_analyzer.construire_bigrams(data_words)
    #Appliquer la  lemmatisation conserver que  noun, adj
    data_lemmatized = tweet_analyzer.lemmatization(data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])
    # Créer le Dictionnaire
    id2word = corpora.Dictionary(data_lemmatized)
    #id2word.filter_extremes (no_below = 3, no_above = 0.5)
    # Créer le Corpus
    texts = data_lemmatized
    # calculer la fréqunce de chaque terme dans le corpus
    corpus = [id2word.doc2bow(text) for text in texts]
    #Appliquer TF-IDF
    tfidf = models.TfidfModel(corpus)
    tfidf_corpus = tfidf[corpus]
   #Choisir le nombre de step selon le nombre de tweets
    Len_tweets= len(df['tweets'])
    if(Len_tweets <= 50) :
        step = 1
    elif Len_tweets > 50 and Len_tweets <= 500:
        step = 3
    else:
        step = 5 
    
    #récuper tous les modèles et leurs valeur de cohérence pour choisir le meilleur modèle par la suite
    print('\nChercher le nombre optimal de topics pour LDA...')
    model_list, valeur_de_coherence = tweet_topics.compute_coherence_values(Dict = id2word, corpus=tfidf_corpus, texts=data_lemmatized, start=2, limit=Len_tweets, step=step, max = 0)
    # Afficher le graph
    #start=2 psq le nombre de sujet pour le corpus doit être superieure à 1 logiquement
    limit=len(valeur_de_coherence)+2; start=2; 
    x = range(start, limit, 1)
    #Visualisation de graph des topics avec leurs coherences
    plt.plot(x, valeur_de_coherence)
    plt.xlabel("Num Topics")
    plt.ylabel("Score de coherence")
    plt.legend(("valeur_de_coherence"), loc='best')
    plt.show()
    # Print the coherence scores
    m = 1
    max = 0
    for m, cv in zip(x, valeur_de_coherence):
        #print("Num Topics =",m , " a la valeur de Coherence ", round(cv, 4))
        if (cv < max):
            max = cv
            M_max = m
    print("Num Topics =",M_max , " a la valeur de Coherence ", round(max, 4))   
    # Selectionner le modele et afficher les Topics
    index = M_max-2 #psq nous avons commencer avec numTopic=2 (la valeur start)
    optimal_model = model_list[index]
    optimal_cv = valeur_de_coherence [index]
    model_topics = optimal_model.show_topics(formatted=False)
    print('\nla distribution : ')
    pprint(optimal_model.print_topics(num_words=10))
    #Trouver le sujet dominant dans chaque tweet
    df_topic_sents_keywords = tweet_topics.format_topics_sentences(ldamodel=optimal_model, corpus=corpus, texts=df['tweets'])
    # Format
    df_dominant_topic = df_topic_sents_keywords.reset_index()
    df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keys_Tweets', 'Text']
    df['Keys_Tweets'] = np.array([a for a in df_dominant_topic['Keys_Tweets']])
    df['Topic_Perc_Contrib'] = np.array([b for b in df_dominant_topic['Topic_Perc_Contrib']])
    df['Tweets_CV'] =  np.array([optimal_cv for b in df_dominant_topic['Topic_Perc_Contrib']])
   
    #Avoir l'intersection entre les mots clés des commentaires et celles des tweets pour une meilleure précisison
    key_ID = 0
    for key in  df['Keys_Tweets'] :
        mot = ''; index =0; listMot = []
        while index < len(key):
            while (index < len(key)) and (key[index]) != ',' :
                   mot = mot + key[index]
                   index = index + 1
            listMot.append(mot)
            mot = ''
            index = index+2
        df.at[key_ID, 'Keys_Tweets'] = listMot
        key_ID = key_ID+1
    
    df['key_Topics'] = np.array(['' for tweet in df['tweets']])
    i= 0
    for tweet in df['tweets']:
        #if(list(set(df['Keys_Tweets'][i]) & set(df['Keys_comments'][i])) ):
            #df.at[i, 'key_Topics'] = list(set(df['Keys_Tweets'][i]) & set(df['Keys_comments'][i]))
        df.at[i, 'key_Topics'] = list(set(df['Keys_Tweets'][i]).union(df['Keys_comments'][i]))
        i=i+1
        
    def predict_category(s, train=train, model=model):
        pred = model.predict([s])
        return train.target_names[pred[0]]
    def convert_list_to_string(org_list, seperator=' '):
        return seperator.join(org_list)
     
    '''full_str = convert_list_to_string(['play', 'ball'])
    print('\nLa catégorie de : ', full_str , ' est-->', predict_category(full_str ))'''
   #inferer le nom de topic
    df['Nom_Topics'] = np.array(['' for tweet in df['tweets']])
    i= 0
    for tweet in df['tweets']:
        if (df['key_Topics'][i] != ''):
        
           full_str = convert_list_to_string(df['key_Topics'][i])
           #print('\nLa catégorie de : ', full_str , ' est-->', predict_category(full_str ))
           var= predict_category(full_str)
           if (var=="sci.med"):
               var="food & health"
           if ((var=="rec.sport.baseball") or (var=="rec.sport.hockey")):
                var="sport & game"
           if (var=='alt.atheism'):
               var ="atheism"
           if (var=="comp.graphics"):
               var="graphics"
           if (var=="comp.os.ms-windows.misc"):
               var="operating system"
           if ((var=="comp.sys.ibm.pc.hardware") or (var =="comp.sys.mac.hardware")): 
               var="hardware"
           if (var=="comp.windows.x"):
               var="computer science"
           if (var=="misc.forsale"):
               var="forsale"
           if ((var=="rec.autos") or (var =="rec.motorcycles")):
               var="automobile & motorcycles"
           if (var=="sci.crypt"):
               var="cryptography"
           if (var=="sci.electronics"):
               var="electronics"
           if (var=="sci.space"):
               var="space"
           if (var=="soc.religion.christian"):
               var="religion & music"     
           if ((var =="talk.politics.guns") or (var=="talk.politics.mideast") or (var=="talk.politics.misc") ):
               var="politics"     
                
           df.at[i, 'Nom_Topics'] = var
           
      #Claculer la valeur de cohérence de code     
        i=i+1
        '''j=0
        somme=0
        for cv in df['Comments_CV']:
            if (cv!=''):
                j=j+1
                somme=somme+cv
        moyen_co=(((somme/j)+coherence_lda)/2)
    print(moyen_co)'''
    
    #pondérer les tweets    
    from math import exp
    from datetime import datetime
    now = datetime.now()
    import datetime      
    
    lamda=0.4
    k= 0
    
    for tweet in df['tweets']:
        Ann_Act = now.strftime("%Y")
        #t = dateo.month-parse(df['date'][k]).month
        Ann_Pub = datetime.datetime.strptime(str(df['date'][k]), "%Y-%m-%d %H:%M:%S").year
        t = int(float(Ann_Act)-float(Ann_Pub))
        if (df['sentiment'][k] == 1):
            sentiment = 0
        elif (df['sentiment'][k] == 0):
            sentiment = 2
        else:
            sentiment = 4
        #print (sentiment)
        #formule de pondération
        df.at[k, 'poids_tmp']= exp(-1*lamda*((t+sentiment)/2))
        
        k=k+1
        #df.drop(df.tail(1).index,inplace=True)
    

    
    
    a=[["food & health",0,[]],
   ["sport & game",0,[]],
   ["atheism",0,[]],
   ["graphics",0,[]],
   ["operating system",0,[]],
   ["hardware",0,[]],
   ["computer science",0,[]],
   ["forsale",0,[]],
   ["automobile & motorcycles",0,[]],
   ["cryptography",0,[]],
   ["electronics",0,[]],
   ["religion & misc",0,[]],
   ["politics",0,[]]
   ]

    t=0 
    while t < len (a):
        i=0
        for tweet in df['tweets']:
            if (df['Nom_Topics'] [i] == a[t][0]):
                a[t][1]= round(a[t][1]+df['poids_tmp'][i],2)
                a[t][2].append(df['key_Topics'][i])
            i = i+1
        
        t=t+1
    #claculer score total
    t=0
    TotalScore=0
    while t < len (a):
        TotalScore= TotalScore+a[t][1]
        t=t+1
        
    t=0
    while t < len (a):
        a[t][1] = a[t][1]*100/TotalScore
        a[t][1] = round( a[t][1], 2)
        my_new_list = []
        # Next we want to iterate over the outer list
        for sub_list in a[t][2]:
            # Now go over each item of the sublist
            for item in sub_list:
                # append it to our new list
                my_new_list.append(item)
            #print('rnakkkkaaa\n', my_new_list)
        a[t][2]= list(set(my_new_list))
        t=t+1
        
    topicsIntersets = [] 
    list_num =[]
    for t in a :
        if (t[1] > 3):
            #t[1]= str(t[1])+'%'
            list_num.append(float(t[1]))
            topicsIntersets.append(t)
            
    list_num.sort(reverse=True)
    Final_Liste =[]
     
    i = 0
    for num in list_num:
        while( topicsIntersets[i][1]!=num):
            i=i+1
            #print(topicsIntersets[i][1])
        
        Final_Liste.append(topicsIntersets[i])
        i=0
    
    for final in  Final_Liste : 
        final[1]= str(final[1])+'%'  
            
    fileXml.create_xml(information_user, compte_user,Final_Liste)
    
    
    
    
  
    
   

