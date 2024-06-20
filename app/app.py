"""
   This is a script representing the interactions as endpoints by using a Flask server. 
   
   https://auth0.com/blog/developing-restful-apis-with-python-and-flask/
   https://www.tutorialspoint.com/how-to-show-all-the-tables-present-in-the-database-and-server-in-mysql-using-python
"""

import os
import logging
import datetime
import numpy as np
import pandas as pd

#----------------------- BIBLIOTECAS PARA LA RED NEURONAl ----------------------

import re
import pickle 
import inflect

#Se importan todas las funcionalidades de Tensorflow para la red neuronal
#recurrente.
import tensorflow

#Para agregar los PAD_SEQUENCES.
from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.initializers import Constant
from tensorflow.keras import Sequential, Model, Input
from tensorflow.keras.layers import LSTM, Embedding, Dense, Dropout, Bidirectional


#--------------------------------------------------------------------------

#To create the directory for saving backups.
from pathlib import Path

#Note: request in Flask is no the same as requests. 
from flask import Flask, jsonify, request 

         #Logger for the exercise.          
#logger = logging.getLogger(__name__)

#https://stackoverflow.com/questions/50981906/change-default-location-log-file-generated-by-logger-in-python
#logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)-8s %(message)s', datefmt='%a, %d %b %Y %H:%M:%S', filename= f'{log_directory}database.log', filemode='w')         
    

#--------------------------------------------------------------------------
#--------------------------------------------------------------------------
#--------------------------------------------------------------------------
#--------------------------------------------------------------------------
#--------------------------------------------------------------------------
#--------------------------------------------------------------------------
#--------------------------------------------------------------------------


word_to_index_file_name = "word_to_index.pkl"
neural_network_weights_file_name = "sentiment_classification_weights.index"
dictionary_words_with_weights_file_name = "dic_tokens_reviews_w2v.pkl"

log_directory = "/home/logs/"
backup_directory = "/home/backups/"
sources_directory = "/home/sources/"

#Se inicializa el objeto para convertir de números a textos.
p = inflect.engine()

#Para poder hacer algunas limpiezas numéricas, se hace uso de expresiones
#regulares (para detectar los patrones).
#https://stackoverflow.com/questions/8586346/python-regex-for-integer
RE_INT = re.compile(r'^[1-9]\d*|0$')

#Con estas variables se hacen los rastreos de información cuando se convierten
#las palabras a un índice numérico y viceversa.
#Se carga el diccionario de palabras word_to_index
word_to_index = pickle.load( open(f'{sources_directory}{word_to_index_file_name}', 'rb'))

#Se carga el diccionario de palabras de las RESEÑAS con sus pesos como parte
#del modelo Word2Vec.
dic_tokens_word2vec = pickle.load( open(f'{sources_directory}{dictionary_words_with_weights_file_name}', 'rb'))

#num_tokens = 68305 + 1
max_pad_sequence = 227
#embedding_dim = 300
#dropout_value = 0.2


sentiment_classification_model = tensorflow.keras.models.load_model(f'{sources_directory}my_model.keras')


#El modelo se debe compilar nuevamente.
sentiment_classification_model.compile(optimizer='adam',         # el optimizador sirve para encontrar los pesos que minimizan la función de pérdida
                                                                 # adam: stochastic gradient descent adaptativo
                                                                 # https://www.tensorflow.org/api_docs/python/tf/keras/optimizers
                                       loss="binary_crossentropy", # función que evalua que tan bien el algoritmo modela el conjunto de datos
                                                                 # https://www.tensorflow.org/api_docs/python/tf/keras/losses
                                       metrics=['accuracy'])


#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

app = Flask(__name__)


#En esta operación se hace la limpieza de los TEXTOS (RESEÑAS), en particular
#cada texto se transforma en una lista de tokens limpios que sirvan para el
#cálculo de los PAD_Tokens y como insumo de la red neuronal.

#Al explorar la base de datos se halló oraciones como:

# -> "I arrived at 2 am"
# -> "It costed me 3 dollars"
# -> "I waited for 10 houes"

#Entonces, esto como propuesta por parte del autor, es posible conservar los
#números y pasarlos a texto con la finalidad de preservar la sensibilidad del
#comentario pues el uso de cuantificadores, incluso con textos, puede intensificar.
#más "el sentimiento".


#Fuentes de consulta para esta función
#Ver cuáles textos contienen números o caracteres raros y tratar de limpiarlos.
#https://stackabuse.com/how-to-split-string-on-multiple-delimiters-in-python/
#https://www.techiedelight.com/remove-non-alphanumeric-characters-string-python/

def transform_review(review):

    #Aquí se guardan los tokens limpios.
    list_review_final = []

    #Se reemplaza todos los caracteres extraños por espacios en blanco.
    replaced_review = re.sub(r'\W+', ' ', review)

    #La tokenización se hace por los espacios en blanco.
    list_replaced_review = replaced_review.split(" ")

    #Por cada token se hace esto.
    for element in list_replaced_review:

        transformed_generic_list = []

        #Si el token coincide con un número identificado por medio
        #de una expresión regular, se hace la transformación.
        if RE_INT.match(element):

           transformed_number = p.number_to_words(element)

           #Dividiendo elemento por varios separadores.
           list_transformed_number = re.split(',|;|-| ', transformed_number)

           #Puede haber el caso de números compuestos, entonces se hace el
           #tratamiento por cada uno.
           for current_transformed_number in list_transformed_number:
               transformed_generic_list.append(current_transformed_number)

        #Si el token es una palabra, sólo se hace la limpieza a lower.
        else:
            transformed_generic_list.append(element.lower())

        #Al final todos los elementos se agregan a la lista final.
        for generic_element in transformed_generic_list:
            if generic_element != "":
               list_review_final.append(generic_element)

    return list_review_final



#Ahora, dado que el modelo requiere que todos los textos se conviertan en números,
#se aplica la siguiente función

def transform_words_to_index(word_list):

    final_list = []

    #Por cada elemento, se hace la transformación numérica y se genera una lista
    #que será la que contenga los índices de las palabras.
    for current_word in word_list:

        try:
           final_list.append(word_to_index[current_word])

        except:
          pass


    return final_list
    



"""
   Index with my personal data.
"""
@app.route('/')
def index():
    final_json = {
                  'Name': 'Aaron Martin Castillo Medina',
                  'Version': '1.0',
                  'Email': 'aa.castillo@svitla.com'
                  }
                  
    return jsonify(final_json)


"""
   The following is a method to INSERT a set of rows in a determined table.
   
   Parameters: 
    table_name - the database to be backed up.
    dataset - the information in list format. 
    
   Returns: 
    A dictionary with the following structure:
      table - the table name.
      status - HTTP status code 
      message - a more particular message depending on the situations during the process. 
      count - the number of rows that were inserted.       
      mimetype - the document type.
"""
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    final_json = {}
    count = 0    
    status = 200
    message = "OK" 
    predictions = []
  
    #In this case, everything will be put in a try-catch block
    #as it's the part that intends to send any insert error into a log.
    try: 
    
       result = request.get_json()
       current_text = result.get("text")    
    
       my_dict = {
                   'text': current_text
                 }

       #Generando el tratamiento de conversión a tokens y números para cada uno de los
       #textos.
       test_df = pd.DataFrame(my_dict)

       test_df["text"] = test_df["text"].apply(lambda x: transform_review(x))
       test_df["text"] = test_df["text"].apply(lambda x: transform_words_to_index(x))
       test_df = pad_sequences(test_df["text"], padding ='pre', maxlen = max_pad_sequence)


       predictions = sentiment_classification_model.predict(test_df,verbose=1).tolist()
       
       final_predictions = []
       
       for element in predictions:
           final_predictions.append(str(element[0]))
       
       #Returning the final json
       final_json = {
                     'predictions': final_predictions,
                     'status': status,
                     'message': message, 
                     'count': len(predictions),
                     'mimetype': 'application/json'
                    }

    except Exception as e:
          #This is the error corresponding to the logger in case something is not inserted. 
          #As a particular design, if some error occurs then the whole process is aborted. 
                    
          #logger.critical(e, exc_info=True) 

          #Returning the error json
          final_json = {
                        'predictions': final_predictions,
                        'status': 500,
                        'message': str(e), 
                        'count': -1,
                        'mimetype': 'application/json'
                       }
    
    return jsonify(final_json)


""" 
   Main method.
"""
if __name__ == '__main__':
    app.run(host='0.0.0.0')
