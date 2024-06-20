"""
   This snippet of code acts a simulation for using the endpoints created through the Flask server.
   
"""

import os
import math
import requests 
import numpy as np
import pandas as pd

#Creating the headers section for all the following requests. 
headers = {
            'Content-type':'application/json', 
            'Accept':'application/json'
          }

    

"""
   This is a function meant to test the backup RESTORATION in .avro format, according
   to the exercise statements. 
   
   Parameters:
     table_name - the table name to be backed up. 
     timestampe - the timestamp IN STRING FORMAT that uniquely identifies the backup.
                  If no timestamp is specified, then the latest backup will be used.     
         
   Returns: 
    A dictionary with the following structure:
      table - the BACKUP file name that was restored.
      status - HTTP status code 
      message - a more particular message depending on the situations during the process. 
      count - the number of rows that contains the BACKUP.       
      mimetype - the document type.

   Important: 
      This will be a FULL RESTORATION, so you should keep that in mind.
      If you don't know the timestamps available, you can use the function get_backup_names instead.
"""    
def test_prediction(): 
    
    additional_data =  {
                        'text': ['My experience was cool. The animation and the graphics were out of this world. I would recommend this place.',
                                 'I am never returning to this place, it sucks',
                                 'I would say that in general I had a pleasant experience, I liked the attention but disliked the food.',
                                 'I liked your hospitality, youre the best guests in the world',
                                 '\"I enjoyed my stay at your restaurant\" - said no one',
                                 'Your food is as good as the global warming',
                                 'My stay at the hotel was as pleasant as a day in jail',
                                 'Fuck you  '
                                 ]
                       }


    response = requests.post('http://127.0.0.1:5000/predict', json=additional_data, headers=headers)
    return response.json()

    

    
      
"""
   This is the section related to the tests. 
"""


print(test_prediction())
# **** BACKUP *****
#print(backup_table("hired_employees"))
#print(backup_table("jobs"))
#print(backup_table("departments"))
#print(backup_table("other")) 

#print(get_backup_names())

# **** RESTORE *****
#print(restore_table("hired_employees"))
#print(restore_table("jobs"))
#print(restore_table("departments"))
#print(restore_table("other")) 


# **** INSERT *****
#print(insert_table("hired_employees"))
#print(insert_table("jobs"))
#print(insert_table("departments"))
#print(insert_table("other")) 

# **** INSERT (MY LOCAL MODE ONLY)*****
#print(insert_table("hired_employees", csv_path = "sources/"))
#print(insert_table("jobs", csv_path = "sources/"))
#print(insert_table("departments", csv_path = "sources/"))
#print(insert_table("other", csv_path = "sources/")) 

