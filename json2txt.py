
# coding: utf-8

# In[6]:


"""Extract TED Talks subtitles from json objects"""

# Author: Sameera Lanka
# Website: www.sameera-lanka.com
# Copyright © TED Conferences, LLC

import pandas as pd
import os
import json 

if not os.path.exists('./TED_transcripts'):
    os.makedirs('./TED_transcripts')
    
i = 0;
filename = './TED_json/transcript.json?language=en.' + str(i)
    
while os.path.isfile(filename):
    with open(filename, 'r') as f:
        
        try:
            talk_html = json.load(f)
            transcript = open("./TED_transcripts/transcript_" + str(i) + ".txt", "w")
            
            for cues in talk_html['paragraphs']:
                for text in cues['cues']:
                    transcript.write(text['text'] + " ")
                    
            transcript.close()
                
        except:
            pass
        
    i = i+1
    filename = './TED_json/transcript.json?language=en.' + str(i)
   
    
    

