
# coding: utf-8

# In[3]:


"""Create a url table for the TED Talks transcript json objects"""

# Author: Sameera Lanka
# Website: www.sameera-lanka.com

import pandas as pd

url = pd.read_csv('talks.csv')
url_transcript = url['public_url'] + '/transcript.json?language=en'
url_transcript.to_csv('talks_url.csv', index=False)

