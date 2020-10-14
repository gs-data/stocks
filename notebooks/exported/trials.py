#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pymongo


# In[ ]:


client = pymongo.MongoClient('localhost', 27017)


# In[ ]:


list(client.list_databases())


# In[ ]:


client.list_database_names()


# In[ ]:


db = client.test_database
collection = db.test_collection


# In[ ]:


collection.find_one()


# In[ ]:


import datetime
post = {"author": "Mike",
        "text": "My first blog post!",
        "tags": ["mongodb", "python", "pymongo"],
        "date": datetime.datetime.utcnow()}


# In[ ]:


post_id = collection.insert_one(post).inserted_id
post_id


# In[ ]:


collection.find_one()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




