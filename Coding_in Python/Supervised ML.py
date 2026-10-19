# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# ///
# MAGIC %md
# MAGIC **Label Encoding**
# MAGIC ##Label encoding translates categorical text or labels into unique integers, which is a standard data preprocessing step in **supervised machine learning** to convert non-numeric inputs or target classes into a format algorithms can process##

# COMMAND ----------

#Importing Liabraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder


# COMMAND ----------

#data ingestion/Original Data
colours=["Red","Blue","Green","Red"]

# COMMAND ----------

# Create a encoder or the converter
encoder=LabelEncoder()

# COMMAND ----------

# MAGIC %md
# MAGIC **__ encoded_colours — A new variable that will store the result (the numeric version of your colours list).
# MAGIC
# MAGIC = — Assignment: store the output of the function call into encoded_colours.
# MAGIC
# MAGIC encoder — The LabelEncoder object created in Cell 4.
# MAGIC
# MAGIC .fit_transform — A method that does two things in one step: fit (learn the mapping from text labels to integers, e.g. Blue→0, Green→1, Red→2) and transform (apply that mapping to produce the actual numbers).
# MAGIC (colours) — The input list from Cell 3 (["Red","Blue","Green","Red"]) that gets converted.
# MAGIC
# MAGIC print(encoded_colours) — Displays the resulting array of integers — [2 0 1 2] — so you can see the encoded output**

# COMMAND ----------

#convert text to numbers
encoded_colours=encoder.fit_transform(colours)
results_df=pd.DataFrame({'Original Colour': colours, 'Encoded Value': encoded_colours})
display(results_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ** Ordinal Encoding**

# COMMAND ----------

#Importing Liabraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OrdinalEncoder

# COMMAND ----------

# data ingestion
education=[["Primary"],["Secondary"],["Degree"],["Masters"],["Phd"]]
encoder=OrdinalEncoder(categories=[["Primary","Secondary","Degree","Masters","Phd"]])
encoded=encoder.fit_transform(education)

# COMMAND ----------

#display the dataframe
education_df=pd.DataFrame({'Education Level': [level[0] for level in education], 'Encoded Value': encoded.flatten()})
display(education_df)

# COMMAND ----------

#Importing Liabraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OrdinalEncoder

# COMMAND ----------

# data ingestion

# Create a sample DataFrame (Ordered data: Sizes)
data = [("Small",), ("Medium",), ("Large",), ("Small",)]
df = spark.createDataFrame(data, ["Size"])

# COMMAND ----------

#ecoding the string to int64 with ordinal ecoder
data = [("Small",), ("Medium",), ("Large",), ("Small",)]
encoder=OrdinalEncoder(categories=[["Small", "Medium", "Large"]])
encoded=encoder.fit_transform(data)
sizes_df=pd.DataFrame({'Original Size': [size[0] for size in data], 'Encoded Value': encoded.flatten()})
display(sizes_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ##unsupervised data

# COMMAND ----------

# MAGIC %md
# MAGIC **When you have unsupervised data, it means you do not have a target "answer" column (a label) to predict. Instead, you are looking for hidden patterns, structures, or groups (like clustering customers or compressing data)**

# COMMAND ----------

#One_Hot Encoding for Heirachical data using OrdinalEncoder
#importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OrdinalEncoder

# COMMAND ----------

# 1. Create unsupervised Spark DataFrame in Databrcks
data = [
    ("Bronze", 20),
    ("Gold", 85),
    ("Silver", 50),
    ("Bronze", 15),
    ("Gold", 90),
    ("Silver", 55)
]
df = spark.createDataFrame(data, ["Customer_Tier", "Spending_Score"])

# 2. Convert to Pandas for Scikit-Learn
df = df.toPandas()

# COMMAND ----------

# hierarchy for OrdinalEncoder
# We list categories from lowest to highest rank so Bronze=0, Silver=1, Gold=2
tier_order = [["Bronze", "Silver", "Gold"]]
encoder = OrdinalEncoder(categories=tier_order)

# COMMAND ----------

# Encode Customer_Tier and display alongside Spending_Score
df['Tier_Encoded'] = encoder.fit_transform(df[['Customer_Tier']])
display(df[['Customer_Tier', 'Tier_Encoded', 'Spending_Score']])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Unsupervised label data using One-Hot Encording

# COMMAND ----------

#One_Hot Encoding for Lbel data using LabelEncoder
#importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OrdinalEncoder

# COMMAND ----------


# 1. Create unsupervised Spark DataFrame into Databricks
categories = [
    ("Electronics", 800),
    ("Clothing", 120),
    ("Groceries", 75),
    ("Electronics", 950),
    ("Clothing", 110),
    ("Groceries", 90)
]
categories = spark.createDataFrame(categories, ["Department", "Purchase_Amount"])

# 2. Convert to Pandas for Scikit-Learn
categories = categories.toPandas()

# COMMAND ----------

# Apply LabelEncoder to the nominal column
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
categories["Dept_Encoded"] = le.fit_transform(categories["Department"])

# COMMAND ----------

# Display the encoded label data
display(categories[['Department', 'Dept_Encoded','Purchase_Amount']])