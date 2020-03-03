#This is for going thru the SQL database on the amazon server and collecting amazon codes of images of craters/aerogel
#To store locally on the machine at Berkeley.
#Does not get the images, just their corresponding codes

import mysql.connector
import os

mydb = mysql.connector.connect(
  host="flair.ssl.berkeley.edu",
  user="stardust",
  passwd="56y$Uq2CY",
  database="stardust"
)

Dir = "/Users/loganjaeger/Desktop/aerogel/"
fname = "aerogel_codes"
file = open(Dir + fname + ".txt", "w")
query = "SELECT amazon_key FROM `real_movie` WHERE exclude = '' AND comment = '' AND tech = 0 AND bad_focus < 50 LIMIT 20000" #Select the type of codes you want
cursor = mydb.cursor()
cursor.execute(query)
result = cursor.fetchall()
i = 0
for key in result:
	i += 1
	file.write(key[0])
	file.write("\n")
print(i)
file.close()

