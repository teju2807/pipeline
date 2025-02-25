import pymysql
from urllib.parse import quote

encoded_password = quote('Aut0t@t321')
conn = pymysql.connect(host='10.0.1.168', user='apdev', password=encoded_password, database='ap_widgets')
print("Connected successfully")
conn.close()
