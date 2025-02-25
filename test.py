import pymysql
# from urllib.parse import quote

# encoded_password = quote('Aut0t@t321')
conn = pymysql.connect(host='serveo.net', user='user', password='password', database='eta')
print("Connected successfully")
conn.close()
