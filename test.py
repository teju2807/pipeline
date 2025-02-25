import pymysql
conn = pymysql.connect(host='10.0.1.168', user='your_user', password='your_password', database='your_db')
print("Connected successfully")
conn.close()
