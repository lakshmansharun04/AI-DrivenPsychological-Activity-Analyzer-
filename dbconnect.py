import pymysql
pymysql.install_as_MySQLdb()
import MySQLdb

def connection():
    conn = MySQLdb.connect(host="localhost",
                           user = "root",
                           passwd = "",
                           port=3306,
                           db = "psychological")
    c = conn.cursor()

    return c, conn		
def inserquery(sql1):
     c, conn = connection()
     c.execute(sql1)
     conn.commit()
     last_id = c.lastrowid

     conn.close()
     return last_id
def inserquerys(sql1,value):
     c, conn = connection()
     c.execute(sql1,value)
     conn.commit()
     last_id = c.lastrowid

     conn.close()
     return last_id


def updatequery(sql1,value):
     c, conn = connection()
     c.execute(sql1,value)
     print(c.rowcount, "record(s) affected")
     conn.commit()
     conn.close()

def recoredselect(sql):
    c, conn = connection()
    c.execute(sql);
    result=c.fetchall();
    return result
