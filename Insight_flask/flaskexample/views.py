from flask import render_template
from flask import request
from flaskexample import app
#from sqlalchemy import create_engine
#from sqlalchemy_utils import database_exists, create_database
import pandas as pd
#import psycopg2
#from a_Model import ModelIt

from flask import session
#session.clear()
#[session.pop(key) for key in list(session.keys())]

#user = 'mac' #add your username here (same as previous postgreSQL)                      
#host = 'localhost'
#dbname = 'birth_db'
#db = create_engine('postgres://%s%s/%s'%(user,host,dbname))
#con = None
#con = psycopg2.connect(database = dbname, user = user)

@app.route('/')
@app.route('/input')
def cesareans_input():
    return render_template("input.html")

@app.route('/Example')
def Example():
  video_url = 'https://www.youtube.com/watch?v=TZeODVlw-_8'
  input2 = video_url
  import webapp_week2
  some_data=webapp_week2.get_dataset(input2)
  #just select the Cesareans  from the birth dtabase for the month that the user inputs
  pred=webapp_week2.get_results(some_data)
  pred=pred[0]
  #runs python script
  exp1=webapp_week2.get_lime_results(some_data)
  rankingtable=webapp_week2.get_imp_table(some_data,exp1,pred)
  #Want to display youtube video on output page
  #Want to display exp1.show_in_notebook(show_table=True, show_all=False) on output page
  #Create chart from rankingtable and put on output page
  pd.set_option('display.max_colwidth',-1)
  rec_table=rankingtable.to_html(justify='left',index=False)
  return render_template("output.html", result2 = pred, tables=[rec_table], titles=['na','Suggestions'])#rankingtable=rankingtable)

   
@app.route('/output')
def cesareans_output():
  #pull 'birth_month' from input field and store it
  input2 = request.args.get('video_url')
  import webapp_week2
  some_data=webapp_week2.get_dataset(input2)
  #just select the Cesareans  from the birth dtabase for the month that the user inputs
  pred=webapp_week2.get_results(some_data)
  pred=pred[0]
  #runs python script
  exp1=webapp_week2.get_lime_results(some_data)
  rankingtable=webapp_week2.get_imp_table(some_data,exp1,pred)
  #Want to display youtube video on output page
  #Want to display exp1.show_in_notebook(show_table=True, show_all=False) on output page
  #Create chart from rankingtable and put on output page
  pd.set_option('display.max_colwidth',-1)
  rec_table=rankingtable.to_html(justify='left',index=False)
  return render_template("output.html", result2 = pred, tables=[rec_table], titles=['na','Suggestions'])#rankingtable=rankingtable)
