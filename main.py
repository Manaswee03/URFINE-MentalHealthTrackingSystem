from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
import neattext.functions as nfx
import pandas as pd
import numpy as np
from flask import Flask, render_template, url_for, redirect, session, g, request
from matplotlib import pyplot as plt
import base64
from io import BytesIO
import mysql.connector as mysql
import pickle
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import csv
import random
from datetime import datetime

app = Flask(__name__)
app.secret_key = "ItShouldBeAnythingButSecret"

conn = mysql.connect(
    host="localhost", database="major_project", user="root", password="manas@2002"
)
cursor = conn.cursor()

lr_model = pickle.load(open("D:/MentalHealthProject/finalized_LRmodel.sav", "rb"))

df = pd.read_csv(
    "C:/Users/manas/Downloads/Text-Emotion-Classification-master/Text-Emotion-Classification-master/text_emotion.csv"
)
df["Clean_Text"] = df["Text"].apply(nfx.remove_stopwords)
df["Clean_Text"] = df["Clean_Text"].apply(nfx.remove_punctuations)
df["Clean_Text"] = df["Clean_Text"].apply(nfx.remove_userhandles)

Xfeatures = df['Clean_Text']
ylabels = df['Emotion']
cv = CountVectorizer()
X = cv.fit_transform(Xfeatures)


def predict_emotion11(sample_text, model):
    vect = cv.transform(sample_text).toarray()
    res = model.predict(vect)
    pred_proba = model.predict_proba(vect)
    pred_percentage_for_all = dict(zip(model.classes_,pred_proba[0]))
    return pred_percentage_for_all

@app.route("/")
def home():
    return render_template("home.html")


@app.route("/activity")
def activity():
    return render_template("activity.html")


@app.route("/quiz1")
def quiz1():
    return render_template("quiz1.html")


@app.route("/mydiary")
def mydiary():
    return render_template("mydiary.html")


@app.route("/books")
def books():
    return render_template("books.html")


@app.route("/music")
def music():
    return render_template("music.html")


@app.route("/pop")
def pop():
    return render_template("pop.html")


@app.route("/sour")
def sour():
    return render_template("sour.html")


@app.route("/worship")
def worship():
    return render_template("worship.html")


@app.route("/radio")
def radio():
    return render_template("radio.html")


@app.route("/christmas")
def christmas():
    return render_template("christmas.html")


@app.route("/disney")
def disney():
    return render_template("disney.html")


@app.route("/bollywood")
def bollywood():
    return render_template("bollywood.html")


@app.route("/exercise")
def exercise():
    return render_template("exercise.html")


@app.route("/feedback")
def feedback():
    return render_template("feedback.html")

@app.route("/daily")
def daily():
    return render_template("daily_quiz.html")

@app.route("/tasks")
def tasks():
    return render_template("task2.html")

@app.route("/exercise_task")
def exercise_task():
    return render_template("exercise_task.html")

@app.route("/music_task")
def music_task():
    return render_template("music_task.html")

@app.route("/meditation")
def meditation():
    return render_template("meditation.html")

@app.route("/selfcare")
def selfcare():
    return render_template("selfcare.html")


@app.route("/feed", methods=["GET", "POST"])
def feed():
    ratings = request.form.get("ratings")
    comments = request.form.get("comments")
    name = request.form.get("name")
    email = request.form.get("email")

    cursor.execute(
        """INSERT INTO `feedback`(`ratings`,`comments`,`name`,`email`) VALUES
                   ('{}', '{}', '{}', '{}') """.format(
            ratings, comments, name, email
        )
    )

    conn.commit()

    if len(name) > 0:
        return render_template("feedback_success.html")
    else:
        return render_template("feedback.html")

@app.route("/login")
def login():
    return render_template("login.html")

@app.route("/login_validation", methods=["POST"])
def login_validation():
    if request.method == "POST":
        email = request.form.get("email")
        password = request.form.get("password")

        # session['email']=request.form['email']
        # if 'email' in session:
        #    s=session['email']

        cursor.execute(
            """SELECT * FROM `reg_users` WHERE `email` LIKE '{}' AND `password` LIKE '{}'""".format(
                email, password
            )
        )

        users = cursor.fetchall()
        if len(users) > 0:
            user1 = users[0]
            g.user = {"email": user1[1], "password": user1[3]}

            if email == g.user["email"] and password == g.user["password"]:
                session["user"] = user1[0]
                return redirect("/dashboard")
        else:
            return render_template("login_fail.html")

    return render_template("dashboard.html")


@app.route("/mark_quiz", methods=["POST"])
def mark_quiz():
    if request.method == "POST":  
       name = g.user
       final_emotion = request.form['final_emotion']
       score = request.form['score']
       day = request.form['day']
       time = request.form['time']
       cursor.execute(
        """INSERT INTO `mark_quiz`(`name`,`final_emotion`,`score`,`Day`,`time`) VALUES
                   ('{}', '{}','{}','{}','{}') """.format(name, final_emotion, score, day, time)
    )
    conn.commit()
    return redirect("/tasks")

@app.route("/dashboard")
def dashboard():
    if g.user:
        return render_template("dashboard.html", user=session["user"])
    return redirect(url_for("login"))


@app.before_request
def before_request():
    g.user = None
    if "user" in session:
        g.user = session["user"]


@app.route("/logout")
def logout():
    session.pop("user", None)
    return redirect("/")


@app.route("/add_user", methods=["POST"])
def add_user():
    name = request.form.get("uname")
    email = request.form.get("uemail")
    phone = request.form.get("phone")
    password = request.form.get("upassword")

    cursor.execute(
        """INSERT INTO `reg_users`(`name`,`email`,`phone`,`password`) VALUES
                   ('{}', '{}', '{}','{}') """.format(
            name, email, phone, password
        )
    )

    conn.commit()

    if len(name) > 0:
        return render_template("reg_success.html")
    else:
        return render_template("reg_fail.html")


@app.route("/result", methods=["GET", "POST"])
def result():

    if request.method == "POST":
        re = request.form
        sample_text = []
        val1 = re["n1"]
        sample_text.append(val1)
        val2 = re["n2"]
        sample_text.append(val2)
        val3 = re["n3"]
        sample_text.append(val3)
        val4 = re["n4"]
        sample_text.append(val4)
        val5 = re["n5"]
        sample_text.append(val5)
        val6 = re["n6"]
        sample_text.append(val6)
        val7 = re["n7"]
        sample_text.append(val7)
        val8 = re["n8"]
        sample_text.append(val8)
        val9 = re["n9"]
        sample_text.append(val9)
        val10 = re["n10"]
        sample_text.append(val10)

    print(sample_text)
    prediction = predict_emotion11(sample_text,lr_model)
    print(prediction)
    keys = prediction.keys()
    values = prediction.values()
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.bar(keys, values,color=[
            "#AA336A",
            "#FFDB58",
            "#FFE5B4",
            "#AFE1AF",
            "#FBCEB1",
            "#D8BFD8",
            "#F88379",
            "#FFFF8F",
            "#9FE2BF",
            "#AA98A9",
            "#FAD5A5",
            "#FAA0A0",
            "#C9A9A6",
        ],
        width=0.8,)
    ax.set_xlabel('Days')
    ax.set_ylabel('Percentage Score')
    buf = BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    buffered_image11 = base64.b64encode(buf.read()).decode('utf-8')
    return render_template("test1.html", image=buffered_image11)

questions = []

with open('D:/MentalHealthProject/static/questions.csv') as f:
    reader = csv.reader(f)
    for row in reader:
        questions.append(row[0])

def get_emotion(score):
    if score >= 0.5:
        return "happy"
    elif score <= -0.5:
        return "sad"
    elif score >= 0.0 and score < 0.5:
        return "worry"
    elif score <= 0.0 and score > -0.5:
        return "anger"
    else:
        return "neutral"

answered_questions = 0
predicted_emotions = []
start_time = None
time_taken_formatted = 0

@app.route('/day', methods=["POST"])
def day():
    global day
    day = request.form.get("day")
    return render_template('daily_quiz.html', day=day)

@app.route('/daily_quiz')
def daily_quiz():
    global answered_questions, start_time
    if answered_questions >= 10:
        return redirect(url_for('quiz_thanks'))
    if answered_questions == 0:
        start_time = datetime.now()
    question = random.choice(questions)
    questions.remove(question)
    answered_questions += 1
    return render_template('daily_quiz.html', question=question)

@app.route('/dailyquiz_post', methods=['POST'])
def dailyquiz_post():
    global time_taken_formatted
    response = request.form['response']
    if isinstance(response, bytes):
        response = response.decode('utf-8')
    sia = SentimentIntensityAnalyzer()
    sentiment = sia.polarity_scores(response)['compound']
    predicted_emotion = get_emotion(sentiment)
    predicted_emotions.append(predicted_emotion)
    if answered_questions == 10:
        end_time = datetime.now()
        time_taken = (end_time - start_time).total_seconds()/ 60.0
        time_taken_formatted = "{:.2f}".format(time_taken)
        print("Time taken:", time_taken_formatted, "minutes")
    return redirect(url_for('daily_quiz'))

@app.route('/quiz_thanks')
def quiz_thanks():
    print(predicted_emotions)
    prediction = predict_emotion(predicted_emotions,lr_model)
    emotion = (prediction[0])
    if emotion=="worry":
        emotion="Worried"
    elif emotion=="neutral":
        emotion="Calm" 
    elif emotion=="happy":
        emotion="Happy" 
    elif emotion=="sad":
        emotion="Sad"            
    score = (round(prediction[1]*100))
    return render_template('quiz_thanks.html', emotion=emotion, score=score, day=day, time=time_taken_formatted)


def predict_emotion(sample_text, model):
    vect = cv.transform(sample_text).toarray()
    res = model.predict(vect)
    pred_proba = model.predict_proba(vect)
    emo = format(res[0])
    emo_score = np.max(pred_proba)
    return emo, emo_score

@app.route('/daily_report')
def daily_report():
    name = g.user
    cursor.execute("SELECT Day, final_emotion, score, time FROM mark_quiz WHERE name ='"+name+"' AND Day IN (1, 2, 3, 4, 5, 6, 7) GROUP BY Day")
    data1 = cursor.fetchall()
    return render_template('daily_report.html',data=data1)     

@app.route("/final_report")
def final_report():
    name = g.user
    cursor.execute("SELECT Day, final_emotion, score, time FROM mark_quiz WHERE name ='"+name+"' AND Day IN (1, 2, 3, 4, 5, 6, 7) GROUP BY Day")
    data1 = cursor.fetchall()
    column1_values = []
    column2_values = []
    column3_values = []
    column4_values = []

    for result in data1:
        column1_values.append(result[0])
        column2_values.append(result[1])
        column3_values.append(result[2])
        column4_values.append(result[3])   
    
    prediction = predict_emotion(column2_values,lr_model)
    emotion = (prediction[0])
    if emotion=="worry":
        emotion="Worried"
    elif emotion=="neutral":
        emotion="Calm" 
    elif emotion=="happiness":
        emotion="Happy" 
    elif emotion=="sad":
        emotion="Sad"            
    score = (round(prediction[1]*100))

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.bar(column1_values, column3_values,color=[
            "#AA336A",
            "#FFDB58",
            "#FFE5B4",
            "#AFE1AF",
            "#FBCEB1",
            "#D8BFD8",
            "#F88379",
        ],
        width=0.8,)
    ax.set_xlabel('Days')
    ax.set_ylabel('Percentage Score')
    ax.set_ylim([0, 60])

    for i, v in enumerate(column3_values):
        ax.text(i+1, v+1, str(column2_values[i]), ha='center', va='bottom') 

    buf = BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)

    buffered_image = base64.b64encode(buf.read()).decode('utf-8')
    column4_values_float = [float(val) for val in column4_values]
    column4_values_int = [int(val) for val in column4_values_float]
    average_time = sum(column4_values_int)/len(column4_values_int)
    average_time_formatted = "{:.2f}".format(average_time)
    return render_template("final_report.html", image=buffered_image, final_emotion=emotion, score=score, time=average_time_formatted)

if __name__ == "__main__":

    app.run(debug=True)
