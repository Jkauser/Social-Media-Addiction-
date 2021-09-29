import streamlit as st
st.set_option('deprecation.showPyplotGlobalUse', False)

# for manipulation
import pandas as pd
import numpy as np

# for data visualization
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')

# to filter warnings
import warnings
warnings.filterwarnings('ignore')

st.title("Social Media Addiction")

# Reading the dataset
data= pd.read_excel('project.xlsx')

# Replacing null values with Not prefer to say..
data.fillna('Not prefer to say', inplace=True)

# Replacing column names
print(data.rename(columns={'Occupation ':'Occupation',
                           '1. From when you started using smartphone?': 'Q1',
                         ' 2. Which of the following social media website do you currently have an account with?': 'Q2',
                          '3. How often do you check-in to your social media accounts in any given week?':'Q3',
                          '4. What is your go-to device to access your social media feed?':'Q4',
                          '5. How often do you use the chat app on your social media accounts?':'Q5',
                          '6. On an average how much time do you spend on social media?':'Q6',
       '7. What is your purpose of using social media website?':'Q7',
       '8. When do you access social media websites?':'Q8',
       '9.  Do you check your social media account before going to bed?':'Q9',
       '10. How does social media affect your physical and mental health?':'Q10',
       '11. How often do you face health issues?':'Q11',
       '12. Do you want to share your opinion about social media addiction?':'Q12'}, inplace= True))


Nv = st.sidebar.radio("Navigator", ["Home","Prediction","Contribute"])

if Nv== "Home":
    #st.write("### Home")
    st.image("project.png", width= 700)
    st.subheader("\nIntroduction\n")
    st.text("\nAs we all know this pandemic is a crucial period. Social media is helping us into this. \nWe all are so dependent on social media that we couldn't see anything above that now a days. \nDue to this pattern of our life we are becoming addicted to this. \nWe students are taking initiative to analyze how social media is affecting our life and our loved ones too. \nHow people are tending towards the addiction of social media\n.")

    if st.checkbox("Show Dataset"):
        st.table(data)

    st.subheader("\nData Visualization")
    if st.checkbox("Show Graphs"):
        Features = st.selectbox("Features",['Gender','Age','Education','Occupation',
                                               'Frequently used Social Media Platforms',
                                               'Popularly used Go-to Devices','Average Time Spent on Social Media',
                                               'When do users access social media websites','Checking social media account before going to bed',
                                               'Effects of social media on physical and mental health','How often users face health issues'])

        if  Features == "Gender":
            plt.figure(figsize=(5, 3))
            data['Gender'].value_counts().plot(kind='pie', explode=[0,0,0.8], autopct='%1.2f%%')
            st.pyplot()

        if  Features == "Age":
            plt.figure(figsize=(5, 3))
            sns.countplot(data=data, x=data['Age'])
            plt.xlabel("Age", fontsize=13)
            plt.ylabel("Count", fontsize=13)
            plt.xticks(rotation=45)
            st.pyplot()
    
        if  Features == "Education":
            plt.figure(figsize=(10, 7))
            sns.countplot(data=data, x='Education', hue='Gender')
            plt.xticks(rotation=45)
            plt.xlabel("Education", fontsize=13)
            plt.ylabel("Count", fontsize=13)
            plt.legend(loc='upper right')
            st.pyplot()
    
        if  Features == "Occupation":
            plt.figure(figsize=(10, 7))
            sns.countplot(data=data, x='Occupation', hue='Gender')
            plt.xticks(rotation=45)
            plt.xlabel("Occupation", fontsize=13)
            plt.ylabel("Count", fontsize=13)
            plt.legend(loc='upper right')
            st.pyplot()

        if  Features == 'Frequently used Social Media Platforms':
            plt.figure(figsize=(12,8))
            sns.countplot(data=data, x='Q2')
            plt.xticks(rotation=90)
            plt.xlabel("\nFrequently used Social Media Platforms", fontsize=13, fontweight= 'bold')
            plt.ylabel("Count", fontsize=13)
            st.pyplot()

        if  Features == 'Popularly used Go-to Devices':
            plt.figure(figsize=(12,8))
            sns.countplot(data=data, x='Q4')
            plt.xticks(rotation=90)
            plt.xlabel("\nPopularly used Go-to Devices", fontsize=13)
            plt.ylabel("Count", fontsize=13)
            st.pyplot()

        if  Features == 'Average Time Spent on Social Media':
            plt.figure(figsize=(12,8))
            sns.countplot(data=data, x='Q6')
            plt.xticks(rotation=90)
            plt.xlabel("Average Time Spent on Social Media", fontsize=13)
            plt.ylabel("Count", fontsize=13)
            st.pyplot()

        if  Features == 'When do users access social media websites':
            plt.figure(figsize=(8,5))
            sns.countplot(data=data, x='Q8')
            plt.xticks(rotation=90)
            plt.xlabel("\nUsers access social media websites during..", fontsize=13, fontweight='bold')
            plt.ylabel("Count", fontsize=13)
            st.pyplot()

        if  Features == 'Checking social media account before going to bed':
            plt.figure(figsize=(8,5))
            data['Q9'].value_counts().plot(kind='pie', explode=[0,0], autopct='%1.2f%%')
            plt.xlabel("Checking social media account before going to bed", fontsize=14, fontweight='bold')
            st.pyplot()

        if  Features == 'Effects of social media on physical and mental health':
            plt.figure(figsize=(8,5))
            sns.countplot(data=data, x='Q10')
            plt.xticks(rotation=90)
            plt.xlabel("\nEffects of social media on physical and mental health", fontsize=13, fontweight='bold')
            plt.ylabel("Count", fontsize=13)
            st.pyplot()


        if  Features == 'How often users face health issues':
            plt.figure(figsize=(8,5))
            sns.countplot(data=data, x='Q11')
            plt.xticks(rotation=90)
            plt.xlabel("\nHow often users face health issues", fontsize=13, fontweight='bold')
            plt.ylabel("Count", fontsize=13)
            st.pyplot()

        


from sklearn.preprocessing import LabelEncoder
for col in data:
    le= LabelEncoder()
    data[col] = le.fit_transform(data[col])

from sklearn.model_selection import train_test_split
x=data[["Gender","Age","Education","Occupation","Q1","Q3","Q5","Q6","Q9"]]
y=data['Q11']
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1, test_size=0.2)

# let's create predictive model
from sklearn.svm import SVC
model= SVC()
model.fit(x_train, y_train)



if Nv == "Prediction":
    st.subheader("\nPredictor\n")
    
    # instantiate

    G = st.radio("Gender: ", ["Male","Female"])

    A = st.radio("Age: ", ["Below 16","16 to 20","20to 30","30 to 50","50 Above"])
    
    E = st.radio("Education: ",["U.G","Graduate","P.G","Other"])
    if E == "Other":
            st.text_input("Enter Education: ")
 
    O= st.radio("Occupation: ", ["Student","Employee","Entrepreneur","Housewife","Service","Other"])
    if O == "Other":
            st.text_input("Enter Occupation: ")
 
    a = st.radio("From when you started using smartphone? ", ["4 to 10 AgeGroup","10 to 20 AgeGroup","20 to 30 AgeGroup","30 to 50 AgeGroup","50 and Above AgeGroup"])
 
    b = st.radio("How often do you check-in to your social media accounts in any given week?", ["Every hour","Daily","Every other day","Every two days","Once a week"])
   
    c = st.radio("How often do you use the chat app on your social media accounts?",["Every 5-15 minutes","Every hour","Every 3-4 hours","Every day","Not at all"])
   
    d = st.radio("On an average how much time do you spend on social media?",["Less than 30 mins","An hour","1-2 hour","3-4 hour","More than 4 hours"])
  
    e= st.radio("Do you check your social media account before going to bed?",["Yes","No"])
 
    G= le.fit_transform([G])
    A= le.fit_transform([A])
    E= le.fit_transform([E])
    O= le.fit_transform([O])
    a= le.fit_transform([a])
    b= le.fit_transform([b])
    c= le.fit_transform([c])
    d= le.fit_transform([d])
    e= le.fit_transform([e])

    #y_pred = model.predict([[G, A, E, O, a,b,c,d,e]])
    st.write("\n\n\n")
    if st.button("Predict"):
        st.subheader(f"\nPredictedion is:")
        #st.success(y_pred)


if Nv == "Contribute":
    st.subheader("Contribute to our Dataset")
    Gender = st.radio("Gender: ", ["Male","Female"])
    Age = st.radio("Age: ", ["Below 16","16 to 20","20to 30","30 to 50","50 Above"])
    Edu = st.radio("Education: ",["U.G","Graduate","P.G","Other"])
    if Edu == "Other":
            st.text_input("Enter Education: ")
    Ocu= st.radio("Occupation: ", ["Student","Employee","Entrepreneur","Housewife","Service","Other"])
    if Ocu == "Other":
            st.text_input("Enter Occupation: ")
    Q1 = st.radio("From when you started using smartphone? ", ["4 to 10 AgeGroup","10 to 20 AgeGroup","20 to 30 AgeGroup","30 to 50 AgeGroup","50 and Above AgeGroup"])
    
    Q2 = st.multiselect("Which of the following social media website do you currently have an account with?",["Facebook","Twitter","WhatsApp",
    "Instagram","LinkedIn","Zoom/Google Meet etc.","Telegram","Netflix","Other: "])
    
    Q3 = st.radio("How often do you check-in to your social media accounts in any given week?", ["Every hour","Daily","Every other day","Every two days","Once a week"])
    
    Q4 = st.multiselect("What is your go-to device to access your social media feed?",["Mobile","Tablet","Laptop","Desktop"])

    Q5 = st.radio("How often do you use the chat app on your social media accounts?",["Every 5-15 minutes","Every hour","Every 3-4 hours","Every day","Not at all"])

    Q6 = st.radio("On an average how much time do you spend on social media?",["Less than 30 mins","An hour","1-2 hour","3-4 hour","More than 4 hours"])
    
    Q7 = st.multiselect("What is your purpose of using social media website?",["To make friends","To socialize casually","To find a suitable date",
    "To promote products/services","Event planning","To find employment","Online Gaming","Studying","Other"])

    Q8= st.multiselect("When do you access social media websites?",["During my free time","While at school/university/work","During social occasions","During meal times","Other"])

    Q9= st.radio("Do you check your social media account before going to bed?",["Yes","No"])

    Q10= st.multiselect("How does social media affect your physical and mental health?",["Back Pain","Headache","Depression","Anxiety","Eye Related Issues","Mood swings"])


    Q11= st.radio("How often do you face health issues?",["Everyday","Once a week","Twice a week","Thrice a month","Once a month"])

    Q12 = st.radio("Do you want to share your opinion about social media addiction?",["No","Other"])
    if Q12 == "Other":
            st.text_input("Comment: ")

    if st.button("Contribute"):
        to_add= {"Gender":[Gender], "Age":[Age], "Education":[Edu], "Occupation":[Ocu], 
        "1. Which of the following social media website do you currently have an account with?":[Q1], 
        "2. Which of the following social media website do you currently have an account with?":[Q2], 
        "3. How often do you check-in to your social media accounts in any given week? ":[Q3], 
        "4. What is your go-to device to access your social media feed?":[Q4],
        "5. How often do you use the chat app on your social media accounts?":[Q5],
        "6. On an average how much time do you spend on social media?":[Q6],
        "7. What is your purpose of using social media website?":[Q7],
        "8. When do you access social media websites?":[Q8],
        "9. Do you check your social media account before going to bed?":[Q9],
        "10. When do you access social media websites?":[Q10],
        "11. How often do you face health issues?":[Q11],
        "12. Do you want to share your opinion about social media addiction?":[Q12]}

        to_add= pd.DataFrame(to_add)
        to_add.to_csv("pro.csv", mode='a', header=False, index=False)
        st.success("Thanks for Your Contribution")
