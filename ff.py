import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split as split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc,roc_auc_score
import xgboost as xgb
from xgboost import plot_importance
import tensorflow as tf
from tensorflow import keras
# from  keras import models
from keras.models import Sequential
from keras.layers import Dense
import h5py
st.set_page_config(layout="wide")
with open("./style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

font = "sans-serif"
st.write(
    """
# Preterm Birth Prediction App

"""
)
st.subheader("Kindly enter your details within the specified range for accurate prediction")
# df=pd.read_csv("data-preterm.csv")

st.sidebar.header("User Input Parameters")
# """GEST: gestational age in weeks at entry into the study
# DILATE: cervical dilation in cm
# EFFACE: cervical effacement (in %)
# CONSIS: cervical consistency (1=soft, 2=medium, 3=firm)
# CONTR: presence (=1) or absence (=2) of contractions
# MEMBRAN: ruptured membranes (=1) or intact (=2) or uncertain (=3)
# AGE: patient's age
# STRAT: stage of pregnancy
# GRAVID: gravidity (number of previous pregnancies including current one)
# PARIT: parity (number of previous full-term pregnancies)
# DIAB: presence (=1) or absence (=2) of diabetes problem, or missing value (=9)
# BEBAGE: gestational age (in days) of the baby at birth
# TRANSF: transfer (1) or non-transfer (2) to a hospital for specialized care
# GEMEL: single (=1) or multiple (=2) pregnancy
# Variable to predict:
# PREMATURE: preterm birth (positive or negative)"""

def user_input_features():
    GEST = st.number_input("Current gestational age in weeks (20 to 35)")
    if(GEST<20 or GEST>45):
        st.write(":red[Please enter a value within the prescribed range (20 to 35)]")
    DILATE = st.number_input("Cervical dilation in cm (0 to 8)")
    if(DILATE<0 or DILATE>8):
        st.write(":red[Please enter a value within the prescribed range (0 to 8)]")
    EFFACE = st.number_input("Cervical effacement (in %) (0 to 100)")
    if(EFFACE<0 or EFFACE>100):
        st.write(":red[Please enter a value within the prescribed range (0 to 100)]")
    # CONSIS=st.number_input("cervical consistency (1=soft, 2=medium, 3=firm)")
    x=st.radio("Cervical consistency",["soft","medium","firm"],horizontal=True)
    if x=="soft":
        CONSIS=1
    elif x=="medium":
        CONSIS=2
    else:
        CONSIS=3
    CONTR = st.radio("Contractions",["present","absent"],horizontal=True)
    if CONTR=="present":
        CONTR=1
    else:
        CONTR=2
    MEMBRAN = st.radio("Membrane",["ruptured","intact","uncertain"],horizontal=True)
    if MEMBRAN=="ruptured":
        MEMBRAN=1
    elif MEMBRAN=="intact":
        MEMBRAN=2
    else:
        MEMBRAN=3
    INFECTION=st.radio("Infection",["yes","no"],horizontal=True)
    if INFECTION=="no":
        INFECTION=0
    else:
        INFECTION=1
    CHR_HYPERTENSION=st.radio("Chronic hypertension",["yes","no"],horizontal=True)
    if CHR_HYPERTENSION=="no":
        CHR_HYPERTENSION=0
    else:
        CHR_HYPERTENSION=1
    AGE = st.number_input("patient's age")
    if AGE<13:
        st.write(":red[invalid]")
    STRAT = st.radio("stage of pregnancy",[1,2,3],horizontal=True)
    GRAVID = st.number_input("number of previous pregnancies including current one")
    if GRAVID<0 or GRAVID>20:
        st.write(":red[invalid]")
    PARIT = st.number_input("number of previous full-term pregnancies")
    if PARIT<0 or PARIT>20:
        st.write(":red[invalid]")
    DIAB = st.radio("Diabetes",["present","absent","don't know"],horizontal=True)
    if DIAB=="present":
        DIAB=1
    elif DIAB=="absent":
        DIAB=2
    else:
        DIAB=9
    # BEBAGE = st.number_input("gestational age (in days) of the baby at birth")
    TRANSF = st.radio("Transfer to a hospital for specialized care",["yes","no"],horizontal=True)
    if TRANSF=="no":
        TRANSF=2
    else:
        TRANSF=1
    GEMEL = st.radio("single (=1) or multiple (=2) pregnancy",["single","multiple"],horizontal=True)
    if GEMEL=="single":
        GEMEL=1
    else:
        GEMEL=2


    data = {
        "GEST": GEST,
        "DILATE": DILATE,
        "EFFACE": EFFACE,
        "CONSIS": CONSIS,
        "CONTR":CONTR,
        "MEMBRAN":MEMBRAN,
        "INFECTION":INFECTION,
        "CHR_HYPERTENSION":CHR_HYPERTENSION,
        "AGE":AGE,
        "STRAT":STRAT,
        "GRAVID":GRAVID,
        "PARIT":PARIT,
        "DIAB":DIAB,
        # "BEBAGE":BEBAGE,
        "TRANSF":TRANSF,
        "GEMEL":GEMEL


    }
    features = pd.DataFrame(data, index=[0])
    return features


df1 = user_input_features()

# st.subheader("User Input parameters")
# st.write(df1)

invalid_input = False

# Check for invalid input ranges
if (df1['GEST'].values < 20 or df1['GEST'].values > 45 or
    df1['DILATE'].values < 0 or df1['DILATE'].values > 8 or
    df1['EFFACE'].values < 0 or df1['EFFACE'].values > 100 or
    df1['AGE'].values < 13 or
    df1['GRAVID'].values < 0 or df1['GRAVID'].values > 20 or
    df1['PARIT'].values < 0 or df1['PARIT'].values > 20):
    invalid_input = True
st.header("Prediction")
if invalid_input==False:
    model = tf.keras.models.load_model('my_model.h5')
    # new_model=tf.keras.Sequential([hub.KerasLayer(model,input_shape=(15,))])
    predictions = (model.predict(df1) > 0.5).astype("int32")
    
    st.subheader("Premature:")
    if(predictions==1):
        st.write('Yes')
    else:
        st.write('No')
else:
    st.write(":red[INVALID DATA ENTERED]")

# st.subheader("Class labels and their corresponding index number")
# st.write(iris.target_names)

# st.subheader("Prediction")
# st.write(iris.target_names[prediction])
# # st.write(prediction)

# st.subheader("Prediction Probability")
# st.write(prediction_proba)
