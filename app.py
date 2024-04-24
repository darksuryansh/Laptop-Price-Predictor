# Import libraries

import streamlit as st
import pandas as pd
import numpy as np
import pickle
from PIL import Image

# Load Cleaned data
df = pd.read_csv('laptop-clean.csv')

# Load Preprocessor
preprocessor = pickle.load(open('preprocessor.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# Inputs
st.title("Laptop Predictor")
# Brand & Type
st.write('- <p style="font-size:26px;"> Laptop Brand & Type</p>',unsafe_allow_html=True)
Company = st.selectbox('Company',df['Company'].unique())
TypeName = st.selectbox('Type',df[df['Company']== Company].groupby('TypeName')['TypeName'].value_counts().index)
WeightKG = st.selectbox('Weight?', np.sort(df[(df['Company']== Company)&(df['TypeName'] ==TypeName)].groupby('WeightKG')['WeightKG'].count().index))
# Screen Inputs
st.write('- <p style="font-size:26px;"> Screen Specs</p>',unsafe_allow_html=True)
TouchScreen = st.selectbox('Does it has a Touch Screen?',('Yes','No'))
if TouchScreen == 'Yes':
        TouchScreen = 1
else:
        TouchScreen = 0
PanelType = st.selectbox(('PanelType'),df['PanelType'].unique())
Resolution = st.selectbox('Resolution',np.sort(df[(df['Company']== Company)&(df['TypeName'] ==TypeName)].groupby('Resolution')['Resolution'].count().index))
Inches = st.number_input('Inches',df[(df['Company']== Company)&(df['TypeName'] ==TypeName)].groupby('Inches')['Inches'].count().index[0],df[(df['Company']== Company)&(df['TypeName'] ==TypeName)].groupby('Inches')['Inches'].count().index[-1])

# Specs
## Processor & Ram
st.write('- <p style="font-size:26px;"> Processor</p>',unsafe_allow_html=True)
RamGB = st.selectbox('Ram GB', np.sort(df[(df['Company']== Company)&(df['TypeName'] ==TypeName)].groupby('RamGB')['RamGB'].count().index))
CpuBrand = st.selectbox('Cpu Brand',np.sort(df[(df['Company']== Company)&(df['TypeName'] ==TypeName)].groupby('CpuBrand')['CpuBrand'].count().index))
GHz = st.selectbox('GHz',np.sort(df[(df['Company']== Company)&(df['TypeName'] ==TypeName)&(df['CpuBrand']==CpuBrand)].groupby('GHz')['GHz'].count().index))
CpuVersion = st.selectbox('Cpu Version',np.sort(df[(df['Company']== Company)&(df['TypeName'] ==TypeName)&(df['CpuBrand']==CpuBrand) &(df['GHz']==GHz)].groupby('CpuVersion')['CpuVersion'].count().index))

## Hard 
st.write('- <p style="font-size:26px;"> Hard disk capacity</p>',unsafe_allow_html=True)
MainMemory = st.selectbox('Main Memory GB',np.sort(df[(df['Company']== Company)&(df['TypeName'] ==TypeName)].groupby('MainMemory')['MainMemory'].count().index))
MainMemoryType = st.selectbox('Main Memory Type',df[(df['Company']== Company)&(df['MainMemory']==MainMemory)].groupby('MainMemoryType')['MainMemoryType'].count().index)
st.write('Does it has an Extra Hard driver?')
secMem = st.checkbox(label='Yes')
if secMem:
        st.write('Please enter Second Memory specs')
        SecondMemory = st.selectbox('Second Memory GB',np.sort(df['SecondMemory'].unique())[1:])
        SecondMemoryType = st.selectbox('Second Memory Type',df['SecondMemoryType'].unique()[1:])
else:
        SecondMemory = 0.0
        SecondMemoryType = 'None'

## Graphics Card
st.write('- <p style="font-size:26px;"> Graphics Card</p>',unsafe_allow_html=True)
GpuBrand = st.selectbox('Gpu Brand',np.sort(df[(df['Company']== Company)&(df['TypeName'] ==TypeName)].groupby('GpuBrand')['GpuBrand'].count().index))
GpuVersion = st.selectbox('Gpu Version',np.sort(df[(df['Company']== Company)&(df['TypeName'] ==TypeName)&(df['GpuBrand']==GpuBrand)].groupby('GpuVersion')['GpuVersion'].count().index))

# Operating system
st.write('- <p style="font-size:22px;"> Operating System</p>',unsafe_allow_html=True)
OpSys = st.selectbox('Operating System',np.sort(df[(df['Company']== Company)].groupby('OpSys')['OpSys'].count().index))


# Preprocessing
new_data = {'Company': Company, 'TypeName': TypeName, 'Inches': Inches, 'RamGB': RamGB, 
        'OpSys': OpSys,
         'WeightKG': WeightKG, 'GHz': GHz, 'CpuBrand': CpuBrand, 'CpuVersion': CpuVersion, 
         'MainMemory': MainMemory, 'SecondMemory': SecondMemory,
         'MainMemoryType':MainMemoryType ,'SecondMemoryType':SecondMemoryType,
         'TouchScreen':TouchScreen,'Resolution':Resolution,
         'PanelType':PanelType , 'GpuBrand':GpuBrand,'GpuVersion':GpuVersion}
new_data = pd.DataFrame(new_data, index=[0])
new_data_preprocessed = preprocessor.transform(new_data)
st.write('- <p style="font-size:26px;"> Laptop Specs</p>',unsafe_allow_html=True)
new_data
# Prediction
log_price = model.predict(new_data_preprocessed) # in log scale
price = np.expm1(log_price) # in original scale
with st.container():
    coll1, coll2, coll3 = st.columns([3,6,1])

    with coll1:
            st.write("     ")

    with coll2:
                # Output
            if st.button('Predict'):
                 st.markdown('# Price of Laptop:')
                 price[0]

    with coll3:
            st.write("")
