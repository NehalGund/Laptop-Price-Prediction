import streamlit as st
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
st.set_option('deprecation.showPyplotGlobalUse', False)

for dirname, _, filenames in os.walk('/laptop dataset'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
dataframe = pd.read_csv(r'/laptop dataset/input/laptop.csv')
def main():
    activities = ['Select', 'Preprocessed Data', 'Prediction', 'About']
    choices = st.sidebar.selectbox("Select choices" ,activities)
    df = pickle.load(open(r'/laptop dataset/pickle file/processed_data.pickle','rb'))
    pipe = pickle.load(open(r'/laptop dataset/pickle file/ml_model_pipe.pickle', 'rb'))
    if choices == 'Select':
        st.title('Welcome to Laptop Price Prediction Web Application')
        st.header('Preprocessed Data -')
        st.info('Preprocessed Data shows Laptop data and Correlation of Price with other specifications')
        st.header('Prediction -')
        st.info('Select your expected laptop specifications and you will get approximate price.')
        st.header('About -')
        st.info('Small description on this project.')
            
    elif choices == 'Preprocessed Data':
        if st.checkbox("Show Details"):
            st.success('Number of columns : {0}'.format(df.shape[1]))
            st.success('Number of records : {0}'.format(df.shape[0]))
            if st.checkbox("Show Columns"):
                all_columns = df.columns.to_list()
                st.write(all_columns)
            if st.checkbox("Show Data"):
                st.dataframe(df.head(10))                
        if st.checkbox("Correlation Plot(Seaborn)"):
            corr_data = pd.DataFrame(df.corr()['Price'])
            plt.figure(figsize = (1,5))
            st.write(sns.heatmap(corr_data,annot=True))
            st.pyplot()           
            
    elif choices == 'Prediction':
        st.title("Laptop Price Prediction")

        # Company name:-
        company = st.selectbox('Laptop Company',['Select','Huawei','Apple','Acer','Asus','HP','Dell','Lenovo','MSI','Microsoft','Toshiba','Razer','Mediacom','Samsung'])
        if company != 'Select':
            st.success(company)
        else:
            company = 'Dell'
            st.info('Please select Laptop Company')

        # Type of laptop:-
        lap_type = st.selectbox('Laptop Type',['Select','Notebook','Ultrabook','2 in 1 Convertible','Netbook','Gaming','Workstation'])
        if lap_type != 'Select':
            st.success(lap_type)
        else:
            lap_type = 'Notebook'
            st.info('Please select Laptop Type')        

        # OS
        os1 = st.radio('Operating System',['Select','No OS','Linux','Mac','Windows','Other OS'])
        if os1 == 'Select':
            os = 'No OS'
            st.info('Please select a Operating System')
        elif os1 == 'No OS':
            os = 'No OS'
            st.success(os)
        elif os1 == 'Linux':
            os = 'Linux'
            st.success(os)
        elif os1 in ['Mac','Windows','Other OS']:
            if os1 == 'Mac':
                os_type = st.radio('Mac Operating System',['Select','Mac','Mac X'])
                if os_type != 'Select':
                    os = os_type
                    st.success(os)
                else:
                    st.info('Please select Mac Operating System')
            elif os1 == 'Windows':
                os_type = st.radio('Windows Operating System',['Select','Windows 7','Windows 10','Windows 10 S'])
                if os_type != 'Select':
                    os = os_type
                    st.success(os)
                else:
                    st.info('Please select Windows Operating System')
            elif os1 == 'Other OS':
                os_type = st.radio('Other Operating System',['Select','Android','Chrome OS'])
                if os_type != 'Select':
                    os = os_type
                    st.success(os)
                else:
                    st.info('Please select any one Operating System')
        
        #cpu
        cpu1 = st.radio('CPU Processor',['Select','Intel','AMD','Samsung Cortex'])
        if cpu1 == 'Select':
            cpu = 'Intel Processor'
            st.info('Please select a CPU Processor')
        elif cpu1 in ['AMD','Intel','Samsung Cortex']:
            if cpu1 == 'Intel':
                cpu2 = st.selectbox('Intel Processor',['Select','Intel i Processor','Intel Core M','Intel Atom','Intel Celeron Dual','Intel Pentium Quad','Other Intel Processor'])
                if cpu2 == 'Select':
                    st.info('Please select Intel Processor')
                elif cpu2 == 'Intel i Processor':
                    cpu_type = st.radio('Intel i Processor',['Select','Intel Core i3','Intel Core i5','Intel Core i7'])
                    if cpu_type != 'Select':
                        cpu = cpu_type
                        st.success(cpu)
                    else:
                        st.info('Please select Intel i Processor')
                elif cpu2 != 'Select'and cpu2 != 'Other Intel Processor':
                    cpu = cpu2
                    st.success(cpu)
                elif cpu2 == 'Other Intel Processor':
                    cpu = 'Intel Processor'
                    st.success(cpu2)
                else:
                    st.info('Please select Intel Processor')
            elif cpu1 == 'AMD':
                cpu_type = st.radio('AMD Processor',['Select','AMD A-Series','AMD E-Series','Other AMD Processor'])
                if cpu1 == 'Select':
                    st.info('Please select Intel CPU')
                elif cpu_type != 'Select' and cpu_type != 'Other AMD Processor':
                    cpu = cpu_type
                    st.success(cpu)
                elif cpu_type == 'Other AMD Processor':
                    cpu = 'AMD Processor'
                    st.success(cpu_type)
                else:
                    st.info('Please select AMD Processor')
            elif (cpu1 == 'Samsung Cortex'):
                cpu = 'Samsung Cortex'  
                st.success(cpu)           

        # CPU Frequency
        cpu_freq1 = st.select_slider('CPU Frequency',['Frequency',1.44,1.5,2.0,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8,2.9,3.0,3.1,3.2,3.6])
        if cpu_freq1 != 'Frequency':
            cpu_freq = cpu_freq1
            st.success(cpu_freq)
        else:
            cpu_freq = 2.2
            st.info('Please select CPU Frequency')
            
        # GPU
        gpu1 = st.radio('Graphics Card',['Select','AMD GPU','ARM GPU','Intel GPU','Nvidia GPU'])
        if gpu1 == 'Select':
            gpu = 'Intel HD Graphics'
            st.info('Please select a GPU Card')
        elif gpu1 in ['AMD GPU','ARM GPU','Intel GPU','Nvidia GPU']:
            if gpu1 == 'AMD GPU':
                gpu2 = st.radio('AMD GPU',['Select','AMD Radeon','AMD FirePro'])
                if gpu2 != 'Select':
                    gpu = gpu2
                    st.success(gpu)
                else:
                    st.info('Please select AMD GPU Card')
            elif gpu1 == 'ARM GPU':
                gpu = 'ARM Mali T860'
                st.success(gpu1)
            elif gpu1 == 'Intel GPU':
                gpu2 = st.radio('Intel GPU',['Select','Intel Iris','Intel HD Graphics','Intel UHD Graphics'])
                if gpu2 != 'Select':
                    gpu = gpu2
                    st.success(gpu)
                else:
                    st.info('Please select Intel GPU Card')
            elif gpu1 == 'Nvidia GPU':
                gpu2 = st.radio('Nvidia GPU',['Select','Nvidia GeForce M','Nvidia GeForce MX','Nvidia GeForce GTX','Nvidia Quadro'])
                if gpu2 != 'Select':
                    gpu = gpu2
                    st.success(gpu)
                else:
                    st.info('Please select Nvidia GPU Card')

        # Inches
        inch = st.select_slider('Enter screen size (inches)',
                                ['Slider',10.1,11.3,11.6,12.0,12.3,12.5,13.0,13.3,13.5,
                                          13.9,14.0,14.1,15.0,15.4,15.6,17.0,17.3,18.4])
        if inch != 'Slider':
            st.success(inch)
        else:
            inch = 15.6

        # HD Quality and Screen Resolutions
        hd = st.radio('Select HD Quality',['Select','Full HD','Quad HD+','4K Ultra HD'])
        full_hd = 0
        quad_hd = 0
        ultra_4k_hd = 0
        if hd != 'Select':
            if hd == 'Full HD':
                full_hd = 1
                scr_res = st.select_slider('Full HD Resolutions',  ['Select','1366x768', '1440x900', '1600x900', '1920x1080'])
                if scr_res != 'Select':
                    res = scr_res.split('x')
                    x_res = res[0]
                    y_res = res[1]
                    st.success(hd)
                    st.success(scr_res)
                else:
                    st.info('Please select Full HD screen resolution') #[]
            elif hd == 'Quad HD+': 
                quad_hd = 1
                scr_res = st.select_slider('Quad HD+ Resolutions',  ['Select','1920x1200','2160x1440','2256x1504','2304x1440','2400x1600','2560x1440','2560x1600'])
                if scr_res != 'Select':
                    res = scr_res.split('x')
                    x_res = res[0]
                    y_res = res[1]
                    st.success(hd)
                    st.success(scr_res)
                else:
                    st.info('Please select Quad HD+ screen resolution')
            elif hd == '4K Ultra HD': 
                ultra_4k_hd = 1
                scr_res = st.select_slider('4K Ultra HD Resolutions',  ['Select','2736x1824','2880x1800','3200x1800','3840x2160'])
                if scr_res != 'Select':
                    res = scr_res.split('x')
                    x_res = res[0]
                    y_res = res[1]
                    st.success(hd)
                    st.success(scr_res)
                else:
                    st.info('Please select 4K Ultra HD+ screen resolution')
        else:
            st.info('Please select HD Quality')
            hd = 'Full HD'
            full_hd = 1
            quad_hd = 0
            ultra_4k_hd = 0
            x_res = 1920
            y_res = 1080 

        # Dispay Type
        disp = st.radio('Select Display Type',['None','IPS','Touch Screen','Both IPS and Touch Screen'])
        ips = 0
        touchscreen = 0
        if disp == 'IPS':
            ips = 1
        elif disp == 'Touch Screen': 
            touchscreen = 1
        elif disp == 'Both IPS and Touch Screen':
            ips = 1
            touchscreen = 1
        else:
            ips = 0
            touchscreen = 0
        st.success(disp)

        # Ram:-
        ram1 = st.select_slider('RAM (GB)', ['Select',2,4,6,8,12,16,24,32,64])
        if ram1 != 'Select':
            ram = ram1
            st.success('RAM = {0} GB'.format(ram))
        else:
            ram = 8
            st.info('Please select RAM (GB)')

        # Drive Type and Size
        st.write('Drive Type:')
        hdd_drive = st.checkbox('HDD')
        ssd_drive = st.checkbox('SSD')
        flash_drive = st.checkbox('Flash Storage Drive')
        hdd = 0
        ssd = 0
        flash = 0
        drive1 = 'No'
        drive2 = 'No'
        drive3 = 'No'
        if (hdd_drive == True) or (ssd_drive == True) or (flash_drive == True):
            if hdd_drive == True:
                drive1 = st.select_slider('Select HDD (GB)',['No HDD',32,128,500,1000,2000])
                if drive1 != 'No HDD':
                    hdd = drive1
                    st.success('HDD = {0} GB'.format(drive1))
                else:
                    hdd = 0
            if ssd_drive == True:
                drive2 = st.select_slider('Select SSD (GB)',['No SSD',8,16,32,128,180,240,512,768,1000])
                if drive2 != 'No SSD':
                    ssd = drive2
                    st.success('SSD = {0} GB'.format(drive2))
                else:
                    ssd = 0
            if flash_drive == True:
                drive3 = st.select_slider('Select Flash Storage (GB)',['No Flash Storage Drive',16,32,64,128,512])
                if drive3 != 'No Flash Storage Drive':
                    flash = drive3
                    st.success('Flash Storage = {0} GB'.format(drive3))
                else:
                    flash = 0
        elif (hdd_drive == False) and (ssd_drive == False) and (flash_drive == False):
            hdd = 1000
            ssd = 0
            flash = 0
            drive1 = 1000
            drive2 = 'No'
            drive3 = 'No'
            st.info('Please select Drive Type')

        # Weight:-
        weight = st.number_input("Enter weight of the Laptop for 0.7  to 4.7")
        if weight != 0.00:
            st.success('{0} Kg'.format(weight))
        else:
            weight = 1.5
            st.info('Please Enter Weight')
        
        # Price prediction
        if st.button("Show Price"):
            query = [[company,lap_type,inch,x_res,y_res,full_hd,quad_hd,ultra_4k_hd,ips,touchscreen,cpu,cpu_freq,ram,hdd,ssd,flash,gpu,os,weight]]
            prediction = str(int(np.exp(pipe.predict(query)[0])))
            st.success("The price for the following configuration is ₹ {0}/-".format(prediction))
            configuration_names = ['Laptop Company','Laptop Type','Operating System','CPU Processor','CPU Frequency (GHz)','Graphics Card','Screen size (inches)','HD Quality','Resolutions','Display Type','RAM (GB)','Drive Type','Weight (kg)']
            configuration = [company,lap_type,os,cpu,cpu_freq,gpu,inch,hd,x_res,disp,ram,drive1,weight]
            for i in range(len(configuration_names)):
                if i == 11:
                    drive_list = ['HDD (GB)','SSD (GB)','Flash Storage Drive (GB)']
                    drive_var = [drive1,drive2,drive3]
                    for k in range(len(drive_list)):
                        st.write('{0} : "{1}"'.format(drive_list[k],drive_var[k]))
                elif i == 8:
                    st.write('Resolutions : "{0}x{1}"'.format(x_res,y_res))                    
                else:    
                    st.write('{0} : "{1}"'.format(configuration_names[i],configuration[i]))
        
    elif choices == 'About':  
        st.info('About project -')
        st.write('''
                1. In this project, a supervised machine learning model is built to predict tentative laptop price based on its specifications.
                2. This model is trained on dataset which is taken from kaggle.
                3. The dataset contains laptop specifications and corresponding prices.
                4. Scikit-learn library is used to build the machine learning model.
                5. Streamlit is used to make a web application that allows users to select the laptop specifications and user gets tentative price of the laptop.
                ''')
        st.warning('Prerequisites :')
        st.write('''
                - numpy
                - pandas
                - matplotlib
                - seaborn
                - scikit-learn
                - pickle
                - streamlit
                ''')
        st.info('Steps to build machine learning model')
        st.write('''
                1. Data preprocessing
                2. EDA
                3. Algorithm selection
                4. Training
                5. Evaluation
                6. Prediction
                ''')
        st.info('Dataset Description:')
        st.write(pd.DataFrame({
            'Column name':['Unnamed: 0',
                            'Company',                                    
                            'TypeName',          
                            'Inches',                                           
                            'ScreenResolution',                  
                            'Cpu',                                                
                            'Ram',                                                 
                            'Memory',                            
                            'Gpu',                                               
                            'OpSys',                                              
                            'Weight',                                                 
                            'Price',
                            ],
            'Description':[ 'Row number',
                            'Laptop manufcturing company names',
                            'Type of laptop (Notebook, Ultrabook, Netbook, Workstation)',
                            'Laptop screen size in inches',
                            'Screen resolutions with screen display type',
                            'CPU name with speed in GHz',
                            'RAM size of laptop in GB',
                            'Memory type and size of memory in GB and TB',
                            'GPU name with their series',
                            'Operating System of laptop',
                            'Weight of laptop in kg',
                            'Laptop price in ( ₹ ) Indian Rupee'
                            ]
                }))
        st.info('Dataset Sample Row:')
        st.write(dataframe.sample())
        st.info('Sample Row of Preprocessed Dataset:')
        st.write(df.sample())

    
    
if __name__ == '__main__':
    main()

                             

                        
                        