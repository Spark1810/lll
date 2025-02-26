import time
import pandas as pd
import joblib
import streamlit as st
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder




st.set_page_config(page_title="Loan Approval Prediction", page_icon='download.png')

def main():
    # st.markdown(""" 
    #     <style>
    #     # .css-1rs6os.edgvbvh3
    #     # {
    #     # visibility: hidden;
    #     # }

    #     .css-1lsmgbg.egzxvld0
    #     {
    #         visibility: hidden;
    #     }
    #     </style>
    # """, unsafe_allow_html=True)

    menu=['Home', 'About System', 'Statistical Information', 'System']
    choice=st.sidebar.selectbox('Menu', menu)

    if(choice=='Home'):
        st.title('Loan Approval Prediction', '\n')

        st.image('12.jpg')
        st.markdown("""
            ### The bank wants to build a system to anticipate the approval or rejection of the loan for the customers who apply for the loan based on some data of the clients.
            ---
        """, unsafe_allow_html=True)
        st.markdown(""" ### The Loan Amount Approval dataset contains information about loan applications and whether they were approved or not. It provides insights into the factors that influence the approval or rejection of loan applications. Here is a description of the typical fields you may find in such a dataset:

    #### Applicant Information: 
    This section includes data about the loan applicants, such as their demographic information, including age, gender, marital status, and employment details like occupation, income, and employment history.

    Loan Details:
    This section includes information about the loan applied for, such as the loan amount requested, the purpose of the loan (e.g., home purchase, business investment, education), the desired loan term, and the type of loan (e.g., personal loan, mortgage, car loan).

    #### Credit History: 
    Credit history information provides insights into the applicant's past credit behavior and includes data such as credit scores, credit utilization ratio, outstanding debts, and payment history. This information is crucial in assessing the applicant's creditworthiness and their ability to repay the loan.

    #### Financial Information:
    Financial information provides an overview of the applicant's financial status. It includes data on assets (e.g., property, investments, savings), liabilities (e.g., existing loans, credit card debts), and financial obligations (e.g., monthly expenses, existing loan repayments).

    #### Co-Applicant Information:
    If applicable, this section includes information about co-applicants or co-signers for the loan. It includes their relationship to the primary applicant, their financial information, and their contribution to the loan application.

    #### Approval Status:
    This field indicates whether the loan application was approved or rejected. It provides information on the final decision made by the lender or financial institution based on the evaluation of the applicant's information and risk assessment.

    #### Other Relevant Factors:
    Additional fields may be included to capture any other relevant factors that influence the loan approval process. These may include the loan-to-value ratio, debt-to-income ratio, employment stability, collateral information, and any specific requirements or criteria set by the lender.

    ### The dataset is typically collected by financial institutions, such as banks or lending agencies, as part of their loan application process. It is important to note that the specific fields and data included in the dataset may vary depending on the institution and the nature of the loans being offered.

    ### This dataset can be used for various purposes, including building credit risk models, assessing loan eligibility criteria, identifying patterns in loan approval decisions, and developing predictive models to automate the loan approval process.
            """)

    elif choice == 'Statistical Information':
        st.title('Loan Approval Prediction', '\n')
        df1 = pd.read_csv('train_ctrUa4K.csv')

        for col in df1[['Gender','Married','Dependents','Education','Self_Employed', 'Credit_History', 'Property_Area', 'Loan_Status']]:
            if col != 'Loan_Status':
                fig = px.histogram(df1, x=col, color='Loan_Status')
                fig.update_layout(
                    title=f"Distribution of Loan Approval Status by {col}",
                    xaxis_title=col,
                    yaxis_title="Count",
                    legend_title="Loan Status"
                )
                st.markdown('---')
                st.plotly_chart(fig)

        st.markdown('---')
        fig = plt.figure(figsize=(10,5))
        sns.countplot(y=df1['Loan_Amount_Term'], order=df1['Loan_Amount_Term'].value_counts().index, hue=df1['Loan_Status'], palette='Set1')
        plt.legend(bbox_to_anchor=(1,1), loc=2)
        plt.title('Count of Loan Amount Term by Loan Status')
        st.pyplot(fig)

        for col in df1[['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount']]:
            fig = px.box(df1, x='Loan_Status', y=col,
                    labels={'Loan_Status': 'Loan Status', col: 'Value'},
                        title=f'{col} by Loan Status')

            st.markdown('---')
            st.plotly_chart(fig)
            st.markdown('---')

    elif choice == 'About System':

        st.title('Loan Approval Prediction', '\n')
        st.markdown("""
                * **The App is Build Purely in Python Programming Language.** \n
                * **This App Uses XGB Classifier Algorithm.** \n
                * **I used many frameworks like:**
                    <ul> <li> Numpy for processing. </il> </ul>
                    <ul> <li> Pandas for processing. </il> </ul>
                    <ul> <li> Matplotlib for visualization. </il> </ul>
                    <ul> <li> Seaborn for visualization. </il> </ul>
                    <ul> <li> plotly for visualization. </il> </ul>
                    <ul> <li> Scikit-learn for machine learning. </il> </ul>
                    <ul> <li> Joblib for save and load model. </il> </ul>
                    <ul> <li> Streamlit for build API. </il> </ul>

                * **The Dataset was Obtained From Kaggle [Click Here for dataset](https://www.kaggle.com/datasets/ashishkumarjayswal/loanamount-approval)**\n
                    <ul> <li> Gender ---> Male/ Female. </il> </ul>
                    <ul> <li> Dependents ---> Number of dependents. </il> </ul>
                    <ul> <li> Married ---> Applicant married (Y/N). </il> </ul>
                    <ul> <li> Education ---> Applicant Education (Graduate/ Under Graduate). </il> </ul>
                    <ul> <li> Self Employed ---> Self-employed (Y/N). </il> </ul>
                    <ul> <li> ApplicantIncome and CoapplicantIncome ---> (TotalIncome). </il> </ul>
                    <ul> <li> Loan Amount ---> Loan amount in thousands. </il> </ul>
                    <ul> <li> Loan Amount Term ---> Term of the loan in months. </il> </ul>
                    <ul> <li> Credit History ---> Credit history meets guidelines. </il> </ul>
                    <ul> <li> Property Area ---> Urban/ Semi-Urban/ Rural. </il> </ul>
                    <ul> <li> Loan Status ---> (Target) Loan approved (Y/N). </il> </ul>
                """ ,unsafe_allow_html=True)

    elif choice == 'System':
        st.title('Loan Approval Prediction', '\n' )
        st.write('---')

        st.markdown("""\n
        ##### The goal of this application is to classify the loan is accepted or not based on the characteristics or data of this person.
        ---

        ##### To predict whether the loan is accepted or not, just follow these steps:
        ##### 1. Enter information describing the person's data.
        ##### 2. Press the "Predict" button and wait for the result.
        """)
        st.image('pexels-kampus-production-8353769.jpg')
        st.video('video.mp4')
        featuress = pd.read_csv("new_data.csv")
        target = featuress.drop(columns=['Loan_Status'])

        def user_input_features():

            st.sidebar.write('# Enter Your Data...')

            Gender = st.sidebar.radio("Gender",
                                        options=(Gender for Gender in featuress.Gender.unique()))

            Married = st.sidebar.radio("Married", 
                                        options=(Married for Married in featuress.Married.unique()))

            Dependents = st.sidebar.selectbox("Dependents",
                                        options=(Dependents for Dependents in featuress.Dependents.unique()))
            
            Education = st.sidebar.radio("Education", 
                                        options=(Education for Education in featuress.Education.unique()))

            Self_Employed = st.sidebar.radio("Self_Employed", 
                                        options=(Self_Employed for Self_Employed in featuress.Self_Employed.unique()))


            Credit_History = st.sidebar.radio("Credit_History", 
                                        options=(Credit_History for Credit_History in featuress.Credit_History.unique()))

            Loan_Amount_Term = st.sidebar.selectbox("Loan_Amount_Term",
                                                options=(Loan_Amount_Term for Loan_Amount_Term in featuress.Loan_Amount_Term.unique()))    

            Property_Area = st.sidebar.selectbox("Property_Area", 
                                            (Property_Area for Property_Area in featuress.Property_Area.unique()))
            
            TotalIncome = st.sidebar.slider('How many Total Income ?', 0, 100000)

            LoanAmount = st.sidebar.slider('How many Loan Amount ?', 0, 700000)

            data = {
                "Gender": Gender,
                "Married": Married,
                "Dependents": Dependents,
                "Education": Education,
                "Self_Employed": Self_Employed,
                "LoanAmount": LoanAmount,
                "Loan_Amount_Term": Loan_Amount_Term,
                "Credit_History": Credit_History,
                "Property_Area": Property_Area,
                "TotalIncome": TotalIncome 
                }

            features = pd.DataFrame(data, index=[0])
            return features

        input_df = user_input_features()

        df = pd.concat([input_df,target],axis=0)

        for col in df[['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Loan_Amount_Term', 'Property_Area']]:
            df[col] = LabelEncoder().fit_transform(df[col])
        
        XGB_MODEL_PATH = joblib.load("final_model.h5")
        XGB_SCALER_PATH = joblib.load("scaler.h5")

        scaled_data = XGB_SCALER_PATH.transform(df)
        prediction_proba = XGB_MODEL_PATH.predict_proba(scaled_data)
        if st.sidebar.button('Predict'):
        
            my_bar = st.sidebar.progress(0)
            for percent_complete in range(100):
                time.sleep(.05)
                my_bar.progress(percent_complete + 1)
            st.sidebar.success(f'# The probability of this Loan is  accepted is: {round(prediction_proba[0][1] * 100, 2)}%')

if __name__ == "__main__":
    main()
