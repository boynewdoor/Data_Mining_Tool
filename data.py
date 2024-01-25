import tkinter as tk
from tabulate import tabulate
import tkinter as tk
from tkinter import filedialog
import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from scipy.stats import chi2_contingency
from tabulate import tabulate
import warnings
from io import StringIO

warnings.filterwarnings('ignore')

def dataset_handling():
    
    def analyse_data():

        browse_button.config(state='disabled')

        print("\nAnalysing data...\n")

        # Importing data file
        data_file = "car.data"
        data_df = pd.read_csv(data_file, header=None)

        # Definition of desired attributes in a specific order
        desired_attributes = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']

        # Creating a dataframe
        combined_df = pd.DataFrame(data_df.values, columns=desired_attributes)

        # Saving a dataframe in .csv format
        combined_df.to_csv("data.csv", index=False)

        # Printing the original data
        #print("\nOriginal Data:")
        #print(data_df.rename(columns=dict(enumerate(desired_attributes))))
        #print("\n")

        # Import .csv file
        file_path = "data.csv"
        df = pd.read_csv(file_path)

        unique_percentages_folder = "unique_percentages"
        os.makedirs(unique_percentages_folder, exist_ok=True)

        print("\nUnique values percentage generation...\n")

        # Calculating the percentage of unique values for each column
        unique_percentages = {}
        total_rows = len(df)

        for column in df.columns:
            counts = df[column].value_counts()
            percentages = (counts / total_rows * 100).round(3).astype(str) + " %"
            unique_percentages[column] = pd.concat([counts, percentages], axis=1, keys=['Count', 'Percentage']).rename_axis(None)

        # Remove the 'class' header
            if 'class' in unique_percentages[column].index:
                unique_percentages[column].index = ['' if idx == 'class' else idx for idx in unique_percentages[column].index]

        # Saving tables has .png images
        for column, result_df in unique_percentages.items():
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.axis('off')
            ax.table(cellText=result_df.values, colLabels=result_df.columns, rowLabels=result_df.index, cellLoc='center', loc='center')
            plt.savefig(os.path.join(unique_percentages_folder, f'tabela_{column}_unique_percentages.png'))
            plt.close()

            result = f"Column: {column}"
            text_area.config(state="normal")
            text_area.insert(tk.END, result + "\n")
            text_area.insert(tk.END, str(result_df) + "\n")
            text_area.insert(tk.END, "\n")
            text_area.config(state="disabled")

        # 1. Descriptive Statistics
        print("\nGenerating Descriptive Statistics...\n")
        result = "Descriptive Statistics:\n"
        descriptive_statistics_folder = "descriptive_statistics"
        os.makedirs(descriptive_statistics_folder, exist_ok=True)
        desc_stats = df.describe()

        random_header = np.random.choice(['Random_Header'])
        desc_stats[random_header] = ['count', 'unique', 'top', 'freq']
        desc_stats = desc_stats[['Random_Header'] + [col for col in desc_stats.columns if col != 'Random_Header']]
        desc_stats.columns = ['' if col == 'Random_Header' else col for col in desc_stats.columns]

        table_desc_stats = desc_stats.to_html
        # Saving HTML table as a .png image
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.axis('off')
        ax.table(cellText=desc_stats.values, colLabels=desc_stats.columns, cellLoc='center', loc='center')
        plt.savefig(os.path.join(descriptive_statistics_folder, 'tabela_desc_stats.png'))
        plt.close()
        result = f"Column: {column}\n"
        text_area.config(state="normal")
        text_area.insert(tk.END, result + desc_stats.to_string() + "\n")
        text_area.insert(tk.END, "\n")  
        text_area.config(state="disabled")

        # Create folders to save the images
        countplot_folder = "countplot"
        boxplot_folder = "boxplot"

        os.makedirs(countplot_folder, exist_ok=True)
        os.makedirs(boxplot_folder, exist_ok=True)

        # 2. Data Visualization
        # - Graphics to categorical variable counting
        for column in df.columns[:-1]: 
            plt.figure(figsize=(8, 5))
            sns.countplot(x=column, data=df, hue='class')
            plt.title(f'Countplot for {column} vs Class')
            
            # Save the countplot image in the countplot folder
            plt.savefig(os.path.join(countplot_folder, f'countplot_{column}_vs_class.png'))
            plt.close()

        # 4. Outliers Analysis
        # - Box plots for each attributes
        for column in df.columns[:-1]:
            plt.figure(figsize=(8, 5))
            sns.boxplot(x='class', y=column, data=df)
            plt.title(f'Boxplot for {column} vs Class')
            
            # Save the boxplot image in the boxplot folder
            plt.savefig(os.path.join(boxplot_folder, f'boxplot_{column}_vs_class.png'))
            plt.close()

        # 5. Missing Data Treatment
        # - Checking missing data presence
        print("\nChecking Missing Data...\n")
        missing_data = df.isnull().sum()
        result_missing = "Missing Data:\n"
        text_area.config(state="normal")
        text_area.insert(tk.END, result_missing + str(missing_data) +"\n")
        text_area.insert(tk.END, "\n")
        text_area.config(state="disabled")

        # 7. Cross-Tabulation

        print("\nGenerating Cross-Tabulations...\n")

        # CROSS-TABULATION -> BUYING x Others
        cross_tab = pd.crosstab(df['buying'], df['class'])
        result_ct = "Cross Tabulation (Buying x Class):\n"
        text_area.config(state="normal")
        text_area.insert(tk.END,result_ct+str(cross_tab)+"\n")
        text_area.insert(tk.END, "\n")
        text_area.config(state="disabled")

        cross_tab1 = pd.crosstab(df['buying'], df['maint'])
        result_ct = "Cross Tabulation (Buying x Maint):\n"
        text_area.config(state="normal")
        text_area.insert(tk.END,result_ct+str(cross_tab1)+"\n")
        text_area.insert(tk.END, "\n")
        text_area.config(state='disabled')

        cross_tab2 = pd.crosstab(df['buying'], df['doors'])
        result_ct = "Cross Tabulation (Buying x Doors):\n"
        text_area.config(state="normal")
        text_area.insert(tk.END,result_ct+str(cross_tab2)+"\n")
        text_area.insert(tk.END, "\n")
        text_area.config(state='disabled')

        cross_tab3 = pd.crosstab(df['buying'], df['persons'])
        result_ct = "Cross Tabulation (Buying x Persons):\n"
        text_area.config(state="normal")
        text_area.insert(tk.END,result_ct+str(cross_tab3)+"\n")
        text_area.insert(tk.END, "\n")
        text_area.config(state='disabled')

        cross_tab4 = pd.crosstab(df['buying'], df['lug_boot'])
        result_ct = "Cross Tabulation (Buying x Lug_Boot):\n"
        text_area.config(state="normal")
        text_area.insert(tk.END,result_ct+str(cross_tab4)+"\n")
        text_area.insert(tk.END, "\n")
        text_area.config(state='disabled')

        cross_tab5 = pd.crosstab(df['buying'], df['safety'])
        result_ct = "Cross Tabulation (Buying x Safety):\n"
        text_area.config(state="normal")
        text_area.insert(tk.END,result_ct+str(cross_tab5)+"\n")
        text_area.insert(tk.END, "\n")
        text_area.config(state='disabled')

        # CROSS-TABULATION -> MAINT x Others
        cross_tab6 = pd.crosstab(df['maint'], df['class'])
        result_ct = "Cross Tabulation (Maint x Class):\n"
        text_area.config(state="normal")
        text_area.insert(tk.END,result_ct+str(cross_tab6)+"\n")
        text_area.insert(tk.END, "\n")
        text_area.config(state='disabled')

        cross_tab7 = pd.crosstab(df['maint'], df['doors'])
        result_ct = "Cross Tabulation (Main x Doors):\n"
        text_area.config(state="normal")
        text_area.insert(tk.END,result_ct+str(cross_tab7)+"\n")
        text_area.insert(tk.END, "\n")
        text_area.config(state='disabled')

        cross_tab8 = pd.crosstab(df['maint'], df['persons'])
        result_ct = "Cross Tabulation (Maint x Persons):\n"
        text_area.config(state="normal")
        text_area.insert(tk.END,result_ct+str(cross_tab8)+"\n")
        text_area.insert(tk.END, "\n")
        text_area.config(state='disabled')

        cross_tab9 = pd.crosstab(df['maint'], df['lug_boot'])
        result_ct = "Cross Tabulation (Maint x Lug_Boot):\n"
        text_area.config(state="normal")
        text_area.insert(tk.END,result_ct+str(cross_tab9)+"\n")
        text_area.insert(tk.END, "\n")
        text_area.config(state='disabled')

        cross_tab10 = pd.crosstab(df['maint'], df['safety'])
        result_ct = "Cross Tabulation (Maint x Safety):\n"
        text_area.config(state="normal")
        text_area.insert(tk.END,result_ct+str(cross_tab10)+"\n")
        text_area.insert(tk.END, "\n")
        text_area.config(state='disabled')

        # CROSS-TABULATION -> DOORS x Others
        cross_tab11 = pd.crosstab(df['doors'], df['class'])
        result_ct = "Cross Tabulation (Doors x Class):\n"
        text_area.config(state="normal")
        text_area.insert(tk.END,result_ct+str(cross_tab11)+"\n")
        text_area.insert(tk.END, "\n")
        text_area.config(state='disabled')

        cross_tab12 = pd.crosstab(df['doors'], df['persons'])
        result_ct = "Cross Tabulation (Doors x Persons):\n"
        text_area.config(state="normal")
        text_area.insert(tk.END,result_ct+str(cross_tab12)+"\n")
        text_area.insert(tk.END, "\n")
        text_area.config(state='disabled')

        cross_tab13 = pd.crosstab(df['doors'], df['lug_boot'])
        result_ct = "Cross Tabulation (Doors x Lug_Boot):\n"
        text_area.config(state="normal")
        text_area.insert(tk.END,result_ct+str(cross_tab13)+"\n")
        text_area.insert(tk.END, "\n")
        text_area.config(state='disabled')

        cross_tab14 = pd.crosstab(df['doors'], df['safety'])
        result_ct = "Cross Tabulation (Doors x Safety):\n"
        text_area.config(state="normal")
        text_area.insert(tk.END,result_ct+str(cross_tab14)+"\n")
        text_area.insert(tk.END, "\n")
        text_area.config(state='disabled')

        # CROSS-TABULATION -> PERSONS x Others
        cross_tab15 = pd.crosstab(df['persons'], df['class'])
        result_ct = "Cross Tabulation (Persons x Class):\n"
        text_area.config(state="normal")
        text_area.insert(tk.END,result_ct+str(cross_tab15)+"\n")
        text_area.insert(tk.END, "\n")
        text_area.config(state='disabled')

        cross_tab16 = pd.crosstab(df['persons'], df['lug_boot'])
        result_ct = "Cross Tabulation (Persons x Lug_Boot):\n"
        text_area.config(state="normal")
        text_area.insert(tk.END,result_ct+str(cross_tab16)+"\n")
        text_area.insert(tk.END, "\n")
        text_area.config(state='disabled')

        cross_tab17 = pd.crosstab(df['persons'], df['safety'])
        result_ct = "Cross Tabulation (Persons x Safety):\n"
        text_area.config(state="normal")
        text_area.insert(tk.END,result_ct+str(cross_tab17)+"\n")
        text_area.insert(tk.END, "\n")
        text_area.config(state='disabled')

        # CROSS-TABULATION -> LUG_BOOT x Others
        cross_tab18 = pd.crosstab(df['lug_boot'], df['class'])
        result_ct = "Cross Tabulation (Lug_Boot x Class):\n"
        text_area.config(state="normal")
        text_area.insert(tk.END,result_ct+str(cross_tab18)+"\n")
        text_area.insert(tk.END, "\n")
        text_area.config(state='disabled')

        cross_tab19 = pd.crosstab(df['lug_boot'], df['safety'])
        result_ct = "Cross Tabulation (Lug_Boot x Safety):\n"
        text_area.config(state="normal")
        text_area.insert(tk.END,result_ct+str(cross_tab19)+"\n")
        text_area.insert(tk.END, "\n")
        text_area.config(state='disabled')

        # CROSS-TABULATION -> SAFETY x Others
        cross_tab20 = pd.crosstab(df['safety'], df['class'])
        result_ct = "Cross Tabulation (Safety x Class):\n"
        text_area.config(state="normal")
        text_area.insert(tk.END,result_ct+str(cross_tab20)+"\n")
        text_area.insert(tk.END, "\n")
        text_area.config(state='disabled')

        # 8. Chi-Square Test

        print("\nGenerating Chi-Square Tests...\n")

        # CHI-SQUARE -> BUYING
        chi2, p, dof, expected = chi2_contingency(cross_tab)
        result_chi2 = "Chi-Square Test (Buying x Class):\n"
        text_area.config(state="normal")
        text_area.insert(tk.END,result_chi2+f"Chi2: "+str(chi2)+" "+f"p-value: "+str(p)+"\n")
        text_area.insert(tk.END, "\n")
        text_area.config(state="disabled")

        chi2, p, dof, expected = chi2_contingency(cross_tab1)
        result_chi2 = "Chi-Square Test (Buying x Maint):\n"
        text_area.config(state="normal")
        text_area.insert(tk.END,result_chi2+f"Chi2:"+str(chi2)+" "+f"p-value:"+str(p)+"\n")
        text_area.insert(tk.END, "\n")
        text_area.config(state="disabled")

        chi2, p, dof, expected = chi2_contingency(cross_tab2)
        result_chi2 = "Chi-Square Test (Buying x Doors):\n"
        text_area.config(state="normal")
        text_area.insert(tk.END,result_chi2+f"Chi2:"+str(chi2)+" "+f"p-value:"+str(p)+"\n")
        text_area.insert(tk.END, "\n")
        text_area.config(state="disabled")

        chi2, p, dof, expected = chi2_contingency(cross_tab3)
        result_chi2 = "Chi-Square Test (Buying x Persons):\n"
        text_area.config(state="normal")
        text_area.insert(tk.END,result_chi2+f"Chi2:"+str(chi2)+" "+f"p-value:"+str(p)+"\n")
        text_area.insert(tk.END, "\n")
        text_area.config(state="disabled")

        chi2, p, dof, expected = chi2_contingency(cross_tab4)
        result_chi2 = "Chi-Square Test (Buying x Lug_Boot):\n"
        text_area.config(state="normal")
        text_area.insert(tk.END,result_chi2+f"Chi2:"+str(chi2)+" "+f"p-value:"+str(p)+"\n")
        text_area.insert(tk.END, "\n")
        text_area.config(state="disabled")

        chi2, p, dof, expected = chi2_contingency(cross_tab5)
        result_chi2 = "Chi-Square Test (Buying x Safety):\n"
        text_area.config(state="normal")
        text_area.insert(tk.END,result_chi2+f"Chi2:"+str(chi2)+" "+f"p-value:"+str(p)+"\n")
        text_area.insert(tk.END, "\n")
        text_area.config(state="disabled")

        # CHI-SQUARE -> MAINT
        chi2, p, dof, expected = chi2_contingency(cross_tab6)
        result_chi2 = "Chi-Square Test (Maint x Class):\n"
        text_area.config(state="normal")
        text_area.insert(tk.END,result_chi2+f"Chi2:"+str(chi2)+" "+f"p-value:"+str(p)+"\n")
        text_area.insert(tk.END, "\n")
        text_area.config(state="disabled")

        chi2, p, dof, expected = chi2_contingency(cross_tab7)
        result_chi2 = "Chi-Square Test (Maint x Doors):\n"
        text_area.config(state="normal")
        text_area.insert(tk.END,result_chi2+f"Chi2:"+str(chi2)+" "+f"p-value:"+str(p)+"\n")
        text_area.insert(tk.END, "\n")
        text_area.config(state="disabled")

        chi2, p, dof, expected = chi2_contingency(cross_tab8)
        result_chi2 = "Chi-Square Test (Maint x Persons):\n"
        text_area.config(state="normal")
        text_area.insert(tk.END,result_chi2+f"Chi2:"+str(chi2)+" "+f"p-value:"+str(p)+"\n")
        text_area.insert(tk.END, "\n")
        text_area.config(state="disabled")

        chi2, p, dof, expected = chi2_contingency(cross_tab9)
        result_chi2 = "Chi-Square Test (Maint x Lug_Boot):\n"
        text_area.config(state="normal")
        text_area.insert(tk.END,result_chi2+f"Chi2:"+str(chi2)+" "+f"p-value:"+str(p)+"\n")
        text_area.insert(tk.END, "\n")
        text_area.config(state="disabled")

        chi2, p, dof, expected = chi2_contingency(cross_tab10)
        result_chi2 = "Chi-Square Test (Maint x Safety):\n"
        text_area.config(state="normal")
        text_area.insert(tk.END,result_chi2+f"Chi2:"+str(chi2)+" "+f"p-value:"+str(p)+"\n")
        text_area.insert(tk.END, "\n")
        text_area.config(state="disabled")

        # CHI-SQUARE -> DOORS
        chi2, p, dof, expected = chi2_contingency(cross_tab11)
        result_chi2 = "Chi-Square Test (Doors x Class):\n"
        text_area.config(state="normal")
        text_area.insert(tk.END,result_chi2+f"Chi2:"+str(chi2)+" "+f"p-value:"+str(p)+"\n")
        text_area.insert(tk.END, "\n")
        text_area.config(state="disabled")

        chi2, p, dof, expected = chi2_contingency(cross_tab12)
        result_chi2 = "Chi-Square Test (Doors x Persons):\n"
        text_area.config(state="normal")
        text_area.insert(tk.END,result_chi2+f"Chi2:"+str(chi2)+" "+f"p-value:"+str(p)+"\n")
        text_area.insert(tk.END, "\n")
        text_area.config(state="disabled")

        chi2, p, dof, expected = chi2_contingency(cross_tab13)
        result_chi2 = "Chi-Square Test (Doors x Lug_Boot):\n"
        text_area.config(state="normal")
        text_area.insert(tk.END,result_chi2+f"Chi2:"+str(chi2)+" "+f"p-value:"+str(p)+"\n")
        text_area.insert(tk.END, "\n")
        text_area.config(state="disabled")

        chi2, p, dof, expected = chi2_contingency(cross_tab14)
        result_chi2 = "Chi-Square Test (Doors x Safety):\n"
        text_area.config(state="normal")
        text_area.insert(tk.END,result_chi2+f"Chi2:"+str(chi2)+" "+f"p-value:"+str(p)+"\n")
        text_area.insert(tk.END, "\n")
        text_area.config(state="disabled")

        # CHI-SQUARE -> PERSONS
        chi2, p, dof, expected = chi2_contingency(cross_tab15)
        result_chi2 = "Chi-Square Test (Persons x Class):\n"
        text_area.config(state="normal")
        text_area.insert(tk.END,result_chi2+f"Chi2:"+str(chi2)+" "+f"p-value:"+str(p)+"\n")
        text_area.insert(tk.END, "\n")
        text_area.config(state="disabled")

        chi2, p, dof, expected = chi2_contingency(cross_tab16)
        result_chi2 = "Chi-Square Test (Persons x Lug_Boot):\n"
        text_area.config(state="normal")
        text_area.insert(tk.END,result_chi2+f"Chi2:"+str(chi2)+" "+f"p-value:"+str(p)+"\n")
        text_area.insert(tk.END, "\n")
        text_area.config(state="disabled")

        chi2, p, dof, expected = chi2_contingency(cross_tab17)
        result_chi2 = "Chi-Square Test (Persons x Safety):\n"
        text_area.config(state="normal")
        text_area.insert(tk.END,result_chi2+f"Chi2:"+str(chi2)+" "+f"p-value:"+str(p)+"\n")
        text_area.insert(tk.END, "\n")
        text_area.config(state="disabled")

        # CHI-SQUARE -> LUG_BOOT
        chi2, p, dof, expected = chi2_contingency(cross_tab18)
        result_chi2 = "Chi-Square Test (Lug_Boot x Class):\n"
        text_area.config(state="normal")
        text_area.insert(tk.END,result_chi2+f"Chi2:"+str(chi2)+" "+f"p-value:"+str(p)+"\n")
        text_area.insert(tk.END, "\n")
        text_area.config(state="disabled")

        chi2, p, dof, expected = chi2_contingency(cross_tab19)
        result_chi2 = "Chi-Square Test (Lug_Boot x Safety):\n"
        text_area.config(state="normal")
        text_area.insert(tk.END,result_chi2+f"Chi2:"+str(chi2)+" "+f"p-value:"+str(p)+"\n")
        text_area.insert(tk.END, "\n")
        text_area.config(state="disabled")

        # CHI-SQUARE -> SAFETY
        chi2, p, dof, expected = chi2_contingency(cross_tab20)
        result_chi2 = "Chi-Square Test (Safety x Class):\n"
        text_area.config(state="normal")
        text_area.insert(tk.END,result_chi2+f"Chi2:"+str(chi2)+" "+f"p-value:"+str(p)+"\n")
        text_area.insert(tk.END, "\n")
        text_area.config(state="disabled")

        ### 9. Predictive Modelling ###

        print("\nGenerating Models Prediction...\n")

        models_folder = "models"

        os.makedirs(models_folder, exist_ok=True)

        X = df.drop('class', axis=1)
        y = df['class']

        # Encode categorical variables
        label_encoder = LabelEncoder()
        X_encoded = X.apply(label_encoder.fit_transform)

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

        ### RANDOM FOREST ###

        print("\nRandom Forest Prediction...\n")

        # Initialize and train a Random Forest classifier
        rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_classifier.fit(X_train, y_train)

        # Make predictions on the test set
        y_pred_rf = rf_classifier.predict(X_test)

        # Evaluate the model
        accuracy_rf = round(accuracy_score(y_test, y_pred_rf), 3)
        accuracy_percentage_rf = "{:.3%}".format(accuracy_rf)
        text_area.config(state="normal")
        text_area.insert(tk.END, f'Accuracy: {accuracy_rf}' + ' = ' + f'{accuracy_percentage_rf}' +"\n")
        text_area.insert(tk.END, "\n")
        text_area.config(state="disabled")

        # Random Forest model evaluation
        classification_report_str = classification_report(y_test, y_pred_rf)
        result_cr = "Classification Report (Random Forest):\n"
        text_area.config(state="normal")
        text_area.insert(tk.END,result_cr+str(classification_report_str)+"\n")
        text_area.insert(tk.END, "\n")
        text_area.config(state="disabled")
        

        # AUC-ROC for a multi-class classification problem
        y_prob = rf_classifier.predict_proba(X_test)  # Class probability
        auc_roc_rf = roc_auc_score(pd.get_dummies(y_test), y_prob, multi_class='ovr').round(3)
        text_area.config(state="normal")
        text_area.insert(tk.END,f'AUC-ROC:' + str(auc_roc_rf) +"\n")
        text_area.insert(tk.END, "\n")
        text_area.config(state="disabled")

        classification_report_df = pd.read_fwf(StringIO(classification_report_str), index_col=0)
        classification_report_df.insert(0, 'Header_Column', ['acc', 'good', 'unacc', 'vgood', 'accuracy', 'macro avg', 'weighted avg'])
        classification_report_df.columns = ['' if col == 'Header_Column' else col for col in classification_report_df.columns]
        classification_report_df = classification_report_df.fillna('')

        # Create an HTML Table for a classification report
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.axis('off')
        ax.table(cellText=classification_report_df.values, colLabels=classification_report_df.columns, cellLoc='center', loc='center')
        plt.savefig(os.path.join(models_folder, 'tabela_classification_report_random_forest.png'))
        plt.close()

        ### SVM - SUPPORT VECTOR MACHINE ###

        print("\nSVM Prediction...\n")

        # Initialize and train a Support Vector Machine classifier
        svm_classifier = SVC(random_state=42)
        svm_classifier.fit(X_train, y_train)

        # Make predictions on the test set
        y_pred_svm = svm_classifier.predict(X_test)

        # Evaluate the model
        accuracy_svm = round(accuracy_score(y_test, y_pred_svm), 3)
        accuracy_percentage_svm = "{:.3%}".format(accuracy_svm)
        text_area.config(state="normal")
        text_area.insert(tk.END, f'Accuracy: {accuracy_svm}' + ' = ' + f'{accuracy_percentage_svm}' +"\n")
        text_area.insert(tk.END, "\n")
        text_area.config(state="disabled")

        # Support Vector Machine model evaluation
        classification_report_str1 = classification_report(y_test, y_pred_svm)
        result_cr1 = "Classification Report (SVM):\n"
        text_area.config(state="normal")
        text_area.insert(tk.END,result_cr1+str(classification_report_str1)+"\n")
        text_area.insert(tk.END, "\n")
        text_area.config(state="disabled")

        # AUC-ROC for a multi-class classification problem
        y_prob = svm_classifier.decision_function(X_test)  # Função de decisão para probabilidades
        auc_roc_svm = roc_auc_score(pd.get_dummies(y_test), y_prob, multi_class='ovr').round(3)
        text_area.config(state="normal")
        text_area.insert(tk.END,f'AUC-ROC:' + str(auc_roc_svm) +"\n")
        text_area.insert(tk.END, "\n")
        text_area.config(state="disabled")

        classification_report_df1 = pd.read_fwf(StringIO(classification_report_str1), index_col=0)
        classification_report_df1.insert(0, 'Header_Column', ['acc', 'good', 'unacc', 'vgood', 'accuracy', 'macro avg', 'weighted avg'])
        classification_report_df1.columns = ['' if col == 'Header_Column' else col for col in classification_report_df1.columns]
        classification_report_df1 = classification_report_df1.fillna('')

        # Create an HTML Table for a classification report
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.axis('off')
        ax.table(cellText=classification_report_df1.values, colLabels=classification_report_df1.columns, cellLoc='center', loc='center')
        plt.savefig(os.path.join(models_folder, 'tabela_classification_report_svm.png'))
        plt.close()

        ### NEURAL NETWORKS ###

        print("\nNeural Networks Prediction...\n")

        # MLP model initialization
        mlp_model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, random_state=42)

        # Model training
        mlp_model.fit(X_train, y_train)

        # Predicting test dataset
        y_pred_mlp = mlp_model.predict(X_test)

        # Model Evaluation
        accuracy_mlp = round(accuracy_score(y_test, y_pred_mlp), 3)
        accuracy_percentage_mlp = "{:.3%}".format(accuracy_mlp)
        text_area.config(state="normal")
        text_area.insert(tk.END, f'Accuracy: {accuracy_mlp}' + ' = ' + f'{accuracy_percentage_mlp}' +"\n")
        text_area.insert(tk.END, "\n")
        text_area.config(state="disabled")

        # Classification Report
        classification_report_str2 = classification_report(y_test, y_pred_mlp)
        result_cr1 = "Classification Report (Neural Networks):\n"
        text_area.config(state="normal")
        text_area.insert(tk.END,result_cr1+str(classification_report_str2)+"\n")
        text_area.insert(tk.END, "\n")
        text_area.config(state="disabled")

        # AUC-ROC
        y_prob_mlp = mlp_model.predict_proba(X_test)  # Probabilidades de classe
        auc_roc_mlp = roc_auc_score(pd.get_dummies(y_test), y_prob_mlp, multi_class='ovr').round(3)
        text_area.config(state="normal")
        text_area.insert(tk.END,f'AUC-ROC:' + str(auc_roc_mlp) +"\n")
        text_area.insert(tk.END, "\n")
        text_area.config(state="disabled")

        classification_report_df2 = pd.read_fwf(StringIO(classification_report_str2), index_col=0)
        classification_report_df2.insert(0, 'Header_Column', ['acc', 'good', 'unacc', 'vgood', 'accuracy', 'macro avg', 'weighted avg'])
        classification_report_df2.columns = ['' if col == 'Header_Column' else col for col in classification_report_df2.columns]
        classification_report_df2 = classification_report_df2.fillna('')

        # Create an HTML Table for a classification report
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.axis('off')
        ax.table(cellText=classification_report_df2.values, colLabels=classification_report_df2.columns, cellLoc='center', loc='center')
        plt.savefig(os.path.join(models_folder, 'tabela_classification_report_neural_networks.png'))
        plt.close()

        ### DECISION TREES ### 

        print("\nDecision Trees Prediction...\n")

        # Decision Tree model initialization
        dt_model = DecisionTreeClassifier(random_state=42)

        # Model training
        dt_model.fit(X_train, y_train)

        # Predicting test dataset
        y_pred_dt = dt_model.predict(X_test)

        # Model evaluation
        accuracy_dt = round(accuracy_score(y_test, y_pred_dt), 3)
        accuracy_percentage_dt = "{:.3%}".format(accuracy_dt)
        text_area.config(state="normal")
        text_area.insert(tk.END, f'Accuracy: {accuracy_dt}' + ' = ' + f'{accuracy_percentage_dt}' +"\n")
        text_area.insert(tk.END, "\n")
        text_area.config(state="disabled")

        # Classification Report
        classification_report_str3 = classification_report(y_test, y_pred_dt)
        result_cr1 = "Classification Report (Decision Trees):\n"
        text_area.config(state="normal")
        text_area.insert(tk.END,result_cr1+str(classification_report_str3)+"\n")
        text_area.insert(tk.END, "\n")
        text_area.config(state="disabled")

        classification_report_df3 = pd.read_fwf(StringIO(classification_report_str3), index_col=0)
        classification_report_df3.insert(0, 'Header_Column', ['acc', 'good', 'unacc', 'vgood', 'accuracy', 'macro avg', 'weighted avg'])
        classification_report_df3.columns = ['' if col == 'Header_Column' else col for col in classification_report_df3.columns]
        classification_report_df3 = classification_report_df3.fillna('')

        # Create an HTML Table for a classification report
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.axis('off')
        ax.table(cellText=classification_report_df3.values, colLabels=classification_report_df3.columns, cellLoc='center', loc='center')
        plt.savefig(os.path.join(models_folder, 'tabela_classification_report_decision_trees.png'))
        plt.close()

        # table of models with precisions

        model_data = [
            ("RANDOM FOREST", accuracy_rf, accuracy_percentage_rf, auc_roc_rf),
            ("SVM", accuracy_svm, accuracy_percentage_svm, auc_roc_svm),
            ("NEURAL NETWORKS", accuracy_mlp, accuracy_percentage_mlp, auc_roc_mlp),
            ("DECISION TREE", accuracy_dt, accuracy_percentage_dt),
        ]

        # Define headers for the table
        headers = ["Model", "Precision Score", "Percentage", "AUC-ROC"]

        # Create the table using tabulate
        table = tabulate(model_data, headers=headers, tablefmt="pipe")

        print("Making Models Comparision...")

        # Print the table
        text_area.config(state="normal")
        text_area.insert(tk.END,str(table)+"\n")
        text_area.insert(tk.END, "\n")
        text_area.config(state="disabled")

        # Compare precision scores
        best_model_precision = max(model_data, key=lambda x: x[1])
        text_area.config(state="normal")
        text_area.insert(tk.END,str(f"\nThe best model based on precision score is {best_model_precision[0]} with a precision score of {best_model_precision[1]:.2f} corresponding to a percentage of {best_model_precision[2]}%.")+"\n")
        text_area.insert(tk.END, "\n")
        text_area.config(state="disabled")

        # Compare AUC-ROC scores
        try:
            best_model_auc_roc = max((model for model in model_data if len(model) == len(headers)), key=lambda x: x[3])
            text_area.config(state="normal")
            text_area.insert(tk.END,str(f"\nThe best model based on AUC-ROC score is {best_model_auc_roc[0]} with an AUC-ROC score of {best_model_auc_roc[3]:.2f}.\n")+"\n")
            text_area.insert(tk.END, "\n")
            text_area.config(state="disabled")
        except ValueError:
            text_area.config(state="normal")
            text_area.insert(tk.END,"Error: AUC-ROC values are not available for any model."+"\n")
            text_area.insert(tk.END, "\n")
            text_area.config(state="disabled")

        # Confusion Matrix
            
        print("\nGenerating Confusion Matrixes...\n")
            
        # Create a directory to save confusion matrixes if it doesn't exist
        output_directory = 'confusion_matrixes'
        os.makedirs(output_directory, exist_ok=True)

        models = {
            'Random Forest': rf_classifier,
            'SVM': svm_classifier,
            'Decision Tree': dt_model,
            'Neural Network': mlp_model
        }

        for model_name, model in models.items():
            # train the model
            model.fit(X_train, y_train)

            # make predictions
            y_pred = model.predict(X_test)

            #calculate the confusion matrix
            cm = confusion_matrix(y_test, y_pred)

            # Print the confusion matrix
            result = f'### CONFUSION MATRIX - {model_name} ###\n'
            text_area.config(state="normal")
            text_area.insert(tk.END,result+"\n")
            text_area.insert(tk.END, "\n")
            text_area.insert(tk.END,str(cm)+"\n")
            text_area.insert(tk.END, "\n")
            text_area.config(state="disabled")

            #Visualize the confusion matrix
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(y))
            disp.plot(cmap='Blues', values_format='d')
            plt.title(f'Matriz de Confusão - {model_name}')
            plt.savefig(os.path.join(output_directory, f'{model_name}_confusion_matrix.png'))
            plt.close()  # Close the plot to avoid displaying it

    def browse_files():
        file_path = filedialog.askopenfilename(filetypes=[("Data Files", "*.data")])
        if file_path:
            data_file = file_path
            analyse_data()

    # Create the main window
    root = tk.Tk()
    root.title("Data Analysis Tool")

    # Create a button to trigger the file browsing operation
    browse_button = tk.Button(root, text="Browse Data File", command=browse_files)
    browse_button.pack(pady=10)

    # Create a Text widget to display the results
    text_area = tk.Text(root, wrap=tk.WORD)
    text_area.pack(expand=True, fill=tk.BOTH)

    # Start the main event loop
    root.mainloop()
    
dataset_handling()

