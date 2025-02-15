from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
import os
import kagglehub
import matplotlib.pyplot as plt
from django.conf import settings

# Use the 'Agg' backend for Matplotlib
import matplotlib
matplotlib.use('Agg')

BASE_DIR = settings.BASE_DIR

# Load the dataset for companies
path = kagglehub.dataset_download("siddheshshivdikar/college-placement")
file_path = os.path.join(path, "variation_3.csv")
data = pd.read_csv(file_path)

label_encoders = {}
for column in ['name of company', 'college name', 'region', 'salary']:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le

X_company = data.drop(columns=['college name', 'salary', 'region'])
y_college = data['college name']
y_region = data['region']
y_salary = data['salary']
years = data['year']

# Ensure that X_company, y_college, y_region, y_salary, and years have the same number of samples
assert len(X_company) == len(y_college) == len(y_region) == len(y_salary) == len(years), "Inconsistent number of samples"

X_train_company, X_test_company, y_college_train, y_college_test, y_region_train, y_region_test, y_salary_train, y_salary_test, years_train, years_test = train_test_split(
    X_company, y_college, y_region, y_salary, years, test_size=0.2, random_state=42
)

scaler_company = StandardScaler()
X_train_company = scaler_company.fit_transform(X_train_company)
X_test_company = scaler_company.transform(X_test_company)

college_model = RandomForestClassifier(random_state=42)
region_model = RandomForestClassifier(random_state=42)
salary_model = RandomForestClassifier(random_state=42)

college_model.fit(X_train_company, y_college_train)
region_model.fit(X_train_company, y_region_train)
salary_model.fit(X_train_company, y_salary_train)

# Load the dataset for colleges
file_path_college = r'C:\Users\haris\test\admission_predictor\main_app\DATA\kaggle_pivot_min_descending.csv'
df_college = pd.read_csv(file_path_college)

df_college = df_college[['stream', 'percentage', 'college_name']]
X_college = df_college[['stream', 'percentage']]
y_college_name = df_college['college_name']

preprocessor_college = ColumnTransformer(
    transformers=[('num', StandardScaler(), ['percentage']),
                  ('cat', OneHotEncoder(handle_unknown='ignore'), ['stream'])]
)

college_predict_model = Pipeline(steps=[('preprocessor', preprocessor_college),
                                        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))])

X_train_college, X_test_college, y_train_college, y_test_college = train_test_split(X_college, y_college_name, test_size=0.2, random_state=42)
college_predict_model.fit(X_train_college, y_train_college)

@csrf_exempt
def index(request):
    return render(request, 'index.html')

@csrf_exempt
def predict_company(request):
    if request.method == 'GET':
        # Render the predict_company.html template for GET requests
        return render(request, 'predict_company.html')
    
    if request.method == 'POST':
        company_name = request.POST.get('company_name')
        if company_name is None:
            return JsonResponse({'error': 'Company name is required'}, status=400)
        
        # Transform the company name using label encoder
        if company_name not in label_encoders['name of company'].classes_:
            return JsonResponse({'error': 'Invalid company name'}, status=400)
        
        company_name_encoded = label_encoders['name of company'].transform([company_name])
        
        # Filter the dataset based on encoded company name
        company_data = data[data['name of company'] == company_name_encoded[0]]
        
        if company_data.empty:
            return JsonResponse({'error': 'No data found for the specified company'}, status=400)
        
        X_company_input = company_data.drop(columns=['college name', 'salary', 'region'])
        X_company_input = scaler_company.transform(X_company_input)
        
        college_pred = college_model.predict(X_company_input)
        region_pred = region_model.predict(X_company_input)
        salary_pred = salary_model.predict(X_company_input)
        years_pred = company_data['year'].tolist()
        
        company_info = {
            'company_name': company_name,
            'college_name': label_encoders['college name'].inverse_transform(college_pred).tolist(),
            'region': label_encoders['region'].inverse_transform(region_pred).tolist(),
            'salary': label_encoders['salary'].inverse_transform(salary_pred).tolist(),
            'years': years_pred
        }
        
        # Ensure the static directory exists
        static_dir = os.path.join(BASE_DIR, 'static', 'generated_graphs')
        if not os.path.exists(static_dir):
            os.makedirs(static_dir)

        # Create a line plot for college name predictions
        plt.figure(figsize=(10, 6))
        plt.plot(company_info['college_name'], marker='o')
        plt.xlabel('Predictions')
        plt.ylabel('College Names')
        plt.title(f'College Name Predictions for {company_info["company_name"]}')
        college_name_plot_path = os.path.join(static_dir, f'{company_info["company_name"]}_college_name_plot.png')
        plt.savefig(college_name_plot_path)
        plt.close()

        # Create a line plot for region predictions
        plt.figure(figsize=(10, 6))
        plt.plot(company_info['region'], marker='o', color='green')
        plt.xlabel('Predictions')
        plt.ylabel('Regions')
        plt.title(f'Region Predictions for {company_info["company_name"]}')
        region_plot_path = os.path.join(static_dir, f'{company_info["company_name"]}_region_plot.png')
        plt.savefig(region_plot_path)
        plt.close()

        # Create a line plot for salary predictions
        plt.figure(figsize=(10, 6))
        plt.plot(company_info['salary'], marker='o', color='red')
        plt.xlabel('Predictions')
        plt.ylabel('Salaries')
        plt.title(f'Salary Predictions for {company_info["company_name"]}')
        salary_plot_path = os.path.join(static_dir, f'{company_info["company_name"]}_salary_plot.png')
        plt.savefig(salary_plot_path)
        plt.close()
        
        # Zip the college names and years to pass to the template
        zipped_colleges_years = zip(company_info['college_name'], company_info['years'])

        return render(request, 'predict_company.html', {
            'company_info': company_info,
            'college_name_plot_path': os.path.join('generated_graphs', f'{company_info["company_name"]}_college_name_plot.png'),
            'region_plot_path': os.path.join('generated_graphs', f'{company_info["company_name"]}_region_plot.png'),
            'salary_plot_path': os.path.join('generated_graphs', f'{company_info["company_name"]}_salary_plot.png'),
            'zipped_colleges_years': zipped_colleges_years
        })
    
    return JsonResponse({'error': 'Invalid request'}, status=400)

@csrf_exempt
def predict_colleges(request):
    if request.method == 'GET':
        # Render the predict_colleges.html template for GET requests
        return render(request, 'predict_colleges.html')
    
    if request.method == 'POST':
        stream = request.POST.get('stream')
        percentage = request.POST.get('percentage')
        if stream is None or percentage is None:
            return JsonResponse({'error': 'Stream and Percentage are required'}, status=400)
        
        input_data = pd.DataFrame({'stream': [stream], 'percentage': [percentage]})
        input_data_transformed = college_predict_model.named_steps['preprocessor'].transform(input_data)
        predictions_proba = college_predict_model.named_steps['classifier'].predict_proba(input_data_transformed)
        top_indices = predictions_proba.argsort()[0, -5:][::-1]
        top_colleges = college_predict_model.named_steps['classifier'].classes_[top_indices].tolist()
        
        # Ensure the static directory exists
        static_dir = os.path.join(BASE_DIR, 'static', 'generated_graphs')
        if not os.path.exists(static_dir):
            os.makedirs(static_dir)

        # Create a line plot of the predicted colleges
        plt.figure(figsize=(10, 6))
        plt.plot(top_colleges, marker='o', color='blue')
        plt.xlabel('Predictions')
        plt.ylabel('Colleges')
        plt.title(f'Top Predicted Colleges for Stream: {stream}, Percentage: {percentage}')
        top_colleges_plot_path = os.path.join(static_dir, 'top_colleges_plot.png')
        plt.savefig(top_colleges_plot_path)
        plt.close()
        
        return render(request, 'predict_colleges.html', {
            'top_colleges': top_colleges,
            'top_colleges_plot_path': os.path.join('generated_graphs', 'top_colleges_plot.png'),
            'stream': stream,
            'percentage': percentage
        })
    
    return JsonResponse({'error': 'Invalid request'}, status=400)