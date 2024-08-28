from faker import Faker
import random
from datetime import date
from dateutil.relativedelta import relativedelta
from Config import Config

fake = Faker()

def generate_customer_data():
    age = random.randint(20, 70)
    gender = random.choice(['Male', 'Female'])
    marital_status = random.choice(['Single', 'Married', 'Divorced', 'Widowed'])
    income_level = random.choice(['Low', 'Medium', 'High'])
    education = random.choice(['High School', 'College', 'University'])
    occupation = fake.job()
    residential_status = random.choice(['Owns house', 'Rents', 'Living with parents'])
    dependents = random.randint(0, 5)
    debt_to_income = round(random.uniform(0.1, 0.5), 2)
    credit_bureau = random.randint(760, 850)

    return {
        'Age': age,
        'Gender': gender,
        'Marital Status': marital_status,
        'Income Level': income_level,
        'Education': education,
        'Occupation': occupation,
        'Residential Status': residential_status,
        'Dependents': dependents,
        'Debt-to-Income': debt_to_income,
        'Credit_Bureau': credit_bureau
    }

def generate_inquiries(last_months):
    inquiries = []
    today = fake.date_this_month()

    for _ in range(random.randint(1, 5)):
        inquiry_date = fake.date_between(start_date=last_months, end_date=today)
        product_type = random.choice(['Personal Loan', 'Credit Card', 'Mortgage'])
        inquiries.append({'product_name': product_type, 'date': inquiry_date})

    return inquiries

def generate_dataset(num_rows, months):
    try:
        data_rows = []

        for _ in range(num_rows):
            customer_data = generate_customer_data()
            last_3_months_inquiries = generate_inquiries(months[0])
            last_6_months_inquiries = generate_inquiries(months[1])

            customer_row = {
                'Customer ID': fake.uuid4(),
                'Age': customer_data['Age'],
                'Gender': customer_data['Gender'],
                'Marital Status': customer_data['Marital Status'],
                'Income Level': customer_data['Income Level'],
                'Education': customer_data['Education'],
                'Occupation': customer_data['Occupation'],
                'Residential Status': customer_data['Residential Status'],
                'Dependents': customer_data['Dependents'],
                'Debt-to-Income': customer_data['Debt-to-Income'],
                'Credit_Bureau': customer_data['Credit_Bureau'],
                'Fixed_Deposits': random.choice(list(Config.FD_RATES.keys())),
                'Fixed_Deposits_Rate': Config.FD_RATES[random.choice(list(Config.FD_RATES.keys()))],
                'Fixed_Deposits_Senior_Rate': Config.FD_RATES[random.choice(list(Config.FD_RATES.keys()))] + Config.FD_SENIOR_CITIZEN_RATE_PREMIUM if customer_data['Age'] >= 60 else Config.FD_RATES[random.choice(list(Config.FD_RATES.keys()))]
            }

            for product_type in ['Personal Loan', 'Credit Card', 'Mortgage']:
                inq_in_last_3_months = any(inq['product_name'] == product_type for inq in last_3_months_inquiries)
                customer_row[f'last_3months_{product_type.replace(" ", "_").lower()}_inq'] = inq_in_last_3_months

            for product_type in ['Personal Loan', 'Credit Card', 'Mortgage']:
                inq_in_last_6_months = any(inq['product_name'] == product_type for inq in last_6_months_inquiries)
                customer_row[f'last_6months_{product_type.replace(" ", "_").lower()}_inq'] = inq_in_last_6_months

            data_rows.append(customer_row)
        return data_rows
    except Exception as e:
        raise Exception(f"Error generating dataset: {str(e)}")