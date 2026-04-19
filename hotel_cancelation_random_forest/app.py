from flask import Flask, request, render_template
import pandas as pd
import joblib

app = Flask(__name__)

# 1. Load the model and the CORRECT expected columns when the app starts
# (Ensure 'random_forest_model.joblib' and 'model_columns (1).joblib' are in the same folder)
model = joblib.load('random_forest_model.joblib')
model_columns = joblib.load('model_columns (1).joblib') # <--- UPDATED FILENAME

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    
    if request.method == 'POST':
        try:
            # 2. Grab ALL 30 features from the HTML form and set the correct data types
            # These are the base features BEFORE get_dummies is applied
            input_dict = {
                'hotel': request.form.get('hotel', ''),
                'lead_time': int(request.form.get('lead_time', 0)),
                'arrival_date_year': int(request.form.get('arrival_date_year', 0)),
                'arrival_date_month': request.form.get('arrival_date_month', ''),
                'arrival_date_week_number': int(request.form.get('arrival_date_week_number', 0)),
                'arrival_date_day_of_month': int(request.form.get('arrival_date_day_of_month', 0)),
                'stays_in_weekend_nights': int(request.form.get('stays_in_weekend_nights', 0)),
                'stays_in_week_nights': int(request.form.get('stays_in_week_nights', 0)),
                'adults': int(request.form.get('adults', 0)),
                'children': float(request.form.get('children', 0.0)),
                'babies': int(request.form.get('babies', 0)),
                'meal': request.form.get('meal', ''),
                'country': request.form.get('country', ''),
                'market_segment': request.form.get('market_segment', ''),
                'distribution_channel': request.form.get('distribution_channel', ''),
                'is_repeated_guest': int(request.form.get('is_repeated_guest', 0)),
                'previous_cancellations': int(request.form.get('previous_cancellations', 0)),
                'previous_bookings_not_canceled': int(request.form.get('previous_bookings_not_canceled', 0)),
                'reserved_room_type': request.form.get('reserved_room_type', ''),
                'assigned_room_type': request.form.get('assigned_room_type', ''),
                'booking_changes': int(request.form.get('booking_changes', 0)),
                'deposit_type': request.form.get('deposit_type', ''),
                'agent': float(request.form.get('agent', 0.0)),       
                'company': float(request.form.get('company', 0.0)),   
                'days_in_waiting_list': int(request.form.get('days_in_waiting_list', 0)),
                'customer_type': request.form.get('customer_type', ''),
                'adr': float(request.form.get('adr', 0.0)),           
                'required_car_parking_spaces': int(request.form.get('required_car_parking_spaces', 0)),
                'total_of_special_requests': int(request.form.get('total_of_special_requests', 0)),
                'reservation_status_date': request.form.get('reservation_status_date', '')
            }
            
            # 3. Convert user input into a Pandas DataFrame
            input_df = pd.DataFrame([input_dict])
            
            # 4. Apply get_dummies to the user input
            # This creates columns like 'hotel_Resort Hotel' or 'country_PRT' based ONLY on what the user typed
            input_dummies = pd.get_dummies(input_df)
            
            # 5. THE MAGIC FIX: Reindex to align exactly with your training data
            # This looks at 'model_columns (1).joblib', adds all the hundreds of missing columns 
            # (like 'country_USA', 'meal_BB', etc.), and sets them to 0 so the model doesn't crash.
            input_aligned = input_dummies.reindex(columns=model_columns, fill_value=0)
            
            # 6. Make the prediction
            pred_value = model.predict(input_aligned)[0]
            prediction = "Likely to Cancel 🚨" if pred_value == 1 else "Likely to Keep Booking ✅"
            
        except Exception as e:
            prediction = f"Error processing input: {e}"

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)