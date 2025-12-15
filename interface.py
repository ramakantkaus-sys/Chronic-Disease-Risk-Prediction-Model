import tkinter as tk
from tkinter import messagebox, ttk
import numpy as np
import pickle

# Load trained Decision Tree model
# Using the correct model file as verified
model = pickle.load(open("decision_treedepression.pkl", "rb"))

# Mappings generated from LabelEncoder on the CSV data
MAPPINGS = {
  "Marital Status": {
    "Divorced": 0, "Married": 1, "Single": 2, "Widowed": 3
  },
  "Education Level": {
    "Associate Degree": 0, "Bachelor's Degree": 1, "High School": 2, "Master's Degree": 3, "PhD": 4
  },
  "Smoking Status": {
    "Current": 0, "Former": 1, "Non-smoker": 2
  },
  "Physical Activity Level": {
    "Active": 0, "Moderate": 1, "Sedentary": 2
  },
  "Employment Status": {
    "Employed": 0, "Unemployed": 1
  },
  "Alcohol Consumption": {
    "High": 0, "Low": 1, "Moderate": 2
  },
  "Dietary Habits": {
    "Healthy": 0, "Moderate": 1, "Unhealthy": 2
  },
  "Sleep Patterns": {
    "Fair": 0, "Good": 1, "Poor": 2
  },
  "History of Mental Illness": {
    "No": 0, "Yes": 1
  },
  "History of Substance Abuse": {
    "No": 0, "Yes": 1
  },
  "Family History of Depression": {
    "No": 0, "Yes": 1
  }
}

def preprocess_input(data):
    """Convert inputs into numerical values using the mappings."""
    # Ensure correct order of feature vector
    # Order: Age, Marital Status, Education Level, Number of Children, Smoking Status, 
    # Physical Activity Level, Employment Status, Income, Alcohol Consumption, Dietary Habits, 
    # Sleep Patterns, History of Mental Illness, History of Substance Abuse, Family History of Depression
    
    vector = []
    
    # Numerical Fields directly
    vector.append(data["Age"])
    
    # Categorical Fields with Mapping
    vector.append(MAPPINGS["Marital Status"][data["Marital Status"]])
    vector.append(MAPPINGS["Education Level"][data["Education Level"]])
    
    # Numerical
    vector.append(data["Number of Children"])
    
    # Categorical
    vector.append(MAPPINGS["Smoking Status"][data["Smoking Status"]])
    vector.append(MAPPINGS["Physical Activity Level"][data["Physical Activity Level"]])
    vector.append(MAPPINGS["Employment Status"][data["Employment Status"]])
    
    # Numerical
    vector.append(data["Income"])
    
    # Categorical
    vector.append(MAPPINGS["Alcohol Consumption"][data["Alcohol Consumption"]])
    vector.append(MAPPINGS["Dietary Habits"][data["Dietary Habits"]])
    vector.append(MAPPINGS["Sleep Patterns"][data["Sleep Patterns"]])
    vector.append(MAPPINGS["History of Mental Illness"][data["History of Mental Illness"]])
    vector.append(MAPPINGS["History of Substance Abuse"][data["History of Substance Abuse"]])
    vector.append(MAPPINGS["Family History of Depression"][data["Family History of Depression"]])
    
    return np.array(vector).reshape(1, -1)

def predict_chronic_disease():
    """Fetch input values, preprocess, and make prediction."""
    try:
        user_data = {
            "Age": int(entries["Age"].get()),
            "Marital Status": entries["Marital Status"].get(),
            "Education Level": entries["Education Level"].get(),
            "Number of Children": int(entries["Number of Children"].get()),
            "Smoking Status": entries["Smoking Status"].get(),
            "Physical Activity Level": entries["Physical Activity Level"].get(),
            "Employment Status": entries["Employment Status"].get(),
            "Income": float(entries["Income"].get()),
            "Alcohol Consumption": entries["Alcohol Consumption"].get(),
            "Dietary Habits": entries["Dietary Habits"].get(),
            "Sleep Patterns": entries["Sleep Patterns"].get(),
            "History of Mental Illness": entries["History of Mental Illness"].get(),
            "History of Substance Abuse": entries["History of Substance Abuse"].get(),
            "Family History of Depression": entries["Family History of Depression"].get()
        }
        
        # Validation
        for key, value in user_data.items():
            if value == "" or value is None:
                raise ValueError(f"Missing value for {key}")
        
        processed_data = preprocess_input(user_data)
        prediction = model.predict(processed_data)[0]
        
        result = "Yes, this person has a serious chronic disease." if prediction == 1 else "No, this person is healthy."
        messagebox.showinfo("Prediction Result", result)
    except Exception as e:
        messagebox.showerror("Error", f"Invalid Input: {e}")

# Create Tkinter Window
root = tk.Tk()
root.title("Chronic Disease Prediction")
root.geometry("500x750")

# Scrollable canvas in case it's too tall
canvas = tk.Canvas(root)
scrollbar = ttk.Scrollbar(root, orient="vertical", command=canvas.yview)
scrollable_frame = ttk.Frame(canvas)

scrollable_frame.bind(
    "<Configure>",
    lambda e: canvas.configure(
        scrollregion=canvas.bbox("all")
    )
)

canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
canvas.configure(yscrollcommand=scrollbar.set)

canvas.pack(side="left", fill="both", expand=True)
scrollbar.pack(side="right", fill="y")

tk.Label(scrollable_frame, text="Enter Patient Details", font=("Arial", 16, "bold"), pady=10).pack()

fields = [
    ("Age", "number"),
    ("Marital Status", "categorical"),
    ("Education Level", "categorical"),
    ("Number of Children", "number"),
    ("Smoking Status", "categorical"),
    ("Physical Activity Level", "categorical"),
    ("Employment Status", "categorical"),
    ("Income", "number"),
    ("Alcohol Consumption", "categorical"),
    ("Dietary Habits", "categorical"),
    ("Sleep Patterns", "categorical"),
    ("History of Mental Illness", "categorical"),
    ("History of Substance Abuse", "categorical"),
    ("Family History of Depression", "categorical")
]

entries = {}

for label, dtype in fields:
    frame = tk.Frame(scrollable_frame, pady=5)
    frame.pack(fill="x", padx=20)
    
    tk.Label(frame, text=label, anchor="w", width=25).pack(side="left")
    
    if dtype == "categorical":
        # Get options from MAPPINGS keys
        options = list(MAPPINGS[label].keys())
        entry = ttk.Combobox(frame, values=options, state="readonly")
        entry.current(0) # Default select first
    else:
        entry = tk.Entry(frame)
    
    entry.pack(side="right", expand=True, fill="x")
    entries[label] = entry

tk.Button(scrollable_frame, text="Predict", command=predict_chronic_disease, bg="green", fg="white", font=("Arial", 12, "bold"), pady=10).pack(pady=20)

root.mainloop()