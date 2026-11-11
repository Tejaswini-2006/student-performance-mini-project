# 🎓 Student Performance Prediction Dashboard

A simple and attractive Machine Learning web application that predicts a student's final mathematics grade (**G3**, on a scale of 0 to 20) based on their study habits, attendance, and demographic background.

Built with **Python**, **Scikit-learn**, and **Streamlit**.

---

## 💡 What This Project Does

- **Predicts Final Grades**: Takes 30 student inputs (like study hours, past failures, and absences) and predicts their final mathematics exam score out of 20.
- **Trained Model**: Uses a **Random Forest Regressor** pipeline with automatic categorical encoding (`OneHotEncoder`).
- **Interactive Dashboard**: A clean single-page dashboard with 1-click preset student profiles and interactive charts.
- **Early Intervention**: Intentionally does not use midterm exam scores ($G1$ and $G2$) so teachers can identify at-risk students right at the start of the term.

---

## 🚀 How to Run Locally

### 1. Clone the repository
```bash
git clone https://github.com/Tejaswini-2006/student-performance-mini-project.git
cd student-performance-mini-project
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit app
```bash
streamlit run app.py
```
Open your browser at **http://localhost:8501**.

---

## 📊 Model Performance

Evaluated on 79 unseen test students from the UCI Student Performance dataset:

| Model | MAE (Lower is better) | RMSE (Lower is better) | R² Score (Higher is better) |
| :--- | :---: | :---: | :---: |
| **Random Forest (Our Model)** | **2.998** | **3.795** | **0.298** |
| Linear Regression | 3.395 | 4.196 | 0.142 |

> **Key Takeaway**: Random Forest outperformed Linear Regression because student grades are influenced non-linearly by absences, past failures, and study time.

---

## 📁 Project Files

```
student-performance-mini-project/
├── app.py                             # Main Streamlit web application
├── student_performance_random_forest.pkl # Saved Random Forest model pipeline
├── student_performance_model.pkl      # Saved Linear Regression model
├── student-mat.csv                    # Mathematics dataset (UCI Repository)
├── requirements.txt                   # Required Python libraries
├── test_app.py                        # Simple automated tests for the model
├── ALML.pdf                           # Lab syllabus & project reference
└── README.md                          # Project documentation
```

---

## 🛠️ Built With

- **Python 3**
- **Streamlit** - Clean web UI
- **Scikit-learn** - Machine learning pipeline
- **Pandas & NumPy** - Data handling
- **Plotly** - Interactive charts

---

## 👩‍💻 Author

**Tejaswini Rakhunde**  
B.Tech / B.E. Information Technology (Third Year)  
GitHub: [@Tejaswini-2006](https://github.com/Tejaswini-2006)
