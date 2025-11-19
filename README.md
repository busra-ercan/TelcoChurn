#<img width="1536" height="1024" alt="ChatGPT Image Nov 19, 2025 at 01_54_50 PM" src="https://github.com/user-attachments/assets/e72fd5f6-09a3-409c-af95-338389627753" />
 Telco Churn Analysis - Machine Learning

This project performs an end-to-end customer churn analysis using the Telco Customer Churn dataset.  
The workflow includes exploratory data analysis (EDA), data preprocessing, feature engineering, model building, and model comparison based on performance metrics.

---

## 📌 Project Workflow

### 🔍 1. Exploratory Data Analysis (EDA)
- Overview of numerical and categorical variables  
- Churn distribution analysis  
- Detection of correlations and important patterns  
- Visualization of key churn drivers  

### 🧹 2. Data Preprocessing
- Identification and handling of missing values  
- Outlier detection and treatment (IQR-based methods)  
- Variable transformations (log, binning, etc.)  
- Encoding of categorical variables  
- Standardization and scaling for model readiness  

### 🏗 3. Feature Engineering
- Creating new features (e.g., tenure groups, total charges ratios, service counts)  
- Enhancing predictive power by combining related variables  
- Handling rare categories  
- Feature selection for better model performance  

### 🤖 4. Machine Learning Models
The following models were trained and compared:

- Logistic Regression  
- Random Forest Classifier  
- Gradient Boosting  
- XGBoost / LightGBM (if applicable)  
- KNN  
- Decision Tree  
- Support Vector Machine  

### 📊 5. Model Evaluation
Models were evaluated using:

- Accuracy  
- Recall  
- Precision  
- F1 Score  
- ROC–AUC  
- Confusion Matrix  

📌 **The best model was selected based on accuracy (and additional metrics where needed).**

### ⭐ Key Results  
- Feature engineering significantly improved model accuracy  
- Contract type, monthly charges, and tenure were among the most important predictors  
- Correctly handling outliers and missing values increased model stability  
- Tree-based models generally performed better than linear models  

## 📊 Model Performance Comparison

| Model           | Accuracy | AUC    | Recall | Precision | F1 Score |
|-----------------|----------|--------|--------|-----------|----------|
| Logistic Regression | 0.8059 | 0.8484 | 0.5388 | 0.6661 | 0.5956 |
| KNN             | 0.7686 | 0.7804 | 0.5265 | 0.5690 | 0.5469 |
| CART (Decision Tree) | 0.7291 | 0.6573 | 0.5029 | 0.4901 | 0.4961 |
| Random Forest   | 0.7890 | 0.8251 | 0.4853 | 0.6340 | 0.5497 |
| XGBoost         | 0.7822 | 0.8237 | 0.5046 | 0.6082 | 0.5514 |
| LightGBM        | 0.7961 | 0.8353 | 0.5211 | 0.6432 | 0.5757 |
| CatBoost        | 0.7975 | 0.8405 | 0.5131 | 0.6503 | 0.5736 |

➡️ **Best overall (balanced performance): Logistic Regression (Accuracy), LightGBM & CatBoost (AUC & Precision).**
## 🔧 Hyperparameter Optimization Results

| Model | Before Accuracy | After Accuracy | Best Params |
|-------|-----------------|----------------|--------------|
| Logistic Regression | 0.8059 | 0.8075 | {'C': 0.1} |
| KNN | 0.7686 | 0.7774 | {'n_neighbors': 7} |
| CART | 0.7264 | 0.7815 | {'max_depth': 7} |
| Random Forest | 0.7872 | 0.7906 | {'n_estimators': 500} |
| XGBoost | 0.7822 | 0.8011 | {'learning_rate': 0.1, 'max_depth': 5} |


--------------------------------------------------------------------------------------------------------------------------------------------

# 🇹🇷 **Telco Müşteri Terk Analizi - Makine Öğrenmesi**

Bu projede Telco Customer Churn veri seti kullanılarak uçtan uca bir müşteri terk analizi yapılmıştır.  
Analiz süreci; keşifsel veri analizi, veri ön işleme, feature engineering, makine öğrenmesi modelleme ve model karşılaştırmalarını içermektedir.

---

## 📌 Proje İş Akışı

### 🔍 1. Keşifsel Veri Analizi (EDA)
- Sayısal ve kategorik değişkenlere genel bakış  
- Churn oranının incelenmesi  
- Korelasyonların analiz edilmesi  
- Önemli churn etkileyicilerinin görselleştirilmesi  

### 🧹 2. Veri Ön İşleme
- Eksik değerlerin tespiti ve doldurulması  
- Aykırı değerlerin belirlenmesi ve müdahalesi (IQR yöntemleri)  
- Değişken dönüşümleri (logaritmik, gruplama vb.)  
- Kategorik değişkenlerin encode edilmesi  
- Standartlaştırma ve ölçeklendirme  

### 🏗 3. Feature Engineering
- Yeni değişkenlerin oluşturulması (ör. tenure grupları, servis sayıları)  
- Değişken birleştirme / ayrıştırma  
- Nadir kategorilerin düzenlenmesi  
- Performansı artıran değişken seçimi  

### 🤖 4. Makine Öğrenmesi Modelleri
Aşağıdaki modeller eğitilip karşılaştırıldı:

- Lojistik Regresyon  
- Random Forest  
- Gradient Boosting  
- XGBoost / LightGBM  
- KNN  
- Decision Tree  
- SVM  

### 📊 5. Model Değerlendirme
Modeller aşağıdaki metriklere göre değerlendirildi:

- Accuracy  
- Recall  
- Precision  
- F1 Skoru  
- ROC–AUC  
- Confusion Matrix  

📌 **En iyi model accuracy ve diğer metriklere göre seçildi.**

---

## ⭐ Önemli Bulgular
- Feature engineering model performansını belirgin şekilde artırdı  
- Sözleşme türü, aylık ücret ve kullanım süresi churn üzerinde kritik öneme sahiptir  
- Outlier ve eksik değer işlemleri model kararlılığını güçlendirdi  
- Ağaç tabanlı modeller lineer modellere göre daha iyi sonuç verdi  

---

## 📂 Repository Content
- `notebooks/` → EDA ve modelleme adımlarının notebook dosyaları  
- `data/` → Dataset (paylaşım koşullarına uygun şekilde)  
- `src/` → Kod dosyaları  
- `README.md` → Proje açıklaması  

---

## ⭐ If you found this project helpful, please consider giving it a star!
