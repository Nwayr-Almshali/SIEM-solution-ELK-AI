#SIEM-solution-ELK-AI
#SaaS 
# نظام كشف الشذوذ في المعاملات المصرفية
# Bank Transaction Anomaly Detection System

### 📋 نظرة عامة على المشروع | Project Overview

**العربية:**
هذا المشروع عبارة عن تطبيق ويب متطور لكشف الشذوذ في المعاملات المصرفية باستخدام تقنيات الذكاء الاصطناعي وتعلم الآلة. يهدف النظام إلى تحديد المعاملات المشبوهة أو الاحتيالية من خلال تحليل أنماط البيانات باستخدام نموذجين مختلفين من نماذج التعلم الآلي.

**English:**
This project is an advanced web application for detecting anomalies in bank transactions using artificial intelligence and machine learning techniques. The system aims to identify suspicious or fraudulent transactions by analyzing data patterns using two different machine learning models.

---

## 🏗️ البنية التقنية للمشروع | Technical Architecture

### التقنيات المستخدمة | Technologies Used:
**العربية:**
- **الخلفية (Backend)**: Flask (Python)
- **الواجهة الأمامية (Frontend)**: HTML5, CSS3, JavaScript
- **تعلم الآلة**: TensorFlow, Scikit-learn
- **معالجة البيانات**: NumPy, Pandas
- **التخزين**: Pickle, Joblib

**English:**
- **Backend**: Flask (Python)
- **Frontend**: HTML5, CSS3, JavaScript
- **Machine Learning**: TensorFlow, Scikit-learn
- **Data Processing**: NumPy, Pandas
- **Storage**: Pickle, Joblib

### هيكل الملفات | File Structure:
```
bank/
├── app.py                          # الملف الرئيسي للخادم
├── requirements.txt                # متطلبات المشروع
├── README.md                      # وثائق المشروع الأساسية
├── templates/
│   └── index.html                 # الواجهة الرئيسية
├── static/
│   ├── css/
│   │   └── style.css             # أنماط التصميم
│   └── js/
│       └── script.js             # منطق الواجهة الأمامية
└── النماذج المدربة:
    ├── preprocessor_bs.pkl        # معالج البيانات
    ├── isolation_forest_model_bs.pkl  # نموذج Isolation Forest
    ├── autoencoder_model_bs.h5    # نموذج Autoencoder
    └── ae_threshold.pkl           # عتبة كشف الشذوذ
```

---

## 🤖 النماذج المستخدمة | Models Used

### 1. نموذج Isolation Forest | Isolation Forest Model
**العربية:**
- **النوع**: خوارزمية تعلم غير مراقب (Unsupervised Learning)
- **المبدأ**: يعمل على عزل النقاط الشاذة من خلال بناء أشجار عشوائية
- **الاستخدام**: كشف المعاملات غير الطبيعية بناءً على الأنماط العامة
- **المخرجات**: Normal أو Fraud

**English:**
- **Type**: Unsupervised Learning Algorithm
- **Principle**: Isolates anomalous points by building random trees
- **Usage**: Detects abnormal transactions based on general patterns
- **Output**: Normal or Fraud

### 2. نموذج Autoencoder | Autoencoder Model
**العربية:**
- **النوع**: شبكة عصبية عميقة (Deep Neural Network)
- **المبدأ**: يتعلم إعادة بناء البيانات الطبيعية ويكشف الشذوذ من خلال خطأ الإعادة البناء
- **المقياس**: Mean Squared Error (MSE)
- **العتبة**: يتم مقارنة MSE بعتبة محددة مسبقاً
- **المخرجات**: Normal أو Fraud + قيمة MSE

**English:**
- **Type**: Deep Neural Network
- **Principle**: Learns to reconstruct normal data and detects anomalies through reconstruction error
- **Metric**: Mean Squared Error (MSE)
- **Threshold**: MSE is compared with a predefined threshold
- **Output**: Normal or Fraud + MSE value

---

## 📊 خصائص البيانات المدخلة | Input Data Features

**العربية:**
النظام يحلل المعاملات بناءً على الخصائص التالية:

**English:**
The system analyzes transactions based on the following features:

### 1. المبلغ | Amount
**العربية:**
- **النوع**: رقم عشري
- **الوصف**: قيمة المعاملة المالية
- **التحقق**: مطلوب، يجب أن يكون رقم موجب

**English:**
- **Type**: Decimal number
- **Description**: Financial transaction value
- **Validation**: Required, must be a positive number

### 2. الفئة | Category
**العربية:**
- **النوع**: قائمة منسدلة
- **الخيارات المتاحة**:
  - `es_transportation` - النقل والمواصلات
  - `es_food` - الطعام والمشروبات
  - `es_health` - الصحة والعلاج
  - `es_leisure` - الترفيه والتسلية
  - `es_shopping` - التسوق
  - `es_home` - المنزل والأسرة
  - `es_other` - أخرى

**English:**
- **Type**: Dropdown list
- **Available Options**:
  - `es_transportation` - Transportation
  - `es_food` - Food & Beverages
  - `es_health` - Health & Medical
  - `es_leisure` - Entertainment & Leisure
  - `es_shopping` - Shopping
  - `es_home` - Home & Family
  - `es_other` - Others

### 3. الجنس | Gender
**العربية:**
- **النوع**: قائمة منسدلة
- **الخيارات**: F (أنثى), M (ذكر)

**English:**
- **Type**: Dropdown list
- **Options**: F (Female), M (Male)

### 4. الفئة العمرية | Age Group
**العربية:**
- **النوع**: قائمة منسدلة مرقمة
- **التقسيمات**:
  - المجموعة 1: 18-24 سنة
  - المجموعة 2: 25-34 سنة
  - المجموعة 3: 35-44 سنة
  - المجموعة 4: 45-54 سنة
  - المجموعة 5: 55-64 سنة
  - المجموعة 6: 65+ سنة

**English:**
- **Type**: Numbered dropdown list
- **Categories**:
  - Group 1: 18-24 years
  - Group 2: 25-34 years
  - Group 3: 35-44 years
  - Group 4: 45-54 years
  - Group 5: 55-64 years
  - Group 6: 65+ years

---

## 🔧 التفاصيل التقنية للخادم | Server Technical Details (app.py)

### إدارة النماذج | Model Management:
```python
# تحميل النماذج المدربة مسبقاً
preprocessor = None
isolation_forest_model = None
autoencoder_model = None
ae_threshold = None
```

### نظام الأمان والاستقرار | Security & Stability System:
**العربية:**
- **تحميل مرن**: يدعم كل من Pickle و Joblib
- **نظام Mock**: في حالة فشل تحميل النماذج، يتم التبديل لنظام محاكاة
- **معالجة الأخطاء**: نظام شامل لتسجيل ومعالجة الأخطاء
- **Logging**: تسجيل مفصل لجميع العمليات

**English:**
- **Flexible Loading**: Supports both Pickle and Joblib
- **Mock System**: Falls back to simulation system when model loading fails
- **Error Handling**: Comprehensive system for logging and error handling
- **Logging**: Detailed logging for all operations

### المسارات | Routes:
**العربية:**
1. **الصفحة الرئيسية** (`/`):
   - عرض واجهة المستخدم
   - التحقق من حالة النماذج

2. **التنبؤ** (`/predict`):
   - استقبال بيانات المعاملة
   - معالجة البيانات
   - تشغيل النماذج
   - إرجاع النتائج

**English:**
1. **Home Page** (`/`):
   - Display user interface
   - Check model status

2. **Prediction** (`/predict`):
   - Receive transaction data
   - Process data
   - Run models
   - Return results

### عملية التنبؤ | Prediction Process:
```python
# معالجة البيانات
processed_data = preprocessor.transform(input_data)

# Isolation Forest
if_prediction = isolation_forest_model.predict(processed_data)[0]
if_result = "Normal" if if_prediction == 1 else "Fraud"

# Autoencoder
reconstructed = autoencoder_model.predict(processed_data)
mse = np.mean(np.power(processed_data - reconstructed, 2), axis=1)[0]
ae_result = "Normal" if mse <= ae_threshold else "Fraud"
```

---

## 🎨 تصميم الواجهة الأمامية | Frontend Design

### الألوان والثيم | Colors & Theme:
```css
:root {
    --primary-color: #2c3e50;      /* الأزرق الداكن */
    --secondary-color: #3498db;     /* الأزرق الفاتح */
    --accent-color: #e74c3c;        /* الأحمر */
    --success-color: #2ecc71;       /* الأخضر */
    --warning-color: #f39c12;       /* البرتقالي */
    --danger-color: #e74c3c;        /* الأحمر */
}
```

### المكونات الرئيسية | Main Components:

#### 1. الهيدر | Header:
**العربية:**
- عنوان التطبيق مع أيقونة الحماية
- وصف مختصر لوظيفة النظام
- تصميم متدرج مع ظلال

**English:**
- Application title with security icon
- Brief description of system function
- Gradient design with shadows

#### 2. نموذج الإدخال | Input Form:
**العربية:**
- حقول مطلوبة للبيانات الأساسية
- قوائم منسدلة للاختيارات
- تحقق من صحة البيانات
- تأثيرات تفاعلية عند التركيز

**English:**
- Required fields for basic data
- Dropdown lists for selections
- Data validation
- Interactive focus effects

#### 3. قسم النتائج | Results Section:
**العربية:**
- عرض نتائج النموذجين جنباً إلى جنب
- مؤشر بصري لخطأ الإعادة البناء (MSE)
- عرض تفاصيل المعاملة المدخلة
- ألوان مميزة للحالات (طبيعي/مشبوه)

**English:**
- Side-by-side display of both model results
- Visual indicator for reconstruction error (MSE)
- Display of input transaction details
- Distinctive colors for states (normal/suspicious)

#### 4. مؤشر MSE | MSE Indicator:
**العربية:**
- شريط تقدم ملون
- علامة العتبة
- قيم رقمية دقيقة
- تغيير اللون حسب النتيجة

**English:**
- Colored progress bar
- Threshold marker
- Precise numerical values
- Color change based on result

---

## 💻 منطق الواجهة الأمامية | Frontend Logic (script.js)

### الوظائف الرئيسية | Main Functions:

#### 1. إرسال النموذج | Form Submission:
```javascript
form.addEventListener('submit', function(e) {
    e.preventDefault();
    // إخفاء النتائج السابقة
    // عرض شاشة التحميل
    // إرسال البيانات للخادم
    // معالجة الاستجابة
});
```

#### 2. تحديث النتائج | Results Update:
```javascript
function updatePrediction(prediction, resultElement, iconElement, predictionBox) {
    // تحديث النص
    // تحديث الأيقونة
    // تطبيق الألوان المناسبة
}
```

#### 3. مؤشر MSE | MSE Indicator:
```javascript
function updateMseMeter(mse, threshold, barElement, markerElement, valueElement, thresholdElement) {
    // حساب النسب المئوية
    // تحديث عرض الشريط
    // وضع علامة العتبة
    // تغيير الألوان
}
```

### معالجة الأخطاء | Error Handling:
**العربية:**
- رسائل خطأ واضحة للمستخدم
- معالجة أخطاء الشبكة
- تسجيل الأخطاء في وحدة التحكم
- واجهة مستخدم متجاوبة مع الأخطاء

**English:**
- Clear error messages for users
- Network error handling
- Error logging in console
- Responsive user interface with errors

---

## 🚀 تشغيل النظام | System Setup

### متطلبات النظام | System Requirements:
```txt
flask==2.0.1
numpy==1.21.0
scikit-learn==1.0.2
tensorflow==2.8.0
pickle-mixin==1.0.2
joblib==1.1.0
```

### خطوات التشغيل | Setup Steps:
**العربية:**
1. **تثبيت المتطلبات**:
   ```bash
   pip install -r requirements.txt
   ```

2. **تشغيل الخادم**:
   ```bash
   python app.py
   ```

3. **الوصول للتطبيق**:
   ```
   http://127.0.0.1:5000/
   ```

**English:**
1. **Install Requirements**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run Server**:
   ```bash
   python app.py
   ```

3. **Access Application**:
   ```
   http://127.0.0.1:5000/
   ```

---

## 🔍 كيفية استخدام النظام | How to Use the System

### خطوات التحليل | Analysis Steps:
**العربية:**
1. **إدخال البيانات**:
   - أدخل مبلغ المعاملة
   - اختر فئة المعاملة
   - حدد الجنس
   - اختر الفئة العمرية

2. **تشغيل التحليل**:
   - انقر على "Analyze Transaction"
   - انتظر معالجة البيانات

3. **مراجعة النتائج**:
   - **Isolation Forest**: نتيجة فورية (Normal/Fraud)
   - **Autoencoder**: نتيجة مع قيمة MSE
   - **مؤشر MSE**: مقارنة بصرية مع العتبة
   - **تفاصيل المعاملة**: البيانات المدخلة

**English:**
1. **Data Input**:
   - Enter transaction amount
   - Select transaction category
   - Specify gender
   - Choose age group

2. **Run Analysis**:
   - Click "Analyze Transaction"
   - Wait for data processing

3. **Review Results**:
   - **Isolation Forest**: Immediate result (Normal/Fraud)
   - **Autoencoder**: Result with MSE value
   - **MSE Indicator**: Visual comparison with threshold
   - **Transaction Details**: Input data

### تفسير النتائج | Results Interpretation:
**العربية:**
- **Normal**: المعاملة تبدو طبيعية وآمنة
- **Fraud**: المعاملة مشبوهة وتحتاج مراجعة
- **MSE منخفض**: المعاملة قريبة من الأنماط الطبيعية
- **MSE مرتفع**: المعاملة تختلف عن الأنماط المعتادة

**English:**
- **Normal**: Transaction appears normal and safe
- **Fraud**: Transaction is suspicious and needs review
- **Low MSE**: Transaction is close to normal patterns
- **High MSE**: Transaction differs from usual patterns

---

## 🛡️ الأمان والموثوقية | Security & Reliability

### ميزات الأمان | Security Features:
**العربية:**
- **تحقق من البيانات**: فحص شامل للمدخلات
- **معالجة الأخطاء**: نظام قوي لمعالجة الاستثناءات
- **Logging**: تسجيل مفصل للعمليات والأخطاء
- **نظام Fallback**: التبديل للمحاكاة عند فشل النماذج

**English:**
- **Data Validation**: Comprehensive input checking
- **Error Handling**: Robust exception handling system
- **Logging**: Detailed logging of operations and errors
- **Fallback System**: Switch to simulation when models fail

### الموثوقية | Reliability:
**العربية:**
- **تحميل مرن للنماذج**: دعم متعدد لطرق التحميل
- **نظام Mock**: ضمان عمل التطبيق حتى بدون النماذج
- **واجهة متجاوبة**: تعامل سلس مع الأخطاء
- **تحقق مستمر**: فحص حالة النماذج

**English:**
- **Flexible Model Loading**: Multiple loading method support
- **Mock System**: Ensures app works even without models
- **Responsive Interface**: Smooth error handling
- **Continuous Verification**: Model status checking

---

## 📈 الأداء والتحسين | Performance & Optimization

### تحسينات الأداء | Performance Improvements:
**العربية:**
- **تحميل مسبق للنماذج**: تحميل النماذج عند بدء التطبيق
- **معالجة فعالة**: استخدام NumPy للعمليات الرياضية
- **ذاكرة محسنة**: إدارة ذكية للذاكرة
- **استجابة سريعة**: واجهة مستخدم متجاوبة

**English:**
- **Pre-loading Models**: Load models at application startup
- **Efficient Processing**: Using NumPy for mathematical operations
- **Optimized Memory**: Smart memory management
- **Fast Response**: Responsive user interface

### مراقبة النظام | System Monitoring:
**العربية:**
- **Logging مفصل**: تسجيل جميع العمليات
- **معالجة الأخطاء**: تتبع وتسجيل الأخطاء
- **حالة النماذج**: مراقبة حالة تحميل النماذج
- **إحصائيات الاستخدام**: تتبع طلبات التنبؤ

**English:**
- **Detailed Logging**: Recording all operations
- **Error Handling**: Tracking and logging errors
- **Model Status**: Monitoring model loading status
- **Usage Statistics**: Tracking prediction requests

---

## 🔮 التطوير المستقبلي | Future Development

### تحسينات مقترحة | Proposed Improvements:
**العربية:**
1. **قاعدة بيانات**: حفظ المعاملات والنتائج
2. **واجهة إدارية**: لوحة تحكم للمشرفين
3. **تقارير**: تقارير دورية عن الأنشطة المشبوهة
4. **API متقدم**: واجهة برمجية شاملة
5. **تحليلات متقدمة**: رسوم بيانية وإحصائيات
6. **تنبيهات فورية**: إشعارات للمعاملات المشبوهة
7. **تعلم مستمر**: تحديث النماذج بناءً على البيانات الجديدة

**English:**
1. **Database**: Save transactions and results
2. **Admin Interface**: Control panel for administrators
3. **Reports**: Periodic reports on suspicious activities
4. **Advanced API**: Comprehensive programming interface
5. **Advanced Analytics**: Charts and statistics
6. **Real-time Alerts**: Notifications for suspicious transactions
7. **Continuous Learning**: Update models based on new data

### التكامل المحتمل | Potential Integration:
**العربية:**
- **أنظمة البنوك**: ربط مع أنظمة إدارة المعاملات
- **قواعد البيانات**: MySQL, PostgreSQL
- **خدمات السحابة**: AWS, Azure, GCP
- **أنظمة المراقبة**: Prometheus, Grafana
- **API Gateway**: لإدارة الطلبات والأمان

**English:**
- **Banking Systems**: Integration with transaction management systems
- **Databases**: MySQL, PostgreSQL
- **Cloud Services**: AWS, Azure, GCP
- **Monitoring Systems**: Prometheus, Grafana
- **API Gateway**: For request management and security

---

## 📝 الخلاصة | Conclusion

**العربية:**
هذا النظام يمثل حلاً متكاملاً وعملياً لكشف الشذوذ في المعاملات المصرفية باستخدام تقنيات الذكاء الاصطناعي المتقدمة. يجمع النظام بين قوة نموذجين مختلفين من تعلم الآلة لتوفير تحليل شامل ودقيق للمعاملات المالية.

**English:**
This system represents an integrated and practical solution for detecting anomalies in bank transactions using advanced artificial intelligence techniques. The system combines the power of two different machine learning models to provide comprehensive and accurate analysis of financial transactions.

### النقاط القوية | Strengths:
**العربية:**
- ✅ **دقة عالية**: استخدام نموذجين مكملين
- ✅ **واجهة سهلة**: تصميم بديهي وسهل الاستخدام
- ✅ **موثوقية**: نظام قوي لمعالجة الأخطاء
- ✅ **مرونة**: قابلية التشغيل حتى بدون النماذج
- ✅ **شفافية**: عرض مفصل للنتائج والعمليات

**English:**
- ✅ **High Accuracy**: Using two complementary models
- ✅ **Easy Interface**: Intuitive and user-friendly design
- ✅ **Reliability**: Robust error handling system
- ✅ **Flexibility**: Can operate even without models
- ✅ **Transparency**: Detailed display of results and operations

### الاستخدامات المناسبة | Suitable Applications:
**العربية:**
- 🏦 **البنوك**: مراقبة المعاملات اليومية
- 💳 **شركات الدفع**: كشف الاحتيال في المدفوعات
- 🛒 **التجارة الإلكترونية**: حماية المعاملات الرقمية
- 📊 **التحليل المالي**: دراسة أنماط المعاملات
- 🎓 **التعليم**: أداة تعليمية لتعلم الآلة

**English:**
- 🏦 **Banks**: Daily transaction monitoring
- 💳 **Payment Companies**: Fraud detection in payments
- 🛒 **E-commerce**: Digital transaction protection
- 📊 **Financial Analysis**: Transaction pattern studies
- 🎓 **Education**: Educational tool for machine learning

**العربية:**
النظام جاهز للاستخدام الفوري ويمكن تطويره وتوسيعه حسب الاحتياجات المحددة.

**English:**
The system is ready for immediate use and can be developed and expanded according to specific needs.
