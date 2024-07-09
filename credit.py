# 이 프로젝트는 구글 코랩 환경에서 실행되었습니다

!apt-get install openjdk-8-jdk-headless #jdk install
!wget -q http://archive.apache.org/dist/spark/spark-3.0.0/spark-3.0.0-bin-hadoop3.2.tgz #spark file
!tar -xf spark-3.0.0-bin-hadoop3.2.tgz
!pip install findspark
!pip install kaggle --upgrade

import os 
import findspark

os.environ['JAVA_HOME'] = "/usr/lib/jvm/java-8-openjdk-amd64"
os.environ['SPARK_HOME'] = '/content/spark-3.0.0-bin-hadoop3.2'

findspark.init() 

from pyspark.sql import SparkSession

spark = (
    SparkSession
    .builder
    .appName('pyspark_test')
    .master('local[*]')
    .getOrCreate()
)

from google.colab import files

files.upload()

!mkdir -p ~/.kaggle/ 
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
!kaggle datasets download -d uciml/default-of-credit-card-clients-dataset
!unzip default-of-credit-card-clients-dataset.zip

df = spark.read.csv(
    path = 'UCI_Credit_Card.csv', header = True, inferSchema = True
)

df.show()

df.printSchema()

df.write.format("parquet").save(
    path = "data_parquet",
    header = True
)

from google.colab import files
download_list = os.listdir('./data_parquet')
for file_name in download_list:
    if file_name[-3:] != 'crc':
        files.download('./data_parquet/' + file_name)

# 1. 데이터 로드 및 Spark 세션 생성
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, count

# Spark 세션 생성
spark = SparkSession.builder \
    .appName("part-00000-ffdb221b-b7ba-4ffc-aa52-04bd396b5f7f-c000.snappy") \
    .getOrCreate()

# 데이터 로드
file_path = "part-00000-ffdb221b-b7ba-4ffc-aa52-04bd396b5f7f-c000.snappy.parquet"
df = spark.read.parquet(file_path)

# 2. 데이터 구조 확인
df.printSchema()
df.show(5)

# 컬럼 이름 변경
df = df.withColumnRenamed("default.payment.next.month", "default_payment_next_month")

# 3. 데이터 클렌징

# 성별 데이터 정리
df = df.withColumn("SEX",
    when(col("SEX") == 'M', 1).when(col("SEX") == 'F', 0).otherwise(col("SEX"))
)

# 결혼 상태 및 교육 수준 데이터 정리
df = df.withColumn("EDUCATION",
    when(col("EDUCATION") == 0, 4).otherwise(col("EDUCATION"))
)

df = df.withColumn("MARRIAGE",
    when(col("MARRIAGE") == 0, 3).otherwise(col("MARRIAGE"))
)

# 결측값 처리
# 결측값 확인
df.select([count(when(col(c).isNull(), c)).alias(c) for c in df.columns]).show()

# 결측값 처리 (예: 결측값이 있는 경우 0으로 채움)
df = df.fillna(0)

# 4. 데이터 검증
df.describe().show()
df.show(5)

# 5. 데이터 저장
output_path = "cleaned_credit_card_data.parquet"
df.write.parquet(output_path, mode='overwrite')



# 1. 데이터 로드 및 Spark 세션 생성
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, count, sum
from pyspark.sql.window import Window
import matplotlib.pyplot as plt

# Spark 세션 생성
spark = SparkSession.builder \
    .appName("part-00000-ffdb221b-b7ba-4ffc-aa52-04bd396b5f7f-c000.snappy") \
    .getOrCreate()

# 데이터 로드
file_path = "part-00000-ffdb221b-b7ba-4ffc-aa52-04bd396b5f7f-c000.snappy.parquet"
df = spark.read.parquet(file_path)

# 2. 데이터 구조 확인
df.printSchema()
df.show(5)

# 컬럼 이름 변경
df = df.withColumnRenamed("default.payment.next.month", "default_payment_next_month")

# 3. 데이터 클렌징

# 성별 데이터 정리
df = df.withColumn("SEX",
    when(col("SEX") == 'M', 1).when(col("SEX") == 'F', 0).otherwise(col("SEX"))
)

# 결혼 상태 및 교육 수준 데이터 정리
df = df.withColumn("EDUCATION",
    when(col("EDUCATION") == 0, 4).otherwise(col("EDUCATION"))
)

df = df.withColumn("MARRIAGE",
    when(col("MARRIAGE") == 0, 3).otherwise(col("MARRIAGE"))
)

# 결측값 처리
# 결측값 확인
df.select([count(when(col(c).isNull(), c)).alias(c) for c in df.columns]).show()

# 결측값 처리 (예: 결측값이 있는 경우 0으로 채움)
df = df.fillna(0)

# 4. 데이터 검증
df.describe().show()
df.show(5)

# 5. 데이터 저장
output_path = "cleaned_credit_card_data.parquet"
df.write.parquet(output_path, mode='overwrite')



### 2. 결혼 상태와 교육 수준에 따른 채무 불이행 분석

# 데이터 로드
df = spark.read.parquet("cleaned_credit_card_data.parquet")

# 결혼 상태와 교육 수준에 따른 불이행 여부 집계
default_by_marriage_education = df.groupBy("MARRIAGE", "EDUCATION", "default_payment_next_month").count()

# 불이행 여부 비율 계산
default_by_marriage_education = default_by_marriage_education.withColumn("percentage",
    col("count") / sum("count").over(Window.partitionBy("MARRIAGE", "EDUCATION"))
)

# 결과 출력
default_by_marriage_education.show()

# 데이터 시각화를 위한 Pandas 변환
default_by_marriage_education_pd = default_by_marriage_education.toPandas()

# 결혼 상태 및 교육 수준에 따른 불이행 비율 시각화
fig, ax = plt.subplots(figsize=(14, 8))

for marriage in default_by_marriage_education_pd['MARRIAGE'].unique():
    for education in default_by_marriage_education_pd['EDUCATION'].unique():
        subset = default_by_marriage_education_pd[(default_by_marriage_education_pd['MARRIAGE'] == marriage) & (default_by_marriage_education_pd['EDUCATION'] == education)]
        ax.bar(subset['default_payment_next_month'], subset['count'], label=f'Marriage {marriage}, Education {education}')

ax.set_xlabel('Default Payment Next Month')
ax.set_ylabel('Count')
ax.set_title('Default Count by Marriage and Education Level')
ax.legend(title='Marriage and Education Level')
plt.show()



### 3. 성별 불이행 분포 분석

# 성별에 따른 불이행 여부 집계
default_by_gender = df.groupBy("SEX", "default_payment_next_month").count().orderBy("SEX", "default_payment_next_month")

# 불이행 여부 비율 계산
default_by_gender = default_by_gender.withColumn("percentage",
    col("count") / sum("count").over(Window.partitionBy("SEX"))
)

# 결과 출력
default_by_gender.show()

# 성별에 따른 불이행 비율 시각화
default_by_gender_pd = default_by_gender.toPandas()

fig, ax = plt.subplots(figsize=(10, 6))

for sex in default_by_gender_pd['SEX'].unique():
    subset = default_by_gender_pd[default_by_gender_pd['SEX'] == sex]
    ax.bar(subset['default_payment_next_month'], subset['count'], label=f'SEX {sex}')

ax.set_xlabel('Default Payment Next Month')
ax.set_ylabel('Count')
ax.set_title('Default Count by Gender')
ax.legend(title='Gender')
plt.show()

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, count, sum
from pyspark.sql.window import Window
import matplotlib.pyplot as plt

# 1. 데이터 로드 및 Spark 세션 생성
spark = SparkSession.builder \
    .appName("Credit Card Default Analysis") \
    .getOrCreate()

# 데이터 로드
file_path = "cleaned_credit_card_data.parquet"
df = spark.read.parquet(file_path)

# 2. 결혼 상태와 교육 수준에 따른 채무 불이행 분석

# 결혼 상태와 교육 수준에 따른 불이행 여부 집계
default_by_marriage_education = df.groupBy("MARRIAGE", "EDUCATION", "default_payment_next_month").count().orderBy("MARRIAGE", "EDUCATION", "default_payment_next_month")

# 불이행 여부 비율 계산
default_by_marriage_education = default_by_marriage_education.withColumn("percentage",
    col("count") / sum("count").over(Window.partitionBy("MARRIAGE", "EDUCATION"))
)

# 결과 출력
default_by_marriage_education.show()

# 성별 불이행 분포 분석
default_by_gender = df.groupBy("SEX", "default_payment_next_month").count().orderBy("SEX", "default_payment_next_month")

# 불이행 여부 비율 계산
default_by_gender = default_by_gender.withColumn("percentage",
    col("count") / sum("count").over(Window.partitionBy("SEX"))
)

# 결과 출력
default_by_gender.show()

# 3. 결과 시각화

# 데이터를 Pandas DataFrame으로 변환
default_by_gender_pd = default_by_gender.toPandas()

# 성별 및 불이행 여부에 따른 분포 시각화
fig, ax = plt.subplots(figsize=(10, 6))

for sex in default_by_gender_pd['SEX'].unique():
    subset = default_by_gender_pd[default_by_gender_pd['SEX'] == sex]
    ax.bar(subset['default_payment_next_month'], subset['count'], label=f'SEX {sex}')

ax.set_xlabel('Defaulted')
ax.set_ylabel('Count')
ax.set_title('Default Count by Gender')
ax.legend(title='Gender')
plt.show()

# 결혼 상태와 교육 수준에 따른 불이행 비율 시각화
default_by_marriage_education_pd = default_by_marriage_education.toPandas()

fig, ax = plt.subplots(figsize=(12, 8))

for marriage in default_by_marriage_education_pd['MARRIAGE'].unique():
    for education in default_by_marriage_education_pd['EDUCATION'].unique():
        subset = default_by_marriage_education_pd[(default_by_marriage_education_pd['MARRIAGE'] == marriage) &
                                                  (default_by_marriage_education_pd['EDUCATION'] == education)]
        if not subset.empty:
            ax.bar(subset['default_payment_next_month'], subset['percentage'], label=f'Marriage {marriage}, Education {education}')

ax.set_xlabel('Defaulted')
ax.set_ylabel('Percentage')
ax.set_title('Default Percentage by Marriage and Education')
ax.legend(title='Marriage and Education')
plt.show()
