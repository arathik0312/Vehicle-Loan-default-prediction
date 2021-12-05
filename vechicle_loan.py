import mlcp.pipeline as pl
import mlcp.classifires as cl
import mlcp.regressors as rg
import numpy as np
import pandas as pd
from datetime import datetime as dt
import math
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean
from datetime import date
import re
from sklearn.preprocessing import LabelEncoder , OneHotEncoder
from sklearn.decomposition import PCA


import warnings
warnings.filterwarnings("ignore")
#execution controls
classification=1; #if regression = 0, classification = 1
read=1
primary_analysis=1 #dev only
visual_analysis=0
observed_corrections=1
feature_engineering=1
analyze_missing_values=1
treat_missing_values=1
define_variables=1
analyze_stats=0; #dev only
analyzed_corrections=0
gaussian_transform=0
polynomial_transform=0
skew_corrections=0
scaling=1 #do for continuous numerical values, don't for binary/ordinal/one-hot
encoding=1
matrix_corrections=0
oversample=1; #dev only
reduce_dim=0
compare=0; #dev only
cross_validate=0; #dev only
grid_search=0; #dev only
bins=1
train_classification=1
train_regression=0
dimension=0


if read==1:
    filepath = "data/train.csv"
    y_name = 'loan_default'
    dtype_file = "veh.txt"
    df = pl.read_data(filepath)
    if classification == 1:
        sample_diff, min_y, max_y = pl.bias_analysis(df,y_name)
        print("sample diff:", sample_diff)
        print("sample ratio:", min_y/max_y)
        print(df[y_name].value_counts())
    else:
        print("y skew--->", df[y_name].skew())
        
 
if primary_analysis==1:
    #consider: unwanted features, numerical conversions (year to no. years), 
    #wrong dtypes, missing values, categorical to ordinal numbers
    df_h = df.head()
    with open(dtype_file, "w") as f:
        for c in df_h:
            line1 = df_h[c]
            line2 = df[c].nunique()
            line3 = df[c].isnull().sum()
            f.write(str(line1) + "\n" + "Unique: " + str(line2) + 
                    ", missing: " + str(line3)
            + "\n\n" + "-----------------"+"\n")
    if classification == 0:
        plt.boxplot(df[y_name]); plt.show()



if visual_analysis==1:
    pl.visualize_y_vs_x(df,y_name)
    pass
               

if observed_corrections==1:
    
    df=df.drop(["UniqueID","branch_id","supplier_id","manufacturer_id",
                "Current_pincode_ID","State_ID","Employee_code_ID"],axis=1)
    
    pres_date='01-01-21'            
    def days_between(d1,d2):
        d1=dt.strptime(d1,'%d-%m-%y')
        d2=dt.strptime(d2,'%d-%m-%y')
        return abs((d2-d1).days)
    
    df['Date.of.Birth']=df['Date.of.Birth'].apply(lambda x:days_between(x,pres_date)/365)
    df['DisbursalDate']=df['DisbursalDate'].apply(lambda x:days_between(x,pres_date)/365)
    
    
if feature_engineering==1:
    
    df['AVERAGE.ACCT.AGE']=df['AVERAGE.ACCT.AGE'].apply(lambda x:(re.sub('[a-z]','',x)).split())
    df["AVERAGE.ACCT.AGE"]=df["AVERAGE.ACCT.AGE"].apply(lambda x:int(x[0])*12 + int(x[1]))
    
    df["CREDIT.HISTORY.LENGTH"]=df["CREDIT.HISTORY.LENGTH"].apply(lambda x:(re.sub('[a-z]',"",x)).split())
    df["CREDIT.HISTORY.LENGTH"]=df["CREDIT.HISTORY.LENGTH"].apply(lambda x:int(x[0])*12 +int(x[1]))
   
if analyze_missing_values==1:
    drop_th = 0.4
    print(df.shape)
    df = pl.missing_value_analysis(df, drop_th)
    print(df.shape)
    before = len(df); df_copy_drop = df.dropna(); after = len(df_copy_drop); 
    print("dropped %--->", round(1-(after/before),2)*100,"%")
    num_df = df.select_dtypes(exclude=['O'])
    

if treat_missing_values==1:
    df["Employment.Type"]=df.fillna(df["Employment.Type"].mode()[0])
   

if define_variables==1:
    y = df[y_name]
    x = df.drop([y_name],axis=1)
    n_dim = x.shape[1]
    print(x.shape)


if analyze_stats==1:
   #find important features and remove correlated features based on low-variance or High-skew
    cors = pl.correlations(x, th=0.7)
    with open(dtype_file, "a") as f:
        f.write("\n\n\n"+str(cors))
    scores = pl.feature_analysis(x,y); print(scores); print("")
#    if classification == 1:
#        ranks = pl.feature_selection(x,y); print(ranks); print("")
    for c in x.columns:
        sd, minv, maxv = pl.bias_analysis(x,c)
        print(c, " = ", sd)
    print("")   
    print("skew in feature:")
    print(x.skew())
    
    
if analyzed_corrections==1:
#    x = x.drop(['antiviral_medication','bought_face_mask', 'cont_child_undr_6_mnths'
#                ],axis=1)
    pass
   
         
if polynomial_transform==1:
   degree=3
   x = pl.polynomial_features(x,degree)
   print("polynomial features:")
   print(x.head(1)); print("")


if gaussian_transform==1:
   n_dim=3
   x = pl.gaussian_features(x,y,n_dim)
   print("Gaussian features:")
   print(x.head(1)); print(x.shape,y.shape);print("")
   

if skew_corrections==1:
    x = pl.skew_correction(x)
 

if scaling==1:
    selective_scaling=0
    
    x_num, x_cat = pl.split_num_cat(x)
    
    if selective_scaling == 1:
        selective_features=[]
        selective_x_num = x_num[selective_features]
        x_num = x_num.drop(selective_features, axis=1)
    else:
        selective_x_num = x_num
    
    if False:
        selective_x_num, fm = pl.max_normalization(selective_x_num); #0-1
    if True:
        selective_x_num = pl.minmax_normalization(selective_x_num) ; #0-1
    if False:
        selective_x_num = pl.Standardization(selective_x_num); #-1 to 1
    
    print("")
    print("after scaling - categorical-->", x_cat.info())
    print("after scaling - numerical-->", x_num.shape)
    
    if selective_scaling == 1:
        x_num = pl.join_num_cat(x_num, selective_x_num)
    else:
        x_num = selective_x_num
        
    x = pl.join_num_cat(x_num,x_cat)
    

if encoding==1:
    x_num, x_cat = pl.split_num_cat(x)
    
    if True:
        x_cat = pl.label_encode(x_cat)
    if False:
        x_cat = pl.onehot_encode(x_cat)
    
    if False:
         x,y,mmd = pl.auto_transform_data(x,y); #best choice if dtypes are fixed
    
    x = pl.join_num_cat(x_num,x_cat)
    print("after encoding--->", x.shape)
      
    
if matrix_corrections==1:   
    x = pl.matrix_correction(x)
      
       
if oversample==1:
    #for only imbalanced data
    x,y = pl.oversampling(x,y)
    print(x.shape); print(y.value_counts())
    
if dimension==1:
    pca=PCA(n_components = 18)
    pca.fit(x) 
    print(plt.plot(np.cumsum(pca.explained_variance_ratio_)))    

if reduce_dim==1:
    x = pl.reduce_dimensions(x, 8); #print(x.shape)
    x = pd.DataFrame(x)
    print("transformed x:")
    print(x.shape); print("")
    

if compare==1:
    #compare models on sample
    n_samples = 500
    df_temp = pd.concat((x,y),axis=1)
    df_sample = pl.stratified_sample(df_temp, y_name, n_samples)
    print("stratified sample:"); print(df_sample[y_name].value_counts())
    y_sample = df_sample[y_name]
    x_sample = df_sample.drop([y_name],axis=1)
    model_meta_data = pl.compare_models(x_sample, y_sample, 111)
    
    
    
if cross_validate==1:
    #deciding the random state
    best_model = cl.GradientBoostingClassifier()
    pl.kfold_cross_validate(best_model, x, y,100)


if grid_search==1:
    #grids
    dtc_param_grid = {"criterion":["gini", "entropy"],
                      "class_weight":[{0:1,1:1.5}],
                      "max_depth":[2,4,6,8,10],
                      "min_samples_leaf":[1,2,3,4,5],
                      "min_samples_split":[2,3,4,5,6],
                      "random_state":[21,111]
                      }
    
    log_param_grid = {"penalty":['l1','l2','elasticnet'],
                      "C":[0.1,0.5,1,2,5,10],
                      "class_weight":[{0:1,1:1}],
                      "solver":['liblinear', 'sag', 'saga'],
                      "max_iter":[100,150,200,300],
                      "random_state":[100,111]
                      }
    
    param_grid = dtc_param_grid
    model = cl.RandomForestClassifier()
    best_param_model = pl.select_best_parameters(model, param_grid, x, y, 111)

if bins==1:
    def cns_score(score):
        if score<100:
            return 0
        elif (score>=100) and (score<200):
            return 1
        elif (score>=200) & (score<300):
            return 2
        elif (score>=300) & (score<400):
            return 3
        elif (score>=400) & (score<500):
            return 4
        elif (score>=500) & (score<600):
            return 5
        elif (score>=600) & (score <700):
            return 6
        elif (score>=700) & (score <800):
            return 7
        elif (score>=800) & (score <900):
            return 8
        elif (score>=900) & (score <1000):
            return 9
        else:
            return 10
    
    df['PERFORM_CNS.SCORE']=df['PERFORM_CNS.SCORE'].map(lambda x:cns_score(x))  
    print(df['PERFORM_CNS.SCORE'].value_counts())  

if train_classification==1:
    
    best_param_model = cl.GradientBoostingClassifier(random_state=100)
    trained_model = pl.clf_train_test(best_param_model,x,y,111,"GBC1",pred_th=None)
    
      
if train_regression==1:
   model = rg.GradientBoostingRegressor()
   pl.reg_train_test(model,x,y,111,"DTR1")
    
