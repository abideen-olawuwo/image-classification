print('Training the model')

import pandas as pd
import numpy as np

import seaborn as sns
from matplotlib import pyplot as plt




df = pd.read_csv('Leads.csv')




len(df)


# ## Initial data preparation



df.head()




df.dtypes




df.isnull().sum()




df.columns = df.columns.str.lower().str.replace(' ', '_')

string_columns = list(df.dtypes[df.dtypes == 'object'].index)

for col in string_columns:
    df[col] = df[col].str.lower().str.replace(' ', '_')




categorical = [
            'lead_origin','lead_source','do_not_email','do_not_call',
            'last_activity','country','specialization', 'how_did_you_hear_about_x_education',
            'what_is_your_current_occupation','what_matters_most_to_you_in_choosing_a_course','search',
            'magazine','newspaper_article','digital_advertisement', 'through_recommendations',
            'receive_more_updates_about_our_courses', 'tags', 'lead_quality','update_me_on_supply_chain_content',
            'lead_profile','city', 'asymmetrique_activity_index','asymmetrique_profile_index',
            'i_agree_to_pay_the_amount_through_cheque', 'a_free_copy_of_mastering_the_interview',
            'last_notable_activity', 'newspaper','x_education_forums','get_updates_on_dm_content'
                ]


numerical = [
    'lead_number', 'totalvisits','total_time_spent_on_website',
    'page_views_per_visit','asymmetrique_activity_score','asymmetrique_profile_score'
            ]



for cat in categorical:
    df[cat]= df[cat].fillna(df[cat].mode()[0])
for num in numerical :
    df[num]= df[num].fillna(df[num].mean())




df.isna().sum()




df.head().T




df.columns = df.columns.str.lower().str.replace(' ', '_')

string_columns = list(df.dtypes[df.dtypes == 'object'].index)

for col in string_columns:
    df[col] = df[col].str.lower().str.replace(' ', '_')




df.head().T




from sklearn.model_selection import train_test_split




df_train_full, df_test = train_test_split(df, test_size=0.2, random_state=1)



df_train, df_val = train_test_split(df_train_full, test_size=0.25, random_state=11)




len(df_train), len(df_val), len(df_test)




df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)




y_train = df_train.converted.values
y_val = df_val.converted.values
y_test = df_test.converted.values



del df_train['converted']
del df_val['converted']
del df_test['converted']


# ## Exploratory data analysis



df_full_train = df_train_full.reset_index(drop=True)




df_full_train.converted.value_counts()




df_full_train.converted.value_counts(normalize=True)




df.isna().sum()




df_train_full.converted.value_counts(normalize=True)




global_mean = df_train_full.converted.mean()
round(global_mean, 3)




categorical = [
            'lead_origin','lead_source','do_not_email','do_not_call',
            'last_activity','country','specialization', 'how_did_you_hear_about_x_education',
            'what_is_your_current_occupation','what_matters_most_to_you_in_choosing_a_course','search',
            'magazine','newspaper_article','digital_advertisement', 'through_recommendations',
            'receive_more_updates_about_our_courses', 'tags', 'lead_quality','update_me_on_supply_chain_content',
            'lead_profile','city', 'asymmetrique_activity_index','asymmetrique_profile_index',
            'i_agree_to_pay_the_amount_through_cheque', 'a_free_copy_of_mastering_the_interview',
            'last_notable_activity', 'newspaper','x_education_forums','get_updates_on_dm_content'
                ]


numerical = [
    'lead_number', 'totalvisits','total_time_spent_on_website',
    'page_views_per_visit','asymmetrique_activity_score','asymmetrique_profile_score'
            ]




categorical




df_train_full[categorical].nunique()


# ## Feature importance



from IPython.display import display




global_mean = df_train_full.converted.mean()
global_mean




for col in categorical:
    df_group = df_train_full.groupby(by=col).converted.agg(['mean'])
    df_group['diff'] = df_group['mean'] - global_mean
    df_group['risk'] = df_group['mean'] / global_mean
    display(df_group)



from sklearn.metrics import mutual_info_score




def calculate_mi(series):
    return mutual_info_score(series, df_train_full.converted)

df_mi = df_train_full[categorical].apply(calculate_mi)
df_mi = df_mi.sort_values(ascending=False).to_frame(name='MI')


display(df_mi.head())
display(df_mi.tail())


# ### Feature Importance: Correlation Coefficient


df_train_full[numerical].corrwith(df_train_full.converted).to_frame('correlation')




df_train_full.groupby(by='converted')[numerical].mean()


# ## One-hot encoding



from sklearn.feature_extraction import DictVectorizer




train_dict = df_train[categorical + numerical].to_dict(orient='records')



df_train[['lead_origin','lead_source']].iloc[:10]



train_dict[0]



dv = DictVectorizer(sparse=False)
dv.fit(train_dict)



X_train = dv.transform(train_dict)




X_train.shape



y_test.shape



dv.get_feature_names()


# ## Training logistic regression



from sklearn.linear_model import LogisticRegression




model = LogisticRegression(solver='liblinear', random_state=1)
model.fit(X_train, y_train)




val_dict = df_val[categorical + numerical].to_dict(orient='records')
X_val = dv.transform(val_dict)




model.predict_proba(X_val)




y_pred = model.predict_proba(X_val)[:, 1]



y_pred



converted = y_pred > 0.5




(y_val == converted).mean()


# ## Model interpretation

model.intercept_[0]




dict(zip(dv.get_feature_names(), model.coef_[0].round(3)))




subset = ['tags', 'lead_number', 'totalvisits']
train_dict_small = df_train[subset].to_dict(orient='records')
dv_small = DictVectorizer(sparse=False)
dv_small.fit(train_dict_small)

X_small_train = dv_small.transform(train_dict_small)

dv_small.get_feature_names()



model_small = LogisticRegression(solver='liblinear', random_state=1)
model_small.fit(X_small_train, y_train)


model_small.intercept_[0]




dict(zip(dv_small.get_feature_names(), model_small.coef_[0].round(3)))




val_dict_small = df_val[subset].to_dict(orient='records')
X_small_val = dv_small.transform(val_dict_small)




y_pred_small = model_small.predict_proba(X_small_val)[:, 1]




y_test


# ## Using the model



prospect = {
'prospect_id': '2f302f24-3151-4763-afdc-a80d9750b3d8',
 'lead_origin': 'lead_add_form',
 'lead_source': 'reference',
 'do_not_email': 'no',
 'do_not_call': 'no',
 'last_activity': 'sms_sent',
 'country': 'india',
 'specialization': 'select',
 'how_did_you_hear_about_x_education': 'select',
 'what_is_your_current_occupation': 'unemployed',
 'what_matters_most_to_you_in_choosing_a_course': 'better_career_prospects',
 'search': 'no',
 'magazine': 'no',
 'newspaper_article': 'no',
 'digital_advertisement': 'no',
 'through_recommendations': 'no',
 'receive_more_updates_about_our_courses': 'no',
 'tags': 'will_revert_after_reading_the_email',
 'lead_quality': 'might_be',
 'update_me_on_supply_chain_content': 'no',
 'lead_profile': 'select',
 'city': 'select',
 'asymmetrique_activity_index': '02.medium',
 'asymmetrique_profile_index': '02.medium',
 'i_agree_to_pay_the_amount_through_cheque': 'no',
 'a_free_copy_of_mastering_the_interview': 'no',
 'last_notable_activity': 'sms_sent',
 'newspaper': 'no',
 'x_education_forums': 'no',
 'get_updates_on_dm_content': 'no',
 'lead_number': 627173,
 'totalvisits': 0.0,
 'total_time_spent_on_website': 0,
 'page_views_per_visit': 0.0,
 'asymmetrique_activity_score': 14.306252489048187,
 'asymmetrique_profile_score': 16.344882516925527
}


# In[64]:


X_test = dv.transform([prospect])
model.predict_proba(X_test)[0, 1]




print(list(X_test[0]))



prospect = {
    'prospect_id': '2f302f24-3151-4763-afdc-a80d9750b3d8',
 'lead_origin': 'lead_add_form',
 'lead_source': 'reference',
 'do_not_email': 'no',
 'do_not_call': 'no',
 'last_activity': 'sms_sent',
 'country': 'india',
 'specialization': 'select',
 'how_did_you_hear_about_x_education': 'select',
 'what_is_your_current_occupation': 'unemployed',
 'what_matters_most_to_you_in_choosing_a_course': 'better_career_prospects',
 'search': 'no',
 'magazine': 'no',
 'newspaper_article': 'no',
 'digital_advertisement': 'no',
 'through_recommendations': 'no',
 'receive_more_updates_about_our_courses': 'no',
 'tags': 'will_revert_after_reading_the_email',
 'lead_quality': 'might_be',
 'update_me_on_supply_chain_content': 'no',
 'lead_profile': 'select',
 'city': 'select',
 'asymmetrique_activity_index': '02.medium',
 'asymmetrique_profile_index': '02.medium',
 'i_agree_to_pay_the_amount_through_cheque': 'no',
 'a_free_copy_of_mastering_the_interview': 'no',
 'last_notable_activity': 'sms_sent',
 'newspaper': 'no',
 'x_education_forums': 'no',
 'get_updates_on_dm_content': 'no',
 'lead_number': 627173,
 'totalvisits': 0.0,
 'total_time_spent_on_website': 0,
 'page_views_per_visit': 0.0,
 'asymmetrique_activity_score': 14.306252489048187,
 'asymmetrique_profile_score': 16.344882516925527

}




X_test = dv.transform([prospect])
model.predict_proba(X_test)[0, 1]


# ### Accuracy


y_pred = model.predict_proba(X_val)[:, 1]
converted = y_pred >= 0.5
(converted == y_val).mean()


from sklearn.metrics import accuracy_score



accuracy_score(y_val, y_pred >= 0.5)



thresholds = np.linspace(0, 1, 11)
thresholds




thresholds = np.linspace(0, 1, 21)

accuracies = []

for t in thresholds:
    acc = accuracy_score(y_val, y_pred >= t)
    accuracies.append(acc)
    print('%0.2f %0.3f' % (t, acc))





converted_small = y_pred_small >= 0.5
(converted_small == y_val).mean()




accuracy_score(y_val, converted_small)



size_val = len(y_val)
baseline = np.repeat(False, size_val)
baseline



accuracy_score(baseline, y_val)


# ## Confusion table



true_positive = ((y_pred >= 0.5) & (y_val == 1)).sum()
false_positive = ((y_pred >= 0.5) & (y_val == 0)).sum()
false_negative = ((y_pred < 0.5) & (y_val == 1)).sum()
true_negative = ((y_pred < 0.5) & (y_val == 0)).sum()



confusion_table = np.array(
     # predict neg    pos
    [[true_negative, false_positive], # actual neg
     [false_negative, true_positive]]) # actual pos

confusion_table



confusion_table / confusion_table.sum()


# ## Precision and recall



precision = true_positive / (true_positive + false_positive)
recall = true_positive / (true_positive + false_negative)
precision, recall




confusion_table / confusion_table.sum()




precision = true_positive / (true_positive + false_positive)
recall = true_positive / (true_positive + false_negative)
precision, recall


# ## ROC and AUC

# TPR and FPR



scores = []

thresholds = np.linspace(0, 1, 101)

for t in thresholds: #B
    tp = ((y_pred >= t) & (y_val == 1)).sum()
    fp = ((y_pred >= t) & (y_val == 0)).sum()
    fn = ((y_pred < t) & (y_val == 1)).sum()
    tn = ((y_pred < t) & (y_val == 0)).sum()
    scores.append((t, tp, fp, fn, tn))

df_scores = pd.DataFrame(scores)
df_scores.columns = ['threshold', 'tp', 'fp', 'fn', 'tn']




df_scores[::10]




df_scores['tpr'] = df_scores.tp / (df_scores.tp + df_scores.fn)
df_scores['fpr'] = df_scores.fp / (df_scores.fp + df_scores.tn)




df_scores[::10]





# Random baseline



def tpr_fpr_dataframe(y_val, y_pred):
    scores = []

    thresholds = np.linspace(0, 1, 101)

    for t in thresholds:
        tp = ((y_pred >= t) & (y_val == 1)).sum()
        fp = ((y_pred >= t) & (y_val == 0)).sum()
        fn = ((y_pred < t) & (y_val == 1)).sum()
        tn = ((y_pred < t) & (y_val == 0)).sum()

        scores.append((t, tp, fp, fn, tn))

    df_scores = pd.DataFrame(scores)
    df_scores.columns = ['threshold', 'tp', 'fp', 'fn', 'tn']

    df_scores['tpr'] = df_scores.tp / (df_scores.tp + df_scores.fn)
    df_scores['fpr'] = df_scores.fp / (df_scores.fp + df_scores.tn)

    return df_scores



np.random.seed(1)
y_rand = np.random.uniform(0, 1, size=len(y_val))
df_rand = tpr_fpr_dataframe(y_val, y_rand)
df_rand[::10]







# Ideal baseline:


num_neg = (y_val == 0).sum()
num_pos = (y_val == 1).sum()

y_ideal = np.repeat([0, 1], [num_neg, num_pos])
y_pred_ideal = np.linspace(0, 1, num_neg + num_pos)

df_ideal = tpr_fpr_dataframe(y_ideal, y_pred_ideal)
df_ideal[::10]







# ROC curve





# Using Scikit-Learn for plotting the ROC curve



from sklearn.metrics import roc_curve
from sklearn.metrics import auc




fpr, tpr, thresholds = roc_curve(y_val, y_pred)





# AUC: Area under the ROC curve



df_scores_small = tpr_fpr_dataframe(y_val, y_pred_small)




auc(df_scores.fpr, df_scores.tpr)




auc(df_scores_small.fpr, df_scores_small.tpr)


# Comparing multiple models with ROC curves


fpr_large, tpr_large, _ = roc_curve(y_val, y_pred)
fpr_small, tpr_small, _ = roc_curve(y_val, y_pred_small)



from sklearn.metrics import roc_auc_score




roc_auc_score(y_val, y_pred)




roc_auc_score(y_val, y_pred_small)


# Interpretation of AUC: the probability that a randomly chosen positive example
# ranks higher than a randomly chosen negative example

neg = y_pred[y_val == 0]
pos = y_pred[y_val == 1]

np.random.seed(1)
neg_choice = np.random.randint(low=0, high=len(neg), size=10000)
pos_choice = np.random.randint(low=0, high=len(pos), size=10000)
(pos[pos_choice] > neg[neg_choice]).mean()


# ## K-fold cross-validation



def train(df, y):
    cat = df[categorical + numerical].to_dict(orient='records')
    
    dv = DictVectorizer(sparse=False)
    dv.fit(cat)

    X = dv.transform(cat)

    model = LogisticRegression(solver='liblinear')
    model.fit(X, y)

    return dv, model


def predict(df, dv, model):
    cat = df[categorical + numerical].to_dict(orient='records')
    
    X = dv.transform(cat)

    y_pred = model.predict_proba(X)[:, 1]

    return y_pred




from sklearn.model_selection import KFold




kfold = KFold(n_splits=10, shuffle=True, random_state=1)



aucs = []

for train_idx, val_idx in kfold.split(df_train_full):
    df_train = df_train_full.iloc[train_idx]
    y_train = df_train.converted.values

    df_val = df_train_full.iloc[val_idx]
    y_val = df_val.converted.values

    dv, model = train(df_train, y_train)
    y_pred = predict(df_val, dv, model)

    rocauc = roc_auc_score(y_val, y_pred)
    aucs.append(rocauc)




np.array(aucs).round(3)



print('auc = %0.3f ± %0.3f' % (np.mean(aucs), np.std(aucs)))


# Tuning the parameter `C`



def train(df, y, C=1.0):
    cat = df[categorical + numerical].to_dict(orient='records')

    dv = DictVectorizer(sparse=False)
    dv.fit(cat)

    X = dv.transform(cat)

    model = LogisticRegression(solver='liblinear', C=C)
    model.fit(X, y)

    return dv, model




nfolds = 10
kfold = KFold(n_splits=nfolds, shuffle=True, random_state=1)

for C in [0.001, 0.01, 0.1, 0.5, 1.0]:
    aucs = []

    for train_idx, val_idx in kfold.split(df_train_full):
        df_train = df_train_full.iloc[train_idx]
        df_val = df_train_full.iloc[val_idx]

        y_train = df_train.converted.values
        y_val = df_val.converted.values

        dv, model = train(df_train, y_train, C=C)
        y_pred = predict(df_val, dv, model)
        
        auc = roc_auc_score(y_val, y_pred)
        aucs.append(auc)

    print('C=%s, auc = %0.3f ± %0.3f' % (C, np.mean(aucs), np.std(aucs)))


# ## Saving the Model



import pickle 




output_file = f'model_C={C}.bin'
output_file




f_out = open(output_file, 'wb')
pickle.dump((dv, model), f_out)
f_out.close()


with open(output_file, 'wb') as f_out:
    pickle.dump((dv, model), f_out)


print(f'Trained model saved as {output_file}')

