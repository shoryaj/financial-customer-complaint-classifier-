"""Reproducible CFPB complaint classification. See README.md for commands."""
import argparse, hashlib, html, json, re, sys
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, ConfusionMatrixDisplay
from sklearn.pipeline import Pipeline

SEED=42
# Explicit harmonization of historical CFPB product names; no silent catch-all.
ALIASES={
 'Credit card':'Credit cards', 'Credit card or prepaid card':'Cards (credit/prepaid)',
 'Prepaid card':'Prepaid cards', 'Mortgage':'Mortgages', 'Debt collection':'Debt collection',
 'Student loan':'Student loans', 'Vehicle loan or lease':'Vehicle loans',
 'Checking or savings account':'Bank accounts', 'Bank account or service':'Bank accounts',
 'Credit reporting':'Credit reporting',
 'Credit reporting, credit repair services, or other personal consumer reports':'Credit reporting',
 'Credit reporting or other personal consumer reports':'Credit reporting',
 'Money transfer, virtual currency, or money service':'Money services',
 'Money transfer, virtual currency, or money services':'Money services',
 'Money transfers':'Money services', 'Virtual currency':'Money services',
 'Payday loan':'Payday loans', 'Payday loan, title loan, or personal loan':'Payday/title/personal loans',
 'Payday loan, title loan, personal loan, or advance loan':'Payday/title/personal loans',
 'Consumer Loan':'Consumer loans', 'Other financial service':'Other financial services',
 'Debt or credit management':'Debt/credit management'}

def clean(text):
    text=html.unescape(str(text)).lower()
    text=re.sub(r'https?://\S+|www\.\S+',' ',text)
    text=re.sub(r'\b\S+@\S+\.\S+\b',' ',text)
    text=re.sub(r'\bx{2,}\b|\d+',' ',text)
    return re.sub(r'\s+',' ',re.sub(r'[^a-z\s\']',' ',text)).strip()

def style(text):
    """Transparent heuristic, never a human-validated formality label."""
    words=re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?",text)
    n=len(words)
    informal=bool(re.search(r"\b(u|ur|pls|plz|lol|wtf|gonna|wanna|idk)\b|!{2,}|\?{2,}",text,re.I))
    formal=bool(re.search(r'\b(furthermore|therefore|pursuant|sincerely|regarding|request|documentation|attached)\b',text,re.I))
    length='short' if n<=50 else ('long' if n>=200 else 'medium')
    group='informal_short_proxy' if n<=50 and informal else ('formal_detailed_proxy' if n>=200 and formal and not informal else 'unassigned')
    return n,length,group

def load_data(path,max_rows):
    headers=pd.read_csv(path,nrows=0).columns
    lookup={re.sub(r'[^a-z]','',c.lower()):c for c in headers}
    needed=['consumercomplaintnarrative','product']
    if not all(k in lookup for k in needed): raise ValueError('CSV needs Consumer complaint narrative and Product columns.')
    keys=needed+[k for k in ['complaintid','datereceived'] if k in lookup]
    # Uniform random priorities across the entire CSV, bounded memory. Blank texts excluded first.
    rng=np.random.default_rng(SEED); pool=None; raw=0; eligible=0
    for chunk in pd.read_csv(path,usecols=[lookup[k] for k in keys],chunksize=50000,dtype=str):
        raw+=len(chunk);chunk=chunk.rename(columns={lookup[k]:k for k in keys})
        chunk=chunk.dropna(subset=needed);chunk=chunk[chunk.consumercomplaintnarrative.str.strip().ne('')]
        eligible+=len(chunk);chunk['_priority']=rng.random(len(chunk))
        pool=chunk if pool is None else pd.concat([pool,chunk],ignore_index=True)
        pool=pool.nsmallest(max_rows,'_priority')
    if pool is None or pool.empty: raise ValueError('No nonempty narratives found.')
    d=pool.rename(columns={'consumercomplaintnarrative':'text'}).copy()
    d['clean']=d.text.map(clean);d=d[d.clean.str.split().str.len().ge(3)].copy()
    d['label']=d['product'].map(lambda x:ALIASES.get(x,x))
    conflict=d.groupby('clean').label.nunique();conflicting=set(conflict[conflict>1].index)
    n_conflict=int(d.clean.isin(conflicting).sum());d=d[~d.clean.isin(conflicting)].drop_duplicates('clean')
    counts=d.label.value_counts();excluded=counts[counts<40].to_dict();d=d[d.label.isin(counts[counts>=40].index)].copy()
    d[['word_count','length_group','style_proxy']]=pd.DataFrame(d.text.map(style).tolist(),index=d.index)
    d['row_key']=d.clean.map(lambda t:hashlib.sha256(t.encode()).hexdigest())
    if d.label.nunique()<2: raise ValueError('Need at least two classes with 40 unique narratives each.')
    return d.reset_index(drop=True),dict(source_rows=raw,eligible_rows=eligible,sample_limit=max_rows,retained=len(d),conflicting_rows_removed=n_conflict,rare_classes_excluded=excluded,unmapped_products=sorted(set(d['product'])-set(ALIASES)))

def scores(y,p):
    return {'n':len(y),'accuracy':float(accuracy_score(y,p)),'macro_f1':float(f1_score(y,p,average='macro',zero_division=0)),'weighted_f1':float(f1_score(y,p,average='weighted',zero_division=0))}

def interval(y,p):
    a=np.asarray(y)==np.asarray(p);rng=np.random.default_rng(SEED)
    return np.quantile([rng.choice(a,len(a),replace=True).mean() for _ in range(1000)],[.025,.975]).tolist()

def audit(test,pred):
    rows=[]
    for column in ['length_group','style_proxy']:
        for group in sorted(test[column].unique()):
            mask=(test[column]==group).to_numpy();y=test.loc[mask,'label'];p=pred[mask]
            s=scores(y,p);s.update(dimension=column,group=group,accuracy_ci_low=interval(y,p)[0],accuracy_ci_high=interval(y,p)[1],small_group=len(y)<30)
            rows.append(s)
    return pd.DataFrame(rows)

def train(args):
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    d,manifest=load_data(args.csv,args.max_rows)
    train,rest=train_test_split(d,test_size=.30,stratify=d.label,random_state=SEED)
    val,test=train_test_split(rest,test_size=.50,stratify=rest.label,random_state=SEED)
    manifest.update(seed=SEED,split_sizes={'train':len(train),'validation':len(val),'test':len(test)},split='70/15/15 stratified random',source_file=Path(args.csv).name)
    sha=hashlib.sha256()
    with open(args.csv,'rb') as f:
        for b in iter(lambda:f.read(1048576),b''):sha.update(b)
    manifest['source_sha256']=sha.hexdigest()
    vector=TfidfVectorizer(ngram_range=(1,2),min_df=2,max_df=.98,max_features=args.max_features,sublinear_tf=True)
    X=vector.fit_transform(train.clean);V=vector.transform(val.clean)
    candidates={'Majority baseline':DummyClassifier(strategy='most_frequent'),
      'Logistic Regression C=1':LogisticRegression(C=1,max_iter=1500,class_weight='balanced',random_state=SEED),
      'Logistic Regression C=4':LogisticRegression(C=4,max_iter=1500,class_weight='balanced',random_state=SEED),
      'Linear SVM C=0.5':LinearSVC(C=.5,class_weight='balanced',random_state=SEED),
      'Linear SVM C=1':LinearSVC(C=1,class_weight='balanced',random_state=SEED),
      'Random Forest':RandomForestClassifier(n_estimators=150,max_depth=40,min_samples_leaf=2,class_weight='balanced_subsample',n_jobs=-1,random_state=SEED)}
    results=[]
    for name,model in candidates.items():
        print('Training:',name,flush=True);model.fit(X,train.label);results.append({'model':name,**scores(val.label,model.predict(V))})
    comparison=pd.DataFrame(results).sort_values('macro_f1',ascending=False);comparison.to_csv(out/'validation_comparison.csv',index=False)
    name=comparison.iloc[0]['model'];model=candidates[name]
    # Keep frozen training-only model: no tuning on the final test set.
    pipeline=Pipeline([('tfidf',vector),('classifier',model)])
    pred=pipeline.predict(test.clean);overall=scores(test.label,pred);overall['selected_model']=name;overall['accuracy_ci']=interval(test.label,pred)
    (out/'test_metrics.json').write_text(json.dumps(overall,indent=2))
    (out/'data_manifest.json').write_text(json.dumps(manifest,indent=2))
    pd.DataFrame(classification_report(test.label,pred,output_dict=True,zero_division=0)).T.to_csv(out/'classification_report.csv')
    a=audit(test,pred);a.to_csv(out/'audit.csv',index=False)
    per=[]
    for dim in ['length_group','style_proxy']:
        for (g,label),part in test.assign(prediction=pred).groupby([dim,'label']):
            per.append({'dimension':dim,'group':g,'label':label,'n':len(part),'recall':float((part.label==part.prediction).mean())})
    pd.DataFrame(per).to_csv(out/'audit_per_class.csv',index=False)
    # Balanced within-class group comparison controls observed product composition.
    controlled=[]
    for dim,g1,g2 in [('length_group','short','long'),('style_proxy','informal_short_proxy','formal_detailed_proxy')]:
        t=test.assign(prediction=pred);rng=np.random.default_rng(SEED);pairs=[]
        for label in sorted(t.label.unique()):
            u=t[(t.label==label)&(t[dim]==g1)];v=t[(t.label==label)&(t[dim]==g2)];n=min(len(u),len(v))
            if n>=5:
                u=u.iloc[rng.choice(len(u),n,replace=False)];v=v.iloc[rng.choice(len(v),n,replace=False)]
                pairs.append((label,n,float((u.label==u.prediction).mean()),float((v.label==v.prediction).mean())))
        controlled.append({'dimension':dim,'group_1':g1,'group_2':g2,'eligible_classes':len(pairs),'class_results':pairs,'equal_class_accuracy_gap_group2_minus_group1':float(np.mean([p[3]-p[2] for p in pairs])) if pairs else None})
    (out/'controlled_audit.json').write_text(json.dumps(controlled,indent=2))
    test_out=test[['row_key','text','label','word_count','length_group','style_proxy']].assign(prediction=pred)
    test_out.to_csv(out/'test_predictions.csv',index=False)
    test_out[test_out.label!=test_out.prediction].head(100).to_csv(out/'error_examples.csv',index=False)
    for split,part in [('train',train),('validation',val),('test',test)]:
        part[['row_key','label']].assign(split=split).to_csv(out/f'{split}_ids.csv',index=False)
    annotation=test_out.groupby('length_group',group_keys=False).sample(frac=.2,random_state=SEED).head(300)[['row_key','text','label','word_count']]
    for col in ['rater_1_formality','rater_2_formality','adjudicated_formality','detail_level']:annotation[col]=''
    annotation.to_csv(out/'human_audit_template.csv',index=False)
    joblib.dump(pipeline,out/'model.joblib')
    # Linear coefficient explanations are exact additive score contributions, not SHAP.
    features=vector.get_feature_names_out()
    linear=model if hasattr(model,'coef_') else candidates['Logistic Regression C=1']
    coefs=linear.coef_ if len(linear.classes_)>2 else np.vstack([-linear.coef_[0],linear.coef_[0]])
    terms=[]
    for label,coef in zip(linear.classes_,coefs):
        for idx in np.argsort(coef)[-20:][::-1]:terms.append({'class':label,'term':features[idx],'coefficient':float(coef[idx]),'explanation_model':name if linear is model else 'Logistic Regression C=1 (separate explanatory model)'})
    pd.DataFrame(terms).to_csv(out/'top_features.csv',index=False)
    fig,ax=plt.subplots(figsize=(12,10));ConfusionMatrixDisplay.from_predictions(test.label,pred,normalize='true',xticks_rotation=90,ax=ax,colorbar=False,values_format='.2f');fig.tight_layout();fig.savefig(out/'confusion_matrix.png',dpi=150);plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,5));ax.barh(comparison.model,comparison.macro_f1);ax.set(xlabel='Validation macro-F1',xlim=(0,1));fig.tight_layout();fig.savefig(out/'model_comparison.png',dpi=150);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4));sub=a[a.dimension=='length_group'];ax.bar(sub.group,sub.accuracy);ax.set(ylim=(0,1),ylabel='Test accuracy',title='Length audit — not a formality measure');fig.tight_layout();fig.savefig(out/'length_audit.png',dpi=150);plt.close(fig)
    report=f'''# Experimental results\n\nSelected using validation macro-F1: **{name}**.\n\nTraining / validation / test rows: {len(train)} / {len(val)} / {len(test)}.\n\nFinal test accuracy: **{overall['accuracy']:.2%}**; macro-F1: **{overall['macro_f1']:.4f}**; weighted-F1: **{overall['weighted_f1']:.4f}**.\n\nAccuracy bootstrap 95% interval: {overall['accuracy_ci'][0]:.2%}–{overall['accuracy_ci'][1]:.2%}.\n\n## Audit\n\n'''
    for row in a.to_dict('records'):report+=f"- {row['group']}: n={row['n']}, accuracy={row['accuracy']:.2%}, macro-F1={row['macro_f1']:.4f}. {'Small group: interpret cautiously.' if row['small_group'] else ''}\n"
    report+='\nProxy groups are heuristic. A difference is an association, not evidence that formality causes errors. Complete the human audit before making a validated formality claim. See controlled_audit.json for equal-class comparisons and audit_per_class.csv for coverage.\n'
    (out/'RESULTS.md').write_text(report)
    print(json.dumps(overall,indent=2))

def predict(args):
    model=joblib.load(args.model);t=clean(args.text)
    if len(t.split())<3:raise ValueError('Enter at least three meaningful words.')
    v=model.named_steps['tfidf'].transform([t])
    if v.nnz==0:raise ValueError('No known vocabulary: route to human review.')
    label=model.predict([t])[0];print('Predicted category:',label)
    clf=model.named_steps['classifier']
    if hasattr(clf,'coef_'):
        idx=list(clf.classes_).index(label);coef=clf.coef_[idx] if len(clf.classes_)>2 else clf.coef_[0]*(1 if idx else -1)
        contributions=v.multiply(coef).toarray()[0];terms=model.named_steps['tfidf'].get_feature_names_out()
        print('Largest positive score contributions (not probabilities):')
        for i in np.argsort(contributions)[-10:][::-1]:
            if contributions[i]>0:print(terms[i],round(float(contributions[i]),4))

def human_audit(args):
    a=pd.read_csv(args.annotations);p=pd.read_csv(args.predictions)
    from sklearn.metrics import cohen_kappa_score
    both=a.dropna(subset=['rater_1_formality','rater_2_formality'])
    print('Double-rated rows:',len(both))
    if len(both):print('Cohen kappa:',cohen_kappa_score(both.rater_1_formality,both.rater_2_formality))
    m=p.merge(a[['row_key','adjudicated_formality','detail_level']],on='row_key',validate='one_to_one')
    groups={'formal_detailed':m.adjudicated_formality.eq('formal')&m.detail_level.eq('detailed'), 'informal_short':m.adjudicated_formality.eq('informal')&m.word_count.le(50)}
    for name,mask in groups.items():
        s=m[mask];print(name,scores(s.label,s.prediction) if len(s) else 'No eligible annotated rows')

if __name__=='__main__':
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='command',required=True)
    t=sub.add_parser('train');t.add_argument('--csv',required=True);t.add_argument('--output',default='outputs');t.add_argument('--max-rows',type=int,default=30000);t.add_argument('--max-features',type=int,default=30000);t.set_defaults(func=train)
    t=sub.add_parser('predict');t.add_argument('--model',default='outputs/model.joblib');t.add_argument('--text',required=True);t.set_defaults(func=predict)
    t=sub.add_parser('human-audit');t.add_argument('--annotations',default='outputs/human_audit_template.csv');t.add_argument('--predictions',default='outputs/test_predictions.csv');t.set_defaults(func=human_audit)
    a=p.parse_args();a.func(a)
